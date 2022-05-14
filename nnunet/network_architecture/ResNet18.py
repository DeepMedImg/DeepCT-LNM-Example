from torchvision.models import resnet18
import torch
import numpy as np
import torchvision
from nnunet.utilities.to_torch import to_cuda, maybe_to_torch
from typing import Union, Tuple, List
from torch import nn
from nnunet.network_architecture.neural_network import NeuralNetwork, ConvNormNonlinPool
from nnunet.network_architecture.custom_modules.encoding import Encoding, View, Normalize
from nnunet.network_architecture.initialization import InitWeights_SideBranch_Addition, InitWeights_SideBranch_Multiplication


def get_bbox_from_mask(mask, inside_value=1):
    mask_voxel_coords = torch.where(mask >= inside_value)
    minxidx = int(torch.min(mask_voxel_coords[0]))
    maxxidx = int(torch.max(mask_voxel_coords[0])) + 1
    minyidx = int(torch.min(mask_voxel_coords[1]))
    maxyidx = int(torch.max(mask_voxel_coords[1])) + 1
    return np.array([[minxidx, maxxidx], [minyidx, maxyidx]])

class ResNet18RoIPool(NeuralNetwork):
    def __init__(self, num_classes, pretrained, if_freeze=None, roi_margin_min=None, roi_margin_max=None, freeze_layernum=3):
        super(ResNet18RoIPool, self).__init__()
        self.num_classes = num_classes
        self.roi_margin_min = roi_margin_min
        self.roi_margin_max = roi_margin_max
        self.do_ds = False
        self.conv_op = nn.Conv2d

        backbone = resnet18(pretrained=True) if pretrained else resnet18(pretrained=False)
        if if_freeze:
            for name, p in backbone.named_parameters():
                if freeze_layernum == 4:
                    if name.startswith('conv1') or name.startswith('bn1') or name.startswith(
                            'layer1') or name.startswith('layer2') or name.startswith('layer3'):
                        p.requires_grad = False

                elif freeze_layernum == 3:
                    if name.startswith('conv1') or name.startswith('bn1') or name.startswith(
                            'layer1') or name.startswith('layer2'):
                        p.requires_grad = False
                elif freeze_layernum == 2:
                    if name.startswith('conv1') or name.startswith('bn1') or name.startswith('layer1'):
                        p.requires_grad = False
                elif freeze_layernum == 1:
                    if name.startswith('conv1') or name.startswith('bn1'):
                        p.requires_grad = False
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # self.global_avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.global_avg_pool = torchvision.ops.RoIAlign(output_size=1, spatial_scale=1. / 32., sampling_ratio=-1, aligned=True)
        self.classifier = torch.nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, x, roi=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        batch_idx = torch.arange(0, x.size(0)).to(x.device).view(-1, 1)
        # roi = torch.cat((batch_idx, roi), dim=1)
        roi = torch.cat((batch_idx, roi[:, 0, 0].view(-1, 1), roi[:, 1, 0].view(-1, 1), roi[:, 0, 1].view(-1, 1), roi[:, 1, 1].view(-1, 1)), dim=1)
        x = self.global_avg_pool(x, roi.type_as(x)).view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def predict_2D(self, x, do_mirroring: bool, mirror_axes: tuple = (0, 1, 2), patch_size: tuple = None, regions_class_order: tuple = None,
                   use_gaussian: bool = False, pad_border_mode: str = "constant",
                   pad_kwargs: dict = None, all_in_gpu: bool = False,
                   verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use this function to predict a 2D image. If this is a 3D U-Net it will crash because you cannot predict a 2D
        image with that (you dummy).

        When running predictions, you need to specify whether you want to run fully convolutional of sliding window
        based inference. We very strongly recommend you use sliding window with the default settings.

        It is the responsibility of the user to make sure the network is in the proper mode (eval for inference!). If
        the network is not in eval mode it will print a warning.

        :param x: Your input data. Must be a nd.ndarray of shape (c, x, y).
        :param do_mirroring: If True, use test time data augmentation in the form of mirroring
        :param mirror_axes: Determines which axes to use for mirroing. Per default, mirroring is done along all three
        axes
        :param use_sliding_window: if True, run sliding window prediction. Heavily recommended! This is also the default
        :param step_size: When running sliding window prediction, the step size determines the distance between adjacent
        predictions. The smaller the step size, the denser the predictions (and the longer it takes!). Step size is given
        as a fraction of the patch_size. 0.5 is the default and means that wen advance by patch_size * 0.5 between
        predictions. step_size cannot be larger than 1!
        :param patch_size: The patch size that was used for training the network. Do not use different patch sizes here,
        this will either crash or give potentially less accurate segmentations
        :param regions_class_order: Fabian only
        :param use_gaussian: (Only applies to sliding window prediction) If True, uses a Gaussian importance weighting
         to weigh predictions closer to the center of the current patch higher than those at the borders. The reason
         behind this is that the segmentation accuracy decreases towards the borders. Default (and recommended): True
        :param pad_border_mode: leave this alone
        :param pad_kwargs: leave this alone
        :param all_in_gpu: experimental. You probably want to leave this as is it
        :param verbose: Do you want a wall of text? If yes then set this to True
        :return:
        """

        if self.conv_op == nn.Conv3d:
            raise RuntimeError("Cannot predict 2d if the network is 3d. Dummy.")

        if verbose: print("debug: mirroring", do_mirroring, "mirror_axes", mirror_axes)

        assert self.get_device() != "cpu", "CPU not implemented"

        if pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        # A very long time ago the mirror axes were (2, 3) for a 2d network. This is just to intercept any old
        # code that uses this convention
        if len(mirror_axes):
            if max(mirror_axes) > 1:
                raise ValueError("mirror axes. duh")

        if self.training:
            print('WARNING! Network is in train mode during inference. This may be intended, or not...')

        assert len(x.shape) == 3, "data must have shape (c,x,y)"

        if self.conv_op == nn.Conv2d:
            res = self._internal_predict_2D_2Dconv(x, patch_size, do_mirroring, mirror_axes, regions_class_order,
                                                       pad_border_mode, pad_kwargs, verbose)
        else:
            raise RuntimeError("Invalid conv op, cannot determine what dimensionality (2d/3d) the network is")

        return res

    def _internal_predict_2D_2Dconv(self, x: np.ndarray, patch_size: Tuple[int, int], do_mirroring: bool,
                                    mirror_axes: tuple = (0, 1, 2), regions_class_order: tuple = None,
                                    pad_border_mode: str = "constant", pad_kwargs: dict = None,
                                    verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:


        if verbose: print("do mirror:", do_mirroring)
        margins = np.array([np.mean([margin_min, margin_max]) for margin_min, margin_max in zip(self.roi_margin_min, self.roi_margin_max)])
        shape = x.shape[1:]
        with torch.no_grad():
            if not np.all(x.shape[1:] == patch_size):
                center_voxel = [x.shape[1] // 2, x.shape[2] // 2]
                x = x[:, center_voxel[0] - patch_size[0] // 2:center_voxel[0] + patch_size[0] // 2 +patch_size[0] % 2,
                                center_voxel[1] - patch_size[1] // 2:center_voxel[1] + patch_size[1] // 2 + patch_size[1] % 2][None]
                with torch.no_grad():
                    x = to_cuda(maybe_to_torch(x), gpu_id=self.get_device())
                    result_torch = torch.zeros([1, self.num_classes], dtype=torch.float).cuda(self.get_device(), non_blocking=True)

                    if do_mirroring:
                        mirror_idx = 4
                        num_results = 2 ** len(mirror_axes)
                    else:
                        mirror_idx = 1
                        num_results = 1

                    for m in range(mirror_idx):
                        if m == 0:
                            bbox_of_nonzero_class = get_bbox_from_mask(x[0, -1], 1)
                            bbox_of_nonzero_class_plus_margin = np.array(bbox_of_nonzero_class) + np.stack([-margins, margins], axis=1)
                            roi = np.array([[int(max(0, minidx)), int(min(shape[idx], maxidx))] for idx, (minidx, maxidx) in
                                            enumerate(bbox_of_nonzero_class_plus_margin)])
                            roi = to_cuda(maybe_to_torch(roi[None]), gpu_id=self.get_device())
                            pred= self.inference_apply_nonlin(self(x, roi))
                            result_torch += 1 / num_results * pred

                        if m == 1 and (1 in mirror_axes):
                            bbox_of_nonzero_class = get_bbox_from_mask(torch.flip(x, (3,))[0, -1], 1)
                            bbox_of_nonzero_class_plus_margin = np.array(bbox_of_nonzero_class) + np.stack(
                                [-margins, margins], axis=1)
                            roi = np.array(
                                [[int(max(0, minidx)), int(min(shape[idx], maxidx))] for idx, (minidx, maxidx) in
                                 enumerate(bbox_of_nonzero_class_plus_margin)])
                            roi = to_cuda(maybe_to_torch(roi[None]), gpu_id=self.get_device())

                            pred= self.inference_apply_nonlin(self(torch.flip(x, (3,)), roi))
                            result_torch += 1 / num_results * pred

                        if m == 2 and (0 in mirror_axes):
                            bbox_of_nonzero_class = get_bbox_from_mask(torch.flip(x, (2,))[0, -1], 1)
                            bbox_of_nonzero_class_plus_margin = np.array(bbox_of_nonzero_class) + np.stack(
                                [-margins, margins], axis=1)
                            roi = np.array(
                                [[int(max(0, minidx)), int(min(shape[idx], maxidx))] for idx, (minidx, maxidx) in
                                 enumerate(bbox_of_nonzero_class_plus_margin)])
                            roi = to_cuda(maybe_to_torch(roi[None]), gpu_id=self.get_device())

                            pred= self.inference_apply_nonlin(self(torch.flip(x, (2,)), roi))
                            result_torch += 1 / num_results * pred

                        if m == 3 and (0 in mirror_axes) and (1 in mirror_axes):
                            bbox_of_nonzero_class = get_bbox_from_mask(torch.flip(x, (3, 2))[0, -1], 1)
                            bbox_of_nonzero_class_plus_margin = np.array(bbox_of_nonzero_class) + np.stack(
                                [-margins, margins], axis=1)
                            roi = np.array(
                                [[int(max(0, minidx)), int(min(shape[idx], maxidx))] for idx, (minidx, maxidx) in
                                 enumerate(bbox_of_nonzero_class_plus_margin)])
                            roi = to_cuda(maybe_to_torch(roi[None]), gpu_id=self.get_device())

                            pred= self.inference_apply_nonlin(self(torch.flip(x, (3, 2)), roi))
                            result_torch += 1 / num_results * pred

        return result_torch

class ResNet18(ResNet18RoIPool):
    def __init__(self, num_classes, pretrained, if_freeze=None, roi_margin_min=None, roi_margin_max=None, freeze_layernum=3):
        super(ResNet18RoIPool, self).__init__()
        self.num_classes = num_classes
        self.roi_margin_min = roi_margin_min
        self.roi_margin_max = roi_margin_max
        self.do_ds = False
        self.conv_op = nn.Conv2d

        backbone = resnet18(pretrained=True) if pretrained else resnet18(pretrained=False)
        if if_freeze:
            for name, p in backbone.named_parameters():
                if freeze_layernum == 4:
                    if name.startswith('conv1') or name.startswith('bn1') or name.startswith(
                            'layer1') or name.startswith('layer2') or name.startswith('layer3'):
                        p.requires_grad = False

                elif freeze_layernum == 3:
                    if name.startswith('conv1') or name.startswith('bn1') or name.startswith(
                            'layer1') or name.startswith('layer2'):
                        p.requires_grad = False
                elif freeze_layernum == 2:
                    if name.startswith('conv1') or name.startswith('bn1') or name.startswith('layer1'):
                        p.requires_grad = False
                elif freeze_layernum == 1:
                    if name.startswith('conv1') or name.startswith('bn1'):
                        p.requires_grad = False
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.global_avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.global_avg_pool = torchvision.ops.RoIAlign(output_size=1, spatial_scale=1. / 32., sampling_ratio=-1, aligned=True)
        self.classifier = torch.nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, x, roi=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_avg_pool(x).view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ResNet18_DeepTEN(ResNet18RoIPool):
    def __init__(self, num_classes, pretrained, if_freeze=None, roi_margin_min=None, roi_margin_max=None, freeze_layernum=3,
                 n_codes=8, len_codes=64, resnet_blocks=4):
        super(ResNet18RoIPool, self).__init__()
        self.num_classes = num_classes
        self.roi_margin_min = roi_margin_min
        self.roi_margin_max = roi_margin_max
        self.do_ds = False
        self.conv_op = nn.Conv2d
        self.resnet_blocks = resnet_blocks

        backbone = resnet18(pretrained=True) if pretrained else resnet18(pretrained=False)
        if if_freeze:
            for name, p in backbone.named_parameters():
                if freeze_layernum == 4:
                    if name.startswith('conv1') or name.startswith('bn1') or name.startswith(
                            'layer1') or name.startswith('layer2') or name.startswith('layer3'):
                        p.requires_grad = False

                elif freeze_layernum == 3:
                    if name.startswith('conv1') or name.startswith('bn1') or name.startswith(
                            'layer1') or name.startswith('layer2'):
                        p.requires_grad = False
                elif freeze_layernum == 2:
                    if name.startswith('conv1') or name.startswith('bn1') or name.startswith('layer1'):
                        p.requires_grad = False
                elif freeze_layernum == 1:
                    if name.startswith('conv1') or name.startswith('bn1'):
                        p.requires_grad = False
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        ### deepten
        self.maskpool0 = nn.MaxPool2d(2)
        self.maskpool1 = nn.MaxPool2d(2)
        self.maskpool2 = nn.MaxPool2d(2)
        self.maskpool3 = nn.MaxPool2d(2)
        self.maskpool4 = nn.MaxPool2d(2)

        if self.resnet_blocks == 1:
            num_channel = self.layer2[0].conv1.in_channels
        elif self.resnet_blocks == 2:
            num_channel = self.layer3[0].conv1.in_channels
        elif self.resnet_blocks == 3:
            num_channel = self.layer4[0].conv1.in_channels
        elif self.resnet_blocks == 4:
            num_channel = backbone.fc.in_features
        else:
            raise ValueError

        self.head = nn.Sequential(
            nn.Conv2d(num_channel, len_codes, 1),
            nn.BatchNorm2d(len_codes),
            nn.ReLU(inplace=True),
        )
        self.encoder = Encoding(D=len_codes, K=n_codes)
        self.viewer = View(-1, len_codes * n_codes)
        self.normalizer = Normalize()

        self.global_avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.classifier = nn.Sequential(
        #     nn.Linear(num_channel + len_codes * n_codes, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, num_class)
        # )
        self.classifier = nn.Sequential(
            nn.Linear(num_channel + len_codes * n_codes, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def deepten_feature(self, x, m):
        # Extract DeepTEN feature
        x = self.head(x)
        x = self.encoder(x, m)
        x = self.viewer(x)
        x = self.normalizer(x)
        return x

    def forward(self, x, m=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        m1, m2, m3, m4 = None, None, None, None
        if m is not None:
            m = self.maskpool0(m)
            m1 = self.maskpool1(m)
            m2 = self.maskpool2(m1)
            m3 = self.maskpool3(m2)
            m4 = self.maskpool4(m3)

        # Extracte deepten feature
        if self.resnet_blocks == 1:
            texture_feature = self.deepten_feature(f1, m1)
            conv_feature = self.global_avg_pool(f1).view(x.size(0), -1)
        elif self.resnet_blocks == 2:
            texture_feature = self.deepten_feature(f2, m2)
            conv_feature = self.global_avg_pool(f2).view(x.size(0), -1)
        elif self.resnet_blocks == 3:
            texture_feature = self.deepten_feature(f3, m3)
            conv_feature = self.global_avg_pool(f3).view(x.size(0), -1)
        elif self.resnet_blocks == 4:
            texture_feature = self.deepten_feature(f4, m4)
            conv_feature = self.global_avg_pool(f4).view(x.size(0), -1)
        else:
            raise ValueError
        total_feature = torch.cat([texture_feature, conv_feature], dim=-1)
        y = self.classifier(total_feature)
        return y

    def _internal_predict_2D_2Dconv(self, x: np.ndarray, patch_size: Tuple[int, int], do_mirroring: bool,
                                    mirror_axes: tuple = (0, 1, 2), regions_class_order: tuple = None,
                                    pad_border_mode: str = "constant", pad_kwargs: dict = None,
                                    verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:


        if verbose: print("do mirror:", do_mirroring)
        margins = np.array([np.mean([margin_min, margin_max]) for margin_min, margin_max in zip(self.roi_margin_min, self.roi_margin_max)])
        shape = x.shape[1:]
        with torch.no_grad():
            if not np.all(x.shape[1:] == patch_size):
                center_voxel = [x.shape[1] // 2, x.shape[2] // 2]
                x = x[:, center_voxel[0] - patch_size[0] // 2:center_voxel[0] + patch_size[0] // 2 +patch_size[0] % 2,
                                center_voxel[1] - patch_size[1] // 2:center_voxel[1] + patch_size[1] // 2 + patch_size[1] % 2][None]
                with torch.no_grad():
                    x = to_cuda(maybe_to_torch(x), gpu_id=self.get_device())
                    result_torch = torch.zeros([1, self.num_classes], dtype=torch.float).cuda(self.get_device(), non_blocking=True)

                    if do_mirroring:
                        mirror_idx = 4
                        num_results = 2 ** len(mirror_axes)
                    else:
                        mirror_idx = 1
                        num_results = 1

                    for m in range(mirror_idx):
                        if m == 0:
                            pred= self.inference_apply_nonlin(self(x, x[:, -1:]))
                            result_torch += 1 / num_results * pred

                        if m == 1 and (1 in mirror_axes):

                            pred= self.inference_apply_nonlin(self(torch.flip(x, (3,)), torch.flip(x, (3,))[:, -1:]))
                            result_torch += 1 / num_results * pred

                        if m == 2 and (0 in mirror_axes):

                            pred= self.inference_apply_nonlin(self(torch.flip(x, (2,)), torch.flip(x, (2,))[:, -1:]))
                            result_torch += 1 / num_results * pred

                        if m == 3 and (0 in mirror_axes) and (1 in mirror_axes):

                            pred= self.inference_apply_nonlin(self(torch.flip(x, (3, 2)), torch.flip(x, (3, 2))[:, -1:]))
                            result_torch += 1 / num_results * pred

        return result_torch


class ResNet18_DeepTEN_MaskSideBranch(ResNet18_DeepTEN):
    def __init__(self, num_classes, pretrained, if_freeze=None, mask_input_layer=None, mask_input_method=None, roi_margin_min=None, roi_margin_max=None,
                 freeze_layernum=3, n_codes=8, len_codes=64, resnet_blocks=4, incorp_CT_N=False, incorp_CT_N_method='v1', incorp_other_info=None):
        super().__init__(num_classes, pretrained, if_freeze, roi_margin_min, roi_margin_max, freeze_layernum, n_codes, len_codes, resnet_blocks)
        self.mask_input_layer = mask_input_layer
        self.mask_input_method = mask_input_method
        self.incorp_CT_N = incorp_CT_N
        self.incorp_CT_N_method = incorp_CT_N_method
        self.incorp_other_info = incorp_other_info

        weightInitializer = None
        if mask_input_method == 'addition':
            weightInitializer = InitWeights_SideBranch_Addition()
        elif mask_input_method == 'multiplication':
            weightInitializer = InitWeights_SideBranch_Multiplication()

        self.mask_sidebranch_list = []
        input_features, output_features = 1, self.conv1.out_channels
        for i in range(max(mask_input_layer)):
            self.mask_sidebranch_list.append(ConvNormNonlinPool(input_features, output_features))
            input_features = output_features
            if i == 0:
                output_features = self.layer2[0].conv1.in_channels
            elif i == 1:
                output_features = self.layer3[0].conv1.in_channels
            elif i == 2:
                output_features = self.layer4[0].conv1.in_channels
            elif i == 3:
                output_features = self.layer4[-1].conv2.out_channels
            else:
                raise ValueError
        if incorp_CT_N:
            if self.incorp_other_info is not None:
                extra_channel_num = 2 + len(self.incorp_other_info)
            else:
                extra_channel_num = 1
            if incorp_CT_N_method == 'v1':
                self.classifier = nn.Sequential(
                    nn.BatchNorm1d(self.layer4[-1].conv2.out_channels + len_codes * n_codes + extra_channel_num),
                    nn.Linear(self.layer4[-1].conv2.out_channels + len_codes * n_codes + 1, 64),
                    nn.ReLU(),
                    nn.Linear(64, num_classes)
                )
            elif incorp_CT_N_method == 'v2':
                self.ctn_embedding = nn.Sequential(
                    nn.BatchNorm1d(extra_channel_num),
                    nn.Linear(extra_channel_num, self.layer4[-1].conv2.out_channels + len_codes * n_codes),
                    nn.ReLU())

                self.classifier = nn.Sequential(
                    nn.Linear(self.layer4[-1].conv2.out_channels + len_codes * n_codes, 64),
                    nn.ReLU(),
                    nn.Linear(64, num_classes)
                )
        self.mask_sidebranch_list = nn.ModuleList(self.mask_sidebranch_list)

        if weightInitializer is not None:
            self.mask_sidebranch_list.apply(weightInitializer)

    def forward(self, x, m=None, n = None, return_feature=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if max(self.mask_input_layer) >= 1:
            m_side = self.mask_sidebranch_list[0](m)
        if 1 in self.mask_input_layer:
            if self.mask_input_method == 'addition':
                x = x + m_side
            elif self.mask_input_method == 'multiplication':
                x = x * m_side
        x = self.maxpool(x)
        f1 = self.layer1(x)
        if max(self.mask_input_layer) >= 2:
            m_side = self.mask_sidebranch_list[1](m_side)
        if 2 in self.mask_input_layer:
            if self.mask_input_method == 'addition':
                f1 = f1 + m_side
            elif self.mask_input_method == 'multiplication':
                f1 = f1 * m_side

        f2 = self.layer2(f1)
        if max(self.mask_input_layer) >= 3:
            m_side = self.mask_sidebranch_list[2](m_side)
        if 3 in self.mask_input_layer:
            if self.mask_input_method == 'addition':
                f2 = f2 + m_side
            elif self.mask_input_method == 'multiplication':
                f2 = f2 * m_side

        f3 = self.layer3(f2)
        if max(self.mask_input_layer) >= 4:
            m_side = self.mask_sidebranch_list[3](m_side)
        if 4 in self.mask_input_layer:
            if self.mask_input_method == 'addition':
                f3 = f3 + m_side
            elif self.mask_input_method == 'multiplication':
                f3 = f3 * m_side
        f4 = self.layer4(f3)
        if max(self.mask_input_layer) >= 5:
            m_side = self.mask_sidebranch_list[4](m_side)
        if 5 in self.mask_input_layer:
            if self.mask_input_method == 'addition':
                f4 = f4 + m_side
            elif self.mask_input_method == 'multiplication':
                f4 = f4 * m_side


        m1, m2, m3, m4 = None, None, None, None
        if m is not None:
            m = self.maskpool0(m)
            m1 = self.maskpool1(m)
            m2 = self.maskpool2(m1)
            m3 = self.maskpool3(m2)
            m4 = self.maskpool4(m3)

        # Extracte deepten feature
        if self.resnet_blocks == 1:
            texture_feature = self.deepten_feature(f1, m1)
            conv_feature = self.global_avg_pool(f1).view(x.size(0), -1)
        elif self.resnet_blocks == 2:
            texture_feature = self.deepten_feature(f2, m2)
            conv_feature = self.global_avg_pool(f2).view(x.size(0), -1)
        elif self.resnet_blocks == 3:
            texture_feature = self.deepten_feature(f3, m3)
            conv_feature = self.global_avg_pool(f3).view(x.size(0), -1)
        elif self.resnet_blocks == 4:
            texture_feature = self.deepten_feature(f4, m4)
            conv_feature = self.global_avg_pool(f4).view(x.size(0), -1)
        else:
            raise ValueError
        if n is not None:
            if self.incorp_CT_N_method == 'v1':
                total_feature = torch.cat([texture_feature, conv_feature, n.view(x.size(0), -1)], dim=-1)
            elif self.incorp_CT_N_method == 'v2':
                n = self.ctn_embedding( n.view(x.size(0), -1))
                total_feature = torch.cat([texture_feature, conv_feature], dim=-1)
                total_feature = total_feature + n
        else:
            total_feature = torch.cat([texture_feature, conv_feature], dim=-1)
        if return_feature:
            for i in range(len(self.classifier)):
                total_feature = self.classifier[i](total_feature)
                if i == 1:
                    return total_feature
        else:
            y = self.classifier(total_feature)
            return y

    def predict_2D(self, x, do_mirroring: bool, mirror_axes: tuple = (0, 1, 2), patch_size: tuple = None, regions_class_order: tuple = None,
                   use_gaussian: bool = False, pad_border_mode: str = "constant",
                   pad_kwargs: dict = None, all_in_gpu: bool = False,
                   verbose: bool = True, CT_N: int = None, return_feature: bool=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use this function to predict a 2D image. If this is a 3D U-Net it will crash because you cannot predict a 2D
        image with that (you dummy).

        When running predictions, you need to specify whether you want to run fully convolutional of sliding window
        based inference. We very strongly recommend you use sliding window with the default settings.

        It is the responsibility of the user to make sure the network is in the proper mode (eval for inference!). If
        the network is not in eval mode it will print a warning.

        :param x: Your input data. Must be a nd.ndarray of shape (c, x, y).
        :param do_mirroring: If True, use test time data augmentation in the form of mirroring
        :param mirror_axes: Determines which axes to use for mirroing. Per default, mirroring is done along all three
        axes
        :param use_sliding_window: if True, run sliding window prediction. Heavily recommended! This is also the default
        :param step_size: When running sliding window prediction, the step size determines the distance between adjacent
        predictions. The smaller the step size, the denser the predictions (and the longer it takes!). Step size is given
        as a fraction of the patch_size. 0.5 is the default and means that wen advance by patch_size * 0.5 between
        predictions. step_size cannot be larger than 1!
        :param patch_size: The patch size that was used for training the network. Do not use different patch sizes here,
        this will either crash or give potentially less accurate segmentations
        :param regions_class_order: Fabian only
        :param use_gaussian: (Only applies to sliding window prediction) If True, uses a Gaussian importance weighting
         to weigh predictions closer to the center of the current patch higher than those at the borders. The reason
         behind this is that the segmentation accuracy decreases towards the borders. Default (and recommended): True
        :param pad_border_mode: leave this alone
        :param pad_kwargs: leave this alone
        :param all_in_gpu: experimental. You probably want to leave this as is it
        :param verbose: Do you want a wall of text? If yes then set this to True
        :return:
        """

        if self.conv_op == nn.Conv3d:
            raise RuntimeError("Cannot predict 2d if the network is 3d. Dummy.")

        if verbose: print("debug: mirroring", do_mirroring, "mirror_axes", mirror_axes)

        assert self.get_device() != "cpu", "CPU not implemented"

        if pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        # A very long time ago the mirror axes were (2, 3) for a 2d network. This is just to intercept any old
        # code that uses this convention
        if len(mirror_axes):
            if max(mirror_axes) > 1:
                raise ValueError("mirror axes. duh")

        if self.training:
            print('WARNING! Network is in train mode during inference. This may be intended, or not...')

        assert len(x.shape) == 3, "data must have shape (c,x,y)"

        if self.conv_op == nn.Conv2d:
            res = self._internal_predict_2D_2Dconv(x, patch_size, do_mirroring, mirror_axes, regions_class_order,
                                                       pad_border_mode, pad_kwargs, verbose, CT_N, return_feature)
        else:
            raise RuntimeError("Invalid conv op, cannot determine what dimensionality (2d/3d) the network is")

        return res

    def _internal_predict_2D_2Dconv(self, x: np.ndarray, patch_size: Tuple[int, int], do_mirroring: bool,
                                    mirror_axes: tuple = (0, 1, 2), regions_class_order: tuple = None,
                                    pad_border_mode: str = "constant", pad_kwargs: dict = None,
                                    verbose: bool = True, CT_N: int = None, return_feature: bool=False) -> Tuple[np.ndarray, np.ndarray]:


        if verbose: print("do mirror:", do_mirroring)
        margins = np.array([np.mean([margin_min, margin_max]) for margin_min, margin_max in zip(self.roi_margin_min, self.roi_margin_max)])
        shape = x.shape[1:]
        with torch.no_grad():
            if not np.all(x.shape[1:] == patch_size):
                center_voxel = [x.shape[1] // 2, x.shape[2] // 2]
                x = x[:, center_voxel[0] - patch_size[0] // 2:center_voxel[0] + patch_size[0] // 2 +patch_size[0] % 2,
                                center_voxel[1] - patch_size[1] // 2:center_voxel[1] + patch_size[1] // 2 + patch_size[1] % 2][None]
                with torch.no_grad():
                    x = to_cuda(maybe_to_torch(x), gpu_id=self.get_device())
                    if CT_N is not None:
                        CT_N =to_cuda(maybe_to_torch(np.array([CT_N])), gpu_id=self.get_device())
                    if return_feature:
                        result_torch = torch.zeros([1, 64], dtype=torch.float).cuda(self.get_device(), non_blocking=True)
                    else:
                        result_torch = torch.zeros([1, self.num_classes], dtype=torch.float).cuda(self.get_device(), non_blocking=True)

                    if do_mirroring:
                        mirror_idx = 4
                        num_results = 2 ** len(mirror_axes)
                    else:
                        mirror_idx = 1
                        num_results = 1

                    for m in range(mirror_idx):
                        if m == 0:
                            pred= self.inference_apply_nonlin(self(x, x[:, -1:], CT_N, return_feature))
                            result_torch += 1 / num_results * pred

                        if m == 1 and (1 in mirror_axes):

                            pred= self.inference_apply_nonlin(self(torch.flip(x, (3,)), torch.flip(x, (3,))[:, -1:], CT_N, return_feature))
                            result_torch += 1 / num_results * pred

                        if m == 2 and (0 in mirror_axes):

                            pred= self.inference_apply_nonlin(self(torch.flip(x, (2,)), torch.flip(x, (2,))[:, -1:], CT_N, return_feature))
                            result_torch += 1 / num_results * pred

                        if m == 3 and (0 in mirror_axes) and (1 in mirror_axes):

                            pred= self.inference_apply_nonlin(self(torch.flip(x, (3, 2)), torch.flip(x, (3, 2))[:, -1:], CT_N, return_feature))
                            result_torch += 1 / num_results * pred

        return result_torch

class ResNet18_MaskSideBranch(ResNet18_DeepTEN):
    def __init__(self, num_classes, pretrained, if_freeze=None, mask_input_layer=None, mask_input_method=None, roi_margin_min=None, roi_margin_max=None,
                 freeze_layernum=3, n_codes=8, len_codes=64, resnet_blocks=4):
        super().__init__(num_classes, pretrained, if_freeze, roi_margin_min, roi_margin_max, freeze_layernum, n_codes, len_codes, resnet_blocks)
        self.mask_input_layer = mask_input_layer
        self.mask_input_method = mask_input_method

        weightInitializer = None
        if mask_input_method == 'addition':
            weightInitializer = InitWeights_SideBranch_Addition()
        elif mask_input_method == 'multiplication':
            weightInitializer = InitWeights_SideBranch_Multiplication()

        self.mask_sidebranch_list = []
        input_features, output_features = 1, self.conv1.out_channels
        for i in range(max(mask_input_layer)):
            self.mask_sidebranch_list.append(ConvNormNonlinPool(input_features, output_features))
            input_features = output_features
            if i == 0:
                output_features = self.layer2[0].conv1.in_channels
            elif i == 1:
                output_features = self.layer3[0].conv1.in_channels
            elif i == 2:
                output_features = self.layer4[0].conv1.in_channels
            elif i == 3:
                output_features = self.layer4[-1].conv2.out_channels
            else:
                raise ValueError
        self.classifier = nn.Sequential(
            nn.Linear(self.layer4[-1].conv2.out_channels, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.mask_sidebranch_list = nn.ModuleList(self.mask_sidebranch_list)

        if weightInitializer is not None:
            self.mask_sidebranch_list.apply(weightInitializer)

    def forward(self, x, m=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if max(self.mask_input_layer) >= 1:
            m_side = self.mask_sidebranch_list[0](m)
        if 1 in self.mask_input_layer:
            if self.mask_input_method == 'addition':
                x = x + m_side
            elif self.mask_input_method == 'multiplication':
                x = x * m_side
        x = self.maxpool(x)
        f1 = self.layer1(x)
        if max(self.mask_input_layer) >= 2:
            m_side = self.mask_sidebranch_list[1](m_side)
        if 2 in self.mask_input_layer:
            if self.mask_input_method == 'addition':
                f1 = f1 + m_side
            elif self.mask_input_method == 'multiplication':
                f1 = f1 * m_side

        f2 = self.layer2(f1)
        if max(self.mask_input_layer) >= 3:
            m_side = self.mask_sidebranch_list[2](m_side)
        if 3 in self.mask_input_layer:
            if self.mask_input_method == 'addition':
                f2 = f2 + m_side
            elif self.mask_input_method == 'multiplication':
                f2 = f2 * m_side

        f3 = self.layer3(f2)
        if max(self.mask_input_layer) >= 4:
            m_side = self.mask_sidebranch_list[3](m_side)
        if 4 in self.mask_input_layer:
            if self.mask_input_method == 'addition':
                f3 = f3 + m_side
            elif self.mask_input_method == 'multiplication':
                f3 = f3 * m_side
        f4 = self.layer4(f3)
        if max(self.mask_input_layer) >= 5:
            m_side = self.mask_sidebranch_list[4](m_side)
        if 5 in self.mask_input_layer:
            if self.mask_input_method == 'addition':
                f4 = f4 + m_side
            elif self.mask_input_method == 'multiplication':
                f4 = f4 * m_side


        conv_feature = self.global_avg_pool(f4).view(x.size(0), -1)
        y = self.classifier(conv_feature)
        return y


class DeepTEN_MaskSideBranch(ResNet18_DeepTEN_MaskSideBranch):
    def __init__(self, num_classes, pretrained, if_freeze=None, mask_input_layer=None, mask_input_method=None, roi_margin_min=None, roi_margin_max=None,
                 freeze_layernum=3, n_codes=8, len_codes=64, resnet_blocks=4, incorp_CT_N=False, incorp_CT_N_method='v1', incorp_other_info=None):
        super().__init__(num_classes, pretrained, if_freeze, mask_input_layer, mask_input_method, roi_margin_min, roi_margin_max, freeze_layernum, n_codes, len_codes, resnet_blocks, incorp_CT_N, incorp_CT_N_method, incorp_other_info)
        self.classifier = nn.Sequential(
            nn.Linear(len_codes * n_codes, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, m=None, n = None, return_feature=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if max(self.mask_input_layer) >= 1:
            m_side = self.mask_sidebranch_list[0](m)
        if 1 in self.mask_input_layer:
            if self.mask_input_method == 'addition':
                x = x + m_side
            elif self.mask_input_method == 'multiplication':
                x = x * m_side
        x = self.maxpool(x)
        f1 = self.layer1(x)
        if max(self.mask_input_layer) >= 2:
            m_side = self.mask_sidebranch_list[1](m_side)
        if 2 in self.mask_input_layer:
            if self.mask_input_method == 'addition':
                f1 = f1 + m_side
            elif self.mask_input_method == 'multiplication':
                f1 = f1 * m_side

        f2 = self.layer2(f1)
        if max(self.mask_input_layer) >= 3:
            m_side = self.mask_sidebranch_list[2](m_side)
        if 3 in self.mask_input_layer:
            if self.mask_input_method == 'addition':
                f2 = f2 + m_side
            elif self.mask_input_method == 'multiplication':
                f2 = f2 * m_side

        f3 = self.layer3(f2)
        if max(self.mask_input_layer) >= 4:
            m_side = self.mask_sidebranch_list[3](m_side)
        if 4 in self.mask_input_layer:
            if self.mask_input_method == 'addition':
                f3 = f3 + m_side
            elif self.mask_input_method == 'multiplication':
                f3 = f3 * m_side
        f4 = self.layer4(f3)
        if max(self.mask_input_layer) >= 5:
            m_side = self.mask_sidebranch_list[4](m_side)
        if 5 in self.mask_input_layer:
            if self.mask_input_method == 'addition':
                f4 = f4 + m_side
            elif self.mask_input_method == 'multiplication':
                f4 = f4 * m_side


        m1, m2, m3, m4 = None, None, None, None
        if m is not None:
            m = self.maskpool0(m)
            m1 = self.maskpool1(m)
            m2 = self.maskpool2(m1)
            m3 = self.maskpool3(m2)
            m4 = self.maskpool4(m3)

        # Extracte deepten feature
        if self.resnet_blocks == 1:
            texture_feature = self.deepten_feature(f1, m1)
            # conv_feature = self.global_avg_pool(f1).view(x.size(0), -1)
        elif self.resnet_blocks == 2:
            texture_feature = self.deepten_feature(f2, m2)
            # conv_feature = self.global_avg_pool(f2).view(x.size(0), -1)
        elif self.resnet_blocks == 3:
            texture_feature = self.deepten_feature(f3, m3)
            # conv_feature = self.global_avg_pool(f3).view(x.size(0), -1)
        elif self.resnet_blocks == 4:
            texture_feature = self.deepten_feature(f4, m4)
            # conv_feature = self.global_avg_pool(f4).view(x.size(0), -1)
        else:
            raise ValueError
        if n is not None:
            if self.incorp_CT_N_method == 'v1':
                total_feature = torch.cat([texture_feature, n.view(x.size(0), -1)], dim=-1)
            elif self.incorp_CT_N_method == 'v2':
                n = self.ctn_embedding( n.view(x.size(0), -1))
                total_feature = texture_feature + n
        else:
            total_feature = texture_feature
        if return_feature:
            for i in range(len(self.classifier)):
                total_feature = self.classifier[i](total_feature)
                if i == 1:
                    return total_feature
        else:
            y = self.classifier(total_feature)
            return y



