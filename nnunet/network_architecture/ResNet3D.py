import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunet.network_architecture.neural_network import NeuralNetwork, ConvNormNonlinPool
from nnunet.network_architecture.initialization import InitWeights_SideBranch_Addition, InitWeights_SideBranch_Multiplication
import numpy as np
from typing import Union, Tuple, List
from nnunet.preprocessing.cropping import get_bbox_from_mask
from nnunet.utilities.to_torch import to_cuda, maybe_to_torch

def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(NeuralNetwork):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400,
                 pretrained=False,
                 if_freeze=False,
                 freeze_layernum=0):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        self.do_ds = False
        self.num_classes = n_classes

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if pretrained:
            pretrained_model = torch.load('/data86/zhengzhilin980/pancreas/3D-ResNet/r3d18_KM_200ep.pth', map_location=torch.device('cpu'))
            curr_state_dict = self.state_dict()
            pretrained_dict = {k: value for k, value in pretrained_model['state_dict'].items() if not k.startswith('fc')}
            curr_state_dict.update(pretrained_dict)
            self.load_state_dict(curr_state_dict)

        if if_freeze:
            for name, p in self.named_parameters():
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

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def _internal_maybe_mirror_and_pred_3D(self, x: Union[np.ndarray, torch.tensor], mirror_axes: tuple,
                    do_mirroring: bool = True, mult: np.ndarray or torch.tensor = None) -> torch.tensor:

        def get_bbox_from_mask(mask, inside_value=1):
            mask_voxel_coords = torch.where(mask >= inside_value)
            minzidx = int(torch.min(mask_voxel_coords[0]))
            maxzidx = int(torch.max(mask_voxel_coords[0])) + 1
            minxidx = int(torch.min(mask_voxel_coords[1]))
            maxxidx = int(torch.max(mask_voxel_coords[1])) + 1
            minyidx = int(torch.min(mask_voxel_coords[2]))
            maxyidx = int(torch.max(mask_voxel_coords[2])) + 1
            return torch.from_numpy(np.array([[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]])).float().cuda()

        assert len(x.shape) == 5, 'x must be (b, c, x, y, z)'
        # everything in here takes place on the GPU. If x and mult are not yet on GPU this will be taken care of here
        # we now return a cuda tensor! Not numpy array!
        with torch.no_grad():
            x = to_cuda(maybe_to_torch(x), gpu_id=self.get_device())
            result_torch = torch.zeros([1, self.num_classes],  dtype=torch.float).cuda(self.get_device(), non_blocking=True)
            x_data = x[:, 0:-1]
            x_seg = x[:, -1:]
            margins = np.array([4, 4, 4]) + np.array([8, 8, 8]) / 2.0
            margins = torch.from_numpy(np.stack([-margins, margins], axis=1)).float().cuda()
            if mult is not None:
                mult = to_cuda(maybe_to_torch(mult), gpu_id=self.get_device())

            if do_mirroring:
                mirror_idx = 8
                num_results = 2 ** len(mirror_axes)
            else:
                mirror_idx = 1
                num_results = 1

            for m in range(mirror_idx):
                if m == 0:
                    # bbox_of_nonzero_class = get_bbox_from_mask(x_seg[0,0], 1)
                    # bbox_of_nonzero_class = bbox_of_nonzero_class + margins
                    pred = self.inference_apply_nonlin(self(x))
                    result_torch += 1 / num_results * pred

                if m == 1 and (2 in mirror_axes):
                    # bbox_of_nonzero_class = get_bbox_from_mask(torch.flip(x_seg, (4,))[0,0], 1)
                    # bbox_of_nonzero_class = bbox_of_nonzero_class + margins
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (4,))))
                    result_torch += 1 / num_results * pred

                if m == 2 and (1 in mirror_axes):
                    # bbox_of_nonzero_class = get_bbox_from_mask(torch.flip(x_seg, (3,))[0,0], 1)
                    # bbox_of_nonzero_class = bbox_of_nonzero_class + margins
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (3,))))
                    result_torch += 1 / num_results * pred

                if m == 3 and (2 in mirror_axes) and (1 in mirror_axes):
                    # bbox_of_nonzero_class = get_bbox_from_mask(torch.flip(x_seg, (4, 3))[0,0], 1)
                    # bbox_of_nonzero_class = bbox_of_nonzero_class + margins
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 3))))
                    result_torch += 1 / num_results * pred

                if m == 4 and (0 in mirror_axes):
                    # bbox_of_nonzero_class = get_bbox_from_mask(torch.flip(x_seg, (2,))[0,0], 1)
                    # bbox_of_nonzero_class = bbox_of_nonzero_class + margins
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (2,))))
                    result_torch += 1 / num_results * pred

                if m == 5 and (0 in mirror_axes) and (2 in mirror_axes):
                    # bbox_of_nonzero_class = get_bbox_from_mask(torch.flip(x_seg, (4, 2))[0,0], 1)
                    # bbox_of_nonzero_class = bbox_of_nonzero_class + margins
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 2))))
                    result_torch += 1 / num_results * pred

                if m == 6 and (0 in mirror_axes) and (1 in mirror_axes):
                    # bbox_of_nonzero_class = get_bbox_from_mask(torch.flip(x_seg, (3, 2))[0,0], 1)
                    # bbox_of_nonzero_class = bbox_of_nonzero_class + margins
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (3, 2))))
                    result_torch += 1 / num_results * pred

                if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes):
                    # bbox_of_nonzero_class = get_bbox_from_mask(torch.flip(x_seg, (4, 3, 2))[0,0], 1)
                    # bbox_of_nonzero_class = bbox_of_nonzero_class + margins
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 3, 2))))
                    result_torch += 1 / num_results * pred

            if mult is not None:
                result_torch[:, :] *= mult

        return result_torch


    def predict_3D_Classification(self, x: np.ndarray, do_mirroring: bool, mirror_axes: Tuple[int, ...] = (0, 1, 2),
                   patch_size: Tuple[int, ...] = None, regions_class_order: Tuple[int, ...] = None,
                   use_gaussian: bool = False, pad_border_mode: str = "constant",
                   pad_kwargs: dict = None, all_in_gpu: bool = False,
                   verbose: bool = True, return_feature : bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
                Use this function to predict a 3D image.
        """
        if verbose: print("debug: mirroring", do_mirroring, "mirror_axes", mirror_axes)

        assert self.get_device() != "cpu", "CPU not implemented"

        if pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        # A very long time ago the mirror axes were (2, 3, 4) for a 3d network. This is just to intercept any old
        # # code that uses this convention
        # if len(mirror_axes):
        #     if self.conv_op == nn.Conv2d:
        #         if max(mirror_axes) > 1:
        #             raise ValueError("mirror axes. duh")
        #     if self.conv_op == nn.Conv3d:
        #         if max(mirror_axes) > 2:
        #             raise ValueError("mirror axes. duh")

        if self.training:
            print('WARNING! Network is in train mode during inference. This may be intended, or not...')

        assert len(x.shape) == 4, "data must have shape (c,x,y,z)"
        torch.cuda.empty_cache()
        with torch.no_grad():
            assert patch_size is not None, "patch_size cannot be None for patch-based prediction"
            data_shape = x.shape
            if verbose:
                print("data shape:", data_shape)
                print("patch size:", patch_size)
                # we only need to compute that once. It can take a while to compute this due to the large sigma in
                # gaussian_filter
            shape = x.shape[1:]
            lb_x = 0
            ub_x = shape[0] - patch_size[0]
            lb_y = 0
            ub_y = shape[1] - patch_size[1]
            lb_z = 0
            ub_z = shape[2] - patch_size[2]

            if x[-1].any():
                bbox_of_nonzero_class = get_bbox_from_mask(x[-1], 0)
                center_voxel = [ (minidx+maxidx)//2 for minidx, maxidx in bbox_of_nonzero_class]
                bbox_x_lb = max(lb_x, center_voxel[0] - patch_size[0] // 2)
                bbox_y_lb = max(lb_y, center_voxel[1] - patch_size[1] // 2)
                bbox_z_lb = max(lb_z, center_voxel[2] - patch_size[2] // 2)

            else:
                print('No lymph node')
                return np.array([-1]), np.array([-1])

            bbox_x_ub = bbox_x_lb + patch_size[0]
            bbox_y_ub = bbox_y_lb + patch_size[1]
            bbox_z_ub = bbox_z_lb + patch_size[2]

            valid_bbox_x_lb = max(0, bbox_x_lb)
            valid_bbox_x_ub = min(shape[0], bbox_x_ub)
            valid_bbox_y_lb = max(0, bbox_y_lb)
            valid_bbox_y_ub = min(shape[1], bbox_y_ub)
            valid_bbox_z_lb = max(0, bbox_z_lb)
            valid_bbox_z_ub = min(shape[2], bbox_z_ub)

            x = x[:, valid_bbox_x_lb:valid_bbox_x_ub,
                            valid_bbox_y_lb:valid_bbox_y_ub,
                            valid_bbox_z_lb:valid_bbox_z_ub]

            x_data = np.pad(x[:-1], ((0, 0),
                              (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                              (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                              (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                                 'constant')

            x_seg = np.pad(x[-1:], ((0, 0),
                            (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                            (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                            (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                               'constant', **{'constant_values': 0})
            x = np.concatenate([x_data, x_seg])
            # x = x_data * (x_seg == 1).astype(float)
            if verbose: print("moving data to GPU")
            x = torch.from_numpy(x).cuda(self.get_device(), non_blocking=True)

            # predicted_score= self._internal_maybe_mirror_and_pred_3D(
            #     x[None, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z], mirror_axes, do_mirroring)[0]

            predicted_score= self._internal_maybe_mirror_and_pred_3D(
                x[None], mirror_axes, do_mirroring)

            predicted_score = predicted_score.cpu().numpy()

        if verbose: print("prediction done")
        return np.argmax(predicted_score), predicted_score

class ResNet_MaskSideBranch(ResNet):
    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400,
                 pretrained=False,
                 if_freeze=False,
                 freeze_layernum=0,
                 mask_input_method='multiplication', mask_input_layer=None):
        super().__init__(block, layers, block_inplanes, n_input_channels,  conv1_t_size, conv1_t_stride, no_max_pool,
                       shortcut_type, widen_factor, n_classes, pretrained, if_freeze, freeze_layernum)

        self.mask_input_method = mask_input_method
        self.mask_input_layer = mask_input_layer
        weightInitializer = None
        if mask_input_method == 'addition':
            weightInitializer = InitWeights_SideBranch_Addition()
        elif mask_input_method == 'multiplication':
            weightInitializer = InitWeights_SideBranch_Multiplication()

        self.mask_sidebranch_list = []
        input_features, output_features = 1, self.conv1.out_channels
        for i in range(max(mask_input_layer)):
            self.mask_sidebranch_list.append(ConvNormNonlinPool(input_features, output_features, conv_op=nn.Conv3d, norm_op=nn.BatchNorm3d, pool_op = nn.MaxPool3d,
                                                                pool_kwargs = {'kernel_size': 3, 'stride': (1,2,2), 'padding': 1}))
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
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        if max(self.mask_input_layer) >= 2:
            m_side = self.mask_sidebranch_list[1](m_side)
        if 2 in self.mask_input_layer:
            if self.mask_input_method == 'addition':
                x = x + m_side
            elif self.mask_input_method == 'multiplication':
                x = x * m_side
        x = self.layer2(x)
        if max(self.mask_input_layer) >= 3:
            m_side = self.mask_sidebranch_list[2](m_side)
        if 3 in self.mask_input_layer:
            if self.mask_input_method == 'addition':
                x = x + m_side
            elif self.mask_input_method == 'multiplication':
                x = x * m_side

        x = self.layer3(x)
        if max(self.mask_input_layer) >= 4:
            m_side = self.mask_sidebranch_list[3](m_side)
        if 4 in self.mask_input_layer:
            if self.mask_input_method == 'addition':
                x = x + m_side
            elif self.mask_input_method == 'multiplication':
                x = x * m_side

        x = self.layer4(x)
        if max(self.mask_input_layer) >= 5:
            m_side = self.mask_sidebranch_list[4](m_side)
        if 5 in self.mask_input_layer:
            if self.mask_input_method == 'addition':
                x = x + m_side
            elif self.mask_input_method == 'multiplication':
                x = x * m_side

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def _internal_maybe_mirror_and_pred_3D(self, x: Union[np.ndarray, torch.tensor], mirror_axes: tuple,
                    do_mirroring: bool = True, mult: np.ndarray or torch.tensor = None) -> torch.tensor:

        def get_bbox_from_mask(mask, inside_value=1):
            mask_voxel_coords = torch.where(mask >= inside_value)
            minzidx = int(torch.min(mask_voxel_coords[0]))
            maxzidx = int(torch.max(mask_voxel_coords[0])) + 1
            minxidx = int(torch.min(mask_voxel_coords[1]))
            maxxidx = int(torch.max(mask_voxel_coords[1])) + 1
            minyidx = int(torch.min(mask_voxel_coords[2]))
            maxyidx = int(torch.max(mask_voxel_coords[2])) + 1
            return torch.from_numpy(np.array([[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]])).float().cuda()

        assert len(x.shape) == 5, 'x must be (b, c, x, y, z)'
        # everything in here takes place on the GPU. If x and mult are not yet on GPU this will be taken care of here
        # we now return a cuda tensor! Not numpy array!
        with torch.no_grad():
            x = to_cuda(maybe_to_torch(x), gpu_id=self.get_device())
            result_torch = torch.zeros([1, self.num_classes],  dtype=torch.float).cuda(self.get_device(), non_blocking=True)
            x_data = x[:, 0:-1]
            x_seg = x[:, -1:]
            margins = np.array([4, 4, 4]) + np.array([8, 8, 8]) / 2.0
            margins = torch.from_numpy(np.stack([-margins, margins], axis=1)).float().cuda()
            if mult is not None:
                mult = to_cuda(maybe_to_torch(mult), gpu_id=self.get_device())

            if do_mirroring:
                mirror_idx = 8
                num_results = 2 ** len(mirror_axes)
            else:
                mirror_idx = 1
                num_results = 1

            for m in range(mirror_idx):
                if m == 0:
                    # bbox_of_nonzero_class = get_bbox_from_mask(x_seg[0,0], 1)
                    # bbox_of_nonzero_class = bbox_of_nonzero_class + margins
                    pred = self.inference_apply_nonlin(self(x, x[:, -1:]))
                    result_torch += 1 / num_results * pred

                if m == 1 and (2 in mirror_axes):
                    # bbox_of_nonzero_class = get_bbox_from_mask(torch.flip(x_seg, (4,))[0,0], 1)
                    # bbox_of_nonzero_class = bbox_of_nonzero_class + margins
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (4,)), torch.flip(x, (4,))[:, -1:]))
                    result_torch += 1 / num_results * pred

                if m == 2 and (1 in mirror_axes):
                    # bbox_of_nonzero_class = get_bbox_from_mask(torch.flip(x_seg, (3,))[0,0], 1)
                    # bbox_of_nonzero_class = bbox_of_nonzero_class + margins
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (3,)), torch.flip(x, (3,))[:, -1:]))
                    result_torch += 1 / num_results * pred

                if m == 3 and (2 in mirror_axes) and (1 in mirror_axes):
                    # bbox_of_nonzero_class = get_bbox_from_mask(torch.flip(x_seg, (4, 3))[0,0], 1)
                    # bbox_of_nonzero_class = bbox_of_nonzero_class + margins
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 3)), torch.flip(x, (4, 3))[:, -1:]))
                    result_torch += 1 / num_results * pred

                if m == 4 and (0 in mirror_axes):
                    # bbox_of_nonzero_class = get_bbox_from_mask(torch.flip(x_seg, (2,))[0,0], 1)
                    # bbox_of_nonzero_class = bbox_of_nonzero_class + margins
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (2,)), torch.flip(x, (2,))[:, -1:]))
                    result_torch += 1 / num_results * pred

                if m == 5 and (0 in mirror_axes) and (2 in mirror_axes):
                    # bbox_of_nonzero_class = get_bbox_from_mask(torch.flip(x_seg, (4, 2))[0,0], 1)
                    # bbox_of_nonzero_class = bbox_of_nonzero_class + margins
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 2)), torch.flip(x, (4, 2))[:, -1:]))
                    result_torch += 1 / num_results * pred

                if m == 6 and (0 in mirror_axes) and (1 in mirror_axes):
                    # bbox_of_nonzero_class = get_bbox_from_mask(torch.flip(x_seg, (3, 2))[0,0], 1)
                    # bbox_of_nonzero_class = bbox_of_nonzero_class + margins
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (3, 2)), torch.flip(x, (3, 2))[:, -1:]))
                    result_torch += 1 / num_results * pred

                if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes):
                    # bbox_of_nonzero_class = get_bbox_from_mask(torch.flip(x_seg, (4, 3, 2))[0,0], 1)
                    # bbox_of_nonzero_class = bbox_of_nonzero_class + margins
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 3, 2)), torch.flip(x, (4, 3, 2))[:, -1:]))
                    result_torch += 1 / num_results * pred

            if mult is not None:
                result_torch[:, :] *= mult

        return result_torch



def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model

def generate_model_masksidebranch(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet_MaskSideBranch(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet_MaskSideBranch(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet_MaskSideBranch(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet_MaskSideBranch(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet_MaskSideBranch(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet_MaskSideBranch(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet_MaskSideBranch(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model