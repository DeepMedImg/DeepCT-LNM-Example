#    Created by zhilin zheng for lymph node metastasis prediction

from copy import deepcopy
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
import torch.nn.functional
from nnunet.network_architecture.neural_network import NeuralNetwork
from nnunet.network_architecture.generic_UNet import ConvDropoutNormNonlin, StackedConvLayers, Upsample
from typing import Union, Tuple, List
from nnunet.preprocessing.cropping import get_bbox_from_mask
from nnunet.utilities.to_torch import to_cuda, maybe_to_torch
from nnunet.network_architecture.custom_modules.roi_pool import MyRoIPooling3D
import torchvision

class Generic_UNet_Classification(NeuralNetwork):
    DEFAULT_BATCH_SIZE_3D = 8
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000  # 505789440

    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_pool_per_axis, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False, fc_dropout_op_kwargs=None,roi_margin_min=None, roi_margin_max=None):
        super().__init__()
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.fc_dropout_op_kwargs = fc_dropout_op_kwargs
        self.num_pool_per_axis = num_pool_per_axis
        self.roi_margin_min = roi_margin_min
        self.roi_margin_max = roi_margin_max

        if conv_op == nn.Conv2d:
            upsample_mode = 'bilinear'
            pool_op = nn.MaxPool2d
            transpconv = nn.ConvTranspose2d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
        elif conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            transpconv = nn.ConvTranspose3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features

        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.td = []
        self.cls_branch = []
        self.tu = []
        self.seg_outputs = []

        output_features = base_num_features
        input_features = input_channels

        for d in range(num_pool):
            # determine the first stride
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions
            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))
            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))

            output_features = min(output_features, self.max_num_features)

        # now the bottleneck.
        # determine the first stride
        if self.convolutional_pooling:
            first_stride = pool_op_kernel_sizes[-1]
        else:
            first_stride = None

        # the output of the last conv must match the number of features from the skip connection if we are not using
        # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
        # done by the transposed conv
        if self.convolutional_upsampling:
            final_num_features = output_features
        else:
            final_num_features = self.conv_blocks_context[-1].output_channels

        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        self.conv_blocks_context.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=basic_block),
            StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, basic_block=basic_block)))

        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        # now lets build the localization pathway
        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[
                -(2 + u)].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1 and not self.convolutional_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            if not self.convolutional_upsampling:
                self.tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
            else:
                self.tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u + 1)],
                                          pool_op_kernel_sizes[-(u + 1)], bias=False))

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]
            self.conv_blocks_localization.append(nn.Sequential(
                StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
            ))
        #
        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(conv_op(self.conv_blocks_localization[ds][-1].output_channels, input_channels,
                                            1, 1, 0, 1, 1, seg_output_use_bias))

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
                                                        mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(lambda x: x)
        #
        # if not dropout_in_localization:
        #     self.dropout_op_kwargs['p'] = old_dropout_p
        # 3d GAP
        # self.cls_branch.append(nn.AdaptiveAvgPool3d(1))
        # RoI Pooling
        self.roi_pool = torchvision.ops.RoIAlign(output_size=1, spatial_scale=1. / 2 ** self.num_pool_per_axis[0], sampling_ratio=-1, aligned=True)
        # self.roi_pool = MyRoIPooling3D(np.array(1.), 1)
        # flatten
        self.cls_branch.append(nn.Flatten())
        # fc layers
        self.cls_branch.append(nn.Linear(self.conv_blocks_context[-1][-1].output_channels,
                                         self.conv_blocks_context[-1][-1].output_channels))
        self.cls_branch.append(nn.BatchNorm1d(self.conv_blocks_context[-1][-1].output_channels, **norm_op_kwargs))
        self.cls_branch.append(self.nonlin(**self.nonlin_kwargs))
        self.cls_branch.append(nn.Dropout(**self.fc_dropout_op_kwargs)) ## 0.5
        self.cls_branch.append(
            nn.Linear(self.conv_blocks_context[-1][-1].output_channels, self.num_classes))
        # self.cls_branch.append(
        #     nn.Linear(self.conv_blocks_context[-1][-1].output_channels, 1))
        # self.cls_branch.append(nn.ReLU())

        # register all modules properly
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.tu = nn.ModuleList(self.tu)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)
        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(
                self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here
        self.cls_branch = nn.Sequential(*self.cls_branch)
        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)
            # self.apply(print_module_training_status)

    def forward(self, x, roi=None):
        batch_idx = torch.arange(0, x.size(0)).to(x.device).view(-1, 1)
        roi = torch.cat((batch_idx, roi[:, 0,0].view(-1, 1), roi[:, 1,0].view(-1, 1), roi[:, 0,1].view(-1, 1), roi[:, 1,1].view(-1, 1)), dim=1)
        skips = []
        seg_outputs = []
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)

        x = self.conv_blocks_context[-1](x)

        cls_ouput = self.roi_pool(x, roi.type_as(x))
        cls_output = self.cls_branch(cls_ouput)

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self._deep_supervision and self.do_ds:
            return cls_output, tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return cls_output, seg_outputs[-1]

    # def _internal_maybe_mirror_and_pred_3D(self, x: Union[np.ndarray, torch.tensor], mirror_axes: tuple,
    #                 do_mirroring: bool = True, mult: np.ndarray or torch.tensor = None) -> torch.tensor:
    #     assert len(x.shape) == 5, 'x must be (b, c, x, y, z)'
    #     # everything in here takes place on the GPU. If x and mult are not yet on GPU this will be taken care of here
    #     # we now return a cuda tensor! Not numpy array!
    #     with torch.no_grad():
    #         x = to_cuda(maybe_to_torch(x), gpu_id=self.get_device())
    #         result_torch = torch.zeros([1, 1],  dtype=torch.float).cuda(self.get_device(), non_blocking=True)
    #
    #         if mult is not None:
    #             mult = to_cuda(maybe_to_torch(mult), gpu_id=self.get_device())
    #
    #         if do_mirroring:
    #             mirror_idx = 8
    #             num_results = 2 ** len(mirror_axes)
    #         else:
    #             mirror_idx = 1
    #             num_results = 1
    #
    #         for m in range(mirror_idx):
    #             if m == 0:
    #                 pred = self(x)
    #                 result_torch += 1 / num_results * pred
    #
    #             if m == 1 and (2 in mirror_axes):
    #                 pred = self(torch.flip(x, (4,)))
    #                 result_torch += 1 / num_results * pred
    #
    #             if m == 2 and (1 in mirror_axes):
    #                 pred = self(torch.flip(x, (3,)))
    #                 result_torch += 1 / num_results * pred
    #
    #             if m == 3 and (2 in mirror_axes) and (1 in mirror_axes):
    #                 pred = self(torch.flip(x, (4, 3)))
    #                 result_torch += 1 / num_results * pred
    #
    #             if m == 4 and (0 in mirror_axes):
    #                 pred = self(torch.flip(x, (2,)))
    #                 result_torch += 1 / num_results * pred
    #
    #             if m == 5 and (0 in mirror_axes) and (2 in mirror_axes):
    #                 pred = self(torch.flip(x, (4, 2)))
    #                 result_torch += 1 / num_results * pred
    #
    #             if m == 6 and (0 in mirror_axes) and (1 in mirror_axes):
    #                 pred = self(torch.flip(x, (3, 2)))
    #                 result_torch += 1 / num_results * pred
    #
    #             if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes):
    #                 pred = self(torch.flip(x, (4, 3, 2)))
    #                 result_torch += 1 / num_results * pred
    #
    #         if mult is not None:
    #             result_torch[:, :] *= mult
    #
    #     return result_torch

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
                    bbox_of_nonzero_class = get_bbox_from_mask(x_seg[0,0], 1)
                    bbox_of_nonzero_class = bbox_of_nonzero_class + margins
                    pred = self.inference_apply_nonlin(self(x_data,bbox_of_nonzero_class[None] ))
                    result_torch += 1 / num_results * pred

                if m == 1 and (2 in mirror_axes):
                    bbox_of_nonzero_class = get_bbox_from_mask(torch.flip(x_seg, (4,))[0,0], 1)
                    bbox_of_nonzero_class = bbox_of_nonzero_class + margins
                    pred = self.inference_apply_nonlin(self(torch.flip(x_data, (4,)), bbox_of_nonzero_class[None]))
                    result_torch += 1 / num_results * pred

                if m == 2 and (1 in mirror_axes):
                    bbox_of_nonzero_class = get_bbox_from_mask(torch.flip(x_seg, (3,))[0,0], 1)
                    bbox_of_nonzero_class = bbox_of_nonzero_class + margins
                    pred = self.inference_apply_nonlin(self(torch.flip(x_data, (3,)), bbox_of_nonzero_class[None]))
                    result_torch += 1 / num_results * pred

                if m == 3 and (2 in mirror_axes) and (1 in mirror_axes):
                    bbox_of_nonzero_class = get_bbox_from_mask(torch.flip(x_seg, (4, 3))[0,0], 1)
                    bbox_of_nonzero_class = bbox_of_nonzero_class + margins
                    pred = self.inference_apply_nonlin(self(torch.flip(x_data, (4, 3)), bbox_of_nonzero_class[None]))
                    result_torch += 1 / num_results * pred

                if m == 4 and (0 in mirror_axes):
                    bbox_of_nonzero_class = get_bbox_from_mask(torch.flip(x_seg, (2,))[0,0], 1)
                    bbox_of_nonzero_class = bbox_of_nonzero_class + margins
                    pred = self.inference_apply_nonlin(self(torch.flip(x_data, (2,)), bbox_of_nonzero_class[None]))
                    result_torch += 1 / num_results * pred

                if m == 5 and (0 in mirror_axes) and (2 in mirror_axes):
                    bbox_of_nonzero_class = get_bbox_from_mask(torch.flip(x_seg, (4, 2))[0,0], 1)
                    bbox_of_nonzero_class = bbox_of_nonzero_class + margins
                    pred = self.inference_apply_nonlin(self(torch.flip(x_data, (4, 2)), bbox_of_nonzero_class[None]))
                    result_torch += 1 / num_results * pred

                if m == 6 and (0 in mirror_axes) and (1 in mirror_axes):
                    bbox_of_nonzero_class = get_bbox_from_mask(torch.flip(x_seg, (3, 2))[0,0], 1)
                    bbox_of_nonzero_class = bbox_of_nonzero_class + margins
                    pred = self.inference_apply_nonlin(self(torch.flip(x_data, (3, 2)), bbox_of_nonzero_class[None]))
                    result_torch += 1 / num_results * pred

                if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes):
                    bbox_of_nonzero_class = get_bbox_from_mask(torch.flip(x_seg, (4, 3, 2))[0,0], 1)
                    bbox_of_nonzero_class = bbox_of_nonzero_class + margins
                    pred = self.inference_apply_nonlin(self(torch.flip(x_data, (4, 3, 2)), bbox_of_nonzero_class[None]))
                    result_torch += 1 / num_results * pred

            if mult is not None:
                result_torch[:, :] *= mult

        return result_torch


    def predict_3D_Classification(self, x: np.ndarray, do_mirroring: bool, mirror_axes: Tuple[int, ...] = (0, 1, 2),
                   patch_size: Tuple[int, ...] = None, regions_class_order: Tuple[int, ...] = None,
                   use_gaussian: bool = False, pad_border_mode: str = "constant",
                   pad_kwargs: dict = None, all_in_gpu: bool = False,
                   verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
                Use this function to predict a 3D image.
        """
        if verbose: print("debug: mirroring", do_mirroring, "mirror_axes", mirror_axes)

        assert self.get_device() != "cpu", "CPU not implemented"

        if pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        # A very long time ago the mirror axes were (2, 3, 4) for a 3d network. This is just to intercept any old
        # code that uses this convention
        if len(mirror_axes):
            if self.conv_op == nn.Conv2d:
                if max(mirror_axes) > 1:
                    raise ValueError("mirror axes. duh")
            if self.conv_op == nn.Conv3d:
                if max(mirror_axes) > 2:
                    raise ValueError("mirror axes. duh")

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

        def get_bbox_from_mask(mask, inside_value=1):
            mask_voxel_coords = torch.where(mask >= inside_value)
            minxidx = int(torch.min(mask_voxel_coords[0]))
            maxxidx = int(torch.max(mask_voxel_coords[0])) + 1
            minyidx = int(torch.min(mask_voxel_coords[1]))
            maxyidx = int(torch.max(mask_voxel_coords[1])) + 1
            return np.array([[minxidx, maxxidx], [minyidx, maxyidx]])


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
                            pred= self.inference_apply_nonlin(self(x, roi)[0])
                            result_torch += 1 / num_results * pred

                        if m == 1 and (1 in mirror_axes):
                            bbox_of_nonzero_class = get_bbox_from_mask(torch.flip(x, (3,))[0, -1], 1)
                            bbox_of_nonzero_class_plus_margin = np.array(bbox_of_nonzero_class) + np.stack(
                                [-margins, margins], axis=1)
                            roi = np.array(
                                [[int(max(0, minidx)), int(min(shape[idx], maxidx))] for idx, (minidx, maxidx) in
                                 enumerate(bbox_of_nonzero_class_plus_margin)])
                            roi = to_cuda(maybe_to_torch(roi[None]), gpu_id=self.get_device())

                            pred= self.inference_apply_nonlin(self(torch.flip(x, (3,)), roi)[0])
                            result_torch += 1 / num_results * pred

                        if m == 2 and (0 in mirror_axes):
                            bbox_of_nonzero_class = get_bbox_from_mask(torch.flip(x, (2,))[0, -1], 1)
                            bbox_of_nonzero_class_plus_margin = np.array(bbox_of_nonzero_class) + np.stack(
                                [-margins, margins], axis=1)
                            roi = np.array(
                                [[int(max(0, minidx)), int(min(shape[idx], maxidx))] for idx, (minidx, maxidx) in
                                 enumerate(bbox_of_nonzero_class_plus_margin)])
                            roi = to_cuda(maybe_to_torch(roi[None]), gpu_id=self.get_device())

                            pred= self.inference_apply_nonlin(self(torch.flip(x, (2,)), roi)[0])
                            result_torch += 1 / num_results * pred

                        if m == 3 and (0 in mirror_axes) and (1 in mirror_axes):
                            bbox_of_nonzero_class = get_bbox_from_mask(torch.flip(x, (3, 2))[0, -1], 1)
                            bbox_of_nonzero_class_plus_margin = np.array(bbox_of_nonzero_class) + np.stack(
                                [-margins, margins], axis=1)
                            roi = np.array(
                                [[int(max(0, minidx)), int(min(shape[idx], maxidx))] for idx, (minidx, maxidx) in
                                 enumerate(bbox_of_nonzero_class_plus_margin)])
                            roi = to_cuda(maybe_to_torch(roi[None]), gpu_id=self.get_device())

                            pred= self.inference_apply_nonlin(self(torch.flip(x, (3, 2)), roi)[0])
                            result_torch += 1 / num_results * pred

        return result_torch











