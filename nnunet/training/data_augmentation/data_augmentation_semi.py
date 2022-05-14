
from copy import deepcopy

import numpy as np
from batchgenerators.dataloading import MultiThreadedAugmenter
from batchgenerators.transforms import DataChannelSelectionTransform, SegChannelSelectionTransform, SpatialTransform, \
    GammaTransform, MirrorTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, BrightnessTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor, CopyTransform
from nnunet.training.data_augmentation.custom_transforms import MaskTransform
from nnunet.training.data_augmentation.my_custom_transform import Convert3DTo2DTransform, Convert2DTo3DTransform, MyMirrorTransform, MoveSDMToData, MoveSDMOutOfData
from nnunet.training.data_augmentation.downsampling import DownsampleSegForDSTransform3, DownsampleSegForDSTransform2
from nnunet.training.data_augmentation.pyramid_augmentations import MoveSegAsOneHotToData, \
    ApplyRandomBinaryOperatorTransform, \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
import os
from nnunet.training.data_augmentation.default_data_augmentation import default_3D_augmentation_params

def get_moreDA_augmentation(dataloader_label, dataloader_unlabel, patch_size, params=default_3D_augmentation_params,
                            border_val_seg=-1,
                            seeds_unlabel=None, seeds_label=None, order_seg=1, order_data=3, deep_supervision_scales=None,
                            soft_ds=False,
                            classes=None, pin_memory=True):
    assert params.get('mirror') is None, "old version of params, use new keyword do_mirror"

    unlabel_transforms = []

    if params.get("selected_data_channels") is not None:
        unlabel_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))

    # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
    if params.get("dummy_2D") is not None and params.get("dummy_2D"):
        ignore_axes = (0,)
        unlabel_transforms.append(Convert3DTo2DTransform())
    else:
        ignore_axes = None

    unlabel_transforms.append(SpatialTransform(
        patch_size, patch_center_dist_from_border=None,
        do_elastic_deform=params.get("do_elastic"), alpha=params.get("elastic_deform_alpha"),
        sigma=params.get("elastic_deform_sigma"),
        do_rotation=params.get("do_rotation"), angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
        angle_z=params.get("rotation_z"), p_rot_per_axis=params.get("rotation_p_per_axis"),
        do_scale=params.get("do_scaling"), scale=params.get("scale_range"),
        border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=order_data,
        border_mode_seg="constant", border_cval_seg=border_val_seg,
        order_seg=order_seg, random_crop=params.get("random_crop"), p_el_per_sample=params.get("p_eldef"),
        p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot"),
        independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis")
    ))

    if params.get("dummy_2D"):
        unlabel_transforms.append(Convert2DTo3DTransform())

    unlabel_transforms.append(CopyTransform({'data': 'ori_data'}, copy=True))

    # we need to put the c'olor augmentations after the dummy 2d part (if applicable). Otherwise the overloaded color
    # channel gets in the way
    if params.get("has_sdm") is not None and params.get("has_sdm"):
        unlabel_transforms.append(MoveSDMOutOfData())

    unlabel_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    unlabel_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                               p_per_channel=0.5))
    unlabel_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))

    if params.get("do_additive_brightness"):
        unlabel_transforms.append(BrightnessTransform(params.get("additive_brightness_mu"),
                                                 params.get("additive_brightness_sigma"),
                                                 True, p_per_sample=params.get("additive_brightness_p_per_sample"),
                                                 p_per_channel=params.get("additive_brightness_p_per_channel")))

    unlabel_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
    unlabel_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                             p_per_channel=0.5,
                                                             order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                             ignore_axes=ignore_axes))

    unlabel_transforms.append(
        GammaTransform(params.get("gamma_range"), True, True, retain_stats=params.get("gamma_retain_stats"),
                       p_per_sample=0.1))  # inverted gamma

    if params.get("do_gamma"):
        unlabel_transforms.append(
            GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"),
                           p_per_sample=params["p_gamma"]))


    if params.get("has_sdm") is not None and params.get("has_sdm"):
        unlabel_transforms.append(MoveSDMToData())

    if params.get("do_mirror") or params.get("mirror"):
        unlabel_transforms.append(MyMirrorTransform(params.get("mirror_axes")))


    unlabel_transforms.append(RenameTransform('data', 'aug_data', True))

    unlabel_transforms.append(NumpyToTensor(['ori_data', 'aug_data'], 'float'))
    unlabel_transforms = Compose(unlabel_transforms)

    batchgenerator_unlabel = MultiThreadedAugmenter(dataloader_unlabel, unlabel_transforms, params.get('num_threads'),
                                                  params.get("num_cached_per_thread"),
                                                  seeds=seeds_unlabel, pin_memory=pin_memory)


    label_transforms = []
    label_transforms.append(RemoveLabelTransform(-1, 0))
    if params.get("selected_data_channels") is not None:
        label_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))
    if params.get("selected_seg_channels") is not None:
        label_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

    if params.get("dummy_2D") is not None and params.get("dummy_2D"):
        ignore_axes = (0,)
        label_transforms.append(Convert3DTo2DTransform())
    else:
        ignore_axes = None

    label_transforms.append(SpatialTransform(
        patch_size, patch_center_dist_from_border=None,
        do_elastic_deform=params.get("do_elastic"), alpha=params.get("elastic_deform_alpha"),
        sigma=params.get("elastic_deform_sigma"),
        do_rotation=params.get("do_rotation"), angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
        angle_z=params.get("rotation_z"), p_rot_per_axis=params.get("rotation_p_per_axis"),
        do_scale=params.get("do_scaling"), scale=params.get("scale_range"),
        border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=order_data,
        border_mode_seg="constant", border_cval_seg=border_val_seg,
        order_seg=order_seg, random_crop=params.get("random_crop"), p_el_per_sample=params.get("p_eldef"),
        p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot"),
        independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis")
    ))

    if params.get("dummy_2D"):
        label_transforms.append(Convert2DTo3DTransform())

    if params.get("has_sdm") is not None and params.get("has_sdm"):
        label_transforms.append(MoveSDMOutOfData())

    label_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    label_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                               p_per_channel=0.5))
    label_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))

    if params.get("do_additive_brightness"):
        label_transforms.append(BrightnessTransform(params.get("additive_brightness_mu"),
                                                 params.get("additive_brightness_sigma"),
                                                 True, p_per_sample=params.get("additive_brightness_p_per_sample"),
                                                 p_per_channel=params.get("additive_brightness_p_per_channel")))

    label_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
    label_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                        p_per_channel=0.5,
                                                        order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                        ignore_axes=ignore_axes))
    label_transforms.append(
        GammaTransform(params.get("gamma_range"), True, True, retain_stats=params.get("gamma_retain_stats"),
                       p_per_sample=0.1))  # inverted gamma

    if params.get("do_gamma"):
        label_transforms.append(GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"),
                           p_per_sample=params["p_gamma"]))

    if params.get("has_sdm") is not None and params.get("has_sdm"):
        label_transforms.append(MoveSDMToData())

    if params.get("do_mirror") or params.get("mirror"):
        label_transforms.append(MirrorTransform(params.get("mirror_axes")))
    if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
        label_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))

    label_transforms.append(RenameTransform('seg', 'target', True))

    if deep_supervision_scales is not None:
        if soft_ds:
            assert classes is not None
            label_transforms.append(DownsampleSegForDSTransform3(deep_supervision_scales, 'target', 'target', classes))
        else:
            label_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, 0, input_key='target',
                                                               output_key='target'))

    label_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    label_transforms = Compose(label_transforms)

    batchgenerator_label = MultiThreadedAugmenter(dataloader_label, label_transforms, params.get('num_threads'), #max(params.get('num_threads') // 2, 1),
                                                params.get("num_cached_per_thread"),
                                                seeds=seeds_label, pin_memory=pin_memory)

    val_transforms = []
    val_transforms.append(RemoveLabelTransform(-1, 0))
    if params.get("selected_data_channels") is not None:
        val_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))
    if params.get("selected_seg_channels") is not None:
        val_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

    if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
        val_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))

    val_transforms.append(RenameTransform('seg', 'target', True))

    if deep_supervision_scales is not None:
        if soft_ds:
            assert classes is not None
            val_transforms.append(DownsampleSegForDSTransform3(deep_supervision_scales, 'target', 'target', classes))
        else:
            val_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, 0, input_key='target',
                                                               output_key='target'))

    val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    val_transforms = Compose(val_transforms)

    batchgenerator_val = MultiThreadedAugmenter(dataloader_label, val_transforms, max(params.get('num_threads') // 2, 1),
                                                params.get("num_cached_per_thread"),
                                                seeds=seeds_label, pin_memory=pin_memory)

    return batchgenerator_unlabel, batchgenerator_label, batchgenerator_val