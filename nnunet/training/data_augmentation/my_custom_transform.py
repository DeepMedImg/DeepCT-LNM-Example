
import numpy as np
from batchgenerators.transforms import AbstractTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform
from batchgenerators.transforms import SpatialTransform
from batchgenerators.augmentations.utils import create_zero_centered_coordinate_mesh, elastic_deform_coordinates,  rotate_coords_3d, scale_coords, rotate_coords_2d, \
    interpolate_img
from batchgenerators.augmentations.crop_and_pad_augmentations import random_crop as random_crop_aug
from batchgenerators.augmentations.crop_and_pad_augmentations import center_crop as center_crop_aug


def convert_3d_to_2d_generator(data_dict):
    shp = data_dict['data'].shape
    data_dict['data'] = data_dict['data'].reshape((shp[0], shp[1] * shp[2], shp[3], shp[4]))
    data_dict['orig_shape_data'] = shp
    if data_dict.get('seg') is not None:
        shp = data_dict['seg'].shape
        data_dict['seg'] = data_dict['seg'].reshape((shp[0], shp[1] * shp[2], shp[3], shp[4]))
        data_dict['orig_shape_seg'] = shp
    return data_dict


def convert_2d_to_3d_generator(data_dict):
    shp = data_dict['orig_shape_data']
    current_shape = data_dict['data'].shape
    data_dict['data'] = data_dict['data'].reshape((shp[0], shp[1], shp[2], current_shape[-2], current_shape[-1]))
    if data_dict.get('seg') is not None:
        shp = data_dict['orig_shape_seg']
        current_shape_seg = data_dict['seg'].shape
        data_dict['seg'] = data_dict['seg'].reshape((shp[0], shp[1], shp[2], current_shape_seg[-2], current_shape_seg[-1]))
    return data_dict


class Convert3DTo2DTransform(AbstractTransform):
    def __init__(self):
        pass

    def __call__(self, **data_dict):
        return convert_3d_to_2d_generator(data_dict)


class Convert2DTo3DTransform(AbstractTransform):
    def __init__(self):
        pass

    def __call__(self, **data_dict):
        return convert_2d_to_3d_generator(data_dict)


######################### spatial transformations ###################################
def augment_mirroring(sample_data, sample_seg=None, axes=(0, 1, 2)):
    if (len(sample_data.shape) != 4) and (len(sample_data.shape) != 5):
        raise Exception(
            "Invalid dimension for sample_data and sample_seg. sample_data and sample_seg should be either "
            "[batchsize, channels, x, y] or [batchsize, channels, x, y, z]")
    do_mirror_axes = []
    if 0 in axes and np.random.uniform() < 0.5:
        sample_data[:, :] = sample_data[:,:, ::-1]
        do_mirror_axes.append(0+2)
        if sample_seg is not None:
            sample_seg[:, :] = sample_seg[:,:, ::-1]
    if 1 in axes and np.random.uniform() < 0.5:
        sample_data[:, :, :] = sample_data[:,:, :, ::-1]
        do_mirror_axes.append(1+2)
        if sample_seg is not None:
            sample_seg[:, :, :] = sample_seg[:,:, :, ::-1]
    if 2 in axes and len(sample_data.shape) == 5:
        if np.random.uniform() < 0.5:
            sample_data[:, :, :, :] = sample_data[:,:, :, :, ::-1]
            do_mirror_axes.append(2+2)
            if sample_seg is not None:
                sample_seg[:, :, :, :] = sample_seg[:,:, :, :, ::-1]
    return sample_data, sample_seg, do_mirror_axes


class MyMirrorTransform(AbstractTransform):
    """ Randomly mirrors data along specified axes. Mirroring is evenly distributed. Probability of mirroring along
    each axis is 0.5

    Args:
        axes (tuple of int): axes along which to mirror

    """

    def __init__(self, axes=(0, 1, 2), data_key="data", label_key="seg", mirror_axes_key="do_mirror_axes"):
        self.data_key = data_key
        self.label_key = label_key
        self.mirror_axes_key = mirror_axes_key
        self.axes = axes
        if max(axes) > 2:
            raise ValueError("MirrorTransform now takes the axes as the spatial dimensions. What previously was "
                             "axes=(2, 3, 4) to mirror along all spatial dimensions of a 5d tensor (b, c, x, y, z) "
                             "is now axes=(0, 1, 2). Please adapt your scripts accordingly.")

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)
        ret_val = augment_mirroring(data, seg, axes=self.axes)

        data_dict[self.data_key] = ret_val[0]
        if seg is not None:
            data_dict[self.label_key] = ret_val[1]
        data_dict[self.mirror_axes_key] = ret_val[2]

        return data_dict

class MoveSDMToData(AbstractTransform):
    '''
    Concat SDM and data
    '''
    def __init__(self, key_origin="sdm", key_target="data", remove_from_origin=True, binarize = False):
        self.remove_from_origin = remove_from_origin
        self.key_target = key_target
        self.key_origin = key_origin
        self.binarize = binarize

    def __call__(self, **data_dict):
        origin = data_dict.get(self.key_origin)
        target = data_dict.get(self.key_target)

        if self.binarize:
            origin = origin.astype(bool).astype(target.dtype)

        target = np.concatenate((target, origin), 1)
        data_dict[self.key_target] = target

        if self.remove_from_origin:
            data_dict[self.key_origin] = None

        return data_dict

class MoveSDMOutOfData(AbstractTransform):
    def __init__(self, key_origin="data", key_target="sdm"):
        self.key_target = key_target
        self.key_origin = key_origin

    def __call__(self, **data_dict):
        origin = data_dict.get(self.key_origin)
        data_dict[self.key_origin] = origin[:, :-1]
        data_dict[self.key_target] = origin[:, -1:]
        return data_dict

class MoveKeyOut(AbstractTransform):
    def __init__(self, key_origin="seg", key_target="attmap", channel=-1):
        self.key_target = key_target
        self.key_origin = key_origin
        self.channel = channel

    def __call__(self, **data_dict):
        origin = data_dict.get(self.key_origin)
        data_dict[self.key_origin] = origin[:, :self.channel]
        data_dict[self.key_target] = origin[:, self.channel:]
        return data_dict

class SelectiveRemoveLabelTransform(AbstractTransform):
    def __init__(self, remove_label, replace_with=0, foreground_num = 20, input_key="seg", output_key="seg"):
        '''
        If foreground class number is less than foreground_num, set background class 0 to replace_with
        '''
        self.output_key = output_key
        self.input_key = input_key
        self.replace_with = replace_with
        self.remove_label = remove_label
        self.foreground_num = foreground_num

    def __call__(self, **data_dict):
        seg = data_dict[self.input_key]
        for b in range(len(seg)):
            if not np.isin(list(range(1, self.foreground_num+1)), data_dict['properties'][b]['classes']).all():
                seg[b][seg[b] == self.remove_label] = self.replace_with
        data_dict[self.output_key] = seg
        return data_dict

class SpatialTransformAttmap(SpatialTransform):
    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        if self.patch_size is None:
            if len(data.shape) == 4:
                patch_size = (data.shape[2], data.shape[3])
            elif len(data.shape) == 5:
                patch_size = (data.shape[2], data.shape[3], data.shape[4])
            else:
                raise ValueError("only support 2D/3D batch data.")
        else:
            patch_size = self.patch_size

        ret_val = augment_spatial_attmap(data, seg, patch_size=patch_size,
                                  patch_center_dist_from_border=self.patch_center_dist_from_border,
                                  do_elastic_deform=self.do_elastic_deform, alpha=self.alpha, sigma=self.sigma,
                                  do_rotation=self.do_rotation, angle_x=self.angle_x, angle_y=self.angle_y,
                                  angle_z=self.angle_z, do_scale=self.do_scale, scale=self.scale,
                                  border_mode_data=self.border_mode_data,
                                  border_cval_data=self.border_cval_data, order_data=self.order_data,
                                  border_mode_seg=self.border_mode_seg, border_cval_seg=self.border_cval_seg,
                                  order_seg=self.order_seg, random_crop=self.random_crop,
                                  p_el_per_sample=self.p_el_per_sample, p_scale_per_sample=self.p_scale_per_sample,
                                  p_rot_per_sample=self.p_rot_per_sample,
                                  independent_scale_for_each_axis=self.independent_scale_for_each_axis,
                                  p_rot_per_axis=self.p_rot_per_axis)
        data_dict[self.data_key] = ret_val[0]
        if seg is not None:
            data_dict[self.label_key] = ret_val[1]

        return data_dict

def augment_spatial_attmap(data, seg, patch_size, patch_center_dist_from_border=30,
                    do_elastic_deform=True, alpha=(0., 1000.), sigma=(10., 13.),
                    do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                    do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
                    border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True, p_el_per_sample=1,
                    p_scale_per_sample=1, p_rot_per_sample=1, independent_scale_for_each_axis=False,
                    p_rot_per_axis: float = 1):
    '''
    add a attmap channel to seg
    '''
    dim = len(patch_size)
    seg_result = None
    if seg is not None:
        if dim == 2:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
        else:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                                  dtype=np.float32)

    if dim == 2:
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
    else:
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                               dtype=np.float32)

    if not isinstance(patch_center_dist_from_border, (list, tuple, np.ndarray)):
        patch_center_dist_from_border = dim * [patch_center_dist_from_border]

    for sample_id in range(data.shape[0]):
        coords = create_zero_centered_coordinate_mesh(patch_size)
        modified_coords = False

        if do_elastic_deform and np.random.uniform() < p_el_per_sample:
            a = np.random.uniform(alpha[0], alpha[1])
            s = np.random.uniform(sigma[0], sigma[1])
            coords = elastic_deform_coordinates(coords, a, s)
            modified_coords = True

        if do_rotation and np.random.uniform() < p_rot_per_sample:

            if np.random.uniform() <= p_rot_per_axis:
                a_x = np.random.uniform(angle_x[0], angle_x[1])
            else:
                a_x = 0

            if dim == 3:
                if np.random.uniform() <= p_rot_per_axis:
                    a_y = np.random.uniform(angle_y[0], angle_y[1])
                else:
                    a_y = 0

                if np.random.uniform() <= p_rot_per_axis:
                    a_z = np.random.uniform(angle_z[0], angle_z[1])
                else:
                    a_z = 0

                coords = rotate_coords_3d(coords, a_x, a_y, a_z)
            else:
                coords = rotate_coords_2d(coords, a_x)
            modified_coords = True

        if do_scale and np.random.uniform() < p_scale_per_sample:
            if not independent_scale_for_each_axis:
                if np.random.random() < 0.5 and scale[0] < 1:
                    sc = np.random.uniform(scale[0], 1)
                else:
                    sc = np.random.uniform(max(scale[0], 1), scale[1])
            else:
                sc = []
                for _ in range(dim):
                    if np.random.random() < 0.5 and scale[0] < 1:
                        sc.append(np.random.uniform(scale[0], 1))
                    else:
                        sc.append(np.random.uniform(max(scale[0], 1), scale[1]))
            coords = scale_coords(coords, sc)
            modified_coords = True

        # now find a nice center location
        if modified_coords:
            for d in range(dim):
                if random_crop:
                    ctr = np.random.uniform(patch_center_dist_from_border[d],
                                            data.shape[d + 2] - patch_center_dist_from_border[d])
                else:
                    ctr = int(np.round(data.shape[d + 2] / 2.))
                coords[d] += ctr
            for channel_id in range(data.shape[1]):
                data_result[sample_id, channel_id] = interpolate_img(data[sample_id, channel_id], coords, order_data,
                                                                     border_mode_data, cval=border_cval_data)
            if seg is not None:
                seg_result[sample_id, 0] = interpolate_img(seg[sample_id, 0], coords, order_seg,
                                                                    border_mode_seg, cval=border_cval_seg,
                                                                    is_seg=True)
                for channel_id in range(1, seg.shape[1]):
                    seg_result[sample_id, channel_id] = interpolate_img(seg[sample_id, channel_id], coords, order_data,
                                                               border_mode_data, cval=border_cval_data)
        else:
            if seg is None:
                s = None
            else:
                s = seg[sample_id:sample_id + 1]
            if random_crop:
                margin = [patch_center_dist_from_border[d] - patch_size[d] // 2 for d in range(dim)]
                d, s = random_crop_aug(data[sample_id:sample_id + 1], s, patch_size, margin)
            else:
                d, s = center_crop_aug(data[sample_id:sample_id + 1], patch_size, s)
            data_result[sample_id] = d[0]
            if seg is not None:
                seg_result[sample_id] = s[0]
    return data_result, seg_result

class GenerateJitteringBBox(AbstractTransform):
    '''
    Generate jittering bounding box for mask
    '''
    def __init__(self, margins_min, margins_max, mask_idx_in_seg=0, data_key="data", seg_key="seg", properities_key = "properties"):
        self.margins_max = margins_max
        self.margins_min = margins_min
        self.seg_key = seg_key
        self.data_key = data_key
        self.mask_idx_in_seg = mask_idx_in_seg
        self.properities_key = properities_key

    def __call__(self, **data_dict):
        seg = data_dict.get(self.seg_key)
        data = data_dict.get(self.data_key)
        roi = []
        for b in range(data.shape[0]):
            mask = seg[b, self.mask_idx_in_seg]
            # if not (mask >= 1).any():
            #     print(data_dict.get('key')[b] + 'has no lymph node')
            assert (mask >= 1).any(), print(data_dict.get('keys')[b] + 'has no lymph node')
            margins = np.array([np.random.uniform(low=margin_min, high=margin_max) for margin_min, margin_max in zip(self.margins_min, self.margins_max)])
            bbox_of_nonzero_class = get_bbox_from_mask(mask, 1)
            bbox_of_nonzero_class_plus_margin = np.array(bbox_of_nonzero_class) + np.stack([-margins, margins], axis=1)
            random_offsets = np.array([np.random.uniform(low = max(margin-margin_max, margin_min - margin), high=min(margin - margin_min,margin_max - margin) )for margin_min, margin_max, margin in zip(self.margins_min, self.margins_max, margins)])
            bbox_of_nonzero_class_plus_margin_offset = bbox_of_nonzero_class_plus_margin + random_offsets[:, None]
            shape = data[b].shape[1:]
            valid_bbox_of_nonzero_class_plus_margin_offset =  [[int(max(0, minidx)), int(min(shape[idx], maxidx))] for idx, (minidx, maxidx) in enumerate(bbox_of_nonzero_class_plus_margin_offset)]
            roi.append(np.array(valid_bbox_of_nonzero_class_plus_margin_offset))
        data_dict['roi'] = np.array(roi)

        return data_dict

def get_bbox_from_mask(mask, inside_value=1):
    mask_voxel_coords = np.where(mask >= inside_value)
    if len(mask_voxel_coords) == 3:
        minzidx = int(np.min(mask_voxel_coords[0]))
        maxzidx = int(np.max(mask_voxel_coords[0])) + 1
        minxidx = int(np.min(mask_voxel_coords[1]))
        maxxidx = int(np.max(mask_voxel_coords[1])) + 1
        minyidx = int(np.min(mask_voxel_coords[2]))
        maxyidx = int(np.max(mask_voxel_coords[2])) + 1
        return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]
    else:
        minxidx = int(np.min(mask_voxel_coords[0]))
        maxxidx = int(np.max(mask_voxel_coords[0])) + 1
        minyidx = int(np.min(mask_voxel_coords[1]))
        maxyidx = int(np.max(mask_voxel_coords[1])) + 1
        return [[minxidx, maxxidx], [minyidx, maxyidx]]


class MaskTransformV2(AbstractTransform):
    def __init__(self, mask_idx_in_seg=1, set_outside_to=0, data_key="data", seg_key="seg", roi_key = "roi"):
        """
        data[bbox(mask) <= 0] = 0
        Sets everything outside the mask to 0. CAREFUL! outside is defined as < 0, not =0 (in the Mask)!!!

        :param dct_for_where_it_was_used:
        :param mask_idx_in_seg:
        :param set_outside_to:
        :param data_key:
        :param seg_key:
        """
        self.seg_key = seg_key
        self.data_key = data_key
        self.roi_key = roi_key
        self.set_outside_to = set_outside_to
        self.mask_idx_in_seg = mask_idx_in_seg

    def __call__(self, **data_dict):
        seg = data_dict.get(self.seg_key)
        if seg is None or seg.shape[1] < self.mask_idx_in_seg:
            raise Warning("mask not found, seg may be missing or seg[:, mask_idx_in_seg] may not exist")
        data = data_dict.get(self.data_key)
        roi = data_dict.get(self.roi_key).astype(int)
        for b in range(data.shape[0]):
            mask = np.zeros_like(seg[b, self.mask_idx_in_seg])
            mask[roi[b, 0, 0]:roi[b, 0,1], roi[b, 1,0]:roi[b, 1,1]] = 1
            for c in range(data.shape[1]):
                data[b, c][mask == 0] = self.set_outside_to
        data_dict[self.data_key] = data
        return data_dict

class Mask2Box(AbstractTransform):
    def __init__(self, mask_idx_in_seg=1, set_outside_to=0, seg_key="seg"):
        """
        mask[bbox(mask) <= 0] = 0
        Sets everything outside the mask to 0. CAREFUL! outside is defined as < 0, not =0 (in the Mask)!!!

        :param dct_for_where_it_was_used:
        :param mask_idx_in_seg:
        :param set_outside_to:
        :param data_key:
        :param seg_key:
        """
        self.seg_key = seg_key
        self.set_outside_to = set_outside_to
        self.mask_idx_in_seg = mask_idx_in_seg

    def __call__(self, **data_dict):
        seg = data_dict.get(self.seg_key)
        if seg is None or seg.shape[1] < self.mask_idx_in_seg:
            raise Warning("mask not found, seg may be missing or seg[:, mask_idx_in_seg] may not exist")
        for b in range(seg.shape[0]):
            mask = np.zeros_like(seg[b, self.mask_idx_in_seg])
            roi = np.array(get_bbox_from_mask(seg[b, self.mask_idx_in_seg]))
            mask[roi[0, 0]:roi[0,1], roi[1,0]:roi[1,1], roi[2,0]:roi[2,1]] = 1
            seg[b, self.mask_idx_in_seg] = mask.copy()
        data_dict[self.seg_key] = seg
        return data_dict


class Mask2JitteringBox(AbstractTransform):
    def __init__(self, mask_idx_in_seg=1, set_outside_to=0, seg_key="seg", roi_key="roi"):
        """
        mask[bbox(mask) <= 0] = 0
        Sets everything outside the mask to 0. CAREFUL! outside is defined as < 0, not =0 (in the Mask)!!!

        :param dct_for_where_it_was_used:
        :param mask_idx_in_seg:
        :param set_outside_to:
        :param data_key:
        :param seg_key:
        """
        self.seg_key = seg_key
        self.roi_key = roi_key
        self.set_outside_to = set_outside_to
        self.mask_idx_in_seg = mask_idx_in_seg

    def __call__(self, **data_dict):
        seg = data_dict.get(self.seg_key)
        if seg is None or seg.shape[1] < self.mask_idx_in_seg:
            raise Warning("mask not found, seg may be missing or seg[:, mask_idx_in_seg] may not exist")
        rois = data_dict.get(self.roi_key).astype(int)
        for b in range(seg.shape[0]):
            mask = np.zeros_like(seg[b, self.mask_idx_in_seg])
            roi = rois[b]
            mask[roi[0, 0]:roi[0,1], roi[1,0]:roi[1,1], roi[2,0]:roi[2,1]] = 1
            seg[b, self.mask_idx_in_seg] = mask.copy()
        data_dict[self.seg_key] = seg
        return data_dict


class MaskTransformV3(AbstractTransform):
    def __init__(self, mask_idx_in_seg=1, set_outside_to=0, data_key="data", seg_key="seg"):
        """
        data[mask <= 0] = 0
        Sets everything outside the mask to 0. CAREFUL! outside is defined as < 0, not =0 (in the Mask)!!!

        :param dct_for_where_it_was_used:
        :param mask_idx_in_seg:
        :param set_outside_to:
        :param data_key:
        :param seg_key:
        """
        self.seg_key = seg_key
        self.data_key = data_key
        self.set_outside_to = set_outside_to
        self.mask_idx_in_seg = mask_idx_in_seg

    def __call__(self, **data_dict):
        seg = data_dict.get(self.seg_key)
        if seg is None or seg.shape[1] < self.mask_idx_in_seg:
            raise Warning("mask not found, seg may be missing or seg[:, mask_idx_in_seg] may not exist")
        data = data_dict.get(self.data_key)
        for b in range(data.shape[0]):
            mask = seg[b, self.mask_idx_in_seg]
            for c in range(data.shape[1]):
                data[b, c][mask <= 0] = self.set_outside_to
        data_dict[self.data_key] = data
        return data_dict