from nnunet.experiment_planning.common_utils import get_pool_and_conv_props
from nnunet.experiment_planning.experiment_planner_baseline_2DUNet_v21 import ExperimentPlanner2D_v21
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.paths import *
import numpy as np
from multiprocessing.pool import Pool
from nnunet.configuration import default_num_threads
from batchgenerators.utilities.file_and_folder_operations import load_pickle, subfiles
from collections import OrderedDict
import pickle
import pandas as pd
from skimage.measure import label, regionprops
from skimage.io import imsave
import SimpleITK as sitk

def add_classes_in_slice_info(args):
    """
    We need this for 2D dataloader with oversampling. As of now it will detect slices that contain specific classes
    at run time, meaning it needs to iterate over an entire patient just to extract one slice. That is obviously bad,
    so we are doing this once beforehand and just give the dataloader the info it needs in the patients pkl file.

    """
    npz_file, pkl_file, all_classes, metastasis_stage, max_pos_volume_per_fold, cropped_dir = args
    all_data = np.load(npz_file)['data']
    if all_data.shape[0] == 4:
        all_data[-2:] = (all_data[-2:]==2).astype(np.float32)
        infer_map = all_data[3]
        label_map = all_data[2]
    elif all_data.shape[0] == 3:
        all_data[-1:] = (all_data[-1:]==2).astype(np.float32)
        label_map = all_data[2]
        infer_map = None
    else:
        label_map = infer_map = None
    # remove but largest one
    # infer_label_map, infer_num = label(infer_map, return_num=True)
    # if infer_num > 1:
    #     print(npz_file)
    #     infer_props = regionprops(infer_label_map)
    #     max_idx = np.argmax([prop.area for prop in infer_props])
    #     max_label = infer_props[max_idx].label
    #     for prop in infer_props:
    #         if prop.label != max_label:
    #             infer_map[infer_label_map == prop.label] = 0
    #     np.savez_compressed(npz_file, data=all_data.astype(np.float32))

    with open(pkl_file, 'rb') as f:
        props = pickle.load(f)
    #if props.get('classes_in_slice_per_axis') is not None:
    # print(pkl_file)
    # this will be a dict of dict where the first dict encodes the axis along which a slice is extracted in its keys.
    # The second dict (value of first dict) will have all classes as key and as values a list of all slice ids that
    # contain this class
    classes_in_slice = OrderedDict()
    for axis in range(3):
        other_axes = tuple([i for i in range(3) if i != axis])
        classes_in_slice[axis] = OrderedDict()
        for c in all_classes:
            classes_in_slice[axis][c] = OrderedDict()
            for seg_idx, seg_map in enumerate([label_map, infer_map]):
                if seg_map is not None:
                    # max_area = np.max(np.sum(seg_map == c, axis=other_axes))
                    # valid_slices = np.where(np.sum(seg_map == c, axis=other_axes) > 0.6 * max_area)[0]  # > 60% of max area is valid
                    if np.sum(seg_map == c) > 0:
                        max_slice_idx = np.argsort(np.sum(seg_map == c, axis=other_axes))[::-1]
                        # valid_slices = np.array([max_slice_idx-1, max_slice_idx, max_slice_idx+1])
                        valid_slices = max_slice_idx[:3]
                        classes_in_slice[axis][c][seg_idx] = valid_slices
                    else:
                        print('%s, %d has none pdac slices' %(npz_file, seg_idx))

    number_of_voxels_per_class = OrderedDict()
    for c in all_classes:
        number_of_voxels_per_class[c] = np.sum(label_map == c)

    props['classes_in_slice_per_axis'] = classes_in_slice
    props['number_of_voxels_per_class'] = number_of_voxels_per_class
    props['metastasis_stage'] = metastasis_stage

    # add info of max positive LN volume
    for f, volume in max_pos_volume_per_fold.items():
        props['seg_ln_vol_fold' + str(f)] = volume

    patch_size = [263, 263]
    for axis in range(3):
        for c in all_classes:
            for seg_idx in classes_in_slice[axis][c].keys():
                for slice in classes_in_slice[axis][c][seg_idx]:
                    identifier = os.path.split(npz_file)[-1][:-4]
                    dst_path = join(cropped_dir, identifier + '_axis%d_cls%d_segidx%d_slice%d.npy'%(axis, c, seg_idx, slice))
                    print(dst_path)
                    if not os.path.exists(dst_path):
                        if axis == 0:
                            case_all_data = all_data[:, slice]
                        elif axis == 1:
                            case_all_data = all_data[:, :, slice]
                        else:
                            case_all_data = all_data[:, :, :, slice]

                        case_all_data = np.vstack((case_all_data[:2], case_all_data[2:][seg_idx: seg_idx + 1]))
                        foreground_voxels = np.array(np.where(case_all_data[-1] == c)) # 1 is PDAC
                        selected_center_voxel = np.mean(foreground_voxels, axis=1).astype(int)

                        lb_x = selected_center_voxel[0] - patch_size[0] // 2
                        ub_x = selected_center_voxel[0] + patch_size[0] // 2 + patch_size[0] % 2
                        lb_y = selected_center_voxel[1] - patch_size[1] // 2
                        ub_y = selected_center_voxel[1] + patch_size[1] // 2 + patch_size[1] % 2
                        valid_lb_x = max(0, lb_x)
                        valid_ub_x = min(case_all_data.shape[1], ub_x)
                        valid_lb_y = max(0, lb_y)
                        valid_ub_y = min(case_all_data.shape[2], ub_y)
                        result = case_all_data[:, valid_lb_x:valid_ub_x, valid_lb_y:valid_ub_y]

                        result_donly = np.pad(result[:-1], ((0, 0),(-min(0, lb_x), max(ub_x - case_all_data.shape[1], 0)),
                                                                          (-min(0, lb_y), max(ub_y - case_all_data.shape[2], 0))),
                                                     'constant', **{'constant_values':0})

                        result_segonly = np.pad(result[-1:], ((0, 0),(-min(0, lb_x), max(ub_x - case_all_data.shape[1], 0)),
                                                                            (-min(0, lb_y), max(ub_y - case_all_data.shape[2], 0))),
                                                       'constant', **{'constant_values': -1})

                        result = np.concatenate((result_donly, result_segonly), axis = 0)

                        np.save(dst_path, result.astype(np.float32))

                        imsave(dst_path.replace('.npy', '_img.png'), result[1])
                        imsave(dst_path.replace('.npy', '_seg.png'), result[2])



    with open(pkl_file, 'wb') as f:
        pickle.dump(props, f)


def get_max_positive_ln_volume(seg_path):
    seg_sitk = sitk.ReadImage(seg_path)
    volume_per_voxel = float(np.prod(seg_sitk.GetSpacing(), dtype=np.float32))
    seg_np = sitk.GetArrayFromImage(seg_sitk)
    seg_map = label(seg_np == 1)
    props = regionprops(seg_map)
    max_area = np.max([prop.area for prop in props]) if len(props) > 0 else 0
    max_volume = max_area * volume_per_voxel
    return max_volume



class ExperimentPlanner2D_v21_pdac_classification(ExperimentPlanner2D_v21):
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder, trained_model_dir, label_table,ln_mask_dir, fold):
        super(ExperimentPlanner2D_v21_pdac_classification, self).__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.data_identifier = "nnUNetData_plans_v2.1_2D"
        self.plans_fname = join(trained_model_dir, "plans.pkl")
        self.unet_featuremap_min_edge_length = 28

        self.preprocessor_name = "GenericPreprocessor"

        self.label_table = label_table

        self.training_df = pd.read_excel(self.label_table)

        self.ln_mask_dir = ln_mask_dir
        self.fold = fold

        self.cropped_slice_dir = join(self.preprocessed_output_folder, self.data_identifier + '_stage0', 'cropped_slice')
        maybe_mkdir_p(self.cropped_slice_dir)

    def get_properties_for_stage(self, current_spacing, original_spacing, original_shape, num_cases,
                                 num_modalities, num_classes):

        new_median_shape = np.round(original_spacing / current_spacing * original_shape).astype(int)

        dataset_num_voxels = np.prod(new_median_shape, dtype=np.int64) * num_cases
        # input_patch_size = new_median_shape[1:]
        input_patch_size = np.array([224, 224])

        # TODO there is a bug here. The pooling operations are determined by the input_patch_size we put into this
        #  PRIOR to padding, so there may be a pooling being left out. This is not detrimental, but not pretty also.
        #  Will be fixed after publication. The bug can be fixed by taking ceil() of current_size in each iteration
        #  of the while loop in get_pool_and_conv_props
        network_numpool, net_pool_kernel_sizes, net_conv_kernel_sizes, input_patch_size, \
        shape_must_be_divisible_by = get_pool_and_conv_props(current_spacing[1:], input_patch_size,
                                                             self.unet_featuremap_min_edge_length,
                                                             self.unet_max_numpool)

        # we pretend to use 30 feature maps. This will yield the same configuration as in V1. The larger memory
        # footpring of 32 vs 30 is mor ethan offset by the fp16 training. We make fp16 training default
        # Reason for 32 vs 30 feature maps is that 32 is faster in fp16 training (because multiple of 8)
        estimated_gpu_ram_consumption = Generic_UNet.compute_approx_vram_consumption(input_patch_size,
                                                                                     network_numpool,
                                                                                     30,
                                                                                     self.unet_max_num_filters,
                                                                                     num_modalities, num_classes,
                                                                                     net_pool_kernel_sizes,
                                                                                     conv_per_stage=self.conv_per_stage)

        batch_size = int(np.floor(Generic_UNet.use_this_for_batch_size_computation_2D /
                                  estimated_gpu_ram_consumption * Generic_UNet.DEFAULT_BATCH_SIZE_2D))
        if batch_size < self.unet_min_batch_size:
            raise RuntimeError("This framework is not made to process patches this large. We will add patch-based "
                               "2D networks later. Sorry for the inconvenience")

        # check if batch size is too large (more than 5 % of dataset)
        max_batch_size = np.round(self.batch_size_covers_max_percent_of_dataset * dataset_num_voxels /
                                  np.prod(input_patch_size, dtype=np.int64)).astype(int)
        batch_size = min(batch_size, max_batch_size)

        plan = {
            'batch_size': batch_size,
            'num_pool_per_axis': network_numpool,
            'patch_size': input_patch_size,
            'median_patient_size_in_voxels': new_median_shape,
            'current_spacing': current_spacing,
            'original_spacing': original_spacing,
            'pool_op_kernel_sizes': net_pool_kernel_sizes,
            'conv_kernel_sizes': net_conv_kernel_sizes,
            'do_dummy_2D_data_aug': False
        }
        return plan

    def run_preprocessing(self, num_threads):
        super().run_preprocessing(num_threads)
        self.add_classes_in_slice_info()

    def add_classes_in_slice_info(self):
        """
        this speeds up oversampling foreground during training
        :return:
        """
        p = Pool(default_num_threads)
        # p = Pool(1)

        # if there is more than one my_data_identifier (different brnaches) then this code will run for all of them if
        # they start with the same string. not problematic, but not pretty
        # stages = [join(self.preprocessed_output_folder, self.data_identifier + "_stage%d" % i) for i in range(len(self.plans_per_stage))]
        stages = [join(self.preprocessed_output_folder, self.data_identifier + "_stage%d" % i) for i in range(1)]

        for s in stages:
            print(s.split("/")[-1])
            list_of_npz_files = subfiles(s, True, None, ".npz", True)
            list_of_pkl_files = [i[:-4]+".pkl" for i in list_of_npz_files]
            cropped_dir = [self.cropped_slice_dir] * len(list_of_pkl_files)
            all_classes, metastasis_stages, list_of_max_pos_volume = [], [], []
            for pk in list_of_pkl_files:
                props = load_pickle(pk)
                # all_classes_tmp = np.array(props['all_classes']) #1 for PDAC
                all_classes_tmp = np.array([0, 1]) #1 for PDAC
                all_classes.append(all_classes_tmp[all_classes_tmp > 0])
                case_identifier = os.path.split(pk)[-1][:-4]
                metastasis_stages.append(self.training_df[self.training_df['LNM_identifier'] == case_identifier]['N'].values[0])
                max_pos_volume_per_fold = {}
                for f in self.fold:
                    ln_seg_path = os.path.join(self.ln_mask_dir, case_identifier+'.nii.gz')
                    max_pos_volume = get_max_positive_ln_volume(ln_seg_path)
                    max_pos_volume_per_fold[f] = max_pos_volume
                list_of_max_pos_volume.append(max_pos_volume_per_fold)
            p.map(add_classes_in_slice_info, zip(list_of_npz_files, list_of_pkl_files, all_classes, metastasis_stages, list_of_max_pos_volume, cropped_dir))
        p.close()
        p.join()