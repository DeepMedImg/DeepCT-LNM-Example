import argparse
from copy import deepcopy
from typing import Tuple, Union, List

import numpy as np
from batchgenerators.augmentations.utils import resize_segmentation
from skimage.transform import resize
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax, save_segmentation_nifti
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Process, Queue
import torch
import SimpleITK as sitk
import shutil
from multiprocessing import Pool
from nnunet.postprocessing.connected_components import load_remove_save, load_postprocessing
from nnunet.training.model_restore import restore_model
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.one_hot_encoding import to_one_hot
from nnunet.inference.predict import check_input_folder_and_return_caseIDs
from skimage.measure import label, regionprops
import copy, collections
from batchgenerators.utilities.file_and_folder_operations import *
import pandas as pd
from nnunet.training.data_augmentation.my_custom_transform import get_bbox_from_mask
from collections import defaultdict
from nnunet.utilities.nd_softmax import softmax_helper

def preprocess_save_to_queue(preprocess_fn, q, list_of_lists, segs, output_files, segs_from_prev_stage, classes,
                             transpose_forward):
    # suppress output
    # sys.stdout = open(os.devnull, 'w')

    errors_in = []
    for i, l in enumerate(list_of_lists):
        try:
            output_file = output_files[i]
            print("preprocessing", output_file)
            d, _, dct = preprocess_fn(l)
            # print(output_file, dct)
            # preprocess attmap
            assert isfile(segs[i]) and segs[i].endswith( ".nii.gz"), "attmaps must point to a nifti file"
            seg_np = sitk.GetArrayFromImage(sitk.ReadImage(segs[i]))
            img = sitk.GetArrayFromImage(sitk.ReadImage(l[0]))
            assert all([i == j for i, j in zip(seg_np.shape, img.shape)]), "image and attmap don't have the same pixel array " \
                                                                             "shape! image: %s, seg_prev: %s" % \
                                                                             (l[0], segs_from_prev_stage[i])
            seg_np = seg_np.transpose(transpose_forward)
            seg_reshaped = resize_segmentation(seg_np, d.shape[1:], order=1, cval=0)
            d = np.vstack((d, seg_reshaped[None])).astype(np.float32)

            if segs_from_prev_stage[i] is not None:
                assert isfile(segs_from_prev_stage[i]) and segs_from_prev_stage[i].endswith(
                    ".nii.gz"), "segs_from_prev_stage" \
                                " must point to a " \
                                "segmentation file"
                seg_prev = sitk.GetArrayFromImage(sitk.ReadImage(segs_from_prev_stage[i]))
                # check to see if shapes match
                img = sitk.GetArrayFromImage(sitk.ReadImage(l[0]))
                assert all([i == j for i, j in zip(seg_prev.shape, img.shape)]), "image and segmentation from previous " \
                                                                                 "stage don't have the same pixel array " \
                                                                                 "shape! image: %s, seg_prev: %s" % \
                                                                                 (l[0], segs_from_prev_stage[i])
                seg_prev = seg_prev.transpose(transpose_forward)
                seg_reshaped = resize_segmentation(seg_prev, d.shape[1:], order=1, cval=0)
                seg_reshaped = to_one_hot(seg_reshaped, classes)
                d = np.vstack((d, seg_reshaped)).astype(np.float32)
            """There is a problem with python process communication that prevents us from communicating obejcts 
            larger than 2 GB between processes (basically when the length of the pickle string that will be sent is 
            communicated by the multiprocessing.Pipe object then the placeholder (\%i I think) does not allow for long 
            enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually 
            patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will 
            then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either 
            filename or np.ndarray and will handle this automatically"""
            print(d.shape)
            if np.prod(d.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save, 4 because float32 is 4 bytes
                print(
                    "This output is too large for python process-process communication. "
                    "Saving output temporarily to disk")
                np.save(output_file[:-7] + ".npy", d)
                d = output_file[:-7] + ".npy"
            q.put((output_file, (d, dct)))
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            print("error in", l)
            print(e)
    q.put("end")
    if len(errors_in) > 0:
        print("There were some errors in the following cases:", errors_in)
        print("These cases were ignored.")
    else:
        print("This worker has ended successfully, no errors to report")
    # restore output
    # sys.stdout = sys.__stdout__

def preprocess_multithreaded(trainer, list_of_lists, segs, output_files, num_processes=2, segs_from_prev_stage=None):
    if segs_from_prev_stage is None:
        segs_from_prev_stage = [None] * len(list_of_lists)

    num_processes = min(len(list_of_lists), num_processes)

    classes = list(range(1, trainer.num_classes))
    assert isinstance(trainer, nnUNetTrainer)
    q = Queue(1)
    processes = []
    for i in range(num_processes):
        pr = Process(target=preprocess_save_to_queue, args=(trainer.preprocess_patient, q,
                                                            list_of_lists[i::num_processes],
                                                            segs[i::num_processes],
                                                            output_files[i::num_processes],
                                                            segs_from_prev_stage[i::num_processes],
                                                            classes, trainer.plans['transpose_forward']))
        pr.start()
        processes.append(pr)

    try:
        end_ctr = 0
        while end_ctr != num_processes:
            item = q.get()
            if item == "end":
                end_ctr += 1
                continue
            else:
                yield item

    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()  # this should not happen but better safe than sorry right
            p.join()

        q.close()

def predict_from_folder(model: str, input_folder: str, seg_folder: str, output_folder: str, folds: Union[Tuple[int], List[int]],
                        save_npz: bool, num_threads_preprocessing: int, num_threads_nifti_save: int,
                        lowres_segmentations: Union[str, None],
                        part_id: int, num_parts: int, tta: bool, fp16: bool = False,
                        overwrite_existing: bool = True, mode: str = 'normal', overwrite_all_in_gpu: bool = None,
                        step_size: float = 0.5, checkpoint_name: str = "model_final_checkpoint",
                        segmentation_export_kwargs: dict = None):
    """
        here we use the standard naming scheme to generate list_of_lists and output_files needed by predict_cases

    :param model:
    :param input_folder:
    :param attmap_folder: folder of attention maps
    :param output_folder:
    :param folds:
    :param save_npz:
    :param num_threads_preprocessing:
    :param num_threads_nifti_save:
    :param lowres_segmentations:
    :param part_id:
    :param num_parts:
    :param tta:
    :param fp16:
    :param overwrite_existing: if not None then it will be overwritten with whatever is in there. None is default (no overwrite)
    :return:
    """
    maybe_mkdir_p(output_folder)
    shutil.copy(join(model, 'plans.pkl'), output_folder)

    assert isfile(join(model, "plans.pkl")), "Folder with saved model weights must contain a plans.pkl file"
    # expected_num_modalities = load_pickle(join(model, "plans.pkl"))['num_modalities']
    expected_num_modalities = load_pickle(join(model, "plans.pkl"))['modalities']
    # check input folder integrity
    # case_ids = check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities)
    case_ids = np.unique([file[:-7] for file in os.listdir(seg_folder) if file.endswith('nii.gz')])

    output_files = [join(output_folder, i + ".nii.gz") for i in case_ids]
    all_files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)
    list_of_lists = [[join(input_folder, i) for i in all_files if i[:len(j)].startswith(j) and
                      len(i) == (len(j) + 12)] for j in case_ids]

    if lowres_segmentations is not None:
        assert isdir(lowres_segmentations), "if lowres_segmentations is not None then it must point to a directory"
        lowres_segmentations = [join(lowres_segmentations, i + ".nii.gz") for i in case_ids]
        assert all([isfile(i) for i in lowres_segmentations]), "not all lowres_segmentations files are present. " \
                                                               "(I was searching for case_id.nii.gz in that folder)"
        lowres_segmentations = lowres_segmentations[part_id::num_parts]
    else:
        lowres_segmentations = None

    # add attmaps
    segs = [join(seg_folder, i + ".nii.gz") for i in case_ids]

    if mode == "normal":
        if overwrite_all_in_gpu is None:
            all_in_gpu = False
        else:
            all_in_gpu = overwrite_all_in_gpu

        return predict_cases(model, list_of_lists[part_id::num_parts], segs[part_id::num_parts], output_files[part_id::num_parts], folds,
                             save_npz, num_threads_preprocessing, num_threads_nifti_save, lowres_segmentations, tta,
                             fp16=fp16, overwrite_existing=overwrite_existing, all_in_gpu=all_in_gpu,
                             step_size=step_size, checkpoint_name=checkpoint_name,
                             segmentation_export_kwargs=segmentation_export_kwargs)
    elif mode == "fast":
        if overwrite_all_in_gpu is None:
            all_in_gpu = True
        else:
            all_in_gpu = overwrite_all_in_gpu

        assert save_npz is False
        # return predict_cases_fast(model, list_of_lists[part_id::num_parts], attmaps[part_id::num_parts], output_files[part_id::num_parts], folds,
        #                           num_threads_preprocessing, num_threads_nifti_save, lowres_segmentations,
        #                           tta, fp16=fp16, overwrite_existing=overwrite_existing, all_in_gpu=all_in_gpu,
        #                           step_size=step_size, checkpoint_name=checkpoint_name,
        #                           segmentation_export_kwargs=segmentation_export_kwargs)
    elif mode == "fastest":
        if overwrite_all_in_gpu is None:
            all_in_gpu = True
        else:
            all_in_gpu = overwrite_all_in_gpu

        assert save_npz is False
        # return predict_cases_fastest(model, list_of_lists[part_id::num_parts], attmaps[part_id::num_parts], output_files[part_id::num_parts], folds,
        #                              num_threads_preprocessing, num_threads_nifti_save, lowres_segmentations,
        #                              tta, fp16=fp16, overwrite_existing=overwrite_existing, all_in_gpu=all_in_gpu,
        #                              step_size=step_size, checkpoint_name=checkpoint_name)
    else:
        raise ValueError("unrecognized mode. Must be normal, fast or fastest")

def load_model_and_checkpoint_files(folder, folds=None, fp16=None, checkpoint_name="model_best"):
    """
    used for if you need to ensemble the five models of a cross-validation. This will restore the model from the
    checkpoint in fold 0, load all parameters of the five folds in ram and return both. This will allow for fast
    switching between parameters (as opposed to loading them form disk each time).

    This is best used for inference and test prediction
    :param folder:
    :param folds:
    :param fp16: if None then we take no action. If True/False we overwrite what the model has in its init
    :return:
    """
    if isinstance(folds, str):
        folds = [join(folder, "all")]
        assert isdir(folds[0]), "no output folder for fold %s found" % folds
    elif isinstance(folds, (list, tuple)):
        if len(folds) == 1 and folds[0] == "all":
            folds = [join(folder, "all")]
        else:
            folds = [join(folder, "fold_%d" % i) for i in folds]
        assert all([isdir(i) for i in folds]), "list of folds specified but not all output folders are present"
    elif isinstance(folds, int):
        folds = [join(folder, "fold_%d" % folds)]
        assert all([isdir(i) for i in folds]), "output folder missing for fold %d" % folds
    elif folds is None:
        print("folds is None so we will automatically look for output folders (not using \'all\'!)")
        folds = subfolders(folder, prefix="fold")
        print("found the following folds: ", folds)
    else:
        raise ValueError("Unknown value for folds. Type: %s. Expected: list of int, int, str or None", str(type(folds)))

    # trainer = restore_model(join(folds[0], "masklayer1_multiplication_tanh", "%s.model.pkl" % checkpoint_name), fp16=fp16)
    # trainer = restore_model(join(folds[0], "masklayer1_multiplication_inlrelu_margin48", "%s.model.pkl" % checkpoint_name), fp16=fp16)
    # trainer = restore_model(join(folds[0], "masklayer1_addition_lrelu", "%s.model.pkl" % checkpoint_name), fp16=fp16)
    trainer = restore_model(join(folds[0],  "%s.model.pkl" % checkpoint_name), fp16=fp16)
    trainer.output_folder = folder
    trainer.output_folder_base = folder
    trainer.update_fold(0)
    trainer.initialize(False)
    # all_best_model_files = [join(i, "masklayer1_multiplication_tanh", "%s.model" % checkpoint_name) for i in folds]
    # all_best_model_files = [join(i, "masklayer1_multiplication_inlrelu_margin48", "%s.model" % checkpoint_name) for i in folds]
    # all_best_model_files = [join(i, "masklayer1_addition_lrelu", "%s.model" % checkpoint_name) for i in folds]
    all_best_model_files = [join(i, "%s.model" % checkpoint_name) for i in folds]
    print("using the following model files: ", all_best_model_files)
    all_params = [torch.load(i, map_location=torch.device('cpu')) for i in all_best_model_files]
    return trainer, all_params

def predict_cases(model, list_of_lists, segs, output_filenames, folds, save_npz, num_threads_preprocessing,
                  num_threads_nifti_save, segs_from_prev_stage=None, do_tta=True, fp16=None, overwrite_existing=False,
                  all_in_gpu=False, step_size=0.5, checkpoint_name="model_final_checkpoint",
                  segmentation_export_kwargs: dict = None):
    """
    :param segmentation_export_kwargs:
    :param model: folder where the model is saved, must contain fold_x subfolders
    :param list_of_lists: [[case0_0000.nii.gz, case0_0001.nii.gz], [case1_0000.nii.gz, case1_0001.nii.gz], ...]
    :param output_filenames: [output_file_case0.nii.gz, output_file_case1.nii.gz, ...]
    :param folds: default: (0, 1, 2, 3, 4) (but can also be 'all' or a subset of the five folds, for example use (0, )
    for using only fold_0
    :param save_npz: default: False
    :param num_threads_preprocessing:
    :param num_threads_nifti_save:
    :param segs_from_prev_stage:
    :param do_tta: default: True, can be set to False for a 8x speedup at the cost of a reduced segmentation quality
    :param overwrite_existing: default: True
    :param fp16: if None then we take no action. If True/False we overwrite what the model has in its init
    :return:
    """
    assert len(list_of_lists) == len(output_filenames)
    assert len(list_of_lists) == len(segs)
    if segs_from_prev_stage is not None: assert len(segs_from_prev_stage) == len(output_filenames)

    pool = Pool(num_threads_nifti_save)
    results = []

    cleaned_output_files = []
    for o in output_filenames:
        dr, f = os.path.split(o)
        if len(dr) > 0:
            maybe_mkdir_p(dr)
        if not f.endswith(".nii.gz"):
            f, _ = os.path.splitext(f)
            f = f + ".nii.gz"
        cleaned_output_files.append(join(dr, f))

    if not overwrite_existing:
        print("number of cases:", len(list_of_lists))
        not_done_idx = [i for i, j in enumerate(cleaned_output_files) if not isfile(j)]

        cleaned_output_files = [cleaned_output_files[i] for i in not_done_idx]
        list_of_lists = [list_of_lists[i] for i in not_done_idx]
        if segs_from_prev_stage is not None:
            segs_from_prev_stage = [segs_from_prev_stage[i] for i in not_done_idx]
        segs = [segs[i] for i in not_done_idx]

        print("number of cases that still need to be predicted:", len(cleaned_output_files))

    print("emptying cuda cache")
    torch.cuda.empty_cache()

    print("loading parameters for folds,", folds)
    trainer, params = load_model_and_checkpoint_files(model, folds, fp16=fp16, checkpoint_name=checkpoint_name)

    if segmentation_export_kwargs is None:
        if 'segmentation_export_params' in trainer.plans.keys():
            force_separate_z = trainer.plans['segmentation_export_params']['force_separate_z']
            interpolation_order = trainer.plans['segmentation_export_params']['interpolation_order']
            interpolation_order_z = trainer.plans['segmentation_export_params']['interpolation_order_z']
        else:
            force_separate_z = None
            interpolation_order = 1
            interpolation_order_z = 0
    else:
        force_separate_z = segmentation_export_kwargs['force_separate_z']
        interpolation_order = segmentation_export_kwargs['interpolation_order']
        interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

    print("starting preprocessing generator")
    preprocessing = preprocess_multithreaded(trainer, list_of_lists, segs, cleaned_output_files, num_threads_preprocessing,
                                             segs_from_prev_stage)
    print("starting prediction...")
    all_output_files = []
    cls_results = collections.defaultdict(dict)
    N_results_dict = {'index':[], 'N_pred':[]}

    ############################################################################################################################################
    # def box1_is_in_box2(box1, box2):
    #     x1_min, y1_min, z1_min, x1_max, y1_max, z1_max = box1
    #     (x2_min, x2_max), (y2_min, y2_max), (z2_min, z2_max) = box2
    #     return x1_min >= x2_min and x1_max <= x2_max and y1_min >= y2_min and y1_max <= y2_max and z1_min >= z2_min and z1_max <= z2_max
    #
    # tmp_preprocessed_data = '/data87/pancreaticCancer/nnUNETFrame/nnUNET_DATASET/nnUNET_preprocessed/Task510_LNMDataset_LN_seg_att_stage2_semi/nnUNetData_plans_v2.1_stage1/'
    # tmp_pkl_dict = collections.defaultdict(list)
    # for filename in os.listdir(tmp_preprocessed_data):
    #     if filename.endswith('pkl') and 'fold%d' % folds[0] in filename:
    #         tmp_pkl_dict[filename[:8]].append(os.path.join(tmp_preprocessed_data, filename))
    ###############################################################################################################################################


    for preprocessed in preprocessing:
        output_filename, (d, dct) = preprocessed
        all_output_files.append(all_output_files)
        if isinstance(d, str):
            data = np.load(d)
            os.remove(d)
            d = data

        print("predicting", output_filename)
        shape = d.shape[1:]

        aggregated_softmax_map = np.ones([trainer.num_classes + 1] + list(shape), dtype=np.float32)
        aggregated_softmax_map = aggregated_softmax_map * np.reshape([1, 0, 0], (3,1,1,1))

        seg = d[-1]
        seg_map = label((seg == 1) | (seg == 2))
        seg_props = regionprops(seg_map)
        pred_classes, sizes = [], []
        margins = np.array([np.mean([margin_min, margin_max]) for margin_min, margin_max in
                            zip(trainer.data_aug_params['margins_min'], trainer.data_aug_params['margins_max'])])

        ########################################################################################################################################
        # tmp_area_pkl_dict = defaultdict(dict)
        # identifier = os.path.basename(output_filename)[:8]
        # for pkl_idx, pkl_filename in enumerate(tmp_pkl_dict[identifier]):
        #     tmp_properties = load_pickle(pkl_filename)
        #     tmp_area_pkl_dict[pkl_idx]['pkl_filename'] = pkl_filename
        #     tmp_area_pkl_dict[pkl_idx]['properties'] = tmp_properties
        ########################################################################################################################################

        for idx, seg_prop in enumerate(seg_props):
            center = seg_prop.centroid
            bbox_x_lb = round(center[0]) - trainer.patch_size[0] // 2
            bbox_y_lb = round(center[1]) - trainer.patch_size[1] // 2
            bbox_z_lb = round(center[2]) - trainer.patch_size[2] // 2

            bbox_x_ub = bbox_x_lb + trainer.patch_size[0]
            bbox_y_ub = bbox_y_lb + trainer.patch_size[1]
            bbox_z_ub = bbox_z_lb + trainer.patch_size[2]

            valid_bbox_x_lb = max(0, bbox_x_lb)
            valid_bbox_x_ub = min(shape[0], bbox_x_ub)
            valid_bbox_y_lb = max(0, bbox_y_lb)
            valid_bbox_y_ub = min(shape[1], bbox_y_ub)
            valid_bbox_z_lb = max(0, bbox_z_lb)
            valid_bbox_z_ub = min(shape[2], bbox_z_ub)

            cropped_data = copy.deepcopy(d[:, valid_bbox_x_lb:valid_bbox_x_ub,
                                         valid_bbox_y_lb:valid_bbox_y_ub,
                                         valid_bbox_z_lb:valid_bbox_z_ub])
            cropped_seg_map = seg_map[valid_bbox_x_lb:valid_bbox_x_ub,
                              valid_bbox_y_lb:valid_bbox_y_ub,
                              valid_bbox_z_lb:valid_bbox_z_ub]
            cropped_data[-1][cropped_seg_map != seg_prop.label] = 0


            case_all_data = np.pad(cropped_data, ((0, 0),
                                         (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                         (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                         (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                                         'constant', **{})

            # case_all_data_ori = case_all_data.copy()
            if trainer.data_aug_params.get("change_mask_to_box") is not None and trainer.data_aug_params.get("change_mask_to_box"):
                print("change_mask_to_box")
                roi = np.array(get_bbox_from_mask(case_all_data[-1]))
                mask = np.zeros_like(case_all_data[-1])
                mask[roi[0, 0]:roi[0, 1], roi[1, 0]:roi[1, 1], roi[2, 0]:roi[2, 1]] = 1
                case_all_data[-1] = mask.copy()

            if trainer.data_aug_params.get("change_jittering_mask_to_box") is not None and trainer.data_aug_params.get("change_jittering_mask_to_box"):
                print("change_jittering_mask_to_box")
                roi = np.array(get_bbox_from_mask(case_all_data[-1]))
                roi = (np.array(roi) + np.stack([-margins, margins], axis=1)).astype(int)
                roi = np.array([[int(max(0, minidx)), int(min(trainer.patch_size[idx], maxidx))] for idx, (minidx, maxidx) in enumerate(roi)])
                mask = np.zeros_like(case_all_data[-1])
                mask[roi[0, 0]:roi[0, 1], roi[1, 0]:roi[1, 1], roi[2, 0]:roi[2, 1]] = 1
                case_all_data[-1] = mask.copy()


            softmax, feature = [], []

            for p in params:
                trainer.load_checkpoint_ram(p, False)
                # trainer.network.inference_apply_nonlin = softmax_helper
                softmax.append(trainer.predict_preprocessed_data_return_class_and_score(
                    case_all_data, do_tta, trainer.data_aug_params['mirror_axes'], use_gaussian=True, all_in_gpu=all_in_gpu)[1])

                ###############################################################################################################
                # trainer.network.inference_apply_nonlin = lambda x: x
                # feature.append(trainer.predict_preprocessed_data_return_class_and_score(
                #     case_all_data, do_tta, trainer.data_aug_params['mirror_axes'], use_gaussian=True, all_in_gpu=all_in_gpu, return_feature=True)[1])
                ###############################################################################################################

            softmax = np.vstack(softmax)
            softmax_mean = np.mean(softmax, 0)

            ###############################################################################################################
            # feature = np.vstack(feature)
            # feature_mean = np.mean(feature, 0)
            ###############################################################################################################

            ###############################################################################################################
        #     for tmp_key, tmp_pkl in tmp_area_pkl_dict.items():
        #         if seg_prop.area == tmp_pkl['properties']['area'] and box1_is_in_box2(seg_prop.bbox, tmp_pkl['properties']['crop_box']):
        #             seg_properties = tmp_pkl['properties']
        #             seg_properties[trainer.__class__.__name__ + '_score'] = softmax_mean[1]
        #             seg_properties[trainer.__class__.__name__ + '_feature'] = feature_mean
        #             save_pickle(seg_properties, tmp_pkl['pkl_filename'])
        #             np.save(tmp_pkl['pkl_filename'].replace('pkl', 'npy'), case_all_data_ori)
        #             del tmp_area_pkl_dict[tmp_key]
        #             break
        #
        # assert not bool(tmp_area_pkl_dict)
            ################################################################################################################




            cls_results[output_filename.split('/')[-1][:-7]][str(idx)] = {'pred_class': float(softmax_mean.argmax(-1)), 'pred_score': softmax_mean.tolist(), 'size':float(seg_prop.area)}
            pred_classes.append(int(softmax_mean.argmax(-1)))
            print(int(softmax_mean.argmax(-1)))
            sizes.append(np.sum(case_all_data[-1] > 0))
            foreground_voxels =  (seg[valid_bbox_x_lb:valid_bbox_x_ub, valid_bbox_y_lb:valid_bbox_y_ub, valid_bbox_z_lb:valid_bbox_z_ub] >0) & \
                                 (seg_map[valid_bbox_x_lb:valid_bbox_x_ub, valid_bbox_y_lb:valid_bbox_y_ub, valid_bbox_z_lb:valid_bbox_z_ub] == seg_prop.label)
            for c in range(trainer.num_classes+1):
                if c == 0:
                    aggregated_softmax_map[c, valid_bbox_x_lb:valid_bbox_x_ub, valid_bbox_y_lb:valid_bbox_y_ub, valid_bbox_z_lb:valid_bbox_z_ub][foreground_voxels]= 0
                else:
                    aggregated_softmax_map[c, valid_bbox_x_lb:valid_bbox_x_ub, valid_bbox_y_lb:valid_bbox_y_ub, valid_bbox_z_lb:valid_bbox_z_ub][foreground_voxels] = softmax_mean[c-1]

        cls_results[output_filename.split('/')[-1][:-7]]['overall_N'] = float((np.array(pred_classes) == 0).any())
        # N_results_dict['index'].append(int(output_filename.split('/')[-1][-11:-7]))
        N_results_dict['index'].append(output_filename.split('/')[-1][5:-7])
        N_results_dict['N_pred'].append(int(((np.array(pred_classes) == 0) &(np.array(sizes) > 500 / (0.8 * 0.683 * 0.683))).any()))

        if save_npz:
            npz_file = output_filename[:-7] + ".npz"
        else:
            npz_file = None

        if hasattr(trainer, 'regions_class_order'):
            region_class_order = trainer.regions_class_order
        else:
            region_class_order = None

        """There is a problem with python process communication that prevents us from communicating obejcts
        larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
        communicated by the multiprocessing.Pipe object then the placeholder (\%i I think) does not allow for long
        enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
        patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will
        then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
        filename or np.ndarray and will handle this automatically"""
        bytes_per_voxel = 4
        if all_in_gpu:
            bytes_per_voxel = 2  # if all_in_gpu then the return value is half (float16)
        if np.prod(aggregated_softmax_map.shape) > (2e9 / bytes_per_voxel * 0.85):  # * 0.85 just to be save
            print(
                "This output is too large for python process-process communication. Saving output temporarily to disk")
            np.save(output_filename[:-7] + ".npy", aggregated_softmax_map)
            aggregated_softmax_map = output_filename[:-7] + ".npy"
        save_segmentation_nifti_from_softmax(aggregated_softmax_map, output_filename, dct, interpolation_order, region_class_order,
                                            None, None,
                                            npz_file, None, force_separate_z, interpolation_order_z)
    if not os.path.exists( os.path.join(os.path.split(output_filenames[0])[0], 'results.json')):
        save_json(cls_results, os.path.join(os.path.split(output_filenames[0])[0], 'results.json'))
    if not os.path.exists( os.path.join(os.path.split(output_filenames[0])[0], 'results.xlsx')):
        N_result_df = pd.DataFrame.from_dict(N_results_dict)
        N_result_df.to_excel(os.path.join(os.path.split(output_filenames[0])[0], 'N_results.xlsx'), index=False)

    print("inference done. Now waiting for the segmentation export to finish...")

    pool.close()
    pool.join()