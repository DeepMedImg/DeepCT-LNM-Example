#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import nnunet
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
from nnunet.experiment_planning.utils import crop_pdac_classification, crop
from nnunet.paths import *
import shutil
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from nnunet.preprocessing.sanity_checks import verify_dataset_integrity
from nnunet.training.model_restore import recursive_find_python_class


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("raw_data_dir", type=str, help="directory of raw images")
    parser.add_argument("cropped_out_dir", type=str, help="directory of cropped output images")
    parser.add_argument("preprocessing_output_dir", type=str, help="directory of prerocessed output images")
    parser.add_argument("fold", type=int, nargs='+', help='0, 1, ..., 5 or \'all\'')
    parser.add_argument("-label_table", type=str,
                        help="path of table containing information about identifiers and metastasis labels")
    parser.add_argument("-ln_mask_dir", help='directory of segmented LN masks')
    parser.add_argument("-pl2d", "--planner2d", type=str, default="ExperimentPlanner2D_v21",
                        help="Name of the ExperimentPlanner class for the 2D U-Net. Default is ExperimentPlanner2D_v21. "
                             "Can be 'None', in which case this U-Net will not be configured")
    parser.add_argument("-no_pp", action="store_true",
                        help="Set this flag if you dont want to run the preprocessing. If this is set then this script "
                             "will only run the experiment planning and create the plans file")
    parser.add_argument("-tl", type=int, required=False, default=8,
                        help="Number of processes used for preprocessing the low resolution data for the 3D low "
                             "resolution U-Net. This can be larger than -tf. Don't overdo it or you will run out of "
                             "RAM")
    parser.add_argument("-tf", type=int, required=False, default=8,
                        help="Number of processes used for preprocessing the full resolution data of the 2D U-Net and "
                             "3D U-Net. Don't overdo it or you will run out of RAM")

    args = parser.parse_args()
    raw_data_dir = args.raw_data_dir
    cropped_out_dir = args.cropped_out_dir
    preprocessing_output_dir = args.preprocessing_output_dir
    trained_model_dir = join(network_training_output_dir, '2d', 'Task501_LNMDataset_PDAC_classification', 'nnUNetTrainerV2_PDAC_classification_resnet_MaskSideBranch_SegLN_v2__nnUNetPlansv2.1')
    label_table = args.label_table
    ln_mask_dir = args.ln_mask_dir
    fold = args.fold
    dont_run_preprocessing = args.no_pp
    tl = args.tl
    tf = args.tf

    planner_name2d = args.planner2d

    if planner_name2d == "None":
        planner_name2d = None

    # we need raw data

    crop_pdac_classification(raw_data_dir, cropped_out_dir, False, tf)


    search_in = join(nnunet.__path__[0], "experiment_planning")


    if planner_name2d is not None:
        planner_2d = recursive_find_python_class([search_in], planner_name2d, current_module="nnunet.experiment_planning")
        if planner_2d is None:
            raise RuntimeError("Could not find the Planner class %s. Make sure it is located somewhere in "
                               "nnunet.experiment_planning" % planner_name2d)
    else:
        planner_2d = None

    #splitted_4d_output_dir_task = os.path.join(nnUNet_raw_data, t)
    #lists, modalities = create_lists_from_splitted_dataset(splitted_4d_output_dir_task)

    # dataset_analyzer = DatasetAnalyzer(cropped_out_dir, overwrite=False, num_processes=tf)  # this class creates the fingerprint
    # _ = dataset_analyzer.analyze_dataset()  # this will write output files that will be used by the ExperimentPlanner
    #
    # maybe_mkdir_p(preprocessing_output_dir_this_task)
    # if not os.path.exists(join(preprocessing_output_dir_this_task, "dataset_properties.pkl")):
    #     shutil.copy(join(cropped_out_dir, "dataset_properties.pkl"), preprocessing_output_dir_this_task)
    # if not os.path.exists(join(preprocessing_output_dir_this_task, "dataset.json")):
    #     shutil.copy(join(nnUNet_raw_data, t, "dataset.json"), preprocessing_output_dir_this_task)

    threads = (tl, tf)

    print("number of threads: ", threads, "\n")


    if planner_2d is not None:
        exp_planner = planner_2d(cropped_out_dir, preprocessing_output_dir, trained_model_dir, label_table, ln_mask_dir, fold)
        # exp_planner.plan_experiment()
        exp_planner.load_my_plans()
        if not dont_run_preprocessing:  # double negative, yooo
            exp_planner.run_preprocessing(threads)


if __name__ == "__main__":
    main()

