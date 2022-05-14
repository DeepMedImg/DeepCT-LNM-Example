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


from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.training.data_augmentation.default_data_augmentation import get_moreDA_augmentation_classification
from nnunet.network_architecture.generic_UNet_classification import Generic_UNet_Classification
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork, NeuralNetwork
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnunet.training.dataloading.dataset_loading import unpack_dataset, DataLoader2DClassification
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
from torch.nn.utils import clip_grad_norm_
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.utilities.tensor_utilities import sum_tensor
from skimage.io import imsave

from nnunet.configuration import default_num_threads
from multiprocessing.pool import Pool
from nnunet.evaluation.evaluator import aggregate_classification_scores
import pandas as pd

try:
    from apex import amp
except ImportError:
    amp = None

def get_bbox_from_mask(mask, inside_value=1):
    mask_voxel_coords = np.where(mask >= inside_value)
    minxidx = int(np.min(mask_voxel_coords[0]))
    maxxidx = int(np.max(mask_voxel_coords[0])) + 1
    minyidx = int(np.min(mask_voxel_coords[1]))
    maxyidx = int(np.max(mask_voxel_coords[1])) + 1
    return np.array([[minxidx, maxxidx], [minyidx, maxyidx]])

class nnUNetTrainerV2_PDAC_classification(nnUNetTrainer):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 100
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None

        self.pin_memory = True
        self.save_latest_only = False
        self.oversample_foreground_percent=1.

        self.rec_loss = nn.MSELoss(reduction='mean')
        self.cls_loss = nn.CrossEntropyLoss(reduction='mean')

        self.debug = False
        self.debug_output_dir = os.path.join(self.output_folder, 'batch_data')
        if self.debug and not os.path.exists(self.debug_output_dir):
            os.makedirs(self.debug_output_dir)

        self.training_table = '/data86/zhengzhilin980/pancreas/PDAC_new/Patients_addN0.xlsx'

        self.dataset_test = None

        self.weight_decay = 5e-4

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            # mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            # weights[~mask] = 0
            # weights = weights / weights.sum()
            # self.ds_loss_weights = weights
            # now wrap the loss
            # self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    # unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation_classification(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            # assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel, NeuralNetwork))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            # norm_op = nn.InstanceNorm2d
            norm_op = nn.BatchNorm2d

        # norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum':0.1}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        fc_dropout_op_kwargs = {'p': 0.5, 'inplace': False}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet_Classification(self.num_input_channels + 1, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes), self.net_pool_per_axis,
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True, fc_dropout_op_kwargs=fc_dropout_op_kwargs,
                                                   roi_margin_min = self.data_aug_params["margins_min"], roi_margin_max = self.data_aug_params["margins_max"])
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = None

    def do_split(self):
        """
        we now allow more than 5 splits. IMPORTANT: and fold > 4 will not be a real split but just another random
        80:20 split of the data. You cannot run X-fold cross-validation with this code. It will always be a 5-fold CV.
        Folds > 4 will be independent from each other
        :return:
        """
        if self.fold == 'all' or self.fold < 5:
            splits_file = join(self.dataset_directory, "splits_final.pkl")
            if not isfile(splits_file):
                self.print_to_log_file("Creating new split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    # splits[-1]['train'] = train_keys
                    # splits[-1]['val'] = test_keys

                    splits[-1]['test'] = test_keys
                    rnd = np.random.RandomState(seed=12345 + i)
                    idx_tr = rnd.choice(len(train_keys), int(len(train_keys) * 0.8), replace=False)
                    idx_val = [i for i in range(len(train_keys)) if i not in idx_tr]
                    splits[-1]['train'] = train_keys[idx_tr]
                    splits[-1]['val'] = train_keys[idx_val]
                save_pickle(splits, splits_file)

            splits = load_pickle(splits_file)

            if self.fold == "all":
                tr_keys = val_keys = list(self.dataset.keys())
            else:
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                testing_keys = splits[self.fold]['test']

            tr_keys.sort()
            val_keys.sort()
            testing_keys.sort()

            self.dataset_tr = OrderedDict()
            for i in tr_keys:
                self.dataset_tr[i] = self.dataset[i]

            self.dataset_val = OrderedDict()
            for i in val_keys:
                self.dataset_val[i] = self.dataset[i]

            self.dataset_test = OrderedDict()
            for i in testing_keys:
                self.dataset_test[i] = self.dataset[i]


        else:
            raise ValueError("fold larger than 5")
            # rnd = np.random.RandomState(seed=12345 + self.fold)
            # keys = np.sort(list(self.dataset.keys()))
            # idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
            # idx_val = [i for i in range(len(keys)) if i not in idx_tr]
            #
            # self.dataset_tr = OrderedDict()
            # for i in idx_tr:
            #     self.dataset_tr[keys[i]] = self.dataset[keys[i]]
            #
            # self.dataset_val = OrderedDict()
            # for i in idx_val:
            #     self.dataset_val[keys[i]] = self.dataset[keys[i]]

    def load_dataset(self):

        def load_dataset(folder):
            # we don't load the actual data but instead return the filename to the np file. the properties are loaded though
            case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith("npy") and (i.find("segFromPrevStage") == -1) and (i.find("fold") == -1)]
            # case_identifiers = get_case_identifiers(folder)
            case_identifiers.sort()
            dataset = OrderedDict()
            for c in case_identifiers:
                dataset[c] = OrderedDict()
                dataset[c]['data_file'] = join(folder, "%s.npz" % c)
                with open(join(folder, "%s.pkl" % c), 'rb') as f:
                    dataset[c]['properties'] = pickle.load(f)
                if dataset[c].get('seg_from_prev_stage_file') is not None:
                    dataset[c]['seg_from_prev_stage_file'] = join(folder, "%s_segs.npz" % c)
            return dataset

        self.dataset = load_dataset(self.folder_with_preprocessed_data)


    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        if self.threeD:
            pass

        else:
            dl_tr = DataLoader2DClassification(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 transpose=None,  # self.plans.get('transpose_forward'),
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides)
            dl_val = DataLoader2DClassification(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
                                  transpose=None,  # self.plans.get('transpose_forward'),
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides)

        return dl_tr, dl_val

    def run_online_evaluation(self, output, target):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """
        with torch.no_grad():
            num_classes = output.shape[1]
            output_softmax = softmax_helper(output)
            output_seg = output_softmax.argmax(1)
            # target = target[:, 0]
            axes = tuple(range(1, len(target.shape)))
            tp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fn_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            for c in range(1, num_classes):
                tp_hard[:, c - 1] = (output_seg == c).float() * (target == c).float()
                fp_hard[:, c - 1] = (output_seg == c).float() * (target != c).float()
                fn_hard[:, c - 1] = (output_seg != c).float() * (target == c).float()

            tp_hard = tp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fp_hard = fp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fn_hard = fn_hard.sum(0, keepdim=False).detach().cpu().numpy()

            self.online_eval_foreground_dc.append(list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)))
            self.online_eval_tp.append(list(tp_hard))
            self.online_eval_fp.append(list(fp_hard))
            self.online_eval_fn.append(list(fn_hard))

    def predict_preprocessed_data_return_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None, use_gaussian: bool = True,
                                                         pad_border_mode: str = 'constant', pad_kwargs: dict = None,
                                                         all_in_gpu: bool = True,
                                                         verbose: bool = True,
                                                         CT_N:int = None, return_feature: bool=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param data:
        :param do_mirroring:
        :param mirror_axes:
        :param use_sliding_window:
        :param step_size:
        :param use_gaussian:
        :param pad_border_mode:
        :param pad_kwargs:
        :param all_in_gpu:
        :param verbose:
        :return:
        """
        if pad_border_mode == 'constant' and pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        if do_mirroring and mirror_axes is None:
            mirror_axes = self.data_aug_params['mirror_axes']

        if do_mirroring:
            assert self.data_aug_params["do_mirror"], "Cannot do mirroring as test time augmentation when training " \
                                                      "was done without mirroring"

        # valid = list((SegmentationNetwork, nn.DataParallel, NeuralNetwork))
        # assert isinstance(self.network, tuple(valid))

        current_mode = self.network.training
        self.network.eval()
        # if CT_N is not None:
        try:
            ret = self.network.predict_2D(data, do_mirroring, mirror_axes, self.patch_size,
                                          self.regions_class_order, use_gaussian, pad_border_mode, pad_kwargs,
                                          all_in_gpu, verbose, CT_N, return_feature)
        except:
            ret = self.network.predict_2D(data, do_mirroring, mirror_axes, self.patch_size,
                                          self.regions_class_order, use_gaussian, pad_border_mode, pad_kwargs,
                                          all_in_gpu, verbose)
        # else:
        #     ret = self.network.predict_2D(data, do_mirroring, mirror_axes, self.patch_size,
        #                                   self.regions_class_order, use_gaussian, pad_border_mode, pad_kwargs,
        #                                   all_in_gpu, verbose, return_feature)
        self.network.train(current_mode)
        return ret

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """

        """
        if debug=True then the temporary files generated for postprocessing determination will be kept
        """

        current_mode = self.network.training
        self.network.eval()

        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_val is None:
            self.load_dataset()
            self.do_split()

        if segmentation_export_kwargs is None:
            if 'segmentation_export_params' in self.plans.keys():
                force_separate_z = self.plans['segmentation_export_params']['force_separate_z']
                interpolation_order = self.plans['segmentation_export_params']['interpolation_order']
                interpolation_order_z = self.plans['segmentation_export_params']['interpolation_order_z']
            else:
                force_separate_z = None
                interpolation_order = 1
                interpolation_order_z = 0
        else:
            force_separate_z = segmentation_export_kwargs['force_separate_z']
            interpolation_order = segmentation_export_kwargs['interpolation_order']
            interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

        # predictions as they come from the network go here
        output_folder = join(self.output_folder, validation_folder_name)
        maybe_mkdir_p(output_folder)
        # this is for debug purposes
        my_input_args = {'do_mirroring': do_mirroring,
                         'use_gaussian': use_gaussian,
                         'overwrite': overwrite,
                         'validation_folder_name': validation_folder_name,
                         'debug': debug,
                         'all_in_gpu': all_in_gpu,
                         'segmentation_export_kwargs': segmentation_export_kwargs,
                         }
        save_json(my_input_args, join(output_folder, "validation_args.json"))

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        pred_gt_tuples = []

        training_df = pd.read_excel(self.training_table)

        export_pool = Pool(default_num_threads)
        margins = np.array([np.mean([margin_min, margin_max]) for margin_min, margin_max in
                            zip(self.data_aug_params['margins_min'], self.data_aug_params['margins_max'])])
        for k in self.dataset_val.keys():
            properties = self.dataset[k]['properties']
            patch_files = [join(self.folder_with_preprocessed_data, 'cropped_slice', filename) for filename in
                           os.listdir(join(self.folder_with_preprocessed_data, 'cropped_slice')) if filename.startswith(k)
                           and filename.endswith('.npy') and 'segidx0' in filename and 'fold' not in filename]
            label = int(bool(training_df[training_df['LNM_identifier'] == k]['N'].values[0]))
            softmax_preds = []
            for patch_file in patch_files:
                data = np.load(patch_file)

                print(k, data.shape)
                data[-1][data[-1] == -1] = 0

                if self.data_aug_params.get("mask_out_of_box_to_zero") is not None and self.data_aug_params.get("mask_out_of_box_to_zero"):
                    bbox_of_nonzero_class = get_bbox_from_mask(data[-1], 1)
                    bbox_of_nonzero_class_plus_margin = (np.array(bbox_of_nonzero_class) + np.stack([-margins, margins], axis=1)).astype(int)
                    mask = np.zeros_like(data[-1])
                    mask[bbox_of_nonzero_class_plus_margin[0,0]:bbox_of_nonzero_class_plus_margin[0,1], bbox_of_nonzero_class_plus_margin[1,0]:bbox_of_nonzero_class_plus_margin[1,1]] = 1
                    for c in range(data[:-1].shape[0]):
                        data[:-1][c][mask == 0] = 0
                if self.data_aug_params.get("mask_out_of_mask_to_zero") is not None and self.data_aug_params.get("mask_out_of_mask_to_zero"):
                    mask = data[-1]
                    for c in range(data[:-1].shape[0]):
                        data[:-1][c][mask <= 0] = 0

                softmax_pred = self.predict_preprocessed_data_return_softmax(
                    data, do_mirroring, mirror_axes, use_gaussian,
                    all_in_gpu=all_in_gpu)


                softmax_preds.append(softmax_pred)
            if len(softmax_preds) > 0:
                # patient_softmax_pred = torch.cat(softmax_preds, dim = 0).mean(dim=0, keepdim=True)
                # patient_pred = torch.argmax(patient_softmax_pred, dim=1).cpu().numpy()[0]
                # patient_softmax_pred = patient_softmax_pred.cpu().numpy()[0]
                patient_preds = torch.argmax(torch.cat(softmax_preds, dim = 0), dim=1).cpu().numpy()
                patient_pred = np.argmax(np.bincount(patient_preds))
                patient_prob = torch.mean(torch.cat(softmax_preds, dim=0), dim=0).cpu().numpy()
            else:
                # patient_softmax_pred, patient_pred = None, None
                patient_pred, patient_prob = None, None
            # pred_gt_tuples.append([patient_softmax_pred, patient_pred, label])
            pred_gt_tuples.append([patient_pred, label, patient_prob[1]])


        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")

        task = self.dataset_directory.split("/")[-1]
        job_name = self.experiment_name
        _ = aggregate_classification_scores(pred_gt_tuples,  list(self.dataset_val.keys()),
                             json_output_file=join(output_folder, "summary.json"),
                             json_name=job_name ,
                             json_author="Zhilin",
                             json_task=task, num_threads=default_num_threads)
        self.network.train(current_mode)

    def test(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'testing_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """

        """
        if debug=True then the temporary files generated for postprocessing determination will be kept
        """

        current_mode = self.network.training
        self.network.eval()

        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_test is None:
            self.load_dataset()
            self.do_split()

        if segmentation_export_kwargs is None:
            if 'segmentation_export_params' in self.plans.keys():
                force_separate_z = self.plans['segmentation_export_params']['force_separate_z']
                interpolation_order = self.plans['segmentation_export_params']['interpolation_order']
                interpolation_order_z = self.plans['segmentation_export_params']['interpolation_order_z']
            else:
                force_separate_z = None
                interpolation_order = 1
                interpolation_order_z = 0
        else:
            force_separate_z = segmentation_export_kwargs['force_separate_z']
            interpolation_order = segmentation_export_kwargs['interpolation_order']
            interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

        # predictions as they come from the network go here
        output_folder = join(self.output_folder, validation_folder_name)
        maybe_mkdir_p(output_folder)
        # this is for debug purposes
        my_input_args = {'do_mirroring': do_mirroring,
                         'use_gaussian': use_gaussian,
                         'overwrite': overwrite,
                         'validation_folder_name': validation_folder_name,
                         'debug': debug,
                         'all_in_gpu': all_in_gpu,
                         'segmentation_export_kwargs': segmentation_export_kwargs,
                         }
        save_json(my_input_args, join(output_folder, "validation_args.json"))

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        pred_gt_tuples = []

        training_df = pd.read_excel(self.training_table)

        export_pool = Pool(default_num_threads)
        margins = np.array([np.mean([margin_min, margin_max]) for margin_min, margin_max in
                            zip(self.data_aug_params['margins_min'], self.data_aug_params['margins_max'])])
        for k in self.dataset_test.keys():
            properties = self.dataset[k]['properties']
            patch_files = [join(self.folder_with_preprocessed_data, 'cropped_slice', filename) for filename in
                           os.listdir(join(self.folder_with_preprocessed_data, 'cropped_slice')) if
                           filename.startswith(k) and filename.endswith('.npy') and 'segidx0' in filename and 'fold' not in filename]
            label = int(bool(training_df[training_df['LNM_identifier'] == k]['N'].values[0]))
            softmax_preds = []
            for patch_file in patch_files:
                data = np.load(patch_file)

                print(k, data.shape)
                data[-1][data[-1] == -1] = 0

                if self.data_aug_params.get("mask_out_of_box_to_zero") is not None and self.data_aug_params.get(
                        "mask_out_of_box_to_zero"):
                    bbox_of_nonzero_class = get_bbox_from_mask(data[-1], 1)
                    bbox_of_nonzero_class_plus_margin = (
                                np.array(bbox_of_nonzero_class) + np.stack([-margins, margins], axis=1)).astype(int)
                    mask = np.zeros_like(data[-1])
                    mask[bbox_of_nonzero_class_plus_margin[0, 0]:bbox_of_nonzero_class_plus_margin[0, 1],
                    bbox_of_nonzero_class_plus_margin[1, 0]:bbox_of_nonzero_class_plus_margin[1, 1]] = 1
                    for c in range(data[:-1].shape[0]):
                        data[:-1][c][mask == 0] = 0
                if self.data_aug_params.get("mask_out_of_mask_to_zero") is not None and self.data_aug_params.get(
                        "mask_out_of_mask_to_zero"):
                    mask = data[-1]
                    for c in range(data[:-1].shape[0]):
                        data[:-1][c][mask <= 0] = 0

                softmax_pred = self.predict_preprocessed_data_return_softmax(
                    data, do_mirroring, mirror_axes, use_gaussian,
                    all_in_gpu=all_in_gpu)

                softmax_preds.append(softmax_pred)
            if len(softmax_preds) > 0:
                # patient_softmax_pred = torch.cat(softmax_preds, dim = 0).mean(dim=0, keepdim=True)
                # patient_pred = torch.argmax(patient_softmax_pred, dim=1).cpu().numpy()[0]
                # patient_softmax_pred = patient_softmax_pred.cpu().numpy()[0]
                patient_preds = torch.argmax(torch.cat(softmax_preds, dim=0), dim=1).cpu().numpy()
                patient_pred = np.argmax(np.bincount(patient_preds))
                patient_prob = torch.mean(torch.cat(softmax_preds, dim=0), dim=0).cpu().numpy()
            else:
                # patient_softmax_pred, patient_pred = None, None
                patient_pred, patient_prob = None, None
            # pred_gt_tuples.append([patient_softmax_pred, patient_pred, label])
            pred_gt_tuples.append([patient_pred, label, patient_prob[1]])

        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")

        task = self.dataset_directory.split("/")[-1]
        job_name = self.experiment_name
        _ = aggregate_classification_scores(pred_gt_tuples, list(self.dataset_test.keys()),
                                            json_output_file=join(output_folder, "summary.json"),
                                            json_name=job_name,
                                            json_author="Zhilin",
                                            json_task=task, num_threads=default_num_threads)
        self.network.train(current_mode)

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True,
                                                         step_size: float = 0.5, use_gaussian: bool = True,
                                                         pad_border_mode: str = 'constant', pad_kwargs: dict = None,
                                                         all_in_gpu: bool = True,
                                                         verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().predict_preprocessed_data_return_seg_and_softmax(data, do_mirroring, mirror_axes,
                                                                       use_sliding_window, step_size, use_gaussian,
                                                                       pad_border_mode, pad_kwargs, all_in_gpu, verbose)
        self.network.do_ds = ds
        return ret

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['cls_target']
        roi = data_dict['roi']

        if self.debug:
            for b in range(data.shape[0]):
                image = data[b, 1].numpy()
                # image = ((image - image.min()) / image.max() * 255).astype(np.uint8)
                seg = data[b, 2].numpy()
                # seg = ((seg - seg.min()) / seg.max() * 255).astype(np.uint8)

                imsave(os.path.join(self.debug_output_dir, data_dict['keys'][b] + '_image_%d.png' % b), image)
                imsave(os.path.join(self.debug_output_dir, data_dict['keys'][b] + '_seg_%d.png' % b), seg)

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)
        roi = maybe_to_torch(roi)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
            roi = to_cuda(roi)

        self.optimizer.zero_grad()

        logits, data_rec = self.network(data, roi)

        rec_loss = self.rec_loss(data_rec, data)
        cls_loss = self.cls_loss(logits, target.long())
        loss = cls_loss + rec_loss
        del data
        if run_online_evaluation:
            self.run_online_evaluation(logits, target)
        del target

        if do_backprop:
            if not self.fp16 or amp is None or not torch.cuda.is_available():
                loss.backward()
            else:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            _ = clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return cls_loss.detach().cpu().numpy()

    def do_split(self):
        """
        we now allow more than 5 splits. IMPORTANT: and fold > 4 will not be a real split but just another random
        80:20 split of the data. You cannot run X-fold cross-validation with this code. It will always be a 5-fold CV.
        Folds > 4 will be independent from each other
        :return:
        """
        if self.fold == 'all' or self.fold < 5:
            splits_file = join(self.dataset_directory, "splits_final.pkl")
            if not isfile(splits_file):
                self.print_to_log_file("Creating new split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    # splits[-1]['train'] = train_keys
                    # splits[-1]['val'] = test_keys

                    splits[-1]['test'] = test_keys
                    rnd = np.random.RandomState(seed=12345 + i)
                    idx_tr = rnd.choice(len(train_keys), int(len(train_keys) * 0.8), replace=False)
                    idx_val = [i for i in range(len(train_keys)) if i not in idx_tr]
                    splits[-1]['train'] = train_keys[idx_tr]
                    splits[-1]['val'] = train_keys[idx_val]
                save_pickle(splits, splits_file)

            splits = load_pickle(splits_file)

            if self.fold == "all":
                tr_keys = val_keys = list(self.dataset.keys())
            else:
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                testing_keys = splits[self.fold]['test']

            tr_keys.sort()
            val_keys.sort()
            testing_keys.sort()

            self.dataset_tr = OrderedDict()
            for i in tr_keys:
                self.dataset_tr[i] = self.dataset[i]

            self.dataset_val = OrderedDict()
            for i in val_keys:
                self.dataset_val[i] = self.dataset[i]

            self.dataset_test = OrderedDict()
            for i in testing_keys:
                self.dataset_test[i] = self.dataset[i]


        else:
            raise ValueError("fold larger than 5")
            # rnd = np.random.RandomState(seed=12345 + self.fold)
            # keys = np.sort(list(self.dataset.keys()))
            # idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
            # idx_val = [i for i in range(len(keys)) if i not in idx_tr]
            #
            # self.dataset_tr = OrderedDict()
            # for i in idx_tr:
            #     self.dataset_tr[keys[i]] = self.dataset[keys[i]]
            #
            # self.dataset_val = OrderedDict()
            # for i in idx_val:
            #     self.dataset_val[keys[i]] = self.dataset[keys[i]]

    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """

        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
            patch_size_for_spatialtransform = self.patch_size[1:]
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            patch_size_for_spatialtransform = self.patch_size

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = patch_size_for_spatialtransform

        self.data_aug_params["num_cached_per_thread"] = 1
        # self.data_aug_params["num_threads"] = 1

        self.data_aug_params["move_seg_chanel_to_data"] = True
        self.data_aug_params["margins_min"] = [8, 8]
        self.data_aug_params["margins_max"] = [16, 16]
        self.data_aug_params["GenerateBBox"] = True

    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs
        :return:
        """
        super().on_epoch_end()
        continue_training = self.epoch < self.max_num_epochs

        # it can rarely happen that the momentum of nnUNetTrainerV2 is too high for some dataset. If at epoch 100 the
        # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        if self.epoch == 100:
            if self.all_val_eval_metrics[-1] == 0:
                self.optimizer.param_groups[0]["momentum"] = 0.95
                self.network.apply(InitWeights_He(1e-2))
                self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
                                       "high momentum. High momentum (0.99) is good for datasets where it works, but "
                                       "sometimes causes issues such as this one. Momentum has now been reduced to "
                                       "0.95 and network weights have been reinitialized")
        return continue_training

    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.do_ds
        self.network.do_ds = False   # no deep supervision zhengzhilin980
        ret = super().run_training()
        self.network.do_ds = ds
        return ret
