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
from nnunet.training.data_augmentation.default_data_augmentation import get_moreDA_augmentation_attmap
from nnunet.network_architecture.generic_UNet_attmap import Generic_UNet_Attmap
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnunet.training.dataloading.dataset_loading import DataLoader3DV2_AttMap, unpack_dataset
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
from torch.nn.utils import clip_grad_norm_
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *
from time import time
from nnunet.postprocessing.connected_components import determine_postprocessing
from nnunet.training.data_augmentation.default_data_augmentation import default_3D_augmentation_params, \
    default_2D_augmentation_params, get_default_augmentation, get_patch_size
from nnunet.configuration import default_num_threads
from nnunet.evaluation.evaluator import aggregate_scores
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
import shutil
from collections import OrderedDict
from multiprocessing import Pool
from time import sleep
from sklearn.model_selection import KFold
from nnunet.training.loss_functions.my_dice_loss import Binary_Tversky_and_CE_loss

try:
    from apex import amp
except ImportError:
    amp = None

from ptflops import get_model_complexity_info

class nnUNetTrainer_Attention_FRCNN_stage1(nnUNetTrainer):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    Info for Zhilin: change get_basic_generators(), add a attmap channel to data, and change forward() function of network
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 1000
        self.initial_lr = 1.5e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None

        self.pin_memory = True

        self.save_latest_only = False

        self.oversample_foreground_percent = 0.5
        self.dataset_test = None
        self.loss = Binary_Tversky_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})

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

    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        if self.threeD:
            dl_tr = DataLoader3DV2_AttMap(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 False, oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides)
            dl_val = DataLoader3DV2_AttMap(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, False,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides)
        else:
            pass
        return dl_tr, dl_val

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation_attmap
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
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation_attmap(
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

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
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
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet_Attmap(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.95, nesterov=True)
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

    def run_online_evaluation(self, output, target):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """
        target = target[0]
        output = output[0]
        return super().run_online_evaluation(output, target.bool().float())

    def finish_online_evaluation(self):
        self.online_eval_tp = np.sum(self.online_eval_tp, 0)
        self.online_eval_fp = np.sum(self.online_eval_fp, 0)
        self.online_eval_fn = np.sum(self.online_eval_fn, 0)

        # global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
        #                                    zip(self.online_eval_tp, self.online_eval_fp, self.online_eval_fn)]
        #                        if not np.isnan(i)]
        # self.all_val_eval_metrics.append(np.mean(global_dc_per_class))
        global_dc_per_class = [i for i in [ i / (i + j) for i, j in zip(self.online_eval_tp, self.online_eval_fn)]
                               if not np.isnan(i)]
        self.all_val_eval_metrics.append(np.mean(global_dc_per_class))

        self.print_to_log_file("Average global foreground Recall:", str(global_dc_per_class))
        self.print_to_log_file("(interpret this as an estimate for the Dice of the different classes. This is not "
                               "exact.)")

        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []

    def train(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, num_parts:int = 1, part_id:int=0):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        current_mode = self.network.training
        self.network.eval()

        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_tr is None:
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
                         'use_sliding_window': use_sliding_window,
                         'step_size': step_size,
                         'save_softmax': save_softmax,
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

        export_pool = Pool(default_num_threads)
        results = []

        # for k in list(self.dataset_tr.keys()):
        for k in list(self.dataset_tr.keys())[part_id::num_parts]:
            properties = self.dataset[k]['properties']
            fname = properties['list_of_data_files'][0].split("/")[-1][:-12]
            if overwrite or (not isfile(join(output_folder, fname + ".nii.gz"))) or \
                    (save_softmax and not isfile(join(output_folder, fname + ".npz"))):
                # data = np.load(self.dataset[k]['data_file'])['data']
                data = np.load(self.dataset[k]['data_file'][:-4] + ".npy")

                print(k, data.shape)
                data[-2][data[-2] == -1] = 0

                softmax_pred = self.predict_preprocessed_data_return_seg_and_softmax(
                    np.concatenate([data[:-2], data[-1:]]), do_mirroring, mirror_axes, use_sliding_window, step_size,
                    use_gaussian,
                    all_in_gpu=all_in_gpu
                )[1]

                softmax_pred = softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])

                if save_softmax:
                    softmax_fname = join(output_folder, fname + ".npz")
                else:
                    softmax_fname = None

                """There is a problem with python process communication that prevents us from communicating obejcts
                larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
                communicated by the multiprocessing.Pipe object then the placeholder (\%i I think) does not allow for long
                enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
                patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will
                then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
                filename or np.ndarray and will handle this automatically"""
                if np.prod(softmax_pred.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save
                    np.save(join(output_folder, fname + ".npy"), softmax_pred)
                    softmax_pred = join(output_folder, fname + ".npy")

                results.append(export_pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                         ((softmax_pred, join(output_folder, fname + ".nii.gz"),
                                                           properties, interpolation_order, self.regions_class_order,
                                                           None, None,
                                                           softmax_fname, None, force_separate_z,
                                                           interpolation_order_z),
                                                          )
                                                         )
                               )

            pred_gt_tuples.append([join(output_folder, fname + ".nii.gz"),
                                   join(self.gt_niftis_folder, fname + "_0003.nii.gz")])

        _ = [i.get() for i in results]
        self.print_to_log_file("finished prediction")

        # # evaluate raw predictions
        # self.print_to_log_file("evaluation of raw predictions")
        # task = self.dataset_directory.split("/")[-1]
        # job_name = self.experiment_name
        # _ = aggregate_scores(pred_gt_tuples, labels=list(range(self.num_classes)),
        #                      json_output_file=join(output_folder, "summary.json"),
        #                      json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
        #                      json_author="Fabian",
        #                      json_task=task, num_threads=default_num_threads)
        #
        # # in the old nnunet we would stop here. Now we add a postprocessing. This postprocessing can remove everything
        # # except the largest connected component for each class. To see if this improves results, we do this for all
        # # classes and then rerun the evaluation. Those classes for which this resulted in an improved dice score will
        # # have this applied during inference as well
        # self.print_to_log_file("determining postprocessing")
        # determine_postprocessing(self.output_folder, self.gt_niftis_folder, validation_folder_name,
        #                          final_subf_name=validation_folder_name + "_postprocessed", debug=debug)
        # # after this the final predictions for the vlaidation set can be found in validation_folder_name_base + "_postprocessed"
        # # They are always in that folder, even if no postprocessing as applied!
        #
        # # detemining postprocesing on a per-fold basis may be OK for this fold but what if another fold finds another
        # # postprocesing to be better? In this case we need to consolidate. At the time the consolidation is going to be
        # # done we won't know what self.gt_niftis_folder was, so now we copy all the niftis into a separate folder to
        # # be used later
        # gt_nifti_folder = join(self.output_folder_base, "gt_niftis")
        # maybe_mkdir_p(gt_nifti_folder)
        # for f in subfiles(self.gt_niftis_folder, suffix=".nii.gz"):
        #     success = False
        #     attempts = 0
        #     e = None
        #     while not success and attempts < 10:
        #         try:
        #             shutil.copy(f, gt_nifti_folder)
        #             success = True
        #         except OSError as e:
        #             attempts += 1
        #             sleep(1)
        #     if not success:
        #         print("Could not copy gt nifti file %s into folder %s" % (f, gt_nifti_folder))
        #         if e is not None:
        #             raise e

        self.network.train(current_mode)

        self.network.do_ds = ds

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
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
                         'use_sliding_window': use_sliding_window,
                         'step_size': step_size,
                         'save_softmax': save_softmax,
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

        export_pool = Pool(default_num_threads)
        results = []

        for k in list(self.dataset_val.keys()):
            properties = self.dataset[k]['properties']
            fname = properties['list_of_data_files'][0].split("/")[-1][:-12]
            if overwrite or (not isfile(join(output_folder, fname + ".nii.gz"))) or \
                    (save_softmax and not isfile(join(output_folder, fname + ".npz"))):
                # data = np.load(self.dataset[k]['data_file'])['data']
                data = np.load(self.dataset[k]['data_file'][:-4] + ".npy")

                print(k, data.shape)
                data[-2][data[-2] == -1] = 0

                softmax_pred = self.predict_preprocessed_data_return_seg_and_softmax(
                    np.concatenate([data[:-2], data[-1:]]), do_mirroring, mirror_axes, use_sliding_window, step_size,
                    use_gaussian,
                    all_in_gpu=all_in_gpu
                )[1]

                softmax_pred = softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])

                if save_softmax:
                    softmax_fname = join(output_folder, fname + ".npz")
                else:
                    softmax_fname = None

                """There is a problem with python process communication that prevents us from communicating obejcts
                larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
                communicated by the multiprocessing.Pipe object then the placeholder (\%i I think) does not allow for long
                enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
                patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will
                then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
                filename or np.ndarray and will handle this automatically"""
                if np.prod(softmax_pred.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save
                    np.save(join(output_folder, fname + ".npy"), softmax_pred)
                    softmax_pred = join(output_folder, fname + ".npy")

                results.append(export_pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                         ((softmax_pred, join(output_folder, fname + ".nii.gz"),
                                                           properties, interpolation_order, self.regions_class_order,
                                                           None, None,
                                                           softmax_fname, None, force_separate_z,
                                                           interpolation_order_z),
                                                          )
                                                         )
                               )

            pred_gt_tuples.append([join(output_folder, fname + ".nii.gz"),
                                   join(self.gt_niftis_folder, fname + "_0003.nii.gz")])

        _ = [i.get() for i in results]
        self.print_to_log_file("finished prediction")

        # # evaluate raw predictions
        # self.print_to_log_file("evaluation of raw predictions")
        # task = self.dataset_directory.split("/")[-1]
        # job_name = self.experiment_name
        # _ = aggregate_scores(pred_gt_tuples, labels=list(range(self.num_classes)),
        #                      json_output_file=join(output_folder, "summary.json"),
        #                      json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
        #                      json_author="Fabian",
        #                      json_task=task, num_threads=default_num_threads)
        #
        # # in the old nnunet we would stop here. Now we add a postprocessing. This postprocessing can remove everything
        # # except the largest connected component for each class. To see if this improves results, we do this for all
        # # classes and then rerun the evaluation. Those classes for which this resulted in an improved dice score will
        # # have this applied during inference as well
        # self.print_to_log_file("determining postprocessing")
        # determine_postprocessing(self.output_folder, self.gt_niftis_folder, validation_folder_name,
        #                          final_subf_name=validation_folder_name + "_postprocessed", debug=debug)
        # # after this the final predictions for the vlaidation set can be found in validation_folder_name_base + "_postprocessed"
        # # They are always in that folder, even if no postprocessing as applied!
        #
        # # detemining postprocesing on a per-fold basis may be OK for this fold but what if another fold finds another
        # # postprocesing to be better? In this case we need to consolidate. At the time the consolidation is going to be
        # # done we won't know what self.gt_niftis_folder was, so now we copy all the niftis into a separate folder to
        # # be used later
        # gt_nifti_folder = join(self.output_folder_base, "gt_niftis")
        # maybe_mkdir_p(gt_nifti_folder)
        # for f in subfiles(self.gt_niftis_folder, suffix=".nii.gz"):
        #     success = False
        #     attempts = 0
        #     e = None
        #     while not success and attempts < 10:
        #         try:
        #             shutil.copy(f, gt_nifti_folder)
        #             success = True
        #         except OSError as e:
        #             attempts += 1
        #             sleep(1)
        #     if not success:
        #         print("Could not copy gt nifti file %s into folder %s" % (f, gt_nifti_folder))
        #         if e is not None:
        #             raise e

        self.network.train(current_mode)

        self.network.do_ds = ds

    def test(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'testing_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None):
        ds = self.network.do_ds
        self.network.do_ds = False


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
                         'use_sliding_window': use_sliding_window,
                         'step_size': step_size,
                         'save_softmax': save_softmax,
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

        export_pool = Pool(default_num_threads)
        results = []

        # for k in list(self.dataset_test.keys())[3::6]:
        for k in self.dataset_test.keys():
            properties = self.dataset[k]['properties']
            fname = properties['list_of_data_files'][0].split("/")[-1][:-12]
            if overwrite or (not isfile(join(output_folder, fname + ".nii.gz"))) or \
                    (save_softmax and not isfile(join(output_folder, fname + ".npz"))):
                data = np.load(self.dataset[k]['data_file'][:-4] + '.npy')

                print(k, data.shape)
                data[-2][data[-2] == -1] = 0

                softmax_pred = self.predict_preprocessed_data_return_seg_and_softmax(
                    np.concatenate([data[:-2], data[-1:]]), do_mirroring, mirror_axes, use_sliding_window, step_size, use_gaussian,
                    all_in_gpu=all_in_gpu
                )[1]

                softmax_pred = softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])

                if save_softmax:
                    softmax_fname = join(output_folder, fname + ".npz")
                else:
                    softmax_fname = None

                """There is a problem with python process communication that prevents us from communicating obejcts
                larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
                communicated by the multiprocessing.Pipe object then the placeholder (\%i I think) does not allow for long
                enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
                patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will
                then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
                filename or np.ndarray and will handle this automatically"""
                if np.prod(softmax_pred.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save
                    np.save(join(output_folder, fname + ".npy"), softmax_pred)
                    softmax_pred = join(output_folder, fname + ".npy")

                results.append(export_pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                         ((softmax_pred, join(output_folder, fname + ".nii.gz"),
                                                           properties, interpolation_order, self.regions_class_order,
                                                           None, None,
                                                           softmax_fname, None, force_separate_z,
                                                           interpolation_order_z),
                                                          )
                                                         )
                               )

            pred_gt_tuples.append([join(output_folder, fname + ".nii.gz"),
                                   join(self.gt_niftis_folder, fname + "_0003.nii.gz")])

        _ = [i.get() for i in results]
        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")
        task = self.dataset_directory.split("/")[-1]
        job_name = self.experiment_name
        _ = aggregate_scores(pred_gt_tuples, labels=list(range(self.num_classes)),
                             json_output_file=join(output_folder, "summary.json"),
                             json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
                             json_author="Fabian",
                             json_task=task, num_threads=default_num_threads)

        # in the old nnunet we would stop here. Now we add a postprocessing. This postprocessing can remove everything
        # except the largest connected component for each class. To see if this improves results, we do this for all
        # classes and then rerun the evaluation. Those classes for which this resulted in an improved dice score will
        # have this applied during inference as well
        self.print_to_log_file("determining postprocessing")
        determine_postprocessing(self.output_folder, self.gt_niftis_folder, validation_folder_name,
                                 final_subf_name=validation_folder_name + "_postprocessed", debug=debug)
        # after this the final predictions for the vlaidation set can be found in validation_folder_name_base + "_postprocessed"
        # They are always in that folder, even if no postprocessing as applied!

        # detemining postprocesing on a per-fold basis may be OK for this fold but what if another fold finds another
        # postprocesing to be better? In this case we need to consolidate. At the time the consolidation is going to be
        # done we won't know what self.gt_niftis_folder was, so now we copy all the niftis into a separate folder to
        # be used later
        gt_nifti_folder = join(self.output_folder_base, "gt_niftis")
        maybe_mkdir_p(gt_nifti_folder)
        for f in subfiles(self.gt_niftis_folder, suffix=".nii.gz"):
            success = False
            attempts = 0
            e = None
            while not success and attempts < 10:
                try:
                    shutil.copy(f, gt_nifti_folder)
                    success = True
                except OSError as e:
                    attempts += 1
                    sleep(1)
            if not success:
                print("Could not copy gt nifti file %s into folder %s" % (f, gt_nifti_folder))
                if e is not None:
                    raise e

        self.network.train(current_mode)

        self.network.do_ds = ds


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

    def get_model_complexity(self):
        def prepare_input(resolution):
            x = torch.FloatTensor(1, 2, *(resolution))
            a = torch.FloatTensor(1, 1, *(resolution))
            return dict(x=x.cuda(), a=a.cuda())

        self.network.do_ds = False
        macs, params = get_model_complexity_info(self.network, input_res=tuple(self.patch_size), input_constructor=prepare_input, as_strings=True, print_per_layer_stat=False)
        print(' - MACs: ' + macs)
        print(' - Params: ' + params)


    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        # time1 = time()
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']
        attmap = data_dict['attmap']
        # time2 = time()
        # print('Generating a batch takes %.2f s, keys: %s, %s' %((time2-time1), data_dict['keys'][0], data_dict['keys'][1]))

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)
        attmap = maybe_to_torch(attmap)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
            attmap = to_cuda(attmap)

        self.optimizer.zero_grad()

        output = self.network(data, attmap)
        # time3 = time()
        # print('Forward pass takes %.2f s' % (time3 - time2))

        del data
        del attmap
        loss = self.loss(output, target)

        if run_online_evaluation:
            self.run_online_evaluation(output, target)
        del target

        if do_backprop:
            if not self.fp16 or amp is None or not torch.cuda.is_available():
                loss.backward()
            else:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            _ = clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        # time4 = time()
        # print('This iteration took %.2f s' % (time4 - time1))

        return loss.detach().cpu().numpy()

    # def do_split(self):
    #     """
    #     we now allow more than 5 splits. IMPORTANT: and fold > 4 will not be a real split but just another random
    #     80:20 split of the data. You cannot run X-fold cross-validation with this code. It will always be a 5-fold CV.
    #     Folds > 4 will be independent from each other
    #     :return:
    #     """
    #     if self.fold == 'all' or self.fold < 5:
    #         return super().do_split()
    #     else:
    #         rnd = np.random.RandomState(seed=12345 + self.fold)
    #         keys = np.sort(list(self.dataset.keys()))
    #         idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
    #         idx_val = [i for i in range(len(keys)) if i not in idx_tr]
    #
    #         self.dataset_tr = OrderedDict()
    #         for i in idx_tr:
    #             self.dataset_tr[keys[i]] = self.dataset[keys[i]]
    #
    #         self.dataset_val = OrderedDict()
    #         for i in idx_val:
    #             self.dataset_val[keys[i]] = self.dataset[keys[i]]

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
        self.data_aug_params['selected_seg_channels'] = [0,1] #[0]
        self.data_aug_params['patch_size_for_spatialtransform'] = patch_size_for_spatialtransform

        self.data_aug_params["num_cached_per_thread"] = 2
        # self.data_aug_params["num_threads"] = 14

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
        self.network.do_ds = True
        ret = super().run_training()
        self.network.do_ds = ds
        return ret

    def load_pretrained_checkpoint(self, fname, train=True):
        self.print_to_log_file("loading checkpoint", fname, "train=", train)
        if not self.was_initialized:
            self.initialize(train)
        pretrained_model = torch.load(fname, map_location=torch.device('cpu'))
        if self.fp16:
            self.network, self.optimizer, self.lr_scheduler = None, None, None
            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

        new_state_dict = OrderedDict()
        curr_state_dict_keys = list(self.network.state_dict().keys())
        for k, value in pretrained_model['state_dict'].items():
            key = k
            if key not in curr_state_dict_keys:
                print("duh")
                key = key[7:]
            new_state_dict[key] = value

        self.network.load_state_dict(new_state_dict)
        self.amp_initialized = False
        self._maybe_init_amp()

    def load_chk_checkpoint(self, chk,train=True):
        if isfile(join(self.output_folder, chk)):
            return self.load_checkpoint(join(self.output_folder, chk), train=train)
        raise RuntimeError("No checkpoint found")