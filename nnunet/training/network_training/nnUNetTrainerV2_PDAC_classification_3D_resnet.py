from nnunet.training.network_training.nnUNetTrainerV2_PDAC_classification_resnet import nnUNetTrainerV2_PDAC_classification_resnet
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params, get_moreDA_augmentation_classification
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.network_architecture.ResNet3D import generate_model, generate_model_masksidebranch
from nnunet.training.dataloading.dataset_loading import DataLoader3D_Classification
from sklearn.metrics import roc_auc_score
from skimage.io import imsave
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from torch.nn.utils import clip_grad_norm_

from nnunet.configuration import default_num_threads
from multiprocessing.pool import Pool
from nnunet.evaluation.evaluator import aggregate_classification_scores
import pandas as pd
from collections import defaultdict
from torch import nn
from collections import OrderedDict
from typing import Tuple
from nnunet.network_architecture.neural_network import SegmentationNetwork, NeuralNetwork
from nnunet.training.dataloading.dataset_loading import load_dataset
from ptflops import get_model_complexity_info
try:
    from apex import amp
except ImportError:
    amp = None


class nnUNetTrainerV2_PDAC_classification_3D_resnet(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 500
        self.initial_lr = 1e-3
        self.oversample_foreground_percent = 1.

        self.cls_loss = nn.CrossEntropyLoss(reduction='mean')

        self.training_table = '/data86/zhengzhilin980/pancreas/PDAC_new/Patients_addN0.xlsx'

        self.dataset_test = None

        self.weight_decay = 5e-4

        self.online_eval_label, self.online_eval_prob, self.online_eval_tn = [], [], []
        self.pretrained = True
        self.if_freeze = False
        self.freeze_layernum = 0

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
        self.data_aug_params["margins_min"] = [8, 8, 8]
        self.data_aug_params["margins_max"] = [16, 16, 16]
        self.data_aug_params["GenerateBBox"] = True

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


        self.network = generate_model(model_depth=18, n_classes=self.num_classes, pretrained=self.pretrained, if_freeze=self.if_freeze, freeze_layernum = self.freeze_layernum)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

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
            dl_tr = DataLoader3D_Classification(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 False, oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, mode='train')
            dl_val = DataLoader3D_Classification(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, False,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, mode='train')
        else:
            # dl_tr = DataLoader2D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
            #                      transpose=None,  # self.plans.get('transpose_forward'),
            #                      oversample_foreground_percent=self.oversample_foreground_percent,
            #                      pad_mode="constant", pad_sides=self.pad_all_sides)
            # dl_val = DataLoader2D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
            #                       transpose=None,  # self.plans.get('transpose_forward'),
            #                       oversample_foreground_percent=self.oversample_foreground_percent,
            #                       pad_mode="constant", pad_sides=self.pad_all_sides)
            pass
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
            tn_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fn_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            for c in range(1, num_classes):
                tp_hard[:, c - 1] = (output_seg == c).float() * (target == c).float()
                tn_hard[:, c - 1] = (output_seg != c).float() * (target != c).float()
                fp_hard[:, c - 1] = (output_seg == c).float() * (target != c).float()
                fn_hard[:, c - 1] = (output_seg != c).float() * (target == c).float()

            tp_hard = tp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            tn_hard = tn_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fp_hard = fp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fn_hard = fn_hard.sum(0, keepdim=False).detach().cpu().numpy()

            self.online_eval_foreground_dc.append(list((tp_hard + tn_hard) / (tp_hard + tn_hard + fp_hard + fn_hard + 1e-8)))
            self.online_eval_tp.append(list(tp_hard))
            self.online_eval_tn.append(list(tn_hard))
            self.online_eval_fp.append(list(fp_hard))
            self.online_eval_fn.append(list(fn_hard))
            self.online_eval_label.extend(list(target.cpu().numpy()))
            self.online_eval_prob.extend(list(output_softmax[:, 1].detach().cpu().numpy()))


    def finish_online_evaluation(self):
        # self.online_eval_tp = np.sum(self.online_eval_tp, 0)
        # self.online_eval_fp = np.sum(self.online_eval_fp, 0)
        # self.online_eval_fn = np.sum(self.online_eval_fn, 0)
        # self.online_eval_tn = np.sum(self.online_eval_tn, 0)
        #
        # global_dc_per_class = [i for i in [(i + j) / (i + j + k + m) for i, j, k, m in
        #                                    zip(self.online_eval_tp, self.online_eval_tn, self.online_eval_fp, self.online_eval_fn)]
        #                        if not np.isnan(i)]
        # self.all_val_eval_metrics.append(np.mean(global_dc_per_class))
        #
        # self.print_to_log_file("Average global foreground Accuracy:", str(global_dc_per_class))

        global_dc_per_class = roc_auc_score(y_true=self.online_eval_label, y_score=self.online_eval_prob)
        self.all_val_eval_metrics.append(global_dc_per_class)

        self.print_to_log_file("Average global foreground AUC:", str(global_dc_per_class))
        self.print_to_log_file("(interpret this as an estimate for the Dice of the different classes. This is not "
                               "exact.)")

        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []
        self.online_eval_tn = []
        self.online_eval_label = []
        self.online_eval_prob = []

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
        # roi = data_dict['roi']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)
        # roi = maybe_to_torch(roi)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
            # roi = to_cuda(roi)

        self.optimizer.zero_grad()

        logits = self.network(data)

        cls_loss = self.cls_loss(logits, target.long())
        loss = cls_loss

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


    def predict_preprocessed_data_return_class_and_score(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None, use_gaussian: bool = True,
                                                         pad_border_mode: str = 'constant', pad_kwargs: dict = None,
                                                         all_in_gpu: bool = True,
                                                         verbose: bool = True, return_feature : bool=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param data:
        :param do_mirroring:
        :param mirror_axes:
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

        valid = list((NeuralNetwork, nn.DataParallel))
        assert isinstance(self.network, tuple(valid))

        current_mode = self.network.training
        self.network.eval()
        ret = self.network.predict_3D_Classification(data, do_mirroring, mirror_axes, self.patch_size,
                                      self.regions_class_order, use_gaussian, pad_border_mode, pad_kwargs,
                                      all_in_gpu, verbose, return_feature=return_feature)
        self.network.train(current_mode)

        return ret



    def validate_classification(self, do_mirroring: bool = True, use_gaussian: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None):
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
        results = OrderedDict()

        for k in self.dataset_val.keys():
            properties = self.dataset[k]['properties']
            fname = properties['list_of_data_files'][0].split("/")[-1][:-12]
            # data = np.load(self.dataset[k]['data_file'])['data']
            data = np.load(self.dataset[k]['data_file'][:-4]+'.npy')
            label = int(bool(training_df[training_df['LNM_identifier'] == k]['N'].values[0]))
            print(k, data.shape)
            data[-1][data[-1] == -1] = 0

            pred = self.predict_preprocessed_data_return_class_and_score(
                data, do_mirroring, mirror_axes, use_gaussian,
                all_in_gpu=all_in_gpu)

            pred_gt_tuples.append([pred[0], label, pred[1][0, 1]])


        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")
        task = self.dataset_directory.split("/")[-1]
        job_name = self.experiment_name
        _ = aggregate_classification_scores(pred_gt_tuples, list(self.dataset_val.keys()),
                                            json_output_file=join(output_folder, "summary.json"),
                                            json_name=job_name,
                                            json_author="Zhilin",
                                            json_task=task, num_threads=default_num_threads)

        self.network.train(current_mode)

    def test_classification(self, do_mirroring: bool = True, use_gaussian: bool = True,
                                validation_folder_name: str = 'validation_raw', debug: bool = False,
                                all_in_gpu: bool = False,
                                segmentation_export_kwargs: dict = None):
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
        results = OrderedDict()

        for k in self.dataset_test.keys():
            properties = self.dataset[k]['properties']
            fname = properties['list_of_data_files'][0].split("/")[-1][:-12]
            # data = np.load(self.dataset[k]['data_file'])['data']
            data = np.load(self.dataset[k]['data_file'][:-4] + '.npy')
            label = int(bool(training_df[training_df['LNM_identifier'] == k]['N'].values[0]))
            print(k, data.shape)
            data[-1][data[-1] == -1] = 0

            pred = self.predict_preprocessed_data_return_class_and_score(
                data, do_mirroring, mirror_axes, use_gaussian,
                all_in_gpu=all_in_gpu)

            pred_gt_tuples.append([pred[0], label, pred[1][0, 1]])

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


    def holdout_test_classification(self, do_mirroring: bool = True, use_gaussian: bool = True,
                                validation_folder_name: str = 'validation_raw', debug: bool = False,
                                all_in_gpu: bool = False,
                                segmentation_export_kwargs: dict = None):
        """
        if debug=True then the temporary files generated for postprocessing determination will be kept
        """

        current_mode = self.network.training
        self.network.eval()

        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        holdout_dataset_test = load_dataset('/data87/pancreaticCancer/nnUNETFrame/nnUNET_DATASET/nnUNET_preprocessed/Task511_SJUDataset_PDAC_classification/nnUNetData_plans_v2.1_stage1/')
        # holdout_dataset_test = load_dataset('/data87/pancreaticCancer/nnUNETFrame/nnUNET_DATASET/nnUNET_preprocessed/Task513_changhai_xiaoyiai_PDAC_classification/nnUNetData_plans_v2.1_2D_stage0/')
        # holdout_dataset_test = load_dataset('/data87/pancreaticCancer/nnUNETFrame/nnUNET_DATASET/nnUNET_preprocessed/Task512_LHGDDataset_PDAC_classification/nnUNetData_plans_v2.1_stage1/')

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

        pred_gt_tuples, keys = [], []

        training_table = '/data87/pancreaticCancer/Media/SJU/patients.xlsx'
        # training_table = '/data87/pancreaticCancer/changhai_xiaoyiai/小胰癌.xlsx'
        # training_table = '/data87/pancreaticCancer/luhong_guangdong/2021-112-tianjing-guangdong-treatment.xlsx'
        training_df = pd.read_excel(training_table)

        export_pool = Pool(default_num_threads)
        results = OrderedDict()

        for k in holdout_dataset_test.keys():
            properties = holdout_dataset_test[k]['properties']
            fname = properties['list_of_data_files'][0].split("/")[-1][:-12]
            data = np.load(holdout_dataset_test[k]['data_file'])['data']
            # data = np.load(self.dataset[k]['data_file'][:-4] + '.npy')
            label = int(bool(training_df[training_df['LNM_identifier'] == k]['N'].values[0]))
            print(k, data.shape)
            data[-1][data[-1] == -1] = 0

            pred = self.predict_preprocessed_data_return_class_and_score(
                data, do_mirroring, mirror_axes, use_gaussian,
                all_in_gpu=all_in_gpu)

            try:
                pred_gt_tuples.append([pred[0], label, pred[1][0, 1]])
                keys.append(k)
            except:
                print("No PDAC")

        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")
        task = self.dataset_directory.split("/")[-1]
        job_name = self.experiment_name
        _ = aggregate_classification_scores(pred_gt_tuples, keys,
                                            json_output_file=join(output_folder, "summary.json"),
                                            json_name=job_name,
                                            json_author="Zhilin",
                                            json_task=task, num_threads=default_num_threads)

        self.network.train(current_mode)

class nnUNetTrainerV2_PDAC_classification_3D_resnet_MaskSideBranch(nnUNetTrainerV2_PDAC_classification_3D_resnet):
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


        self.network = generate_model_masksidebranch(model_depth=18, n_classes=self.num_classes, pretrained=self.pretrained, if_freeze=self.if_freeze,
                                                     freeze_layernum = self.freeze_layernum, mask_input_method='multiplication', mask_input_layer=[1])
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def get_model_complexity(self):
        def prepare_input(resolution):
            x = torch.FloatTensor(1, 3, *(resolution))
            a = torch.FloatTensor(1, 1, *(resolution))
            return dict(x=x.cuda(), m=a.cuda())

        self.network.do_ds = False
        macs, params = get_model_complexity_info(self.network, input_res=tuple(self.patch_size), input_constructor=prepare_input, as_strings=True, print_per_layer_stat=True)
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
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['cls_target']
        # roi = data_dict['roi']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)
        # roi = maybe_to_torch(roi)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
            # roi = to_cuda(roi)

        self.optimizer.zero_grad()

        logits = self.network(data, data[:, -1:])

        cls_loss = self.cls_loss(logits, target.long())
        loss = cls_loss

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




