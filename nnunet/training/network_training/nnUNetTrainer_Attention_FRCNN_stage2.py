#    Created by zhilin zheng for lymph node metastasis prediction

from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.training.data_augmentation.default_data_augmentation import get_moreDA_augmentation_classification
from nnunet.network_architecture.generic_Classification_roipool import Generic_Classification_RoIPooling, Generic_Classification_MaskSideBranch
from nnunet.network_architecture.initialization import InitWeights_He_Classification
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnunet.network_architecture.neural_network import NeuralNetwork
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
from torch.nn.utils import clip_grad_norm_
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.dataloading.dataset_loading import  DataLoader3D_Classification
from nnunet.training.loss_functions.focal_loss import FocalLoss
from multiprocessing.pool import Pool
from nnunet.configuration import default_num_threads
from collections import OrderedDict
from datetime import datetime
import hashlib
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc, balanced_accuracy_score
import pandas as pd
from nnunet.training.dataloading.dataset_loading_semi import  load_dataset_LN_stage2
from nnunet.training.data_augmentation.my_custom_transform import get_bbox_from_mask
try:
    from apex import amp
except ImportError:
    amp = None
from ptflops import  get_model_complexity_info

def aggregate_scores(results,
                     json_output_file=None,
                     json_name="",
                     json_description="",
                     json_author="Zhilin",
                     json_task="",
                     num_threads=2):
    """
    test = predicted image
    :param test_ref_pairs:
    :param evaluator:
    :param labels: must be a dict of int-> str or a list of int
    :param nanmean:
    :param json_output_file:
    :param json_name:
    :param json_description:j
    :param json_author:
    :param json_task:
    :param metric_kwargs:
    :return:
    """
    all_scores = results

    # all_label = [results[k]['gt_class'] for k in results.keys() if results[k]['pred_class'] != -1 and results[k]['gt_class'] !=-1]
    # all_pred =  [results[k]['pred_class'] for k in results.keys() if results[k]['pred_class'] != -1 and results[k]['gt_class'] !=-1]
    # all_prop = [results[k]['pred_score'][0][1] for k in results.keys() if results[k]['pred_class'] != -1 and results[k]['gt_class'] !=-1]

    all_label = [results[k]['gt_class'] for k in results.keys() ]
    all_prop = [results[k]['pred_score'][0][1] for k in results.keys()]

    fpr, tpr, thresholds = roc_curve(y_true=all_label, y_score=all_prop)
    auc_score = auc(fpr, tpr)
    # maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))
    # threshold = thresholds[maxindex]

    # all_pred = [int(results[k]['pred_score'][0][1] >= threshold) for k in results.keys()]
    all_pred = [int(np.argmax(results[k]['pred_score'][0])) for k in results.keys()]
    for idx, k in enumerate(results.keys()):
        results[k]['pred_class'] = all_pred[idx]
    confusion_matrix = metrics.confusion_matrix(all_label, all_pred)

    target_names = ['N_1', 'N_0']
    classification_report = metrics.classification_report(all_label, all_pred, target_names=target_names)
    accuracy = metrics.accuracy_score(all_label, all_pred)
    balanced_accuracy = metrics.balanced_accuracy_score(all_label, all_pred)

    # auc = roc_auc_score(y_true=all_label, y_score=all_prop)
    # all_scores['overall'] = {"confusion_matrix":confusion_matrix.tolist(), "classification_report": classification_report, "accuracy": accuracy,
    #                          "balanced accuracy":balanced_accuracy, "auc": auc_score, "threshold":threshold}
    all_scores['overall'] = {"confusion_matrix": confusion_matrix.tolist(),
                             "classification_report": classification_report, "accuracy": accuracy,
                             "balanced accuracy": balanced_accuracy, "auc": auc_score}

    # save to file if desired
    # we create a hopefully unique id by hashing the entire output dictionary
    if json_output_file is not None:
        json_dict = OrderedDict()
        json_dict["name"] = json_name
        json_dict["description"] = json_description
        timestamp = datetime.today()
        json_dict["timestamp"] = str(timestamp)
        json_dict["task"] = json_task
        json_dict["author"] = json_author
        json_dict["results"] = all_scores
        json_dict["id"] = hashlib.md5(json.dumps(json_dict).encode("utf-8")).hexdigest()[:12]
        save_json(json_dict, json_output_file)


    return all_scores

class nnUNetTrainer_Attention_FRCNN_stage2(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, initial_lr=2e-5, pretrain = False, opt = 'SGD', weight_decay=1e-4):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 500  # 1000  modified by zhengzhilin980
        self.loss = nn.CrossEntropyLoss(ignore_index=-1) # 216
        # self.loss =FocalLoss(alpha=torch.Tensor([0.6, 0.6, 1.8]).cuda(), gamma=2, ignore_index=-1) # 215
        # self.loss = MSELoss_Ignore(ignore_index=-1)
        self.online_eval_tn = []
        self.initial_lr = initial_lr
        self.pretrain = pretrain
        self.opt = opt
        self.weight_decay = weight_decay
        self.save_every = 10
        self.num_batches_per_epoch = 50
        self.num_val_batches_per_epoch = 10
        self.dataset_test = None
        self.online_eval_label, self.online_eval_prob = [], []

    def setup_DA_params(self):
        super().setup_DA_params()
        self.deep_supervision_scales = None
        self.data_aug_params['move_last_seg_chanel_to_data'] = False

        self.data_aug_params['do_rotation'] = True
        self.data_aug_params['do_scaling'] = False #True
        self.data_aug_params['random_crop'] = False

        self.data_aug_params['MaskTransform'] = 'V4'
        self.data_aug_params['margins_min'] = [4, 4, 4]
        self.data_aug_params['margins_max'] = [8, 8, 8]
        self.data_aug_params['GenerateBBox'] = True

        self.data_aug_params["num_cached_per_thread"] = 2
        # self.data_aug_params["num_threads"] = 24

    # def do_split(self):
    #     """
    #     This is a suggestion for if your dataset is a dictionary (my personal standard)
    #     :return:
    #     """
    #     splits_file = join(self.dataset_directory, "splits_final.pkl")
    #     if not isfile(splits_file):
    #         self.print_to_log_file("Creating new split...")
    #         splits = [OrderedDict()]
    #         all_keys_sorted = np.sort(list(self.dataset.keys()))
    #         splits[0]['train'] = np.array([key for key in all_keys_sorted if key.startswith('pdac_')])
    #         splits[0]['val'] = np.array([key for key in all_keys_sorted if key.startswith('pdactestset_')])
    #         save_pickle(splits, splits_file)
    #
    #     splits = load_pickle(splits_file)
    #
    #     if self.fold == "all":
    #         tr_keys = val_keys = list(self.dataset.keys())
    #     else:
    #         tr_keys = splits[self.fold]['train']
    #         val_keys = splits[self.fold]['val']
    #
    #     tr_keys.sort()
    #     val_keys.sort()
    #
    #     self.dataset_tr = OrderedDict()
    #     for i in tr_keys:
    #         self.dataset_tr[i] = self.dataset[i]
    #
    #     self.dataset_val = OrderedDict()
    #     for i in val_keys:
    #         self.dataset_val[i] = self.dataset[i]

    def do_split(self):
        """
        This is a suggestion for if your dataset is a dictionary (my personal standard)
        :return:
        """
        splits_file = join(self.dataset_directory, "splits_final.pkl")
        image_splits_file = join(self.dataset_directory, "image_splits_final.pkl")
        # if not isfile(splits_file):
        image_splits = load_pickle(image_splits_file)
        self.print_to_log_file("Creating new split...")
        splits = []
        all_keys_sorted = np.sort(list(self.dataset.keys()))
        for i in range(5):
            image_train_keys = image_splits[i]['train']
            image_val_keys = image_splits[i]['val']
            image_test_keys = image_splits[i]['test']
            splits.append(OrderedDict())
            splits[-1]['train'] = np.array([key for key in all_keys_sorted if key[:8] in image_train_keys])
            splits[-1]['val'] = np.array([key for key in all_keys_sorted if key[:8] in image_val_keys])
            splits[-1]['test'] = np.array([key for key in all_keys_sorted if key[:8] in image_test_keys])
        save_pickle(splits, splits_file)

        # splits = load_pickle(splits_file)

        if self.fold == "all":
            tr_keys = val_keys =test_keys= list(self.dataset.keys())
        else:
            tr_keys = splits[self.fold]['train']
            val_keys = splits[self.fold]['val']
            test_keys = splits[self.fold]['test']

        tr_keys.sort()
        val_keys.sort()
        test_keys.sort()

        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]

        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

        self.dataset_test = OrderedDict()
        for i in test_keys:
            self.dataset_test[i] = self.dataset[i]

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

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        if self.pretrain:
            # if self.opt == 'SGD':
            #     self.optimizer = torch.optim.SGD([{'params':self.network.conv_blocks_context.parameters(), 'lr':self.initial_lr * 0.1}, {'params':self.network.cls_branch.parameters()}], self.initial_lr, weight_decay=self.weight_decay,
            #                                      momentum=0.99, nesterov=True)
            # elif self.opt == 'Adam':
            #     self.optimizer = torch.optim.Adam([{'params':self.network.conv_blocks_context.parameters(), 'lr':self.initial_lr * 0.1}, {'params':self.network.cls_branch.parameters()}], self.initial_lr, weight_decay=self.weight_decay)
            if self.opt == 'SGD':
                self.optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad is not False, self.network.parameters()), self.initial_lr, weight_decay=self.weight_decay,
                    momentum=0.95, nesterov=True)
            elif self.opt == 'Adam':
                self.optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad is not False, self.network.parameters()), self.initial_lr, weight_decay=self.weight_decay)

        else:
            if self.opt == 'SGD':
                self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                                 momentum=0.95, nesterov=True)
            elif self.opt == 'Adam':
                self.optimizer = torch.optim.Adam(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay)

        self.lr_scheduler = None

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
        # if self.pretrain:
        #     self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr*0.1, 0.9)
        #     self.optimizer.param_groups[1]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        #     self.print_to_log_file("encoder lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))
        #     self.print_to_log_file("cls lr:", np.round(self.optimizer.param_groups[1]['lr'], decimals=6))
        # else:
        #     self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        #     self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation_classification
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
            self.num_classes = self.plans['num_classes']
            # self.num_input_channels +=1

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            # weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

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
                    unpack_dataset(self.folder_with_preprocessed_data)
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

            assert isinstance(self.network, (NeuralNetwork, nn.DataParallel))
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
            # norm_op = nn.BatchNorm3d ##219 old

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d
            # norm_op = nn.BatchNorm2d ##219 old

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        # dropout_op_kwargs = {'p': 0, 'inplace': True}
        dropout_op_kwargs = {'p': 0.0, 'inplace': True}  # 207
        # fc_dropout_op_kwargs = {'p': 0.9, 'inplace': False} #207....
        fc_dropout_op_kwargs = {'p': 0.0, 'inplace': False}  # 206, 219
        # fc_dropout_op_kwargs = {'p': 0.0, 'inplace': False}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        # net_nonlin = nn.ReLU    # 219 old
        # net_nonlin_kwargs = {'inplace': True}
        self.network = Generic_Classification_RoIPooling(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes), self.net_pool_per_axis,
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He_Classification(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True, fc_dropout_op_kwargs=fc_dropout_op_kwargs)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

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
        # target = []
        # roi = []
        # for idx, data_prop in enumerate(data_dict['properties']):
        #     if data_prop.get('roi') is not None:
        #         roi.append(data_prop['roi'])
        #         target.append(int(data_prop['cls_label']))
        #     else:
        #         roi.append(np.array([[0,16], [0,64], [0,64]])) ### random values
        #         target.append(-1)
        # target = np.array(target)
        # roi = np.array(roi)

        data = maybe_to_torch(data)
        roi = maybe_to_torch(roi)
        target = maybe_to_torch(target)

        # def processClsTarget(cls_target, seg):
        #     has_pdac = any_tensor(seg !=0,axes = tuple(range(1, len(seg.shape))))
        #     return cls_target * has_pdac + ~has_pdac* (-1)
        # target = processClsTarget(target, data[:, -1])

        if torch.cuda.is_available():
            data = to_cuda(data)
            roi = to_cuda(roi)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        output = self.network(data, roi)

        del data
        del roi
        # loss = self.loss(output.squeeze(), target)
        loss = self.loss(output.squeeze(), target.long())

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

        return loss.detach().cpu().numpy()

    # def run_online_evaluation(self, output, target):
    #     with torch.no_grad():
    #         num_classes = output.shape[1]
    #         output_softmax = softmax_helper(output)
    #         output_seg = output_softmax.argmax(1) # (batch_size, )
    #
    #         tp_hard = torch.zeros((target.shape[0], num_classes)).to(output_seg.device.index)
    #         fp_hard = torch.zeros((target.shape[0], num_classes)).to(output_seg.device.index)
    #         fn_hard = torch.zeros((target.shape[0], num_classes)).to(output_seg.device.index)
    #         tn_hard = torch.zeros((target.shape[0], num_classes)).to(output_seg.device.index)
    #         for c in range(num_classes):
    #             tp_hard[:, c ] = (output_seg == c).float() * (target == c).float()
    #             fp_hard[:, c] = (output_seg == c).float() * (target != c).float()
    #             fn_hard[:, c] = (output_seg != c).float() * (target == c).float()
    #             tn_hard[:, c] = (output_seg != c).float() * (target != c).float()
    #
    #         tp_hard = tp_hard.sum(0, keepdim=False).detach().cpu().numpy()
    #         fp_hard = fp_hard.sum(0, keepdim=False).detach().cpu().numpy()
    #         fn_hard = fn_hard.sum(0, keepdim=False).detach().cpu().numpy()
    #         tn_hard = tn_hard.sum(0, keepdim=False).detach().cpu().numpy()
    #
    #         self.online_eval_foreground_dc.append(list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)))
    #         self.online_eval_tp.append(list(tp_hard))
    #         self.online_eval_fp.append(list(fp_hard))
    #         self.online_eval_fn.append(list(fn_hard))
    #         self.online_eval_tn.append(list(tn_hard))

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
            nonnan_idx = np.isnan(output_softmax[:, 1].detach().cpu().numpy())
            self.online_eval_label.extend(list(target.cpu().numpy()[~nonnan_idx]))
            self.online_eval_prob.extend(list(output_softmax[:, 1].detach().cpu().numpy()[~nonnan_idx]))

    # def finish_online_evaluation(self):
    #     self.online_eval_tp = np.sum(self.online_eval_tp, 0)
    #     self.online_eval_fp = np.sum(self.online_eval_fp, 0)
    #     self.online_eval_fn = np.sum(self.online_eval_fn, 0)
    #     self.online_eval_tn = np.sum(self.online_eval_tn, 0)
    #
    #     global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
    #                                        zip(self.online_eval_tp, self.online_eval_fp, self.online_eval_fn)]
    #                            if not np.isnan(i)]
    #     self.all_val_eval_metrics.append(np.mean(global_dc_per_class))
    #     self.print_to_log_file("Average global dice:", np.mean(global_dc_per_class))
    #     # global_acc_per_class = [i for i in [i / (i+j+k+m) for i, j, k, m in
    #     #                                    zip(self.online_eval_tp, self.online_eval_fp, self.online_eval_fn, self.online_eval_tn)]
    #     #                        if not np.isnan(i)]
    #     # self.all_val_eval_metrics.append(np.sum(global_acc_per_class))
    #
    #     # self.print_to_log_file("Average global accuracy:", np.sum(global_acc_per_class))
    #     self.print_to_log_file("(interpret this as an estimate for the Dice of the different classes. This is not "
    #                            "exact.)")
    #
    #     self.online_eval_foreground_dc = []
    #     self.online_eval_tp = []
    #     self.online_eval_fp = []
    #     self.online_eval_fn = []
    #     self.online_eval_tn = []

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

        export_pool = Pool(default_num_threads)
        results = OrderedDict()

        for k in self.dataset_val.keys():
            properties = self.dataset[k]['properties']
            fname = properties['list_of_data_files'][0].split("/")[-1][:-12]
            # data = np.load(self.dataset[k]['data_file'])['data']
            data = np.load(self.dataset[k]['data_file'][:-4]+'.npy')

            print(k, data.shape)
            data[-1][data[-1] == -1] = 0

            pred = self.predict_preprocessed_data_return_class_and_score(
                data, do_mirroring, mirror_axes, use_gaussian,
                all_in_gpu=all_in_gpu)

            # results[k] = {'pred_class': float(pred[0][0]), 'pred_score': float(pred[1][0]), 'gt_class':properties['cls_label']}
            # results[k] = {'pred_class': float(pred[0]), 'pred_score': pred[1].tolist(), 'gt_class': properties['cls_target']-1}
            if not np.isnan(pred[1]).any():
                results[k] = {'pred_class': None, 'pred_score': pred[1].tolist(), 'gt_class': properties['cls_target']-1}

                print(pred[1].tolist(), properties['cls_target']-1)

        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")
        task = self.dataset_directory.split("/")[-1]
        job_name = self.experiment_name
        _ = aggregate_scores(results,
                             json_output_file=join(output_folder, "summary.json"),
                             json_name=job_name,
                             json_author="Fabian",
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

        export_pool = Pool(default_num_threads)
        results = OrderedDict()

        for k in self.dataset_test.keys():
            properties = self.dataset[k]['properties']
            fname = properties['list_of_data_files'][0].split("/")[-1][:-12]
            # data = np.load(self.dataset[k]['data_file'])['data']
            data = np.load(self.dataset[k]['data_file'][:-4] + '.npy')

            print(k, data.shape)
            data[-1][data[-1] == -1] = 0

            pred = self.predict_preprocessed_data_return_class_and_score(
                data, do_mirroring, mirror_axes, use_gaussian,
                all_in_gpu=all_in_gpu)

            # results[k] = {'pred_class': float(pred[0][0]), 'pred_score': float(pred[1][0]), 'gt_class':properties['cls_label']}
            # results[k] = {'pred_class': float(pred[0]), 'pred_score': pred[1].tolist(), 'gt_class': properties['cls_target']-1}
            if not np.isnan(pred[1]).any():
                results[k] = {'pred_class': None, 'pred_score': pred[1].tolist(), 'gt_class': properties['cls_target'] - 1}
                print(pred[1].tolist(), properties['cls_target'] - 1)

        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")
        task = self.dataset_directory.split("/")[-1]
        job_name = self.experiment_name
        _ = aggregate_scores(results,
                             json_output_file=join(output_folder, "summary.json"),
                             json_name=job_name,
                             json_author="Fabian",
                             json_task=task, num_threads=default_num_threads)

        self.network.train(current_mode)

    def load_pretrained_checkpoint(self, fname, train=True):
        self.pretrain = True
        self.print_to_log_file("loading checkpoint", fname, "train=", train)
        if not self.was_initialized:
            self.initialize(train)
        pretrained_model = torch.load(fname, map_location=torch.device('cpu'))

        curr_state_dict = self.network.state_dict()
        # filter out unnecessary keys
        pretrained_dict = {k: value for k, value in pretrained_model['state_dict'].items() if k in curr_state_dict}
        # overwrite entries in the existing state dict
        curr_state_dict.update(pretrained_dict)

        self.network.load_state_dict(curr_state_dict)
        # for name, p in self.network.named_parameters():
        #     if name.startswith('conv_blocks_context'):
        #         p.requires_grad = False
        # self.optimizer.param_groups[0]['params'] = list(filter(lambda x: x.requires_grad is not False, self.network.parameters()))
        self.amp_initialized = False
        self._maybe_init_amp()

class nnUNetTrainer_Attention_FRCNN_stage2_MaskSideBranch(nnUNetTrainer_Attention_FRCNN_stage2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, initial_lr=1e-3, pretrain=False, opt='SGD',
                 weight_decay=1e-4):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                 unpack_data, deterministic, fp16, initial_lr, pretrain, opt, weight_decay)
        self.mask_input_layer = [1]
        self.mask_input_method = 'multiplication'
        self.mask_nonlinear = 'lrelu'
        self.mask_norm = 'in'
        self.initial_lr = 5e-4
        self.output_folder = join(self.output_folder_base, 'fold_%d' % fold,
                                  'masklayer%s_%s_%s%s' % (','.join(map(str, self.mask_input_layer)), self.mask_input_method, self.mask_norm, self.mask_nonlinear)) if fold != 'all' else \
            join(self.output_folder_base, 'all',
                                  'masklayer%s_%s_%s%s' % (','.join(map(str, self.mask_input_layer)), self.mask_input_method, self.mask_norm, self.mask_nonlinear))

        self.max_num_epochs = 500


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
            # norm_op = nn.BatchNorm3d ##219 old

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d
            # norm_op = nn.BatchNorm2d ##219 old

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        # dropout_op_kwargs = {'p': 0, 'inplace': True}
        dropout_op_kwargs = {'p': 0.0, 'inplace': True}  # 207
        # fc_dropout_op_kwargs = {'p': 0.9, 'inplace': False} #207....
        fc_dropout_op_kwargs = {'p': 0.0, 'inplace': False}  # 206, 219
        # fc_dropout_op_kwargs = {'p': 0.0, 'inplace': False}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        # net_nonlin = nn.ReLU    # 219 old
        # net_nonlin_kwargs = {'inplace': True}
        self.network = Generic_Classification_MaskSideBranch(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes), self.net_pool_per_axis,
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He_Classification(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True, fc_dropout_op_kwargs=fc_dropout_op_kwargs,
                                    mask_input_layer=self.mask_input_layer, mask_input_method=self.mask_input_method, mask_nonlinear=self.mask_nonlinear,
                                                             mask_norm = self.mask_norm)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def get_model_complexity(self):
        def prepare_input(resolution):
            x = torch.FloatTensor(1, 2, *(resolution))
            a = torch.FloatTensor(1, 1, *(resolution))
            return dict(x=x.cuda(), m=a.cuda())

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
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['cls_target']
        seg = data_dict['seg']
        # target = []
        # roi = []
        # for idx, data_prop in enumerate(data_dict['properties']):
        #     if data_prop.get('roi') is not None:
        #         roi.append(data_prop['roi'])
        #         target.append(int(data_prop['cls_label']))
        #     else:
        #         roi.append(np.array([[0,16], [0,64], [0,64]])) ### random values
        #         target.append(-1)
        # target = np.array(target)
        # roi = np.array(roi)

        data = maybe_to_torch(data)
        seg = maybe_to_torch(seg)
        target = maybe_to_torch(target)

        # def processClsTarget(cls_target, seg):
        #     has_pdac = any_tensor(seg !=0,axes = tuple(range(1, len(seg.shape))))
        #     return cls_target * has_pdac + ~has_pdac* (-1)
        # target = processClsTarget(target, data[:, -1])

        if torch.cuda.is_available():
            data = to_cuda(data)
            seg = to_cuda(seg)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        output = self.network(data, seg)

        del data
        del seg
        # loss = self.loss(output.squeeze(), target)
        loss = self.loss(output.squeeze(), target.long())

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

        return loss.detach().cpu().numpy()

    def save_validation_feature2npz(self, do_mirroring: bool = False, use_gaussian: bool = True,
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

        patient_identifiers = np.unique([key[:8] for key in self.dataset_val.keys()])
        data_df = pd.read_excel('/data86/zhengzhilin980/pancreas/PDAC_new/Patients_addN0.xlsx')
        for patient_identifier in patient_identifiers:
            N = int(bool(data_df[data_df['LNM_identifier'] == patient_identifier]['N'].values[0]))
            instance_identifiers = [key for key in self.dataset_val.keys() if key.startswith(patient_identifier)]
            instances = []
            for k in instance_identifiers:
                data = np.load(self.dataset[k]['data_file'][:-4]+'.npy')
                data[-1][data[-1] == -1] = 0
                instances.append(data)

            instances = np.stack(instances)
            print(patient_identifier, instances.shape)

            feature = self.network.get_3D_Classification_feature(instances, do_mirroring, mirror_axes, self.patch_size)

            print("saving: ", os.path.join(output_folder, "%s.npz" % patient_identifier))
            np.savez_compressed(os.path.join(output_folder, "%s.npz" % patient_identifier),
                                feature=feature.astype(np.float32), label=N)

        self.print_to_log_file("finished prediction")
        self.network.train(current_mode)


    def save_testing_feature2npz(self, do_mirroring: bool = False, use_gaussian: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
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

        patient_identifiers = np.unique([key[:8] for key in self.dataset_test.keys()])
        data_df = pd.read_excel('/data86/zhengzhilin980/pancreas/PDAC_new/Patients_addN0.xlsx')
        for patient_identifier in patient_identifiers:
            if not os.path.exists(os.path.join(output_folder, "%s.npz" % patient_identifier)):
                N = int(bool(data_df[data_df['LNM_identifier'] == patient_identifier]['N'].values[0]))
                instance_identifiers = [key for key in self.dataset_test.keys() if key.startswith(patient_identifier)]
                instances = []
                for k in instance_identifiers:
                    data = np.load(self.dataset[k]['data_file'][:-4]+'.npy')
                    data[-1][data[-1] == -1] = 0
                    instances.append(data)

                instances = np.stack(instances)
                print(patient_identifier, instances.shape)

                feature = self.network.get_3D_Classification_feature(instances, do_mirroring, mirror_axes, self.patch_size)

                print("saving: ", os.path.join(output_folder, "%s.npz" % patient_identifier))
                np.savez_compressed(os.path.join(output_folder, "%s.npz" % patient_identifier),
                                    feature=feature.astype(np.float32), label=N)

        self.print_to_log_file("finished prediction")
        self.network.train(current_mode)


    def save_training_feature2npz(self, do_mirroring: bool = False, use_gaussian: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None):
        """
        if debug=True then the temporary files generated for postprocessing determination will be kept
        """

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

        patient_identifiers = np.unique([key[:8] for key in self.dataset_tr.keys()])
        data_df = pd.read_excel('/data86/zhengzhilin980/pancreas/PDAC_new/Patients_addN0.xlsx')
        for patient_identifier in patient_identifiers:
            N = int(bool(data_df[data_df['LNM_identifier'] == patient_identifier]['N'].values[0]))
            instance_identifiers = [key for key in self.dataset_tr.keys() if key.startswith(patient_identifier)]
            instances = []
            for k in instance_identifiers:
                data = np.load(self.dataset[k]['data_file'][:-4]+'.npy')
                data[-1][data[-1] == -1] = 0
                instances.append(data)

            instances = np.stack(instances)
            print(patient_identifier, instances.shape)

            feature = self.network.get_3D_Classification_feature(instances, do_mirroring, mirror_axes, self.patch_size)

            print("saving: ", os.path.join(output_folder, "%s.npz" % patient_identifier))
            np.savez_compressed(os.path.join(output_folder, "%s.npz" % patient_identifier),
                                feature=feature.astype(np.float32), label=N)

        self.print_to_log_file("finished prediction")
        self.network.train(current_mode)

    # def load_dataset(self):
    #     folder = '/data86/pancreaticCancer/nnUNETFrame/nnUNET_DATASET/nnUNET_preprocessed/Task510_LNMDataset_LN_seg_att_stage2_semi/nnUNetData_plans_v2.1_stage1/'
    #     fold = self.fold
    #
    #     def get_case_identifiers_fold(folder, fold):
    #         case_identifiers = [i[:-4] for i in os.listdir(folder) if
    #                             i.endswith("npy") and (i.find("segFromPrevStage") == -1)
    #                             and i.find('fold%d' % fold) != -1]
    #         return case_identifiers
    #
    #     # we don't load the actual data but instead return the filename to the np file. the properties are loaded though
    #     case_identifiers = get_case_identifiers_fold(folder, fold)
    #     # case_identifiers = get_case_identifiers(folder)
    #     case_identifiers.sort()
    #     dataset = OrderedDict()
    #     for c in case_identifiers:
    #         dataset[c] = OrderedDict()
    #         dataset[c]['data_file'] = join(folder, "%s.npz" % c)
    #         with open(join(folder, "%s.pkl" % c), 'rb') as f:
    #             dataset[c]['properties'] = pickle.load(f)
    #         if dataset[c].get('seg_from_prev_stage_file') is not None:
    #             dataset[c]['seg_from_prev_stage_file'] = join(folder, "%s_segs.npz" % c)
    #
    #     self.dataset = dataset

class nnUNetTrainer_Attention_FRCNN_stage2_MaskSideBranch_v2(nnUNetTrainer_Attention_FRCNN_stage2_MaskSideBranch):
    def load_dataset(self):
        self.dataset = load_dataset_LN_stage2(self.folder_with_preprocessed_data, self.fold)

    def do_split(self):
        """
        This is a suggestion for if your dataset is a dictionary (my personal standard)
        :return:
        """
        splits_file = join(self.dataset_directory, "splits_final_fold%d.pkl" % self.fold)
        image_splits_file = join(self.dataset_directory, "image_splits_final.pkl")
        if not isfile(splits_file):
            image_splits = load_pickle(image_splits_file)
            self.print_to_log_file("Creating new split...")
            splits = OrderedDict()
            all_keys_sorted = np.sort(list(self.dataset.keys()))
            image_train_keys = image_splits[self.fold]['train']
            image_val_keys = image_splits[self.fold]['val']
            image_test_keys = image_splits[self.fold]['test']
            splits['train'] = np.array([key for key in all_keys_sorted if key[:8] in image_train_keys])
            splits['val'] = np.array([key for key in all_keys_sorted if key[:8] in image_val_keys])
            splits['test'] = np.array([key for key in all_keys_sorted if key[:8] in image_test_keys])
            save_pickle(splits, splits_file)

        splits = load_pickle(splits_file)

        if self.fold == "all":
            tr_keys = val_keys =test_keys= list(self.dataset.keys())
        else:
            tr_keys = splits['train']
            val_keys = splits['val']
            test_keys = splits['test']

        tr_keys.sort()
        val_keys.sort()
        test_keys.sort()

        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]

        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

        self.dataset_test = OrderedDict()
        for i in test_keys:
            self.dataset_test[i] = self.dataset[i]

class nnUNetTrainer_Attention_FRCNN_stage2_BoxSideBranch(nnUNetTrainer_Attention_FRCNN_stage2_MaskSideBranch):
    def setup_DA_params(self):
        super().setup_DA_params()
        self.data_aug_params['change_mask_to_box'] = True


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

        export_pool = Pool(default_num_threads)
        results = OrderedDict()

        for k in self.dataset_val.keys():
            properties = self.dataset[k]['properties']
            fname = properties['list_of_data_files'][0].split("/")[-1][:-12]
            # data = np.load(self.dataset[k]['data_file'])['data']
            data = np.load(self.dataset[k]['data_file'][:-4]+'.npy')

            print(k, data.shape)
            data[-1][data[-1] == -1] = 0
            roi = np.array(get_bbox_from_mask(data[-1]))
            mask = np.zeros_like(data[-1])
            mask[roi[0, 0]:roi[0, 1], roi[1, 0]:roi[1, 1], roi[2, 0]:roi[2, 1]] = 1
            data[-1] = mask.copy()

            pred = self.predict_preprocessed_data_return_class_and_score(
                data, do_mirroring, mirror_axes, use_gaussian,
                all_in_gpu=all_in_gpu)

            # results[k] = {'pred_class': float(pred[0][0]), 'pred_score': float(pred[1][0]), 'gt_class':properties['cls_label']}
            # results[k] = {'pred_class': float(pred[0]), 'pred_score': pred[1].tolist(), 'gt_class': properties['cls_target']-1}
            if not np.isnan(pred[1]).any():
                results[k] = {'pred_class': None, 'pred_score': pred[1].tolist(), 'gt_class': properties['cls_target']-1}

                print(pred[1].tolist(), properties['cls_target']-1)

        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")
        task = self.dataset_directory.split("/")[-1]
        job_name = self.experiment_name
        _ = aggregate_scores(results,
                             json_output_file=join(output_folder, "summary.json"),
                             json_name=job_name,
                             json_author="Fabian",
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

        export_pool = Pool(default_num_threads)
        results = OrderedDict()

        for k in self.dataset_test.keys():
            properties = self.dataset[k]['properties']
            fname = properties['list_of_data_files'][0].split("/")[-1][:-12]
            # data = np.load(self.dataset[k]['data_file'])['data']
            data = np.load(self.dataset[k]['data_file'][:-4] + '.npy')

            print(k, data.shape)
            data[-1][data[-1] == -1] = 0
            roi = np.array(get_bbox_from_mask(data[-1]))
            mask = np.zeros_like(data[-1])
            mask[roi[0, 0]:roi[0, 1], roi[1, 0]:roi[1, 1], roi[2, 0]:roi[2, 1]] = 1
            data[-1] = mask.copy()

            pred = self.predict_preprocessed_data_return_class_and_score(
                data, do_mirroring, mirror_axes, use_gaussian,
                all_in_gpu=all_in_gpu)

            # results[k] = {'pred_class': float(pred[0][0]), 'pred_score': float(pred[1][0]), 'gt_class':properties['cls_label']}
            # results[k] = {'pred_class': float(pred[0]), 'pred_score': pred[1].tolist(), 'gt_class': properties['cls_target']-1}
            if not np.isnan(pred[1]).any():
                results[k] = {'pred_class': None, 'pred_score': pred[1].tolist(), 'gt_class': properties['cls_target'] - 1}
                print(pred[1].tolist(), properties['cls_target'] - 1)

        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")
        task = self.dataset_directory.split("/")[-1]
        job_name = self.experiment_name
        _ = aggregate_scores(results,
                             json_output_file=join(output_folder, "summary.json"),
                             json_name=job_name,
                             json_author="Fabian",
                             json_task=task, num_threads=default_num_threads)

        self.network.train(current_mode)

class nnUNetTrainer_Attention_FRCNN_stage2_BoxSideBranch_v2(nnUNetTrainer_Attention_FRCNN_stage2_BoxSideBranch):
    def load_dataset(self):
        self.dataset = load_dataset_LN_stage2(self.folder_with_preprocessed_data, self.fold)

    def do_split(self):
        """
        This is a suggestion for if your dataset is a dictionary (my personal standard)
        :return:
        """
        splits_file = join(self.dataset_directory, "splits_final_fold%d.pkl" % self.fold)
        image_splits_file = join(self.dataset_directory, "image_splits_final.pkl")
        if not isfile(splits_file):
            image_splits = load_pickle(image_splits_file)
            self.print_to_log_file("Creating new split...")
            splits = OrderedDict()
            all_keys_sorted = np.sort(list(self.dataset.keys()))
            image_train_keys = image_splits[self.fold]['train']
            image_val_keys = image_splits[self.fold]['val']
            image_test_keys = image_splits[self.fold]['test']
            splits['train'] = np.array([key for key in all_keys_sorted if key[:8] in image_train_keys])
            splits['val'] = np.array([key for key in all_keys_sorted if key[:8] in image_val_keys])
            splits['test'] = np.array([key for key in all_keys_sorted if key[:8] in image_test_keys])
            save_pickle(splits, splits_file)

        splits = load_pickle(splits_file)

        if self.fold == "all":
            tr_keys = val_keys =test_keys= list(self.dataset.keys())
        else:
            tr_keys = splits['train']
            val_keys = splits['val']
            test_keys = splits['test']

        tr_keys.sort()
        val_keys.sort()
        test_keys.sort()

        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]

        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

        self.dataset_test = OrderedDict()
        for i in test_keys:
            self.dataset_test[i] = self.dataset[i]

class nnUNetTrainer_Attention_FRCNN_stage2_JitteringBoxSideBranch(nnUNetTrainer_Attention_FRCNN_stage2_MaskSideBranch):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, initial_lr=1e-3, pretrain=False, opt='SGD',
                 weight_decay=1e-4):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                 unpack_data, deterministic, fp16, initial_lr, pretrain, opt, weight_decay)

        self.margins_min = 4
        self.margins_max = 8
        self.output_folder = join(self.output_folder_base, 'fold_%d' % fold)

    def setup_DA_params(self):
        super().setup_DA_params()
        self.data_aug_params['change_jittering_mask_to_box'] = True
        self.data_aug_params['margins_min'] = [self.margins_min, self.margins_min, self.margins_min]
        self.data_aug_params['margins_max'] = [self.margins_max, self.margins_max, self.margins_max]


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
            # nonnan_idx = np.isnan(output_softmax[:, 1].detach().cpu().numpy())
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

        export_pool = Pool(default_num_threads)
        results = OrderedDict()
        margins = np.array([np.mean([margin_min, margin_max]) for margin_min, margin_max in
                            zip(self.data_aug_params['margins_min'], self.data_aug_params['margins_max'])])

        for k in self.dataset_val.keys():
            properties = self.dataset[k]['properties']
            fname = properties['list_of_data_files'][0].split("/")[-1][:-12]
            # data = np.load(self.dataset[k]['data_file'])['data']
            data = np.load(self.dataset[k]['data_file'][:-4]+'.npy')

            print(k, data.shape)
            data[-1][data[-1] == -1] = 0
            roi = np.array(get_bbox_from_mask(data[-1]))
            roi = (np.array(roi) + np.stack([-margins, margins], axis=1)).astype(int)
            mask = np.zeros_like(data[-1])
            mask[roi[0, 0]:roi[0, 1], roi[1, 0]:roi[1, 1], roi[2, 0]:roi[2, 1]] = 1
            data[-1] = mask.copy()

            pred = self.predict_preprocessed_data_return_class_and_score(
                data, do_mirroring, mirror_axes, use_gaussian,
                all_in_gpu=all_in_gpu)

            # results[k] = {'pred_class': float(pred[0][0]), 'pred_score': float(pred[1][0]), 'gt_class':properties['cls_label']}
            # results[k] = {'pred_class': float(pred[0]), 'pred_score': pred[1].tolist(), 'gt_class': properties['cls_target']-1}
            if not np.isnan(pred[1]).any():
                results[k] = {'pred_class': None, 'pred_score': pred[1].tolist(), 'gt_class': properties['cls_target']-1}

                print(pred[1].tolist(), properties['cls_target']-1)

        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")
        task = self.dataset_directory.split("/")[-1]
        job_name = self.experiment_name
        _ = aggregate_scores(results,
                             json_output_file=join(output_folder, "summary.json"),
                             json_name=job_name,
                             json_author="Fabian",
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

        export_pool = Pool(default_num_threads)
        results = OrderedDict()
        margins = np.array([np.mean([margin_min, margin_max]) for margin_min, margin_max in
                            zip(self.data_aug_params['margins_min'], self.data_aug_params['margins_max'])])

        for k in self.dataset_test.keys():
            properties = self.dataset[k]['properties']
            fname = properties['list_of_data_files'][0].split("/")[-1][:-12]
            # data = np.load(self.dataset[k]['data_file'])['data']
            data = np.load(self.dataset[k]['data_file'][:-4] + '.npy')

            print(k, data.shape)
            data[-1][data[-1] == -1] = 0
            roi = np.array(get_bbox_from_mask(data[-1]))
            roi = (np.array(roi) + np.stack([-margins, margins], axis=1)).astype(int)
            mask = np.zeros_like(data[-1])
            mask[roi[0, 0]:roi[0, 1], roi[1, 0]:roi[1, 1], roi[2, 0]:roi[2, 1]] = 1
            data[-1] = mask.copy()

            pred = self.predict_preprocessed_data_return_class_and_score(
                data, do_mirroring, mirror_axes, use_gaussian,
                all_in_gpu=all_in_gpu)

            # results[k] = {'pred_class': float(pred[0][0]), 'pred_score': float(pred[1][0]), 'gt_class':properties['cls_label']}
            # results[k] = {'pred_class': float(pred[0]), 'pred_score': pred[1].tolist(), 'gt_class': properties['cls_target']-1}
            if not np.isnan(pred[1]).any():
                results[k] = {'pred_class': None, 'pred_score': pred[1].tolist(), 'gt_class': properties['cls_target'] - 1}
                print(pred[1].tolist(), properties['cls_target'] - 1)

        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")
        task = self.dataset_directory.split("/")[-1]
        job_name = self.experiment_name
        _ = aggregate_scores(results,
                             json_output_file=join(output_folder, "summary.json"),
                             json_name=job_name,
                             json_author="Fabian",
                             json_task=task, num_threads=default_num_threads)

        self.network.train(current_mode)

    def get_classification_score(self, preprocessed_data_dir, do_mirroring: bool = True, use_gaussian: bool = True,
                                validation_folder_name: str = 'validation_raw', debug: bool = False,
                                all_in_gpu: bool = False,
                                segmentation_export_kwargs: dict = None):
        """
        if debug=True then the temporary files generated for postprocessing determination will be kept
        """

        def load_dataset(folder, fold, split_keys=None):
            # we don't load the actual data but instead return the filename to the np file. the properties are loaded though
            if split_keys is None:
                case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith("npy") and i.find('fold%d'%fold) != -1]
            else:
                case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith("npy") and i.find('fold%d' % fold) != -1 and i[:8] in split_keys]
            # case_identifiers = get_case_identifiers(folder)
            case_identifiers.sort()
            dataset = OrderedDict()
            for c in case_identifiers:
                dataset[c] = OrderedDict()
                dataset[c]['data_file'] = join(folder, "%s.npy" % c)
                with open(join(folder, "%s.pkl" % c), 'rb') as f:
                    dataset[c]['properties'] = pickle.load(f)
                if dataset[c].get('seg_from_prev_stage_file') is not None:
                    dataset[c]['seg_from_prev_stage_file'] = join(folder, "%s_segs.npz" % c)
            return dataset



        current_mode = self.network.training
        self.network.eval()

        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        # image_splits_file = join(self.dataset_directory, "image_splits_final.pkl")
        # image_splits = load_pickle(image_splits_file)

        # dataset = load_dataset(preprocessed_data_dir, self.fold, split_keys=image_splits[self.fold]['test'])
        dataset = load_dataset(preprocessed_data_dir, self.fold)


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
        # output_folder = join(self.output_folder, validation_folder_name)
        # maybe_mkdir_p(output_folder)
        # this is for debug purposes
        # my_input_args = {'do_mirroring': do_mirroring,
        #                  'use_gaussian': use_gaussian,
        #                  'validation_folder_name': validation_folder_name,
        #                  'debug': debug,
        #                  'all_in_gpu': all_in_gpu,
        #                  'segmentation_export_kwargs': segmentation_export_kwargs,
        #                  }
        # save_json(my_input_args, join(output_folder, "validation_args.json"))

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        # export_pool = Pool(default_num_threads)
        # results = OrderedDict()
        margins = np.array([np.mean([margin_min, margin_max]) for margin_min, margin_max in
                            zip(self.data_aug_params['margins_min'], self.data_aug_params['margins_max'])])

        for k in dataset.keys():
            properties = dataset[k]['properties']
            data = np.load(dataset[k]['data_file'][:-4] + '.npy')

            print(k, data.shape)
            data[-1][data[-1] == -1] = 0
            roi = np.array(get_bbox_from_mask(data[-1]))
            roi = (np.array(roi) + np.stack([-margins, margins], axis=1)).astype(int)
            mask = np.zeros_like(data[-1])
            mask[roi[0, 0]:roi[0, 1], roi[1, 0]:roi[1, 1], roi[2, 0]:roi[2, 1]] = 1
            data[-1] = mask.copy()

            pred = self.predict_preprocessed_data_return_class_and_score(
                data, do_mirroring, mirror_axes, use_gaussian,
                all_in_gpu=all_in_gpu)

            # results[k] = {'pred_class': float(pred[0][0]), 'pred_score': float(pred[1][0]), 'gt_class':properties['cls_label']}
            # results[k] = {'pred_class': float(pred[0]), 'pred_score': pred[1].tolist(), 'gt_class': properties['cls_target']-1}
            # if not np.isnan(pred[1]).any():
            #     results[k] = {'pred_class': None, 'pred_score': pred[1].tolist(), 'gt_class': properties['cls_target'] - 1}
            #     print(pred[1].tolist(), properties['cls_target'] - 1)

            properties[self.__class__.__name__ + '_score'] = pred[1][0, 1]
            write_pickle(properties, dataset[k]['data_file'][:-4] + '.pkl')

        self.print_to_log_file("finished prediction")

        self.network.train(current_mode)

class nnUNetTrainer_Attention_FRCNN_stage2_JitteringBoxSideBranch_v2(nnUNetTrainer_Attention_FRCNN_stage2_JitteringBoxSideBranch):
    def load_dataset(self):
        self.dataset = load_dataset_LN_stage2(self.folder_with_preprocessed_data, self.fold)

    def do_split(self):
        """
        This is a suggestion for if your dataset is a dictionary (my personal standard)
        :return:
        """
        splits_file = join(self.dataset_directory, "splits_final_fold%d.pkl" % self.fold)
        image_splits_file = join(self.dataset_directory, "image_splits_final.pkl")
        if not isfile(splits_file):
            image_splits = load_pickle(image_splits_file)
            self.print_to_log_file("Creating new split...")
            splits = OrderedDict()
            all_keys_sorted = np.sort(list(self.dataset.keys()))
            image_train_keys = image_splits[self.fold]['train']
            image_val_keys = image_splits[self.fold]['val']
            image_test_keys = image_splits[self.fold]['test']
            splits['train'] = np.array([key for key in all_keys_sorted if key[:8] in image_train_keys])
            splits['val'] = np.array([key for key in all_keys_sorted if key[:8] in image_val_keys])
            splits['test'] = np.array([key for key in all_keys_sorted if key[:8] in image_test_keys])
            save_pickle(splits, splits_file)

        splits = load_pickle(splits_file)

        if self.fold == "all":
            tr_keys = val_keys =test_keys= list(self.dataset.keys())
        else:
            tr_keys = splits['train']
            val_keys = splits['val']
            test_keys = splits['test']

        tr_keys.sort()
        val_keys.sort()
        test_keys.sort()

        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]

        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

        self.dataset_test = OrderedDict()
        for i in test_keys:
            self.dataset_test[i] = self.dataset[i]
