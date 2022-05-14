from nnunet.training.network_training.nnUNetTrainerV2_PDAC_classification_resnet import nnUNetTrainerV2_PDAC_classification_resnet
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.network_architecture.ResNet18 import ResNet18, ResNet18_DeepTEN, ResNet18_DeepTEN_MaskSideBranch, ResNet18_MaskSideBranch, DeepTEN_MaskSideBranch
from sklearn.metrics import roc_auc_score
from skimage.io import imsave
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from torch.nn.utils import clip_grad_norm_

from nnunet.configuration import default_num_threads
from multiprocessing.pool import Pool
from nnunet.evaluation.evaluator import aggregate_classification_scores
import pandas as pd
from collections import defaultdict
from nnunet.training.dataloading.dataset_loading import load_dataset
from ptflops import get_model_complexity_info

try:
    from apex import amp
except ImportError:
    amp = None

class nnUNetTrainerV2_PDAC_classification_resnet_maskout(nnUNetTrainerV2_PDAC_classification_resnet):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 300
        self.pretrained = True
        self.if_freeze = True
        self.initial_lr = 1e-4
        self.weight_decay = 5e-4 #5e-3
        self.freeze_layernum = 3 #3
        self.output_folder = join(self.output_folder, 'freezenum%d_weightdecay%f' % (self.freeze_layernum, self.weight_decay))
        self.online_eval_tn = []
        self.num_val_batches_per_epoch = 16
        self.best_val_eval_criterion = None
        self.online_eval_label, self.online_eval_prob = [], []

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        # self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
        #                                  momentum=0.95, nesterov=True)
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay)
        self.lr_scheduler = None

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
        self.data_aug_params["mask_out_of_box_to_zero"] = True

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

    def manage_patience(self):
        # update patience
        continue_training = True
        if self.patience is not None:
            # if best_MA_tr_loss_for_patience and best_epoch_based_on_MA_tr_loss were not yet initialized,
            # initialize them
            if self.best_MA_tr_loss_for_patience is None:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA

            if self.best_epoch_based_on_MA_tr_loss is None:
                self.best_epoch_based_on_MA_tr_loss = self.epoch

            if self.best_val_eval_criterion_MA is None:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA

            if self.best_val_eval_criterion is None:
                self.best_val_eval_criterion = self.all_val_eval_metrics[-1]

            # check if the current epoch is the best one according to moving average of validation criterion. If so
            # then save 'best' model
            # Do not use this for validation. This is intended for test set prediction only.
            #self.print_to_log_file("current best_val_eval_criterion_MA is %.4f0" % self.best_val_eval_criterion_MA)
            #self.print_to_log_file("current val_eval_criterion_MA is %.4f" % self.val_eval_criterion_MA)

            if self.val_eval_criterion_MA > self.best_val_eval_criterion_MA:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA
                #self.print_to_log_file("saving best epoch checkpoint...")
                self.save_checkpoint(join(self.output_folder, "model_ma_best.model"))

            if self.all_val_eval_metrics[-1] > self.best_val_eval_criterion:
                self.best_val_eval_criterion = self.all_val_eval_metrics[-1]
                self.save_checkpoint(join(self.output_folder, "model_ep_best.model"))

            # Now see if the moving average of the train loss has improved. If yes then reset patience, else
            # increase patience
            if self.train_loss_MA + self.train_loss_MA_eps < self.best_MA_tr_loss_for_patience:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA
                self.best_epoch_based_on_MA_tr_loss = self.epoch
                #self.print_to_log_file("New best epoch (train loss MA): %03.4f" % self.best_MA_tr_loss_for_patience)
            else:
                pass
                #self.print_to_log_file("No improvement: current train MA %03.4f, best: %03.4f, eps is %03.4f" %
                #                       (self.train_loss_MA, self.best_MA_tr_loss_for_patience, self.train_loss_MA_eps))

            # if patience has reached its maximum then finish training (provided lr is low enough)
            if self.epoch - self.best_epoch_based_on_MA_tr_loss > self.patience:
                if self.optimizer.param_groups[0]['lr'] > self.lr_threshold:
                    #self.print_to_log_file("My patience ended, but I believe I need more time (lr > 1e-6)")
                    self.best_epoch_based_on_MA_tr_loss = self.epoch - self.patience // 2
                else:
                    #self.print_to_log_file("My patience ended")
                    continue_training = False
            else:
                pass
                #self.print_to_log_file(
                #    "Patience: %d/%d" % (self.epoch - self.best_epoch_based_on_MA_tr_loss, self.patience))

        return continue_training

    def load_MA_best_checkpoint(self, train=True):
        if self.fold is None:
            raise RuntimeError("Cannot load best checkpoint if self.fold is None")
        if isfile(join(self.output_folder, "model_ma_best.model")):
            self.load_checkpoint(join(self.output_folder, "model_ma_best.model"), train=train)
        else:
            self.print_to_log_file("WARNING! model_best.model does not exist! Cannot load best checkpoint. Falling "
                                   "back to load_latest_checkpoint")
            # self.load_latest_checkpoint(train)

    def load_ep_best_checkpoint(self, train=True):
        if self.fold is None:
            raise RuntimeError("Cannot load best checkpoint if self.fold is None")
        if isfile(join(self.output_folder, "model_ep_best.model")):
            self.load_checkpoint(join(self.output_folder, "model_ep_best.model"), train=train)
        else:
            self.print_to_log_file("WARNING! model_best.model does not exist! Cannot load best checkpoint. Falling "
                                   "back to load_latest_checkpoint")
            # self.load_latest_checkpoint(train)

class nnUNetTrainerV2_PDAC_classification_resnet_woroipool_maskout(nnUNetTrainerV2_PDAC_classification_resnet_maskout):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 200
        self.pretrained = True
        self.if_freeze = True
        self.initial_lr = 1e-4
        self.weight_decay = 5e-5 #5e-3
        self.freeze_layernum = 4 #3
        self.output_folder = join(self.output_folder_base, 'fold_%d'% fold, 'freezenum%d_weightdecay%f' % (self.freeze_layernum, self.weight_decay))
        self.online_eval_tn = []
        self.num_val_batches_per_epoch = 16
        self.best_val_eval_criterion = None
        self.debug = False

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


        self.network = ResNet18(self.num_classes, self.pretrained, self.if_freeze,
                                        roi_margin_min = self.data_aug_params["margins_min"], roi_margin_max = self.data_aug_params["margins_max"],
                                       freeze_layernum = self.freeze_layernum)
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

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        logits = self.network(data, None)

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

class nnUNetTrainerV2_PDAC_classification_resnet_deepten_woroipool_maskout(nnUNetTrainerV2_PDAC_classification_resnet_woroipool_maskout):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 200
        self.pretrained = True
        self.if_freeze = True
        self.initial_lr = 5e-4
        self.weight_decay = 5e-4 #5e-3
        self.freeze_layernum = 4 #3
        self.n_codes = 8 # 4
        self.len_codes = 64
        self.output_folder = join(self.output_folder_base, 'fold_%d'% fold, 'freezenum%d_weightdecay%f_ncodes%d_lencodes%d' %
                                  (self.freeze_layernum, self.weight_decay, self.n_codes, self.len_codes))
        self.online_eval_tn = []
        self.num_val_batches_per_epoch = 16
        self.best_val_eval_criterion = None

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


        self.network = ResNet18_DeepTEN(self.num_classes, self.pretrained, self.if_freeze,
                                        roi_margin_min = self.data_aug_params["margins_min"], roi_margin_max = self.data_aug_params["margins_max"],
                                       freeze_layernum = self.freeze_layernum, n_codes=self.n_codes, len_codes=self.len_codes)
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

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

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


class nnUNetTrainerV2_PDAC_classification_resnet_woroipool_maskoutV2(nnUNetTrainerV2_PDAC_classification_resnet_woroipool_maskout):

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 300
        self.pretrained = True
        self.if_freeze = True
        self.initial_lr = 1e-4
        self.weight_decay = 5e-4 #5e-3
        self.freeze_layernum = 1 #3
        self.output_folder = join(self.output_folder_base, 'fold_%d'% fold, 'freezenum%d_weightdecay%f' % (self.freeze_layernum, self.weight_decay))
        self.online_eval_tn = []
        self.num_val_batches_per_epoch = 16
        self.best_val_eval_criterion = None

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
        self.data_aug_params["mask_out_of_mask_to_zero"] = True


class nnUNetTrainerV2_PDAC_classification_resnet_MaskSideBranch(nnUNetTrainerV2_PDAC_classification_resnet_deepten_woroipool_maskout):

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.mask_input_layer = [1]
        self.mask_input_method = 'multiplication' #'addition'
        self.save_every = 5
        self.max_num_epochs = 500 #50
        self.pretrained = True
        self.if_freeze = True
        self.initial_lr = 5e-4
        self.weight_decay = 5e-4 #5e-3
        self.freeze_layernum = 4
        self.n_codes = 8 # 4
        self.len_codes = 64
        self.output_folder = join(self.output_folder_base, 'fold_%d'% fold)
        self.online_eval_tn = []
        self.num_val_batches_per_epoch = 16
        self.best_val_eval_criterion = None

    def setup_DA_params(self):
        super().setup_DA_params()
        self.data_aug_params['mask_out_of_box_to_zero'] = False

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

        self.network = ResNet18_DeepTEN_MaskSideBranch(self.num_classes, self.pretrained, self.if_freeze, self.mask_input_layer, self.mask_input_method,
                        roi_margin_min = self.data_aug_params["margins_min"], roi_margin_max = self.data_aug_params["margins_max"],
                        freeze_layernum = self.freeze_layernum)
        # self.network = ResNet18_MaskSideBranch(self.num_classes, self.pretrained, self.if_freeze,
        #                                                self.mask_input_layer, self.mask_input_method,
        #                                                roi_margin_min=self.data_aug_params["margins_min"],
        #                                                roi_margin_max=self.data_aug_params["margins_max"],
        #                                                freeze_layernum=self.freeze_layernum)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper


    def get_deep_signature(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                            step_size: float = 0.5, use_gaussian: bool = True, overwrite: bool = True,
                            debug: bool = False, all_in_gpu: bool = False,
                            segmentation_export_kwargs: dict = None, output_excel= 'deep_signature.xlsx'):
        current_mode = self.network.training
        self.network.eval()

        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"

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
        output_excel = join(self.output_folder, output_excel.replace('.xlsx', '_fold%d.xlsx'% self.fold))
        writer = pd.ExcelWriter(output_excel)
        # this is for debug purposes
        my_input_args = {'do_mirroring': do_mirroring,
                         'use_gaussian': use_gaussian,
                         'overwrite': overwrite,
                         'debug': debug,
                         'all_in_gpu': all_in_gpu,
                         'segmentation_export_kwargs': segmentation_export_kwargs,
                         }
        # save_json(my_input_args, join(output_folder, "validation_args.json"))

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()


        training_df = pd.read_excel(self.training_table)

        margins = np.array([np.mean([margin_min, margin_max]) for margin_min, margin_max in
                            zip(self.data_aug_params['margins_min'], self.data_aug_params['margins_max'])])

        for split_name, split_keys in {'train':list(self.dataset_tr.keys())+list(self.dataset_val.keys()), 'test':list(self.dataset_test.keys())}.items():
            output_dict = defaultdict(list)
            for k in split_keys:
                properties = self.dataset[k]['properties']
                CT_N = properties['CT_reported_N']
                patch_files = [join(self.folder_with_preprocessed_data, 'cropped_slice', filename) for filename in
                               os.listdir(join(self.folder_with_preprocessed_data, 'cropped_slice')) if
                               filename.startswith(k) and filename.endswith('.npy') and 'segidx0' in filename and 'fold' not in filename]
                match_k = training_df[training_df['LNM_identifier'] == k]
                label = int(bool(match_k['N'].values[0]))
                patient_id = match_k['PatientID'].values[0]
                date = match_k['date'].values[0]
                output_dict['PatientID'].append(patient_id)
                output_dict['date'].append(date)
                output_dict['LNM_identifier'].append(k)
                output_dict['N'].append(label)
                output_dict['CT_report_N'].append(CT_N)

                for patch_idx, patch_file in enumerate(patch_files):
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

                    self.network.inference_apply_nonlin = lambda x: x
                    feature_pred = self.predict_preprocessed_data_return_softmax(
                        data, do_mirroring, mirror_axes, use_gaussian,
                        all_in_gpu=all_in_gpu, return_feature=True)

                    for feature_id, feature_value in enumerate(feature_pred[0].cpu().numpy()):
                        output_dict['slice%d_feature%d' % (patch_idx, feature_id)].append(feature_value)

                    self.network.inference_apply_nonlin = softmax_helper
                    softmax_pred = self.predict_preprocessed_data_return_softmax(
                        data, do_mirroring, mirror_axes, use_gaussian,
                        all_in_gpu=all_in_gpu, return_feature=False)
                    output_dict['slice%d_score' % (patch_idx)].append(softmax_pred.cpu().numpy()[0, 1])

            feature_df = pd.DataFrame.from_dict(output_dict)
            feature_df.to_excel(writer, index=False, sheet_name=split_name)
        writer.close()

        self.print_to_log_file("finished prediction")


    def get_model_complexity(self):
        def prepare_input(resolution):
            x = torch.FloatTensor(1, 3, *(resolution))
            a = torch.FloatTensor(1, 1, *(resolution))
            return dict(x=x.cuda(), m=a.cuda())

        self.network.do_ds = False
        macs, params = get_model_complexity_info(self.network, input_res=tuple(self.patch_size), input_constructor=prepare_input, as_strings=True, print_per_layer_stat=True)
        print(' - MACs: ' + macs)
        print(' - Params: ' + params)

class nnUNetTrainerV2_PDAC_classification_resnet_MaskSideBranch_wodeepten(nnUNetTrainerV2_PDAC_classification_resnet_MaskSideBranch):
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

        self.network = ResNet18_MaskSideBranch(self.num_classes, self.pretrained, self.if_freeze, self.mask_input_layer, self.mask_input_method,
                        roi_margin_min = self.data_aug_params["margins_min"], roi_margin_max = self.data_aug_params["margins_max"],
                        freeze_layernum = self.freeze_layernum)
        # self.network = ResNet18_MaskSideBranch(self.num_classes, self.pretrained, self.if_freeze,
        #                                                self.mask_input_layer, self.mask_input_method,
        #                                                roi_margin_min=self.data_aug_params["margins_min"],
        #                                                roi_margin_max=self.data_aug_params["margins_max"],
        #                                                freeze_layernum=self.freeze_layernum)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

class nnUNetTrainerV2_PDAC_classification_deepten_MaskSideBranch(nnUNetTrainerV2_PDAC_classification_resnet_MaskSideBranch):
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

        self.network = DeepTEN_MaskSideBranch(self.num_classes, self.pretrained, self.if_freeze, self.mask_input_layer, self.mask_input_method,
                        roi_margin_min = self.data_aug_params["margins_min"], roi_margin_max = self.data_aug_params["margins_max"],
                        freeze_layernum = self.freeze_layernum)
        # self.network = ResNet18_MaskSideBranch(self.num_classes, self.pretrained, self.if_freeze,
        #                                                self.mask_input_layer, self.mask_input_method,
        #                                                roi_margin_min=self.data_aug_params["margins_min"],
        #                                                roi_margin_max=self.data_aug_params["margins_max"],
        #                                                freeze_layernum=self.freeze_layernum)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper


class nnUNetTrainerV2_PDAC_classification_resnet_MaskSideBranch_CTN(nnUNetTrainerV2_PDAC_classification_resnet_MaskSideBranch):
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

        self.network = ResNet18_DeepTEN_MaskSideBranch(self.num_classes, self.pretrained, self.if_freeze, self.mask_input_layer, self.mask_input_method,
                        roi_margin_min = self.data_aug_params["margins_min"], roi_margin_max = self.data_aug_params["margins_max"],
                        freeze_layernum = self.freeze_layernum, incorp_CT_N=True, incorp_CT_N_method='v1')
        # self.network = ResNet18_MaskSideBranch(self.num_classes, self.pretrained, self.if_freeze,
        #                                                self.mask_input_layer, self.mask_input_method,
        #                                                roi_margin_min=self.data_aug_params["margins_min"],
        #                                                roi_margin_max=self.data_aug_params["margins_max"],
        #                                                freeze_layernum=self.freeze_layernum)
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

        CT_N = []
        for k in range(len(data_dict['properties'])):
            CT_N.append(data_dict['properties'][k]['CT_reported_N'])
        CT_N = np.array(CT_N)

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
        CT_N = maybe_to_torch(CT_N)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
            CT_N = to_cuda(CT_N)

        self.optimizer.zero_grad()

        logits = self.network(data, data[:, -1:], CT_N)

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
            CT_N = properties['CT_reported_N']
            patch_files = [join(self.folder_with_preprocessed_data, 'cropped_slice', filename) for filename in
                           os.listdir(join(self.folder_with_preprocessed_data, 'cropped_slice')) if filename.startswith(k) and filename.endswith('.npy') and 'segidx0' in filename]
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
                    all_in_gpu=all_in_gpu, CT_N=CT_N)


                softmax_preds.append(softmax_pred)
            if len(softmax_preds) > 0:
                # patient_softmax_pred = torch.cat(softmax_preds, dim = 0).mean(dim=0, keepdim=True)
                # patient_pred = torch.argmax(patient_softmax_pred, dim=1).cpu().numpy()[0]
                # patient_softmax_pred = patient_softmax_pred.cpu().numpy()[0]
                patient_preds = torch.argmax(torch.cat(softmax_preds, dim = 0), dim=1).cpu().numpy()
                patient_pred = np.argmax(np.bincount(patient_preds))
            else:
                # patient_softmax_pred, patient_pred = None, None
                patient_pred = None
            # pred_gt_tuples.append([patient_softmax_pred, patient_pred, label])
            pred_gt_tuples.append([patient_pred, label])


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
            CT_N = properties['CT_reported_N']
            patch_files = [join(self.folder_with_preprocessed_data, 'cropped_slice', filename) for filename in
                           os.listdir(join(self.folder_with_preprocessed_data, 'cropped_slice')) if filename.startswith(k) and filename.endswith('.npy') and 'segidx0' in filename]
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
                    all_in_gpu=all_in_gpu, CT_N=CT_N)


                softmax_preds.append(softmax_pred)
            if len(softmax_preds) > 0:
                # patient_softmax_pred = torch.cat(softmax_preds, dim = 0).mean(dim=0, keepdim=True)
                # patient_pred = torch.argmax(patient_softmax_pred, dim=1).cpu().numpy()[0]
                # patient_softmax_pred = patient_softmax_pred.cpu().numpy()[0]
                patient_preds = torch.argmax(torch.cat(softmax_preds, dim = 0), dim=1).cpu().numpy()
                patient_pred = np.argmax(np.bincount(patient_preds))
            else:
                # patient_softmax_pred, patient_pred = None, None
                patient_pred = None
            # pred_gt_tuples.append([patient_softmax_pred, patient_pred, label])
            pred_gt_tuples.append([patient_pred, label])


        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")

        task = self.dataset_directory.split("/")[-1]
        job_name = self.experiment_name
        _ = aggregate_classification_scores(pred_gt_tuples,  list(self.dataset_test.keys()),
                             json_output_file=join(output_folder, "summary.json"),
                             json_name=job_name ,
                             json_author="Zhilin",
                             json_task=task, num_threads=default_num_threads)
        self.network.train(current_mode)

class nnUNetTrainerV2_PDAC_classification_resnet_MaskSideBranch_CTN_v2(nnUNetTrainerV2_PDAC_classification_resnet_MaskSideBranch_CTN):
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

        self.network = ResNet18_DeepTEN_MaskSideBranch(self.num_classes, self.pretrained, self.if_freeze, self.mask_input_layer, self.mask_input_method,
                        roi_margin_min = self.data_aug_params["margins_min"], roi_margin_max = self.data_aug_params["margins_max"],
                        freeze_layernum = self.freeze_layernum, incorp_CT_N=True, incorp_CT_N_method='v2')
        # self.network = ResNet18_MaskSideBranch(self.num_classes, self.pretrained, self.if_freeze,
        #                                                self.mask_input_layer, self.mask_input_method,
        #                                                roi_margin_min=self.data_aug_params["margins_min"],
        #                                                roi_margin_max=self.data_aug_params["margins_max"],
        #                                                freeze_layernum=self.freeze_layernum)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper


class nnUNetTrainerV2_PDAC_classification_resnet_MaskSideBranch_SegLN_v2(nnUNetTrainerV2_PDAC_classification_resnet_MaskSideBranch):
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

        self.network = ResNet18_DeepTEN_MaskSideBranch(self.num_classes, self.pretrained, self.if_freeze, self.mask_input_layer, self.mask_input_method,
                        roi_margin_min = self.data_aug_params["margins_min"], roi_margin_max = self.data_aug_params["margins_max"],
                        freeze_layernum = self.freeze_layernum, incorp_CT_N=True, incorp_CT_N_method='v2')
        # self.network = ResNet18_MaskSideBranch(self.num_classes, self.pretrained, self.if_freeze,
        #                                                self.mask_input_layer, self.mask_input_method,
        #                                                roi_margin_min=self.data_aug_params["margins_min"],
        #                                                roi_margin_max=self.data_aug_params["margins_max"],
        #                                                freeze_layernum=self.freeze_layernum)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def get_model_complexity(self):
        def prepare_input(resolution):
            x = torch.FloatTensor(1, 3, *(resolution))
            a = torch.FloatTensor(1, 1, *(resolution))
            n = torch.FloatTensor(1, 1)
            return dict(x=x.cuda(), m=a.cuda(), n=n.cuda())

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

        ln_vol = []
        for k in range(len(data_dict['properties'])):
            ln_vol.append(data_dict['properties'][k]['seg_ln_vol_fold'+str(self.fold)])
        ln_vol = np.array(ln_vol)

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
        ln_vol = maybe_to_torch(ln_vol)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
            ln_vol = to_cuda(ln_vol)

        self.optimizer.zero_grad()

        logits = self.network(data, data[:, -1:], ln_vol)

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
            ln_vol = properties['seg_ln_vol_fold'+str(self.fold)]
            patch_files = [join(self.folder_with_preprocessed_data, 'cropped_slice', filename) for filename in
                           os.listdir(join(self.folder_with_preprocessed_data, 'cropped_slice')) if filename.startswith(k) and
                           filename.endswith('.npy') and 'segidx0' in filename and 'fold' not in filename]
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
                    all_in_gpu=all_in_gpu, CT_N=ln_vol)


                softmax_preds.append(softmax_pred)
            if len(softmax_preds) > 0:
                # patient_softmax_pred = torch.cat(softmax_preds, dim = 0).mean(dim=0, keepdim=True)
                # patient_pred = torch.argmax(patient_softmax_pred, dim=1).cpu().numpy()[0]
                # patient_softmax_pred = patient_softmax_pred.cpu().numpy()[0]
                patient_preds = torch.argmax(torch.cat(softmax_preds, dim = 0), dim=1).cpu().numpy()
                patient_pred = np.argmax(np.bincount(patient_preds))
                patient_prob = torch.mean(torch.cat(softmax_preds, dim = 0), dim=0).cpu().numpy()
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
            ln_vol = properties['seg_ln_vol_fold'+str(self.fold)]
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
                    all_in_gpu=all_in_gpu, CT_N=ln_vol)


                softmax_preds.append(softmax_pred)
            if len(softmax_preds) > 0:
                # patient_softmax_pred = torch.cat(softmax_preds, dim = 0).mean(dim=0, keepdim=True)
                # patient_pred = torch.argmax(patient_softmax_pred, dim=1).cpu().numpy()[0]
                # patient_softmax_pred = patient_softmax_pred.cpu().numpy()[0]
                patient_preds = torch.argmax(torch.cat(softmax_preds, dim = 0), dim=1).cpu().numpy()
                patient_pred = np.argmax(np.bincount(patient_preds))
                patient_prob = torch.mean(torch.cat(softmax_preds, dim = 0), dim=0).cpu().numpy()
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
        _ = aggregate_classification_scores(pred_gt_tuples,  list(self.dataset_test.keys()),
                             json_output_file=join(output_folder, "summary.json"),
                             json_name=job_name ,
                             json_author="Zhilin",
                             json_task=task, num_threads=default_num_threads)
        self.network.train(current_mode)

    def holdout_set_test(self, preprocessed_data_dir, do_mirroring: bool = True, use_sliding_window: bool = True,
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

        holdout_dataset_test = load_dataset(join(preprocessed_data_dir, 'nnUNetData_plans_v2.1_2D_stage0'))
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

        pred_gt_tuples, keys = [], []

        export_pool = Pool(default_num_threads)
        margins = np.array([np.mean([margin_min, margin_max]) for margin_min, margin_max in
                            zip(self.data_aug_params['margins_min'], self.data_aug_params['margins_max'])])
        for k in holdout_dataset_test.keys():
            properties = holdout_dataset_test[k]['properties']
            ln_vol = properties['seg_ln_vol_fold'+str(self.fold)]
            patch_files = [join(self.folder_with_preprocessed_data, 'cropped_slice', filename) for filename in
                           os.listdir(join(self.folder_with_preprocessed_data, 'cropped_slice')) if filename.startswith(k)
                           and filename.endswith('.npy') and 'segidx0' in filename and 'fold' not in filename]
            label = int(bool(properties['metastasis_stage']))
            softmax_preds = []
            if len(patch_files) > 0:
                keys.append(k)
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
                        data, do_mirroring, mirror_axes, use_gaussian, all_in_gpu=all_in_gpu, CT_N=ln_vol)


                    softmax_preds.append(softmax_pred)

                # patient_softmax_pred = torch.cat(softmax_preds, dim = 0).mean(dim=0, keepdim=True)
                # patient_pred = torch.argmax(patient_softmax_pred, dim=1).cpu().numpy()[0]
                # patient_softmax_pred = patient_softmax_pred.cpu().numpy()[0]
                patient_preds = torch.argmax(torch.cat(softmax_preds, dim = 0), dim=1).cpu().numpy()
                patient_pred = np.argmax(np.bincount(patient_preds))
                patient_prob = torch.mean(torch.cat(softmax_preds, dim = 0), dim=0).cpu().numpy()
                # pred_gt_tuples.append([patient_softmax_pred, patient_pred, label])
                pred_gt_tuples.append([patient_pred, label, patient_prob[1]])


        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")

        task = self.output_folder.split("/")[-3]
        job_name = self.experiment_name
        _ = aggregate_classification_scores(pred_gt_tuples,  keys,
                             json_output_file=join(output_folder, "summary.json"),
                             json_name=job_name ,
                             json_author="Zhilin",
                             json_task=task, num_threads=default_num_threads)
        self.network.train(current_mode)


    def get_deep_signature(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                            step_size: float = 0.5, use_gaussian: bool = True, overwrite: bool = True,
                            debug: bool = False, all_in_gpu: bool = False,
                            segmentation_export_kwargs: dict = None, output_excel= 'deep_signature.xlsx'):
        current_mode = self.network.training
        self.network.eval()

        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        self.network.inference_apply_nonlin = lambda x: x
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
        output_excel = join(self.output_folder, output_excel.replace('.xlsx', '_fold%d.xlsx'% self.fold))
        writer = pd.ExcelWriter(output_excel)
        # this is for debug purposes
        my_input_args = {'do_mirroring': do_mirroring,
                         'use_gaussian': use_gaussian,
                         'overwrite': overwrite,
                         'debug': debug,
                         'all_in_gpu': all_in_gpu,
                         'segmentation_export_kwargs': segmentation_export_kwargs,
                         }
        # save_json(my_input_args, join(output_folder, "validation_args.json"))

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()


        training_df = pd.read_excel(self.training_table)

        margins = np.array([np.mean([margin_min, margin_max]) for margin_min, margin_max in
                            zip(self.data_aug_params['margins_min'], self.data_aug_params['margins_max'])])

        for split_name, split_keys in {'train':list(self.dataset_tr.keys())+list(self.dataset_val.keys()), 'test':list(self.dataset_test.keys())}.items():
            output_dict = defaultdict(list)
            for k in split_keys:
                properties = self.dataset[k]['properties']
                CT_N = properties['CT_reported_N']
                ln_vol = properties['seg_ln_vol_fold' + str(self.fold)]
                patch_files = [join(self.folder_with_preprocessed_data, 'cropped_slice', filename) for filename in
                               os.listdir(join(self.folder_with_preprocessed_data, 'cropped_slice')) if
                               filename.startswith(k) and filename.endswith('.npy') and 'segidx0' in filename and 'fold' not in filename]
                match_k = training_df[training_df['LNM_identifier'] == k]
                label = int(bool(match_k['N'].values[0]))
                patient_id = match_k['PatientID'].values[0]
                date = match_k['date'].values[0]
                output_dict['PatientID'].append(patient_id)
                output_dict['date'].append(date)
                output_dict['LNM_identifier'].append(k)
                output_dict['N'].append(label)
                output_dict['CT_report_N'].append(CT_N)

                for patch_idx, patch_file in enumerate(patch_files):
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

                    feature_pred = self.predict_preprocessed_data_return_softmax(
                        data, do_mirroring, mirror_axes, use_gaussian,
                        all_in_gpu=all_in_gpu, CT_N=ln_vol, return_feature=True)

                    for feature_id, feature_value in enumerate(feature_pred[0].cpu().numpy()):
                        output_dict['slice%d_feature%d' % (patch_idx, feature_id)].append(feature_value)
            feature_df = pd.DataFrame.from_dict(output_dict)
            feature_df.to_excel(writer, index=False, sheet_name=split_name)
        writer.close()





        self.print_to_log_file("finished prediction")


    def get_deep_score(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                            step_size: float = 0.5, use_gaussian: bool = True, overwrite: bool = True,
                            debug: bool = False, all_in_gpu: bool = False,
                            segmentation_export_kwargs: dict = None, output_excel= 'deep_signature_all.xlsx'):
        current_mode = self.network.training
        self.network.eval()

        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
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
        output_excel = join('/data86/pancreaticCancer/nnUNETFrame/nnUNET_DATASET/nnUNET_trained_models/nnUNet/deep_signature_all/', output_excel.replace('.xlsx', '_fold%d.xlsx'% self.fold))
        deep_feature_df = pd.read_excel(output_excel, None)
        writer = pd.ExcelWriter(output_excel)
        # this is for debug purposes
        my_input_args = {'do_mirroring': do_mirroring,
                         'use_gaussian': use_gaussian,
                         'overwrite': overwrite,
                         'debug': debug,
                         'all_in_gpu': all_in_gpu,
                         'segmentation_export_kwargs': segmentation_export_kwargs,
                         }
        # save_json(my_input_args, join(output_folder, "validation_args.json"))

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()


        training_df = pd.read_excel(self.training_table)

        margins = np.array([np.mean([margin_min, margin_max]) for margin_min, margin_max in
                            zip(self.data_aug_params['margins_min'], self.data_aug_params['margins_max'])])

        for split_name, split_keys in {'train':list(self.dataset_tr.keys())+list(self.dataset_val.keys()), 'test':list(self.dataset_test.keys())}.items():
            output_dict = defaultdict(list)
            for k in split_keys:
                properties = self.dataset[k]['properties']
                ln_vol = properties['seg_ln_vol_fold' + str(self.fold)]
                patch_files = [join(self.folder_with_preprocessed_data, 'cropped_slice', filename) for filename in
                               os.listdir(join(self.folder_with_preprocessed_data, 'cropped_slice')) if
                               filename.startswith(k) and filename.endswith('.npy') and 'segidx0' in filename and 'fold' not in filename]
                output_dict['LNM_identifier'].append(k)

                softmax_preds = []
                for patch_idx, patch_file in enumerate(patch_files):
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
                        all_in_gpu=all_in_gpu, CT_N=ln_vol, return_feature=False)
                    softmax_preds.append(softmax_pred.cpu().numpy())

                softmax_preds = np.vstack(softmax_preds)
                softmax_mean = np.mean(softmax_preds, 0)
                output_dict['overall_score'].append(softmax_mean[1])
            feature_df = pd.merge(deep_feature_df[split_name], pd.DataFrame.from_dict(output_dict), how='inner', on=['LNM_identifier'])
            feature_df.to_excel(writer, index=False, sheet_name=split_name)
        writer.close()

        self.print_to_log_file("finished prediction")


class nnUNetTrainerV2_PDAC_classification_resnet_MaskSideBranch_GTMaskLN_v2(nnUNetTrainerV2_PDAC_classification_resnet_MaskSideBranch):
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

        self.network = ResNet18_DeepTEN_MaskSideBranch(self.num_classes, self.pretrained, self.if_freeze, self.mask_input_layer, self.mask_input_method,
                        roi_margin_min = self.data_aug_params["margins_min"], roi_margin_max = self.data_aug_params["margins_max"],
                        freeze_layernum = self.freeze_layernum, incorp_CT_N=True, incorp_CT_N_method='v2')
        # self.network = ResNet18_MaskSideBranch(self.num_classes, self.pretrained, self.if_freeze,
        #                                                self.mask_input_layer, self.mask_input_method,
        #                                                roi_margin_min=self.data_aug_params["margins_min"],
        #                                                roi_margin_max=self.data_aug_params["margins_max"],
        #                                                freeze_layernum=self.freeze_layernum)
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

        ln_vol = []
        for k in range(len(data_dict['properties'])):
            ln_vol.append(data_dict['properties'][k]['gtmask_ln_vol_fold'+str(self.fold)])
        ln_vol = np.array(ln_vol)

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
        ln_vol = maybe_to_torch(ln_vol)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
            ln_vol = to_cuda(ln_vol)

        self.optimizer.zero_grad()

        logits = self.network(data, data[:, -1:], ln_vol)

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
            ln_vol = properties['gtmask_ln_vol_fold'+str(self.fold)]
            patch_files = [join(self.folder_with_preprocessed_data, 'cropped_slice', filename) for filename in
                           os.listdir(join(self.folder_with_preprocessed_data, 'cropped_slice')) if filename.startswith(k) and filename.endswith('.npy') and 'segidx0' in filename]
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
                    all_in_gpu=all_in_gpu, CT_N=ln_vol)


                softmax_preds.append(softmax_pred)
            if len(softmax_preds) > 0:
                # patient_softmax_pred = torch.cat(softmax_preds, dim = 0).mean(dim=0, keepdim=True)
                # patient_pred = torch.argmax(patient_softmax_pred, dim=1).cpu().numpy()[0]
                # patient_softmax_pred = patient_softmax_pred.cpu().numpy()[0]
                patient_preds = torch.argmax(torch.cat(softmax_preds, dim = 0), dim=1).cpu().numpy()
                patient_pred = np.argmax(np.bincount(patient_preds))
            else:
                # patient_softmax_pred, patient_pred = None, None
                patient_pred = None
            # pred_gt_tuples.append([patient_softmax_pred, patient_pred, label])
            pred_gt_tuples.append([patient_pred, label])


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
            ln_vol = properties['gtmask_ln_vol_fold'+str(self.fold)]
            patch_files = [join(self.folder_with_preprocessed_data, 'cropped_slice', filename) for filename in
                           os.listdir(join(self.folder_with_preprocessed_data, 'cropped_slice')) if filename.startswith(k) and filename.endswith('.npy') and 'segidx0' in filename]
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
                    all_in_gpu=all_in_gpu, CT_N=ln_vol)


                softmax_preds.append(softmax_pred)
            if len(softmax_preds) > 0:
                # patient_softmax_pred = torch.cat(softmax_preds, dim = 0).mean(dim=0, keepdim=True)
                # patient_pred = torch.argmax(patient_softmax_pred, dim=1).cpu().numpy()[0]
                # patient_softmax_pred = patient_softmax_pred.cpu().numpy()[0]
                patient_preds = torch.argmax(torch.cat(softmax_preds, dim = 0), dim=1).cpu().numpy()
                patient_pred = np.argmax(np.bincount(patient_preds))
            else:
                # patient_softmax_pred, patient_pred = None, None
                patient_pred = None
            # pred_gt_tuples.append([patient_softmax_pred, patient_pred, label])
            pred_gt_tuples.append([patient_pred, label])


        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")

        task = self.dataset_directory.split("/")[-1]
        job_name = self.experiment_name
        _ = aggregate_classification_scores(pred_gt_tuples,  list(self.dataset_test.keys()),
                             json_output_file=join(output_folder, "summary.json"),
                             json_name=job_name ,
                             json_author="Zhilin",
                             json_task=task, num_threads=default_num_threads)
        self.network.train(current_mode)


class nnUNetTrainerV2_PDAC_classification_resnet_MaskSideBranch_GTMaskLN_Other_v2(nnUNetTrainerV2_PDAC_classification_resnet_MaskSideBranch):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.incorp_info_key = ['age', 'sex', 'degree', 'PDAC_volume']

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

        self.network = ResNet18_DeepTEN_MaskSideBranch(self.num_classes, self.pretrained, self.if_freeze, self.mask_input_layer, self.mask_input_method,
                        roi_margin_min = self.data_aug_params["margins_min"], roi_margin_max = self.data_aug_params["margins_max"],
                        freeze_layernum = self.freeze_layernum, incorp_CT_N=True, incorp_CT_N_method='v2', incorp_other_info=self.incorp_info_key)
        # self.network = ResNet18_MaskSideBranch(self.num_classes, self.pretrained, self.if_freeze,
        #                                                self.mask_input_layer, self.mask_input_method,
        #                                                roi_margin_min=self.data_aug_params["margins_min"],
        #                                                roi_margin_max=self.data_aug_params["margins_max"],
        #                                                freeze_layernum=self.freeze_layernum)
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

        info = []
        for k in range(len(data_dict['properties'])):
            tmp = [data_dict['properties'][k]['gtmask_ln_vol_fold'+str(self.fold)]]
            for info_key in self.incorp_info_key:
                tmp.append(data_dict['properties'][k][info_key])
            info.append(tmp)
        info = np.array(info)

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
        info = maybe_to_torch(info)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
            info = to_cuda(info)

        self.optimizer.zero_grad()

        logits = self.network(data, data[:, -1:], info)

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
            info = [properties['gtmask_ln_vol_fold'+str(self.fold)]]
            for info_key in self.incorp_info_key:
                info.append(properties[info_key])
            patch_files = [join(self.folder_with_preprocessed_data, 'cropped_slice', filename) for filename in
                           os.listdir(join(self.folder_with_preprocessed_data, 'cropped_slice')) if filename.startswith(k) and filename.endswith('.npy') and 'segidx0' in filename]
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
                    all_in_gpu=all_in_gpu, CT_N=info)


                softmax_preds.append(softmax_pred)
            if len(softmax_preds) > 0:
                # patient_softmax_pred = torch.cat(softmax_preds, dim = 0).mean(dim=0, keepdim=True)
                # patient_pred = torch.argmax(patient_softmax_pred, dim=1).cpu().numpy()[0]
                # patient_softmax_pred = patient_softmax_pred.cpu().numpy()[0]
                patient_preds = torch.argmax(torch.cat(softmax_preds, dim = 0), dim=1).cpu().numpy()
                patient_pred = np.argmax(np.bincount(patient_preds))
            else:
                # patient_softmax_pred, patient_pred = None, None
                patient_pred = None
            # pred_gt_tuples.append([patient_softmax_pred, patient_pred, label])
            pred_gt_tuples.append([patient_pred, label])


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
            info = [properties['gtmask_ln_vol_fold' + str(self.fold)]]
            for info_key in self.incorp_info_key:
                info.append(properties[info_key])
            patch_files = [join(self.folder_with_preprocessed_data, 'cropped_slice', filename) for filename in
                           os.listdir(join(self.folder_with_preprocessed_data, 'cropped_slice')) if filename.startswith(k) and filename.endswith('.npy') and 'segidx0' in filename]
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
                    all_in_gpu=all_in_gpu, CT_N=info)


                softmax_preds.append(softmax_pred)
            if len(softmax_preds) > 0:
                # patient_softmax_pred = torch.cat(softmax_preds, dim = 0).mean(dim=0, keepdim=True)
                # patient_pred = torch.argmax(patient_softmax_pred, dim=1).cpu().numpy()[0]
                # patient_softmax_pred = patient_softmax_pred.cpu().numpy()[0]
                patient_preds = torch.argmax(torch.cat(softmax_preds, dim = 0), dim=1).cpu().numpy()
                patient_pred = np.argmax(np.bincount(patient_preds))
            else:
                # patient_softmax_pred, patient_pred = None, None
                patient_pred = None
            # pred_gt_tuples.append([patient_softmax_pred, patient_pred, label])
            pred_gt_tuples.append([patient_pred, label])


        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")

        task = self.dataset_directory.split("/")[-1]
        job_name = self.experiment_name
        _ = aggregate_classification_scores(pred_gt_tuples,  list(self.dataset_test.keys()),
                             json_output_file=join(output_folder, "summary.json"),
                             json_name=job_name ,
                             json_author="Zhilin",
                             json_task=task, num_threads=default_num_threads)
        self.network.train(current_mode)

class nnUNetTrainerV2_PDAC_classification_resnet_MaskSideBranch_SegLN_Other_v2(nnUNetTrainerV2_PDAC_classification_resnet_MaskSideBranch):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.weight_decay = 5e-4
        self.freeze_layernum = 4
        self.n_codes = 8  # 4
        self.len_codes = 64
        self.incorp_info_key = ['CT_reported_N']
        self.output_folder = join(self.output_folder_base, 'fold_%d' % fold, 'freezenum%d_weightdecay%f_ncodes%d_lencodes%d_masklayer%s_%s_%s' %
                                  (self.freeze_layernum, self.weight_decay, self.n_codes, self.len_codes,
                                   ','.join(map(str, self.mask_input_layer)), self.mask_input_method, ','.join(self.incorp_info_key)))

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

        self.network = ResNet18_DeepTEN_MaskSideBranch(self.num_classes, self.pretrained, self.if_freeze, self.mask_input_layer, self.mask_input_method,
                        roi_margin_min = self.data_aug_params["margins_min"], roi_margin_max = self.data_aug_params["margins_max"],
                        freeze_layernum = self.freeze_layernum, incorp_CT_N=True, incorp_CT_N_method='v2', incorp_other_info=self.incorp_info_key)

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

        info = []
        for k in range(len(data_dict['properties'])):
            tmp = [data_dict['properties'][k]['seg_ln_vol_fold' + str(self.fold)]]
            for info_key in self.incorp_info_key:
                tmp.append(data_dict['properties'][k][info_key])
            info.append(tmp)
        info = np.array(info)


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
        info = maybe_to_torch(info)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
            info = to_cuda(info)

        self.optimizer.zero_grad()

        logits = self.network(data, data[:, -1:], info)

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
            info = [properties['seg_ln_vol_fold'+str(self.fold)]]
            for info_key in self.incorp_info_key:
                info.append(properties[info_key])
            patch_files = [join(self.folder_with_preprocessed_data, 'cropped_slice', filename) for filename in
                           os.listdir(join(self.folder_with_preprocessed_data, 'cropped_slice')) if filename.startswith(k) and filename.endswith('.npy') and 'segidx0' in filename]
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
                    all_in_gpu=all_in_gpu, CT_N=info)


                softmax_preds.append(softmax_pred)
            if len(softmax_preds) > 0:
                # patient_softmax_pred = torch.cat(softmax_preds, dim = 0).mean(dim=0, keepdim=True)
                # patient_pred = torch.argmax(patient_softmax_pred, dim=1).cpu().numpy()[0]
                # patient_softmax_pred = patient_softmax_pred.cpu().numpy()[0]
                patient_preds = torch.argmax(torch.cat(softmax_preds, dim = 0), dim=1).cpu().numpy()
                patient_pred = np.argmax(np.bincount(patient_preds))
            else:
                # patient_softmax_pred, patient_pred = None, None
                patient_pred = None
            # pred_gt_tuples.append([patient_softmax_pred, patient_pred, label])
            pred_gt_tuples.append([patient_pred, label])


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
            info = [properties['seg_ln_vol_fold'+str(self.fold)]]
            for info_key in self.incorp_info_key:
                info.append(properties[info_key])
            patch_files = [join(self.folder_with_preprocessed_data, 'cropped_slice', filename) for filename in
                           os.listdir(join(self.folder_with_preprocessed_data, 'cropped_slice')) if filename.startswith(k) and filename.endswith('.npy') and 'segidx0' in filename]
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
                    all_in_gpu=all_in_gpu, CT_N=info)


                softmax_preds.append(softmax_pred)
            if len(softmax_preds) > 0:
                # patient_softmax_pred = torch.cat(softmax_preds, dim = 0).mean(dim=0, keepdim=True)
                # patient_pred = torch.argmax(patient_softmax_pred, dim=1).cpu().numpy()[0]
                # patient_softmax_pred = patient_softmax_pred.cpu().numpy()[0]
                patient_preds = torch.argmax(torch.cat(softmax_preds, dim = 0), dim=1).cpu().numpy()
                patient_pred = np.argmax(np.bincount(patient_preds))
            else:
                # patient_softmax_pred, patient_pred = None, None
                patient_pred = None
            # pred_gt_tuples.append([patient_softmax_pred, patient_pred, label])
            pred_gt_tuples.append([patient_pred, label])


        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")

        task = self.dataset_directory.split("/")[-1]
        job_name = self.experiment_name
        _ = aggregate_classification_scores(pred_gt_tuples,  list(self.dataset_test.keys()),
                             json_output_file=join(output_folder, "summary.json"),
                             json_name=job_name ,
                             json_author="Zhilin",
                             json_task=task, num_threads=default_num_threads)
        self.network.train(current_mode)