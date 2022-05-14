#    Created by zhilin zheng for lymph node metastasis prediction

from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.training.data_augmentation.default_data_augmentation import get_moreDA_augmentation_classification
from nnunet.network_architecture.generic_Classification_roipool import Generic_Classification_RoIPooling
from nnunet.network_architecture.initialization import InitWeights_He_Classification
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnunet.network_architecture.neural_network import NeuralNetwork
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.network_training.nnUNetTrainer_Attention_FRCNN_stage2 import nnUNetTrainer_Attention_FRCNN_stage2
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
from torch.optim.lr_scheduler import _LRScheduler

class nnUNetTrainer_Attention_FRCNN_stage2_weightfreeze(nnUNetTrainer_Attention_FRCNN_stage2):

    def load_pretrained_checkpoint(self, fname, train=True):
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
        for name, p in self.network.named_parameters():
            if name.startswith('conv_blocks_context'):
                p.requires_grad = False
        self.optimizer.param_groups[0]['params'] = list(filter(lambda x: x.requires_grad is not False, self.network.parameters()))
        self.amp_initialized = False
        self._maybe_init_amp()

    def load_checkpoint_ram(self, saved_model, train=True):
        """
        used for if the checkpoint is already in ram
        :param saved_model:
        :param train:
        :return:
        """
        if not self.was_initialized:
            self.initialize(train)

        new_state_dict = OrderedDict()
        curr_state_dict_keys = list(self.network.state_dict().keys())
        # if state dict comes form nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        for k, value in saved_model['state_dict'].items():
            key = k
            if key not in curr_state_dict_keys:
                print("duh")
                key = key[7:]
            new_state_dict[key] = value

        # if we are fp16, then we need to reinitialize the network and the optimizer. Otherwise amp will throw an error
        if self.fp16:
            self.network, self.optimizer, self.lr_scheduler = None, None, None
            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

        self.network.load_state_dict(new_state_dict)
        self.epoch = saved_model['epoch']

        if train:
            for name, p in self.network.named_parameters():
                if name.startswith('conv_blocks_context'):
                    p.requires_grad = False
            optimizer_state_dict = saved_model['optimizer_state_dict']
            if optimizer_state_dict is not None:
                self.optimizer.param_groups[0]['params'] = list(filter(lambda x: x.requires_grad is not False, self.network.parameters()))
                self.optimizer.load_state_dict(optimizer_state_dict)

            if self.lr_scheduler is not None and hasattr(self.lr_scheduler, 'load_state_dict') and saved_model[
                'lr_scheduler_state_dict'] is not None:
                self.lr_scheduler.load_state_dict(saved_model['lr_scheduler_state_dict'])

            if issubclass(self.lr_scheduler.__class__, _LRScheduler):
                self.lr_scheduler.step(self.epoch)

        self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode, self.all_val_eval_metrics = saved_model[
            'plot_stuff']

        # after the training is done, the epoch is incremented one more time in my old code. This results in
        # self.epoch = 1001 for old trained models when the epoch is actually 1000. This causes issues because
        # len(self.all_tr_losses) = 1000 and the plot function will fail. We can easily detect and correct that here
        if self.epoch != len(self.all_tr_losses):
            self.print_to_log_file("WARNING in loading checkpoint: self.epoch != len(self.all_tr_losses). This is "
                                   "due to an old bug and should only appear when you are loading old models. New "
                                   "models should have this fixed! self.epoch is now set to len(self.all_tr_losses)")
            self.epoch = len(self.all_tr_losses)
            self.all_tr_losses = self.all_tr_losses[:self.epoch]
            self.all_val_losses = self.all_val_losses[:self.epoch]
            self.all_val_losses_tr_mode = self.all_val_losses_tr_mode[:self.epoch]
            self.all_val_eval_metrics = self.all_val_eval_metrics[:self.epoch]

        self.amp_initialized = False
        self._maybe_init_amp()
