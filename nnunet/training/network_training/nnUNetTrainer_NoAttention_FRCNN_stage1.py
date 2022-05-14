from nnunet.training.network_training.nnUNetTrainerV2_nestedCV import nnUNetTrainerV2_nestedCV
from nnunet.training.loss_functions.my_dice_loss import Binary_Tversky_and_CE_loss
import numpy as np
import torch
from ptflops import get_model_complexity_info

class nnUNetTrainer_NoAttention_FRCNN_stage1(nnUNetTrainerV2_nestedCV):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 1000
        self.initial_lr = 1e-2
        self.oversample_foreground_percent = 0.5
        self.loss = Binary_Tversky_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})

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

    def get_model_complexity(self):
        def prepare_input(resolution):
            x = torch.FloatTensor(1, 2, *(resolution))
            return dict(x=x.cuda())

        self.network.do_ds = False
        macs, params = get_model_complexity_info(self.network, input_res=tuple(self.patch_size), input_constructor=prepare_input, as_strings=True, print_per_layer_stat=False)
        print(' - MACs: ' + macs)
        print(' - Params: ' + params)
