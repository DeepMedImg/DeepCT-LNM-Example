from nnunet.training.network_training.nnUNetTrainerV2_PDAC_classification import nnUNetTrainerV2_PDAC_classification
from nnunet.network_architecture.ResNet18 import ResNet18RoIPool, ResNet18_DeepTEN_MaskSideBranch, ResNet18_DeepTEN
import torch
from nnunet.utilities.nd_softmax import softmax_helper
from skimage.io import imsave
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
import os
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import roc_auc_score

try:
    from apex import amp
except ImportError:
    amp = None

class nnUNetTrainerV2_PDAC_classification_resnet(nnUNetTrainerV2_PDAC_classification):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.pretrained = True
        self.if_freeze = False
        self.initial_lr = 1e-4
        self.freeze_layernum = 3

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


        self.network = ResNet18RoIPool(self.num_classes, self.pretrained, self.if_freeze,
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

        logits = self.network(data, roi)

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






