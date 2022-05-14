from nnunet.training.loss_functions.dice_loss import SoftDiceLoss, SoftDiceLossSquared

import torch
from nnunet.training.loss_functions.ND_Crossentropy import CrossentropyND
from nnunet.training.loss_functions.TopK_loss import TopKLoss
from nnunet.training.loss_functions.focal_loss import FocalLoss
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor
from torch import nn
import numpy as np
from torch import Tensor
from torch.nn import functional as F
from nnunet.training.loss_functions.dice_loss import get_tp_fp_fn_tn

class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, x: Tensor, y: Tensor, loss_mask:Tensor = None) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            y = y.permute(0, *range(2, y.ndim), 1).reshape(-1, c)
            if loss_mask is not None:
                mask = loss_mask.permute(0, *range(2, loss_mask.ndim), 1).reshape(-1)

        log_p = F.log_softmax(x, dim=-1)
        if loss_mask is not None:
            loss = - torch.sum(y * log_p, dim=1) * mask
        else:
            loss = - torch.sum(y * log_p, dim=1)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

class CrossEntropyLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, x: Tensor, y: Tensor, loss_mask:Tensor = None) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            y = y.view(-1, 1)
            if loss_mask is not None:
                mask = loss_mask.view(-1)

        y_onehot = torch.zeros(x.shape)
        if x.device.type == "cuda":
            y_onehot = y_onehot.cuda(x.device.index)
        y_onehot.scatter_(1, y, 1)

        log_p = F.log_softmax(x, dim=-1)
        if loss_mask is not None:
            loss = - torch.sum(y_onehot * log_p, dim=1) * mask
        else:
            loss = - torch.sum(y_onehot * log_p, dim=1)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss



class WeightedSoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        add weight to fp and fn
        """
        super(WeightedSoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, fp_weight=1., fn_weight=1., loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        if not fp_weight + fn_weight == 2:
            fp_weight, fn_weight = 2*(fp_weight + self.smooth) / (fp_weight + fn_weight + 2 * self.smooth), 2*(fn_weight + self.smooth) / (fp_weight + fn_weight + 2 * self.smooth)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp_weight * fp + fn_weight * fn + self.smooth

        dc = nominator / denominator

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc


class Soft_DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(Soft_DC_and_CE_loss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = SoftCrossEntropyLoss(**ce_kwargs)
        if not square_dice:
            self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        else:
            self.dc = SoftDiceLossSquared(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target, mask):
        dc_loss = self.dc(net_output, target, loss_mask = mask) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target, loss_mask = mask) if self.weight_ce != 0 else 0
        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result

class Hard_DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(Hard_DC_and_CE_loss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = CrossEntropyLoss(**ce_kwargs)
        if not square_dice:
            self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        else:
            self.dc = SoftDiceLossSquared(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target, mask):
        dc_loss = self.dc(net_output, target, loss_mask = mask) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0].long(), loss_mask = mask) if self.weight_ce != 0 else 0
        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result

class WeightedDC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(WeightedDC_and_CE_loss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = nn.CrossEntropyLoss(**ce_kwargs)
        if not square_dice:
            self.dc = WeightedSoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        else:
            self.dc = None

    def forward(self, net_output, target, fp_weight=1., fn_weight=1.):
        dc_loss = self.dc(net_output, target, fp_weight, fn_weight) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result


class MultipleOutputLoss2_withMask(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2_withMask, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y, mask=None):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        l = weights[0] * self.loss(x[0], y[0], mask[0] if mask else None)
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y[i], mask[i] if mask else None)
        return l

class MultipleOutputLoss2_withFPFNWeight(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2_withFPFNWeight, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y, fp_weight=1., fn_weight=1.):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        l = weights[0] * self.loss(x[0], y[0], fp_weight=fp_weight, fn_weight=fn_weight)
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y[i], fp_weight=fp_weight, fn_weight=fn_weight)
        return l

def get_weighted_tp_fp_fn_tn(net_output, gt, fn_mask, axes=None, mask=None, square=False, ignore_index=None):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            if ignore_index is not None:
                gt[gt == ignore_index] = 0 #added by zhengzhilin980
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot * fn_mask
    fp = net_output * (1 - y_onehot) * fn_mask
    fn = (1 - net_output) * y_onehot * fn_mask
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn

class TverskyLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        """
        super(TverskyLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None, pos_weight=4.):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        fn_mask = torch.where(y == 1, torch.tensor(pos_weight, dtype=y.dtype, device=y.device), torch.tensor(1., dtype=y.dtype, device=y.device))
        tp, fp, fn, _ = get_weighted_tp_fp_fn_tn(x, y.bool().float(), fn_mask, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + 0.5 * fp + 1.5 * fn + self.smooth

        dc = nominator / denominator

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc

class Binary_Tversky_and_CE_loss(nn.Module):
    '''
    Convert target to binary
    add weight to fn in dice loss
    '''
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(Binary_Tversky_and_CE_loss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = nn.CrossEntropyLoss(**ce_kwargs)
        if not square_dice:
            self.dc = TverskyLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        else:
            pass
            # self.dc = SoftDiceLossSquared(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0].bool().long()) if self.weight_ce != 0 else 0
        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result