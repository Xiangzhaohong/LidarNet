import numba
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@numba.jit(nopython=True)
def circle_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    scores = dets[:, 2]
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep = []
    for _i in range(ndets):
        i = order[_i]  # start with highest score box
        if suppressed[i] == 1:  # if any box have enough iou with this, remove it
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            # calculate center distance between i and j box
            dist = (x1[i]-x1[j])**2 + (y1[i]-y1[j])**2

            # ovr = inter / areas[j]
            if dist <= thresh:
                suppressed[j] = 1
    return keep


def Center_FocalLoss(pred, gt, alpha=2, beta=4):
    ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        Arguments:
          pred (batch, W, H, C)
          gt_regr (batch, W, H, C)
    '''
    # print(pred.shape, gt.shape)

    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weight = torch.pow(1 - gt, beta)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weight * neg_inds

    num_pos = pos_inds.float().sum()

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        # TODO
        # 这样相当于把batch_size 和 num_pos 混一起算了, 会不会有问题呢?
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def Center_RegLoss(pred, gt, mask=None):
    '''Regression loss for an output tensor
        Arguments:
          pred (batch, max_objects,  dim)
          gt (batch, max_objects, dim)
          mask (batch, max_objects)
    '''
    num = pred.size(0) * pred.size(1)
    if mask is not None:
        num = mask.float().sum()
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # isnotnan = (~torch.isnan(gt)).float()
        # mask *= isnotnan
        pred = pred * mask
        gt = gt * mask

    loss = torch.abs(pred - gt)

    loss = loss.float().sum()
    loss = loss / (num + 1e-4)

    return loss


def balanced_l1_loss(pred, target, beta=1.0, alpha=0.5, gamma=1.5, reduction="none"):

    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0

    diff = torch.abs(pred - target)
    b = np.e ** (gamma / alpha) - 1
    loss = torch.where(
        diff < beta,
        alpha / b * (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff,
        gamma * diff + gamma / b - alpha * beta,)
    if reduction == "none":
        loss = loss
    elif reduction == "sum":
        loss = loss.sum()
    elif reduction == "mean":
        loss = loss.mean()
    else:
        raise NotImplementedError
    return loss


class BalancedL1Loss(nn.Module):
    """Balanced L1 Loss
    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)
    """
    def __init__(self, loss_weight=1.0, beta=1.0, alpha=0.5, gamma=1.5, reduction="none"):
        super(BalancedL1Loss, self).__init__()
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.weight = loss_weight
        self.reduction = reduction
        assert reduction == "none", "only none reduction is support!"

    def forward(self, pred, gt, mask=None):
        num = pred.size(0) * pred.size(1)
        if mask is not None:
            num = mask.float().sum()
            mask = mask.unsqueeze(2).expand_as(pred).float()
            # isnotnan = (~torch.isnan(gt)).float()
            # mask *= isnotnan
            pred = pred * mask
            gt = gt * mask

        loss = balanced_l1_loss(pred, gt, beta=self.beta, alpha=self.alpha, gamma=self.gamma,
                                reduction=self.reduction)
        loss = loss.sum() / (num + 1e-4) * self.weight
        return loss