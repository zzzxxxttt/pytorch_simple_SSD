import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiBoxLoss(nn.Module):
  def __init__(self, num_classes, neg_pos_ratio):
    super(MultiBoxLoss, self).__init__()
    self.num_classes = num_classes + 1
    self.neg_pos_ratio = neg_pos_ratio

  def forward(self, reg_pred, cls_pred, reg_targets, cls_targets):
    # reg_pred: [B, 8732, 4]
    # cls_pred: [B, 8732, 21]
    # reg_targets: [B, 8732, 4]
    # cls_targets: [B, 8732]
    batch_size = reg_pred.size(0)
    cls_targets = torch.clamp(cls_targets, min=0, max=None)

    pos_ind = cls_targets > 0  # shape: [B, 8732]
    num_pos = pos_ind.sum(1, keepdim=True)  # shape: [B, 1]

    # extract reg_targets of postive examples and calculate smooth L1 loss
    pos_ind_expand = pos_ind[:, :, None].expand_as(reg_pred)  # shape: [B, 8732, 4]
    loc_loss = F.smooth_l1_loss(reg_pred[pos_ind_expand].view(-1, 4),
                                reg_targets[pos_ind_expand].view(-1, 4),
                                reduction='sum')

    # get the class probability of each anchor box
    cls_probs = \
      F.softmax(cls_pred.view(-1, self.num_classes), -1).gather(1, cls_targets.view(-1, 1))
    # here we use negative log probability, because these probs are for negative samples
    # so a low probability means it is not been classified as background
    # which is wrong for negative samples
    cls_probs = -cls_probs.log()

    # Hard Negative Mining
    cls_probs[pos_ind.view(-1, 1)] = 0  # filter out pos boxes for now
    cls_probs = cls_probs.view(batch_size, -1)  # shape: [B, 8732]
    # sort the prob by ascending order
    _, prob_idx = cls_probs.sort(1, descending=True)  # shape: [B, 8732]
    # sort the index by descending order
    _, idx_rank = prob_idx.sort(1)  # shape: [B, 8732]
    # value = 1 means negative samples
    neg_ind = idx_rank < self.neg_pos_ratio * num_pos

    used_ind = ((pos_ind + neg_ind) > 0).detach()

    # Confidence Loss Including Positive and Negative Examples
    cls_loss = F.cross_entropy(cls_pred[used_ind].view(-1, self.num_classes),
                               cls_targets[used_ind],
                               reduction='sum')

    # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
    loc_loss /= num_pos.sum().float()
    cls_loss /= num_pos.sum().float()
    return loc_loss, cls_loss


# if __name__ == '__main__':
#   from nets.vgg_base import *
#   from nets.ssd import *
#
#   net = SSD()
#   # net = vgg16base()
#   # net = MobileNet()
