import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiBoxLoss(nn.Module):
  def __init__(self, num_classes, neg_pos_ratio):
    super(MultiBoxLoss, self).__init__()
    self.num_classes = num_classes + 1
    self.neg_pos_ratio = neg_pos_ratio

  def forward(self, reg_pred, cls_pred, reg_targets, cls_targets):
    batch_size = reg_pred.size(0)
    # match priors (default boxes) and ground truth boxes

    pos_ind = cls_targets > 0
    # non_neg_ind = cls_targets >= 0

    # Localization Loss (Smooth L1)
    # Shape: [batch,num_priors,4]
    pos_ind_expand = pos_ind[:, :, None].expand_as(reg_pred)  # 正样本对应的行值为1
    loc_loss = F.smooth_l1_loss(reg_pred[pos_ind_expand].view(-1, 4),
                                reg_targets[pos_ind_expand].view(-1, 4), size_average=False)

    # Compute max prob across batch for hard negative mining
    batch_prob = cls_pred.view(-1, self.num_classes)  # [num_batch*8732, 21]

    # 计算每个ancher对应的gtbox的分类概率,正样本是对应类别的概率，负样本是对应背景的概率
    cls_loss = torch.log(torch.sum(torch.exp(batch_prob - batch_prob.max()), 1, keepdim=True)) + batch_prob.max()
    cls_loss = cls_loss - batch_prob.gather(1, torch.clamp(cls_targets, min=0, max=None).view(-1, 1))

    # Hard Negative Mining
    cls_loss[pos_ind.view(-1, 1)] = 0  # filter out pos boxes for now
    cls_loss = cls_loss.view(batch_size, -1)  # [num_batch,8732]
    _, loss_idx = cls_loss.sort(1, descending=True)  # 每个batch的概率从大到小排序 [batch_size, 8732]
    _, idx_rank = loss_idx.sort(1)  # 把排序的index再做排序 [batch_size, 8732]
    num_pos = pos_ind.long().sum(1, keepdim=True)  # 计算每个batch的正样本数量 [batch_size,1]
    # 负样本数量最多为正样本的3倍
    num_neg = torch.clamp(self.neg_pos_ratio * num_pos, min=None, max=pos_ind.size(1) - 1)
    # 负样本的index
    neg_ind = idx_rank < num_neg

    # Confidence Loss Including Positive and Negative Examples
    pos_ind = pos_ind[:, :, None]
    neg_ind = neg_ind[:, :, None]
    cls_pred_selected = cls_pred[(pos_ind + neg_ind) > 0].view(-1, self.num_classes)
    cls_targets_selected = torch.clamp(cls_targets[(pos_ind + neg_ind) > 0], min=0, max=None)
    cls_loss = F.cross_entropy(cls_pred_selected, cls_targets_selected, size_average=False)

    # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
    loc_loss /= num_pos.data.sum()
    cls_loss /= num_pos.data.sum()
    return loc_loss, cls_loss


def sparse_mask(net, sparse_type, threshold, include_fc=False):
  p_mask = {}
  # 取出所有线性组合卷积权重
  paras = {v[0]: v[1] for v in net.named_parameters() if 'weight' in v[0] and 'bn' not in v[0]}
  for key in paras.keys():
    # mask是线性组合的权重
    p = paras[key]

    if 'base' in key or 'extra' in key:
      # 如果某个group权重均值大于threshold，就设置成1
      if sparse_type == 'filter':
        abs_sum = p.abs().sum(1).sum(1).sum(1) / (p.size(1) * p.size(2) * p.size(3))
      elif sparse_type == 'channel':
        abs_sum = p.abs().sum(0).sum(1).sum(1) / (p.size(0) * p.size(2) * p.size(3))
      elif sparse_type == 'shape':
        abs_sum = p.abs().sum(0).sum(0) / (p.size(0) * p.size(1))
      else:
        assert False, 'type unknown !'

    elif 'fc' in key and include_fc:
      # 如果某个group权重均值大于threshold，就设置成1
      if sparse_type == 'filter':
        abs_sum = p.abs().sum(1) / p.size(1)
      elif sparse_type == 'channel':
        abs_sum = p.abs().sum(0) / p.size(0)
      elif sparse_type != 'shape':
        continue
      else:
        assert False, 'type unknown !'

    else:
      continue

    p_mask[key] = (abs_sum > threshold).type(p.data.type())
    _, max_ind = abs_sum.max(0)
    p_mask[key].index_fill_(0, max_ind, 1)

  return p_mask


def group_sparsity_loss(net, sparse_type, include_fc=False):
  group_lasso_losses = []
  # 取出所有线性组合的权重
  paras = [v for v in net.named_parameters() if 'weight' in v[0] and 'bn' not in v[0]]
  for p in paras:
    if 'base' in p[0] or 'extra' in p[0]:
      p = p[1]
      if sparse_type == 'filter':
        group_lasso_losses.append(torch.sum(torch.sqrt(torch.pow(p, 2).sum(1).sum(1).sum(1))))
      elif sparse_type == 'channel':
        group_lasso_losses.append(torch.sum(torch.sqrt(torch.pow(p, 2).sum(0).sum(1).sum(1))))
      elif sparse_type == 'shape':
        group_lasso_losses.append(torch.sum(torch.sqrt(torch.pow(p, 2).sum(0).sum(0))))
      else:
        assert False, 'type unknown !'

    elif 'fc' in p[0] and include_fc:
      p = p[1]
      if sparse_type == 'filter':
        group_lasso_losses.append(torch.sum(torch.sqrt(torch.pow(p, 2).sum(1))))
      elif sparse_type == 'channel':
        group_lasso_losses.append(torch.sum(torch.sqrt(torch.pow(p, 2).sum(0))))
      elif sparse_type == 'shape':
        pass
      else:
        assert False, 'type unknown !'

  # 对所有层的group lasso损失值求平均
  group_lasso_losses = sum(group_lasso_losses) / len(group_lasso_losses)
  return group_lasso_losses


def sparse_summary(p_mask, summary_writer, step, sparse_type):
  for key in p_mask.keys():
    summary_writer.add_scalar('glasso/' + key + sparse_type, torch.sum(p_mask[key]).data[0], step)


if __name__ == '__main__':
  from nets.mobilenet_base import *
  from nets.vgg_base import *
  from nets.ssd import *

  net = SSD()
  # net = vgg16base()
  # net = MobileNet()
  group_sparsity_loss(net, 'filter')
