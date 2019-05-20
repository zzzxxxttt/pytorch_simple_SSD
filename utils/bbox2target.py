# -*- coding: utf-8 -*-
import torch


def xywh_to_xyxy(boxes):
  return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,  # xmin, ymin
                    boxes[:, :2] + boxes[:, 2:] / 2), 1)  # xmax, ymax


def xyxy_to_xywh(boxes):
  return torch.cat((boxes[:, 2:] + boxes[:, :2]) / 2,  # cx, cy
                   boxes[:, 2:] - boxes[:, :2], 1)  # w, h


def IoU(box_a, box_b):
  # 输入xyxy格式的box
  max_xy = torch.min(box_a[:, 2:].unsqueeze(1), box_b[:, 2:].unsqueeze(0))
  min_xy = torch.max(box_a[:, :2].unsqueeze(1), box_b[:, :2].unsqueeze(0))
  inter = torch.clamp((max_xy - min_xy), min=0, max=None)
  inter = inter[:, :, 0] * inter[:, :, 1]

  area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1)  # [A,B]
  area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0)  # [A,B]
  union = area_a + area_b - inter
  return inter / union  # [A,B]


def encode(gt_boxes, priors):
  # 输入是xyxy形式的gt_boxes和xywh形式的priors
  # dist b/t match center and prior's center
  # gt_boxes的中心与priors的中心距离除以priors的长宽
  g_cxcy = ((gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2 - priors[:, :2]) / (priors[:, 2:] * 0.1)
  # match wh / prior wh
  # gt_boxes的长宽除以priors的长宽再取对数
  g_wh = torch.log((gt_boxes[:, 2:] - gt_boxes[:, :2]) / priors[:, 2:]) / 0.2

  # return target for smooth_l1_loss
  return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors):
  # 输入为网络的输出和xywh格式的priors
  boxes = torch.cat((priors[:, :2] + loc[:, :2] * 0.1 * priors[:, 2:],  # 乘以priors的长宽再加上priors的中心坐标
                     priors[:, 2:] * torch.exp(loc[:, 2:] * 0.2)), 1)  # 取exp然后再乘以priors的长宽
  # 转换为xyxy格式
  boxes[:, :2] -= boxes[:, 2:] / 2
  boxes[:, 2:] += boxes[:, :2]
  return boxes


def decode_batch(loc, priors):
  # 输入为网络的输出 [batch_size, num_priors, 4]
  # xywh格式的priors[num_priors, 4]
  boxes = torch.cat((
    # 乘以priors的长宽再加上priors的中心坐标
    priors[:, :2][None, :, :] + loc[:, :, :2] * 0.1 * priors[:, 2:][None, :, :],
    # 取exp然后再乘以priors的长宽
    priors[:, 2:][None, :, :] * torch.exp(loc[:, :, 2:] * 0.2)), 2)

  # 转换为xyxy格式
  boxes[:, :, :2] -= boxes[:, :, 2:] / 2
  boxes[:, :, 2:] += boxes[:, :, :2]
  return boxes
