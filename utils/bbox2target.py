# -*- coding: utf-8 -*-
import torch


def xywh_to_xyxy(boxes):
  return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,  # xmin, ymin
                    boxes[:, :2] + boxes[:, 2:] / 2), 1)  # xmax, ymax


def xyxy_to_xywh(boxes):
  return torch.cat((boxes[:, 2:] + boxes[:, :2]) / 2,  # cx, cy
                   boxes[:, 2:] - boxes[:, :2], 1)  # w, h


def IoU(box_a, box_b):
  # box_a and box_b are in xyxy form
  max_xy = torch.min(box_a[:, 2:].unsqueeze(1), box_b[:, 2:].unsqueeze(0))
  min_xy = torch.max(box_a[:, :2].unsqueeze(1), box_b[:, :2].unsqueeze(0))
  inter = torch.clamp((max_xy - min_xy), min=0, max=None)
  inter = inter[:, :, 0] * inter[:, :, 1]

  area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1)  # [A,B]
  area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0)  # [A,B]
  union = area_a + area_b - inter
  return inter / union  # [A,B]


def encode(gt_boxes, priors):
  # gt_boxes and priors are in xyxy form
  # distance between match center and prior's center
  g_cxcy = ((gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2 - priors[:, :2]) / (priors[:, 2:] * 0.1)
  # match wh / prior wh
  g_wh = torch.log((gt_boxes[:, 2:] - gt_boxes[:, :2]) / priors[:, 2:]) / 0.2

  # return target for smooth_l1_loss
  return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors):
  # loc and priors are in xywh form
  boxes = torch.cat((priors[:, :2] + loc[:, :2] * 0.1 * priors[:, 2:],
                     priors[:, 2:] * torch.exp(loc[:, 2:] * 0.2)), 1)
  # xywh -> xyxy
  boxes[:, :2] -= boxes[:, 2:] / 2
  boxes[:, 2:] += boxes[:, :2]
  return boxes


def decode_batch(loc, priors):
  # loc and priors are in xywh form
  # loc shape: [batch_size, num_priors, 4]
  # priors shape: [num_priors, 4]
  boxes = torch.cat((
    priors[:, :2][None, :, :] + loc[:, :, :2] * 0.1 * priors[:, 2:][None, :, :],
    priors[:, 2:][None, :, :] * torch.exp(loc[:, :, 2:] * 0.2)), 2)

  # xywh -> xyxy
  boxes[:, :, :2] -= boxes[:, :, 2:] / 2
  boxes[:, :, 2:] += boxes[:, :, :2]
  return boxes
