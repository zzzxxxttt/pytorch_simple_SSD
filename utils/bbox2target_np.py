import numpy as np


def xywh_to_xyxy_np(boxes):
  return np.concatenate((boxes[:, :2] - boxes[:, 2:] / 2,  # x - w / 2, y - w / 2
                         boxes[:, :2] + boxes[:, 2:] / 2), 1)  # x + w / 2, y + w / 2


def xyxy_to_xywh_np(boxes):
  return np.concatenate(((boxes[:, 2:] + boxes[:, :2]) / 2,  # (min + max) / 2
                         boxes[:, 2:] - boxes[:, :2]), 1)  # max - min


def IoU_np(box_a, box_b):
  min_xy_a, max_xy_a = box_a[:, :2], box_a[:, 2:]
  min_xy_b, max_xy_b = box_b[:, :2], box_b[:, 2:]

  # boardcast [Na, 1, 2] with [1, Nb, 2], result in [Na, Nb, 2]
  # this means compare each box in box_a with each box in box_b
  # thus there are Na x Nb minimum/maximum (x, y)
  max_xy = np.minimum(max_xy_a[:, None, :], max_xy_b[None, :, :])
  min_xy = np.maximum(min_xy_a[:, None, :], min_xy_b[None, :, :])
  # if two boxes do not overlap, the IOU should be 0
  inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)
  # [Na, Nb] x [Na, Nb] result in [Na, Nb]
  # which means Na x Nb overlaps for each pair of boxes
  inter = inter[:, :, 0] * inter[:, :, 1]

  # box = [x_min, y_min, x_max, y_max]
  area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]))[:, None]  # [Na, 1]
  area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1]))[None, :]  # [1, Nb]
  union = area_a + area_b - inter  # [Na, Nb]
  return inter / union  # [Na, Nb]


def match_np(pos_threshold, gt_boxes, anchor_boxes, cls_labels):
  # jaccard index
  overlaps = IoU_np(gt_boxes, xywh_to_xyxy_np(anchor_boxes))

  # [1, num_objects] best anchor for each ground truth
  # best_anchor_overlap = np.amax(overlaps, 1) # overlap
  best_anchor_idx = np.argmax(overlaps, 1)  # ind

  # [1, 8732] best ground truth for each anchor
  best_gt_overlap = np.amax(overlaps, 0)  # overlap
  best_gt_idx = np.argmax(overlaps, 0)  # ind

  # set the iou of anchors that have matched boxes to 2
  # note that other iou are all below 1
  best_gt_overlap[best_anchor_idx] = 2

  # ensure every gt matches with its anchor of max overlap
  # best_anchor_idx coresponding to gt box 1, 2, ..., N
  best_gt_idx[best_anchor_idx] = np.arange(best_anchor_idx.shape[0])

  # find the gtboxes of each anchor
  gt_matches = gt_boxes[best_gt_idx]  # Shape: [8732, 4]

  # find the class label of each anchor
  cls_target = cls_labels[best_gt_idx] + 1  # Shape: [8732]

  # set label 0 to anchors that have a low iou
  cls_target[best_gt_overlap < pos_threshold] = 0  # label as background

  # distance between matched gt box center and anchor's center
  g_cxcy = (gt_matches[:, :2] + gt_matches[:, 2:]) / 2 - anchor_boxes[:, :2]
  # distance / anchor_box_size, and encode variance
  g_cxcy /= (anchor_boxes[:, 2:] * 0.1)
  # matched gt_box_size / anchor_box_size
  g_wh = (gt_matches[:, 2:] - gt_matches[:, :2]) / anchor_boxes[:, 2:]
  # apply log, and encode variance
  g_wh = np.log(g_wh) / 0.2

  # return target for smooth_l1_loss
  reg_target = np.concatenate([g_cxcy, g_wh], 1)  # [8732, 4]

  return reg_target, cls_target


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode_np(loc, anchors):
  boxes = np.concatenate((anchors[:, :2] + loc[:, :2] * 0.1 * anchors[:, 2:],
                          anchors[:, 2:] * np.exp(loc[:, 2:] * 0.2)), 1)
  boxes[:, :2] -= boxes[:, 2:] / 2
  boxes[:, 2:] += boxes[:, :2]
  return boxes


def decode_batch_np(loc, anchors):
  boxes = np.concatenate((anchors[:, :, :2] + loc[:, :, :2] * 0.1 * anchors[:, :, 2:],
                          anchors[:, :, 2:] * np.exp(loc[:, :, 2:] * 0.2)), 1)
  boxes[:, :, :2] -= boxes[:, :, 2:] / 2
  boxes[:, :, 2:] += boxes[:, :, :2]
  return boxes


def detect_np(boxes_pred, probs_pred, top_k, cls_thresh, nms_thresh):
  '''
  :param boxes_pred: [batch_size, 8732, 4]
  :param probs_pred: [batch_size, 8732, 21]
  :param top_k:
  :param cls_thresh:
  :param nms_thresh:
  :return:
  '''
  batch_size = probs_pred.shape[0]
  num_classes = probs_pred.shape[2]

  max_probs = np.amax(probs_pred, axis=-1)
  argmax_probs = np.argmax(probs_pred, axis=-1)

  batch_result = []
  for i in range(batch_size):
    result = []

    prob_filter = np.logical_and(max_probs[i] > cls_thresh, argmax_probs[i] > 0)
    if np.sum(prob_filter) < 1:
      batch_result.append(None)
      continue
    boxes = boxes_pred[i, prob_filter, :]
    clses = argmax_probs[i, prob_filter]
    probs = max_probs[i, prob_filter]

    for c in range(1, num_classes):
      # 找出等于当前类的anchor

      cls_filter = clses == c
      if np.sum(cls_filter) < 1:
        continue

      # idx of highest scoring and non-overlapping boxes per class
      # 非极大值抑制
      nms_boxes, nms_probs, count = nms_np(boxes[cls_filter, :], probs[cls_filter], nms_thresh, top_k)
      # 拿出抑制之后的bbox和概率
      # output[i, cls, :count] = torch.cat((scores[nms_idx[:count]].unsqueeze(1), boxes[nms_idx[:count]]), 1)
      result.append(np.hstack([np.clip(nms_boxes, 0.0, 1.0), nms_probs[:, None], np.ones([count, 1]) * (c - 1)]))
    batch_result.append(np.vstack(result))

  return batch_result


def nms_np(boxes, scores, overlap=0.5, top_k=200):
  # 如果没有box就返回全0
  if boxes.shape[0] <= 1:
    return boxes, scores, boxes.shape[0]

  area = np.prod(boxes[:, 2:] - boxes[:, :2], axis=-1)
  argsort_prob = np.argsort(scores)[::-1]

  keep = []
  count = 0
  while argsort_prob.shape[0] > 0:
    i = argsort_prob[-1]  # index of current largest val
    # keep.append(i)
    keep.append(i)
    count += 1
    if argsort_prob.shape[0] == 1:
      break
    argsort_prob = argsort_prob[:-1]  # remove kept element from view
    # load bboxes of next highest vals
    # 除了当前box之外其他的box
    max_xy = np.minimum(boxes[i, 2:][None, :], boxes[argsort_prob, 2:])
    min_xy = np.maximum(boxes[i, :2][None, :], boxes[argsort_prob, :2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)
    inter = inter[:, 0] * inter[:, 1]

    # IoU = i / (area(a) + area(b) - i)
    rem_areas = area[argsort_prob]  # load remaining areas)
    union = (rem_areas - inter) + area[i]
    IoU = inter / union  # store result in iou
    # keep only elements with an IoU <= overlap
    argsort_prob = argsort_prob[IoU < overlap]
  return boxes[keep, :], scores[keep], count

# if __name__ == '__main__':
#   from nets.anchors import *
#   import pickle
#
#   with open('../debug.pickle', 'rb') as handle:
#     debug = pickle.load(handle)
#
#   for i in range(len(debug)):
#     out = detect_np(debug[i][0], debug[i][1], 200, 0.01, 0.45)
