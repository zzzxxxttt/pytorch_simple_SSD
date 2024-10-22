import torch
from torch.autograd import Function


class Detect(Function):
  """At test time, Detect is the final layer of SSD.  Decode location preds,
  apply non-maximum suppression to location predictions based on conf
  scores and threshold to a top_k number of output predictions for both
  confidence score and locations.
  """

  def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
    self.num_classes = num_classes
    self.background_label = bkg_label
    self.top_k = top_k
    # Parameters used in nms.
    self.nms_thresh = nms_thresh
    if nms_thresh <= 0:
      raise ValueError('nms_threshold must be non negative.')
    self.conf_thresh = conf_thresh

  def forward(self, reg_pred, cls_pred, prior_data):
    """
    Args:
        reg_pred: (tensor) Loc preds from loc layers
            Shape: [batch,num_priors, 4]
        cls_pred: (tensor) Shape: Conf preds from conf layers
            Shape: [batch, num_priors,num_classes]
        prior_data: (tensor) Prior boxes and variances from priorbox layers
            Shape: [num_priors,4]
    """
    batch_size = reg_pred.size(0)  # batch size
    num_anchors = prior_data.size(0)
    output = torch.zeros(batch_size, self.num_classes, self.top_k, 5).cuda()
    conf_preds = cls_pred.transpose(2, 1)

    # Decode predictions into bboxes.
    for i in range(batch_size):
      decoded_boxes = decode(reg_pred[i], prior_data)
      # For each class, perform nms
      # conf_scores = conf_preds[i].clone()

      for cls in range(1, self.num_classes):
        # find the index of anchors with large confidence
        c_mask = conf_preds[i, cls] > self.conf_thresh # [8732]
        # extract the corresponding probabilities
        scores = conf_preds[i, cls][c_mask] # [num_objects]
        if scores.shape[0] == 0:
          continue
        l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes) # [8732, 4]
        # extract the corresponding bboxes
        boxes = decoded_boxes[l_mask].view(-1, 4)
        # idx of highest scoring and non-overlapping boxes per class
        # non-maximum suppression
        ids, count = nms(decoded_boxes[l_mask].view(-1, 4), scores, self.nms_thresh, self.top_k)
        # bboxes and probabilities after NMS
        output[i, cls, :count] = \
          torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)

    return output


def nms(boxes, scores, overlap=0.5, top_k=200):
  """Apply non-maximum suppression at test time to avoid detecting too many
  overlapping bounding boxes for a given object.
  Args:
      boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
      scores: (tensor) The class predscores for the img, Shape:[num_priors].
      overlap: (float) The overlap thresh for suppressing unnecessary boxes.
      top_k: (int) The Maximum number of box preds to consider.
  Return:
      The indices of the kept boxes with respect to num_priors.
  """

  keep = scores.new(scores.size(0)).zero_().long()
  # return 0 if there is no bbox
  if boxes.numel() == 0:
    return keep
  x1 = boxes[:, 0]
  y1 = boxes[:, 1]
  x2 = boxes[:, 2]
  y2 = boxes[:, 3]
  area = torch.mul(x2 - x1, y2 - y1)
  v, idx = scores.sort(0)  # sort in ascending order
  # I = I[v >= 0.01]
  idx = idx[-top_k:]  # indices of the top-k largest vals
  xx1 = boxes.new()
  yy1 = boxes.new()
  xx2 = boxes.new()
  yy2 = boxes.new()
  w = boxes.new()
  h = boxes.new()

  # keep = torch.Tensor()
  count = 0
  while idx.numel() > 0:
    i = idx[-1]  # index of current largest val
    # keep.append(i)
    keep[count] = i
    count += 1
    if idx.size(0) == 1:
      break
    idx = idx[:-1]  # remove kept element from view
    # load bboxes of next highest vals
    # boxes except the current box
    torch.index_select(x1, 0, idx, out=xx1)
    torch.index_select(y1, 0, idx, out=yy1)
    torch.index_select(x2, 0, idx, out=xx2)
    torch.index_select(y2, 0, idx, out=yy2)
    # store element-wise max with next highest score
    xx1 = torch.clamp(xx1, min=x1[i])
    yy1 = torch.clamp(yy1, min=y1[i])
    xx2 = torch.clamp(xx2, max=x2[i])
    yy2 = torch.clamp(yy2, max=y2[i])
    w.resize_as_(xx2)
    h.resize_as_(yy2)
    w = xx2 - xx1
    h = yy2 - yy1
    # check sizes of xx1 and xx2.. after each iteration
    w = torch.clamp(w, min=0.0)
    h = torch.clamp(h, min=0.0)
    inter = w * h
    # IoU = i / (area(a) + area(b) - i)
    rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
    union = (rem_areas - inter) + area[i]
    IoU = inter / union  # store result in iou
    # keep only elements with an IoU <= overlap
    idx = idx[IoU.le(overlap)]
  return keep, count


def decode(loc, priors):
  # loc and prior are in xywh form
  boxes = torch.cat((priors[:, :2] + loc[:, :2] * 0.1 * priors[:, 2:],
                     priors[:, 2:] * torch.exp(loc[:, 2:] * 0.2)), 1)
  # xywh -> xyxy
  boxes[:, :2] -= boxes[:, 2:] / 2
  boxes[:, 2:] += boxes[:, :2]
  return boxes

# if __name__ == '__main__':
#   from nets.anchors import *
#
#   with open('../debug.pickle', 'rb') as handle:
#     debug = pickle.load(handle)
#
#   priors = AnchorBox(v2).forward()
#   detect = Detect(21, 0, 200, 0.01, 0.45)
#   for i in range(len(debug)):
#     out = detect.forward(torch.from_numpy(debug[i][0]),
#                          torch.from_numpy(debug[i][1]),
#                          torch.from_numpy(priors).type(torch.FloatTensor))
