import os
import xml.etree.ElementTree as ET

import torch.utils.data as data

from dataset.augmentation import *
from nets.anchors import *
from utils.bbox2target_np import *

VOC_CLASSES = (  # always index 0
  'aeroplane', 'bicycle', 'bird', 'boat',
  'bottle', 'bus', 'car', 'cat', 'chair',
  'cow', 'diningtable', 'dog', 'horse',
  'motorbike', 'person', 'pottedplant',
  'sheep', 'sofa', 'train', 'tvmonitor')

NUM_CLASSES = 20


class readXML(object):
  def __init__(self, keep_difficult):
    self.class_to_ind = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
    self.keep_difficult = keep_difficult

  def __call__(self, target, width, height):
    res = []
    # names = []
    for obj in target.iter('object'):
      difficult = int(obj.find('difficult').text) == 1
      if not self.keep_difficult and difficult:
        continue
      obj_name = obj.find('name').text.lower().strip()
      obj_bbox = obj.find('bndbox')

      bbox = [(int(obj_bbox.find('xmin').text) - 1) / width,
              (int(obj_bbox.find('ymin').text) - 1) / height,
              (int(obj_bbox.find('xmax').text) - 1) / width,
              (int(obj_bbox.find('ymax').text) - 1) / height,
              self.class_to_ind[obj_name]]

      res.append(bbox)  # [xmin, ymin, xmax, ymax, label_ind]
      # names.append(obj_name)
      # img_id = target.find('filename').text[:-4]

    return res  # [[xmin, ymin, xmax, ymax, label_ind, obj_name], ... ]


class VOCDetection(data.Dataset):
  def __init__(self, root_dir, image_sets, keep_difficult=True, positive_threshold=0.5,
               transform=None, target_transform=readXML(keep_difficult=True)):
    self.root_dir = root_dir
    if transform is not None:
      self.transform = transform

    if target_transform is not None:
      self.target_transform = target_transform

    self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
    self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
    # todo what's this?
    self.prior_boxes = PriorBox(v2)()
    self.positive_threshold = positive_threshold
    self.ids = list()
    self.keep_difficult = keep_difficult

    for (year, name) in image_sets:
      rootpath = os.path.join(self.root_dir, 'VOC' + year)
      for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
        self.ids.append((rootpath, line.strip()))

  def __getitem__(self, index):
    return self.pull_item(index)

  def __len__(self):
    return len(self.ids)

  def pull_item(self, index):
    img_id = self.ids[index]

    raw_target = ET.parse(self._annopath % img_id).getroot()
    img = cv2.imread(self._imgpath % img_id)
    height, width, channels = img.shape

    target = self.target_transform(raw_target, width, height)

    if self.transform is not None:
      target = np.array(target)
      # 对图像做变换需要同时改变target
      img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
      # cv2在读取数据的时候，图像是BGR格式的！如果不注释下面就会把BGR转换成了RGB！和预训练的VGG权重不匹配了！
      # is it？
      # to rgb
      # img = img[:, :, (2, 1, 0)]
      # img = img.transpose(2, 0, 1)
      # target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

    # match priors (default boxes) and ground truth boxes
    loc_target, conf_target = match_np(self.positive_threshold, boxes, self.prior_boxes, labels)

    # 把[H,W,C]变成[C,H,W]
    return torch.from_numpy(img).permute(2, 0, 1), \
           torch.from_numpy(loc_target).float(), \
           torch.from_numpy(conf_target).long(), \
           {'id': img_id, 'h': height, 'w': width}


def detection_collate(batch):
  imgs = []
  bbox_targets = []
  cls_targets = []
  meta_datas = []
  for sample in batch:
    imgs.append(sample[0])
    bbox_targets.append(sample[1])
    cls_targets.append(sample[2])
    meta_datas.append(sample[3])
  return torch.stack(imgs, 0), \
         torch.stack(bbox_targets, 0), \
         torch.stack(cls_targets, 0), \
         meta_datas


if __name__ == '__main__':
  import time

  dataset = VOCDetection('F:/VOCdevkit',
                         image_sets=[('2007', 'trainval'), ('2012', 'trainval')])
  data_loader = data.DataLoader(dataset, 1, num_workers=0, shuffle=False,
                                collate_fn=detection_collate, pin_memory=True)

  prior_boxes = PriorBox(v2).forward()

  start_time = time.perf_counter()
  for batch_idx, (images, bbox_targets, cls_targets, _) in enumerate(dataset):
    # print(np.amax(bbox_targets.numpy()),np.amin(bbox_targets.numpy()),
    #       np.amax(cls_targets.numpy()), np.amin(cls_targets.numpy()))

    img = images.numpy().transpose([1, 2, 0])
    img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    img = cv2.resize(img, (500, 500))

    pos_ind = cls_targets.numpy() > 0
    bboxes = decode_np(bbox_targets.numpy(), prior_boxes)
    classes = cls_targets.numpy()[pos_ind] - 1
    bboxes = bboxes[pos_ind]
    for box, cls in zip(bboxes, classes):
      cv2.rectangle(img,
                    (int(img.shape[1] * box[0]), int(img.shape[0] * box[1])),
                    (int(img.shape[1] * box[2]), int(img.shape[0] * box[3])),
                    color=COLORS[int(cls)], thickness=2)
      cv2.rectangle(img,
                    (int(img.shape[1] * box[0]), int(img.shape[0] * box[1]) - 15),
                    (int(img.shape[1] * box[0]) + len(VOC_CLASSES[int(cls)]) * 8, int(img.shape[0] * box[1])),
                    color=COLORS[int(cls)], thickness=-1)
      cv2.putText(img, VOC_CLASSES[int(cls)], (int(img.shape[1] * box[0]), int(img.shape[0] * box[1]) - 3),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
      print(box, VOC_CLASSES[int(cls)])

    cv2.imshow('img', img)
    cv2.waitKey()

    # for images, targets, names in dataset:
    #   img = images.numpy().transpose([1, 2, 0])
    #   img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    #   img = cv2.resize(img, (500, 500))
    #
    #   for i, box in enumerate(targets):
    #     cv2.rectangle(img, (int(img.shape[1] * box[0]), int(img.shape[0] * box[1])),
    #                   (int(img.shape[1] * box[2]), int(img.shape[0] * box[3])),
    #                   color=COLORS[int(box[-1])], thickness=2)
    #     cv2.putText(img, names[i], (int(img.shape[1] * box[0]), int(img.shape[0] * box[1])),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    #
    #   cv2.imshow('img', img)
    #   cv2.waitKey()
    #   pass
