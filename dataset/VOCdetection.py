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
    # generate all anchor boxes
    # shape: [N, 4]
    self.prior_boxes = AnchorBox(v2)()
    self.positive_threshold = positive_threshold
    self.img_names = list()
    self.keep_difficult = keep_difficult

    for (year, name) in image_sets:
      rootpath = os.path.join(self.root_dir, 'VOC' + year)
      for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
        self.img_names.append((rootpath, line.strip()))
    return

  def __getitem__(self, index):
    img_root, img_name = self.img_names[index]

    raw_target = ET.parse(self._annopath % (img_root, img_name)).getroot()
    img = cv2.imread(self._imgpath % (img_root, img_name))
    height, width, channels = img.shape

    target = self.target_transform(raw_target, width, height)

    if self.transform is not None:
      target = np.array(target)
      img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])

      # image read by cv2.imread is BGR
      # to rgb
      # img = img[:, :, (2, 1, 0)]
      # target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
    else:
      boxes, labels = target[:, :4], target[:, 4]

    # match priors (default boxes) and ground truth boxes
    reg_target, cls_target = match_np(self.positive_threshold, boxes, self.prior_boxes, labels)

    # [H,W,C] -> [C,H,W]
    img = torch.from_numpy(img).permute(2, 0, 1)
    reg_target = torch.from_numpy(reg_target).float()
    cls_target = torch.from_numpy(cls_target).long()
    meta_data = {'id': (img_root, img_name), 'h': height, 'w': width}
    return img, reg_target, cls_target, meta_data

  def __len__(self):
    return len(self.img_names)


def detection_collate(batch):
  imgs, reg_targets, cls_targets, meta_datas = zip(*batch)
  imgs = torch.stack(imgs, 0)
  reg_targets = torch.stack(reg_targets, 0)
  cls_targets = torch.stack(cls_targets, 0)
  return imgs, reg_targets, cls_targets, meta_datas


if __name__ == '__main__':
  import time
  import seaborn as sns

  colors = sns.color_palette("hls", NUM_CLASSES)

  dataset = VOCDetection('E:\\VOCdevkit',
                         image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                         transform=imageAugmentation(train=True,
                                                     to_01=False,
                                                     to_rgb=False,
                                                     mean=[104, 117, 123], std=[1, 1, 1]))
  data_loader = data.DataLoader(dataset, 2, num_workers=0, shuffle=False,
                                collate_fn=detection_collate, pin_memory=True)

  prior_boxes = AnchorBox(v2)()

  start_time = time.perf_counter()
  for batch_idx, (images, bbox_targets, cls_targets, _) in enumerate(dataset):

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
                    color=colors[int(cls)], thickness=2)
      cv2.rectangle(img,
                    (int(img.shape[1] * box[0]), int(img.shape[0] * box[1]) - 15),
                    (int(img.shape[1] * box[0]) + len(VOC_CLASSES[int(cls)]) * 8, int(img.shape[0] * box[1])),
                    color=colors[int(cls)], thickness=-1)
      cv2.putText(img, VOC_CLASSES[int(cls)], (int(img.shape[1] * box[0]), int(img.shape[0] * box[1]) - 3),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
      print(box, VOC_CLASSES[int(cls)])

    cv2.imshow('img', img)
    cv2.waitKey()
