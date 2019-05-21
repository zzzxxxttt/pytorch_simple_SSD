import os
import cv2
import time
import pickle
import argparse
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

from nets.ssd import *
from dataset.VOCdetection import *
from nets.anchors import *
from utils.bbox2target_np import *
from utils.io import *

parser = argparse.ArgumentParser()

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--eval_data_dir', type=str, default='../VOCdevkit_test')
parser.add_argument('--pretrain_dir', type=str, default='./ckpt/vgg16/checkpoint.t7')

parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--use_gpu', type=str, default='0')
parser.add_argument('--workers', type=int, default=5)

cfg = parser.parse_args()

cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.log_name)
cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)

os.makedirs(cfg.ckpt_dir, exist_ok=True)
os.makedirs(cfg.log_dir, exist_ok=True)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.use_gpu


def main():
  eval_set = VOCDetection('/data1/zhuxt/VOCdevkit_test', image_sets=[('2007', 'test')],
                          transform=imageAugmentation(train=False))
  eval_loader = data.DataLoader(eval_set, cfg.batch_size, num_workers=cfg.workers,
                                shuffle=False, collate_fn=detection_collate, pin_memory=True)

  model = SSD().cuda()
  # model = torch.nn.DataParallel(model.cuda())
  load_pretrain(model, cfg.pretrain_dir)

  model.eval()
  model.phase = 'eval'

  results = {k: [] for k in VOC_CLASSES}

  with torch.no_grad():
    for inputs, bbox_targets, cls_targets, meta_data in tqdm(eval_loader):
      inputs = inputs.cuda()

      img_id = meta_data[0]['id']
      width = meta_data[0]['w']
      height = meta_data[0]['h']
      outputs = model(inputs)

      # print(time.perf_counter() - start)
      img_id = meta_data[0]['id']
      width = meta_data[0]['w']
      height = meta_data[0]['h']
      outputs = outputs[0].cpu().data.numpy()
      for i in range(1, outputs.shape[0]):
        for j in range(outputs.shape[1]):
          if outputs[i, j, 0] > 0.01:
            box = (outputs[i, j, 1:] * [width, height, width, height]).astype(np.int32)
            results[VOC_CLASSES[i - 1]].append(
              '%s %.6f %d %d %d %d' % (img_id[-1], outputs[i, j, 0], box[0], box[1], box[2], box[3]))
          else:
            break

      # img = images[0].cpu().data.numpy().transpose([1, 2, 0])
      # img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
      # img = cv2.resize(img, (500, 500))
      #
      # if outputs[0] is not None:
      #   for i in range(outputs[0].shape[0]):
      #     box = outputs[0][i, :4]
      #     prob = outputs[0][i, 4]
      #     cls = outputs[0][i, 5]
      #
      #     cv2.rectangle(img,
      #                   (int(img.shape[1] * box[0]), int(img.shape[0] * box[1])),
      #                   (int(img.shape[1] * box[2]), int(img.shape[0] * box[3])),
      #                   color=COLORS[int(cls)], thickness=2)
      #     cv2.rectangle(img,
      #                   (int(img.shape[1] * box[0]), int(img.shape[0] * box[1]) - 15),
      #                   (int(img.shape[1] * box[0]) + len(VOC_CLASSES[int(cls)]) * 8, int(img.shape[0] * box[1])),
      #                   color=COLORS[int(cls)], thickness=-1)
      #     cv2.putText(img, VOC_CLASSES[int(cls)],
      #                 (int(img.shape[1] * box[0]), int(img.shape[0] * box[1]) - 3),
      #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
      #
      # cv2.imshow('img', img)
      # cv2.waitKey()

  for key in results.keys():
    with open('comp3_det_test_%s.txt' % key, 'w') as file:
      for line in results[key]:
        print(line, end='\n', file=file)


if __name__ == '__main__':
  main()
