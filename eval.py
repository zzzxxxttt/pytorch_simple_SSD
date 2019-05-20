import os
import cv2
import torch
import time
import pickle
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from nets.ssd import *
from dataset.VOCdetection import *
from nets.anchors import *
from utils.bbox2target_np import *
from utils.io import *

from config import cfg

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '2'


def main():
  eval_set = VOCDetection('/data1/zhuxt/VOCdevkit_test', image_sets=[('2007', 'test')],
                          transform=aug_generator(base_network='vgg16', train=False))
  eval_loader = data.DataLoader(eval_set, cfg.eval_batch_size, num_workers=5, shuffle=False,
                                collate_fn=detection_collate, pin_memory=True)

  network = SSD
  model = network(base='vgg16')
  # model.weight_init()
  # model = torch.nn.DataParallel(model.cuda())
  cudnn.benchmark = True
  model.cuda()
  # load pretrained model
  load_pretrain(model, './model/vgg16_0.01_S')

  model.eval()
  model.phase = 'eval'

  results = {k: [] for k in VOC_CLASSES}

  def eval():
    debug = []
    with tqdm(total=len(eval_set)) as pbar:
      for batch_idx, (images, bbox_targets, cls_targets, meta_data) in enumerate(eval_loader):
        images = Variable(images.cuda())

        img_id = meta_data[0]['id']
        width = meta_data[0]['w']
        height = meta_data[0]['h']
        outputs = model(images)
        # loc_pred, prob_pred = model(images)
        # debug.append((loc_pred.data.cpu().numpy(), prob_pred.data.cpu().numpy()))
        # loc_pred = loc_pred.cpu().numpy()
        # prob_pred = prob_pred.cpu().numpy()
        # # start = time.perf_counter()
        # outputs = detect_np(loc_pred, prob_pred, 200, 0.01, 0.45)
        # for i in range(len(outputs)):
        #   if outputs[i] is not None:
        #     for j in range(outputs[i].shape[0]):
        #       if outputs[i][j, 4] > 0.6:
        #         box = (outputs[i][j, 0:4] * [width, height, width, height]).astype(np.int32)
        #         results[VOC_CLASSES[int(outputs[i][j, 5])]]. \
        #           append('%s %.6f %d %d %d %d' % (img_id[-1], outputs[i][j, 4], box[0], box[1], box[2], box[3]))

        # print(time.perf_counter() - start)
        img_id = meta_data[0]['id']
        width = meta_data[0]['w']
        height = meta_data[0]['h']
        outputs = outputs[0].cpu().data.numpy()
        for i in range(1, outputs.shape[0]):
          for j in range(outputs.shape[1]):
            if outputs[i, j, 0] > 0.01:
              box = (outputs[i, j, 1:] * [width, height, width, height]).astype(np.int32)
              results[VOC_CLASSES[i - 1]]. \
                append('%s %.6f %d %d %d %d' % (img_id[-1], outputs[i, j, 0], box[0], box[1], box[2], box[3]))
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
        #
        pbar.update(cfg.eval_batch_size)
        # if batch_idx==100:
        #   break

    for key in results.keys():
      with open('comp3_det_test_%s.txt' % key, 'w') as file:
        for line in results[key]:
          print(line, end='\n', file=file)
          # with open('debug.pickle', 'wb') as handle:
          #   pickle.dump(debug, handle, protocol=pickle.HIGHEST_PROTOCOL)

  eval()


if __name__ == '__main__':
  main()
