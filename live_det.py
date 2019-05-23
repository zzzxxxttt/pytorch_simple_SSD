import os
import cv2
import time
import pickle
from tqdm import tqdm
import seaborn as sns

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from dataset.VOCdetection import *
from nets.ssd import *
from nets.anchors import *
from utils.i_o import *
from utils.bbox2target_np import *

use_gpu = True

colors=sns.color_palette("hls", NUM_CLASSES)

def draw(img, box, cls_id, prob):
  cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]),
                color=colors[int(cls_id)], thickness=2)
  cv2.rectangle(img,
                (box[0], box[1] - 15),
                (box[0] + len(VOC_CLASSES[int(cls_id)] + str(np.round(prob, 2))) * 8, box[1]),
                color=colors[int(cls_id)], thickness=-1)
  cv2.putText(img, VOC_CLASSES[int(cls_id)] + str(np.round(prob, 2)),
              (box[0], box[1] - 3),
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))


def draw_fps(img, fps):
  cv2.putText(img, 'fps: ' + str(np.round(fps, 1)),
              (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


def main():
  model = SSD()
  if use_gpu:
    model = nn.DataParallel(model).cuda()
    model.module.phase = 'eval'
  else:
    model.phase = 'eval'
  # load pretrained model
  load_pretrain(model, './model')

  model.eval()

  cap = cv2.VideoCapture(0)
  while True:
    start = time.perf_counter()
    # get a frame
    ret, frame = cap.read()

    height, width = frame.shape[0], frame.shape[1]
    img = cv2.resize(frame, (300, 300)) - [104, 117, 123]
    # images = cv2.resize(frame.astype(np.float32), (300, 300))[:,:,(2,1,0)]/255
    # images=(images - (0.485, 0.456, 0.406))/(0.229, 0.224, 0.225)
    img = torch.from_numpy(img.transpose([2, 0, 1])[None, :, :, :]).float()
    if use_gpu:
      img.cuda()

    outputs = model(img)

    outputs = outputs[0].cpu().data.numpy()
    for i in range(outputs.shape[0]):
      for j in range(outputs.shape[1]):
        if outputs[i, j, 0] > 0.5:
          box = (outputs[i, j, 1:] * [width, height, width, height]).astype(np.int32)
          draw(frame, box, i - 1, outputs[i, j, 0])

    draw_fps(frame, 1 / (time.perf_counter() - start))
    # show a frame
    cv2.imshow("capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  cap.release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
  main()
