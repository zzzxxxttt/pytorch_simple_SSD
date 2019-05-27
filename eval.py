import argparse
from tqdm import tqdm

import torch.backends.cudnn as cudnn

cudnn.benchmark = True

import tensorflow as tf
from utils.nms_tf import bboxes_nms_batch

from nets.ssd import *

from utils.i_o import *
from utils.eval_mAP import *
from utils.bbox2target_np import *

parser = argparse.ArgumentParser()

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--eval_data_dir', type=str, default='E:\\VOCdevkit_test')
parser.add_argument('--pretrain_dir', type=str, default='./ckpt/SSD_07_12_baseline/checkpoint.t7')

parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--use_gpu', type=str, default='0')
parser.add_argument('--workers', type=int, default=2)

cfg = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.use_gpu


def main():
  eval_set = VOCDetection(cfg.eval_data_dir, image_sets=[('2007', 'test')],
                          transform=imageAugmentation(train=False))
  eval_loader = data.DataLoader(eval_set, cfg.batch_size, num_workers=cfg.workers,
                                shuffle=False, collate_fn=detection_collate, pin_memory=True)
  map_util = mAP(cfg.eval_data_dir)

  model = SSD().cuda()
  # model = torch.nn.DataParallel(model.cuda())
  load_pretrain(model, cfg.pretrain_dir)

  model.eval()
  model.phase = 'eval'

  regs = tf.placeholder(tf.float32, [None, 8732, 4], 'reg')
  clss = tf.placeholder(tf.float32, [None, 8732, NUM_CLASSES], 'cls')

  scores, bboxes = bboxes_nms_batch({i: clss[:, :, i] for i in range(NUM_CLASSES)},
                                    {i: regs for i in range(NUM_CLASSES)},
                                    nms_threshold=0.45,
                                    keep_top_k=200, parallel=10)

  results = {k: [] for k in VOC_CLASSES}
  with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
    with torch.no_grad():
      for inputs, bbox_targets, cls_targets, meta_data in tqdm(eval_loader):
        inputs = inputs.cuda()

        reg_pred, cls_pred = model(inputs)
        scores_nms, bboxes_nms = sess.run([scores, bboxes],
                                          feed_dict={regs: reg_pred.cpu().numpy(),
                                                     clss: cls_pred.cpu().numpy()[:, :, 1:]})

        for c in scores_nms.keys():
          for i in range(scores_nms[c].shape[0]):
            img_id, width, height = meta_data[i]['id'], meta_data[i]['w'], meta_data[i]['h']
            for j in range(scores_nms[c].shape[1]):
              if scores_nms[c][i, j] > 0.01:
                box = (bboxes_nms[c][i, j] * [width, height, width, height]).astype(np.int32)
                results[VOC_CLASSES[c]].append(
                  '%s %.6f %d %d %d %d' % (img_id[-1], scores_nms[c][i, j], box[0], box[1], box[2], box[3]))
              else:
                break

  filename = os.path.join(cfg.eval_data_dir, 'results', 'VOC2007', 'Main', 'comp3_det_test_%s.txt')
  for key in results.keys():
    with open(filename % key, 'w') as file:
      for line in results[key]:
        print(line, end='\n', file=file)

  map_util.do_python_eval()


if __name__ == '__main__':
  main()
