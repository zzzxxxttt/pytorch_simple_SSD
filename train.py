import os
import time
import argparse
from datetime import datetime

import torch.optim as optim
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

from tensorboardX import SummaryWriter

from dataset.VOCdetection import *
from dataset.augmentation import *

from nets.ssd import *

from utils.i_o import *
from utils.losses import *
# from utils.nms_tf import *
from utils.eval_mAP import *

parser = argparse.ArgumentParser()

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='../VOCdevkit')
parser.add_argument('--eval_data_dir', type=str, default='../VOCdevkit_test')
parser.add_argument('--log_name', type=str, default='SSD_07_12_baseline')
parser.add_argument('--pretrain_dir', type=str, default='./ckpt/vgg16/checkpoint.t7')

parser.add_argument('--neg_pos_ratio', type=float, default=3.0)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument('--wd', type=float, default=5e-4)

parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--eval_batch_size', type=int, default=5)
parser.add_argument('--max_epoch', type=int, default=300)
parser.add_argument('--epochs_per_eval', type=int, default=10)

parser.add_argument('--workers', type=int, default=5)
parser.add_argument('--gpu', type=str, default='0,1')
parser.add_argument('--log_interval', type=int, default=10)

cfg = parser.parse_args()

cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.log_name)
cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)

os.makedirs(cfg.ckpt_dir, exist_ok=True)
os.makedirs(cfg.log_dir, exist_ok=True)

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu


def main():
  train_set = VOCDetection(cfg.data_dir,
                           image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                           transform=imageAugmentation(train=True))
  train_loader = data.DataLoader(train_set, cfg.train_batch_size,
                                 num_workers=cfg.workers, shuffle=True,
                                 collate_fn=detection_collate, pin_memory=True)

  # eval_set = VOCDetection(cfg.eval_data_dir,
  #                         image_sets=[('2007', 'test')],
  #                         transform=imageAugmentation(train=False))
  # eval_loader = data.DataLoader(eval_set, cfg.eval_batch_size,
  #                               num_workers=cfg.workers, shuffle=False,
  #                               collate_fn=detection_collate, pin_memory=True)

  # map_util = mAP(cfg.eval_data_dir)

  model = SSD()
  load_pretrain(model, cfg.pretrain_dir)
  model = nn.DataParallel(model.cuda())

  optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.wd)
  scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 200, 230], gamma=0.1)
  criterion = MultiBoxLoss(num_classes=NUM_CLASSES, neg_pos_ratio=cfg.neg_pos_ratio).cuda()

  summary_writer = SummaryWriter(cfg.log_dir)

  # sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
  #                                         device_count={'CPU': cfg.workers}))

  def train(epoch):
    print('\n%s Epoch: %d' % (str(datetime.now())[:-7], epoch))
    model.train()
    model.module.phase = 'train'

    tic = time.perf_counter()
    for idx, (inputs, reg_targets, cls_targets, _) in enumerate(train_loader):
      inputs = inputs.cuda()
      reg_targets = reg_targets.cuda(non_blocking=True)
      cls_targets = cls_targets.cuda(non_blocking=True)

      reg_pred, cls_pred = model(inputs)

      optimizer.zero_grad()  # it took me one week to find out that I forgot this...
      reg_loss, cls_loss = criterion(reg_pred=reg_pred, cls_pred=cls_pred,
                                     reg_targets=reg_targets, cls_targets=cls_targets)

      loss = cfg.alpha * reg_loss + cls_loss
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), 5.0)
      optimizer.step()

      if idx % cfg.log_interval == 0:
        toc = time.perf_counter() - tic
        print('epoch: %d step: %d cls_loss= %.5f reg_loss= %.5f (%d imgs/sec %.2f sec/batch)' % \
              (epoch, idx, cls_loss.item(), reg_loss.item(),
               cfg.log_interval * cfg.train_batch_size / toc, toc / cfg.log_interval))
        tic = time.perf_counter()

        step = epoch * len(train_loader) + idx
        summary_writer.add_scalar('reg_loss', reg_loss.item(), step)
        summary_writer.add_scalar('cls_loss', cls_loss.item(), step)
        summary_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], step)

  # def eval(epoch):
  #   if epoch % cfg.epochs_per_eval != 0:
  #     return
  #
  #   model.eval()
  #   model.module.phase = 'eval'
  #
  #   regs = tf.placeholder(tf.float32, [None, 8732, 4], 'reg')
  #   clss = tf.placeholder(tf.float32, [None, 8732, NUM_CLASSES], 'cls')
  #
  #   scores, bboxes = bboxes_nms_batch({i: clss[:, :, i] for i in range(NUM_CLASSES)},
  #                                     {i: regs for i in range(NUM_CLASSES)},
  #                                     nms_threshold=0.45, keep_top_k=200, parallel=5)
  #
  #   results = {k: [] for k in VOC_CLASSES}
  #   with torch.no_grad():
  #     for inputs, _, _, meta_data in eval_loader:
  #       reg_pred, cls_pred = model(inputs.cuda())
  #       reg_pred = reg_pred.cpu().numpy()
  #       cls_pred = cls_pred.cpu().numpy()[:, :, 1:]
  #       scores_nms, bboxes_nms = sess.run([scores, bboxes], feed_dict={regs: reg_pred, clss: cls_pred})
  #
  #       for c in scores_nms.keys():
  #         for i in range(scores_nms[c].shape[0]):
  #           img_id, width, height = meta_data[i]['id'], meta_data[i]['w'], meta_data[i]['h']
  #           for j in range(scores_nms[c].shape[1]):
  #             if scores_nms[c][i, j] > 0.01:
  #               box = (bboxes_nms[c][i, j] * [width, height, width, height]).astype(np.int32)
  #               results[VOC_CLASSES[c]].append(
  #                 '%s %.6f %d %d %d %d' % (img_id[-1], scores_nms[c][i, j], box[0], box[1], box[2], box[3]))
  #             else:
  #               break
  #
  #   result_dir = os.path.join(cfg.ckpt_dir, 'results', 'VOC2007', 'Main')
  #   os.makedirs(result_dir, exist_ok=True)
  #   filename = os.path.join(result_dir, 'comp3_det_test_%s.txt')
  #   for key in results.keys():
  #     with open(filename % key, 'w') as file:
  #       for line in results[key]:
  #         print(line, end='\n', file=file)
  #
  #   aps, mean_ap = map_util.do_python_eval(results_root=cfg.ckpt_dir)
  #
  #   for ap, cls in zip(aps, VOC_CLASSES):
  #     summary_writer.add_scalar('AP/%s_ap' % cls, ap, epoch)
  #   summary_writer.add_scalar('mean_AP', mean_ap, epoch)

  for epoch in range(cfg.max_epoch):
    scheduler.step()
    train(epoch)
    # eval(epoch)
    torch.save(model.state_dict(), os.path.join(cfg.ckpt_dir, 'checkpoint.t7'))
    print('model saved in %s' % os.path.join(cfg.ckpt_dir, 'checkpoint.t7'))

  summary_writer.close()
  # sess.close()


if __name__ == '__main__':
  main()
