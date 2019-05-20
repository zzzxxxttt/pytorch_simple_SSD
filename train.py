import os
import sys
import cv2
import time
import argparse
from collections import OrderedDict

from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from dataset.VOCdetection import *
from nets.ssd import *
from utils.io import *
from utils.losses import *
from eval_mAP import *

parser = argparse.ArgumentParser()

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='E:\\VOCdevkit')
parser.add_argument('--eval_data_dir', type=str, default='E:\\VOCdevkit_test')
parser.add_argument('--log_name', type=str, default='SSD_07_12_glasso_0.01')

parser.add_argument('--pretrain_dir', type=str, default='./model/vgg16')

parser.add_argument('--neg_pos_ratio', type=float, default=3.0)
parser.add_argument('--alpha', type=float, default=1.0)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--wd', type=float, default=5e-4)

parser.add_argument('--train_batch_size', type=int, default=32)
parser.add_argument('--eval_batch_size', type=int, default=1)
parser.add_argument('--max_epoch', type=int, default=300)
parser.add_argument('--epochs_per_eval', type=int, default=10)

parser.add_argument('--use_gpu', type=str, default='0')
parser.add_argument('--workers', type=int, default=5)
parser.add_argument('--log_interval', type=int, default=10)

cfg = parser.parse_args()

cfg.model_dir = os.path.join(cfg.root_dir, 'model', cfg.log_name)
cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)

os.makedirs(cfg.model_dir, exist_ok=True)
os.makedirs(cfg.log_dir, exist_ok=True)

if not cfg.cluster:
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
  os.environ["CUDA_VISIBLE_DEVICES"] = cfg.use_gpu


def main():
  train_set = VOCDetection(cfg.data_dir,
                           image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                           transform=aug_generator(base_network=cfg.base_network, train=True))
  train_loader = data.DataLoader(train_set, cfg.train_batch_size,
                                 num_workers=cfg.workers, shuffle=True,
                                 collate_fn=detection_collate, pin_memory=True)

  eval_set = VOCDetection(cfg.eval_data_dir,
                          image_sets=[('2007', 'test')],
                          transform=aug_generator(base_network=cfg.base_network, train=False))
  eval_loader = data.DataLoader(eval_set, cfg.eval_batch_size,
                                num_workers=cfg.workers, shuffle=False,
                                collate_fn=detection_collate, pin_memory=True)

  model = SSD(base=cfg.base_network).cuda()
  model.weight_init()
  # model = torch.nn.DataParallel(model.cuda())
  # load pretrained model
  load_pretrain(model, cfg.base_dir)

  optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.wd)
  scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 200, 230], gamma=0.1)
  criterion = MultiBoxLoss(num_classes=NUM_CLASSES, neg_pos_ratio=cfg.neg_pos_ratio).cuda()

  summary_writer = SummaryWriter(cfg.log_dir)

  def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    model.phase = 'train'

    start_time = time.perf_counter()
    for batch_idx, (images, reg_targets, cls_targets, _) in enumerate(train_loader):
      images, reg_targets, cls_targets = images.cuda(), reg_targets.cuda(), cls_targets.cuda()

      reg_pred, cls_pred = model(images)

      optimizer.zero_grad()  # 我有一句...不知当不当讲...
      reg_loss, cls_loss = criterion(reg_pred=reg_pred, cls_pred=cls_pred,
                                     reg_targets=reg_targets, cls_targets=cls_targets)

      losses = OrderedDict([('loc_loss', reg_loss.item()), ('cls_loss', cls_loss.item())])

      loss = cfg.alpha * reg_loss + cls_loss
      loss.backward()
      optimizer.step()

      if batch_idx % cfg.log_interval == 0:
        duration = time.perf_counter() - start_time
        print_train_log(epoch, batch_idx, losses,
                        cfg.log_interval * cfg.train_batch_size / duration, duration / cfg.log_interval)
        start_time = time.perf_counter()

        step = epoch * len(train_loader) + batch_idx
        for key in losses.keys():
          summary_writer.add_scalar(key, losses[key], step)
        summary_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], step)

  def eval(epoch):
    model.eval()
    model.phase = 'eval'

    results = {k: [] for k in VOC_CLASSES}
    with tqdm(total=len(eval_loader)) as pbar:
      for batch_idx, (images, reg_targets, cls_targets, meta_data) in enumerate(eval_loader):
        images = images.cuda()
        img_id = meta_data[0]['id']
        width = meta_data[0]['w']
        height = meta_data[0]['h']
        outputs = model(images)

        outputs = outputs[0].cpu().data.numpy()
        for i in range(1, outputs.shape[0]):
          for j in range(outputs.shape[1]):
            if outputs[i, j, 0] > 0.6:
              box = (outputs[i, j, 1:] * [width, height, width, height]).astype(np.int32)
              results[VOC_CLASSES[i - 1]]. \
                append('%s %.6f %d %d %d %d' % (img_id[-1], outputs[i, j, 0], box[0], box[1], box[2], box[3]))
            else:
              break
        pbar.update(cfg.eval_batch_size)

    filename = os.path.join(cfg.eval_data_dir, 'results', 'VOC2007', 'Main', 'comp3_det_test_%s.txt')
    for key in results.keys():
      with open(filename % key, 'w') as file:
        for line in results[key]:
          print(line, end='\n', file=file)

    aps, mean_ap = do_python_eval()
    print_eval_log(aps, VOC_CLASSES, mean_ap, epoch)
    for ap, cls in zip(aps, VOC_CLASSES):
      summary_writer.add_scalar('AP/%s_ap' % cls, ap, epoch)
    summary_writer.add_scalar('mean_AP', mean_ap, epoch)

  for epoch in range(cfg.max_epoch):
    scheduler.step()
    train(epoch)
    if epoch % cfg.epochs_per_eval == 0:
      eval(epoch)
    save_model(model, optimizer, cfg.model_dir)

  summary_writer.close()


if __name__ == '__main__':
  main()
