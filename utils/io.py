import os
import time
from collections import OrderedDict
from datetime import datetime
# from termcolor import colored
# import colorama

import torch


def load_pretrain(net, pretrain_dir, pretrain_name='checkpoint.t7'):
  pretrain_file = os.path.join(pretrain_dir, pretrain_name)
  if os.path.isfile(pretrain_file):
    model_state = net.state_dict()
    pretrained_state = torch.load(pretrain_file, map_location=lambda storage, loc: storage)
    merged_state = OrderedDict()
    for k in model_state.keys():
      if k in pretrained_state.keys():
        merged_state[k] = pretrained_state[k]
      elif 'module.' + k in pretrained_state.keys():
        merged_state[k] = pretrained_state['module.' + k]
      else:
        print('%s not in pretrained model!' % k)
        merged_state[k] = model_state[k]
    net.load_state_dict(merged_state)
    print("=> loaded checkpoint '%s'" % pretrain_file)
  else:
    print("=> no checkpoint found at '%s'" % pretrain_file)


def save_model(net, optimizer, checkpoint_dir, name='checkpoint'):
  torch.save(net.state_dict(), os.path.join(checkpoint_dir, '%s.t7' % name))
  print('model saved in %s' % checkpoint_dir)


def print_train_log(epoch, step, losses, exp_per_sec, batch_per_sec):
  format_str = '%s epoch: %d step: %d ' % (str(datetime.now())[:-7], epoch, step)
  for key in losses.keys():
    format_str += '%s= %.5f ' % (key, losses[key])
  format_str += ' (%d samples/sec %.2f sec/batch)' % (exp_per_sec, batch_per_sec)
  print(format_str)


def print_eval_log(aps, classes, mean_ap, epoch):
  for cls, ap in zip(classes, aps):
    print('%s : %.3f' % (cls, ap * 100))
  format_str = '%s epoch: %d mAP: %.3f' % (str(datetime.now())[:-7], epoch, mean_ap * 100)
  print(format_str)
