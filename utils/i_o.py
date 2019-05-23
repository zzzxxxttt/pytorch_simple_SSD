import os
from collections import OrderedDict

import torch


def load_pretrain(net, pretrain_dir):
  if os.path.isfile(pretrain_dir):
    model_state = net.state_dict()
    pretrained_state = torch.load(pretrain_dir, map_location=lambda storage, loc: storage)
    merged_state = OrderedDict()
    for k in model_state.keys():
      if k in pretrained_state.keys():
        merged_state[k] = pretrained_state[k]
      elif 'module.%s' % k in pretrained_state.keys():
        merged_state[k] = pretrained_state['module.%s' % k]
      else:
        print('%s not in pretrained model!' % k)
        merged_state[k] = model_state[k]
    net.load_state_dict(merged_state)
    print("=> loaded checkpoint '%s'" % pretrain_dir)
  else:
    print("=> no checkpoint found at '%s'" % pretrain_dir)
