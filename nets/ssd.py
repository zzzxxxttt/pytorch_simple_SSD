import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from nets.anchors import *
from nets.vgg_base import *
from nets.mobilenet_base import *
from utils.bbox2target import *
from utils import old_detect

base = {'300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
                512, 512, 512],
        '512': [], }

extras = {'300': [(256, 1, 1), (512, 3, 2),
                  (128, 1, 1), (256, 3, 2),
                  (128, 1, 1), (256, 3, 1), (128, 1, 1), (256, 3, 1)],
          '512': []}

heads = {'300': [(512, 4), (1024, 6), (512, 6),
                 (256, 6), (256, 4), (256, 4)],  # number of boxes per feature map location
         '512': []}


class SSD(nn.Module):
  def __init__(self, base='vgg16', extra_config=extras['300'], head_config=heads['300'], num_classes=20):
    super(SSD, self).__init__()
    self.num_clsses = num_classes + 1
    self.phase = 'train'

    if base == 'vgg16':
      self.base_network = vgg16base()
    elif base == 'mobilenet':
      self.base_network = MobileNet()
    else:
      assert False, 'base unknown !'
    self.extra_features = nn.ModuleList()
    self.reg_layers = nn.ModuleList()  # 回归网络
    self.cls_layers = nn.ModuleList()  # 分类网络
    self.register_buffer('priors', torch.from_numpy(PriorBox(v2)()).float())

    # 使用额外增加的feature map
    in_channels = self.base_network.out_channels
    for v in extra_config:
      if v[2] == 2:
        self.extra_features.append(nn.Conv2d(in_channels, v[0], kernel_size=v[1], stride=2, padding=1))
      else:
        self.extra_features.append(nn.Conv2d(in_channels, v[0], kernel_size=v[1]))
      in_channels = v[0]

    for v in head_config:
      self.reg_layers.append(nn.Conv2d(v[0], v[1] * 4, kernel_size=3, padding=1))
      self.cls_layers.append(nn.Conv2d(v[0], v[1] * (num_classes + 1), kernel_size=3, padding=1))

    # todo remove this after debug
    self.detect = old_detect.Detect(num_classes + 1, 0, 200, 0.5, 0.45)

  def forward(self, x):
    features = []
    out = self.base_network(x)
    features.append(out[0])
    features.append(out[1])

    out = out[1]
    for i, layer in enumerate(self.extra_features):
      out = F.relu(layer(out), inplace=True)
      if i % 2 == 1:
        features.append(out)

    reg_outputs = []
    cls_outputs = []
    for feature, reg_layer, conf_layer in zip(features, self.reg_layers, self.cls_layers):
      out = reg_layer(feature).permute(0, 2, 3, 1).contiguous()
      reg_outputs.append(out.view(out.size(0), -1))
      out = conf_layer(feature).permute(0, 2, 3, 1).contiguous()
      cls_outputs.append(out.view(out.size(0), -1))

    reg_outputs = torch.cat(reg_outputs, dim=1)
    reg_outputs = reg_outputs.view(reg_outputs.size(0), -1, 4)
    cls_outputs = torch.cat(cls_outputs, dim=1)
    cls_outputs = cls_outputs.view(cls_outputs.size(0), -1, self.num_clsses)

    if self.phase == 'eval':
      # probs = F.softmax(cls_outputs, dim=-1)
      # decoded_boxes = decode_batch(loc_outputs.data, self.priors)
      # return decoded_boxes, probs.data
      # todo remove this after debug
      output = self.detect(reg_outputs, F.softmax(cls_outputs, dim=-1), Variable(self.priors))
      return output
    else:
      return reg_outputs, cls_outputs

  def weight_init(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        init.xavier_uniform(m.weight.data)
        if m.bias is not None:
          m.bias.data.zero_()


if __name__ == '__main__':
  from torch.autograd import Variable

  features = []


  def hook(self, input, output):
    print(output.data.cpu().numpy().shape)
    features.append(output.data.cpu().numpy())


  net = SSD(extra_config=extras['300'], head_config=heads['300'], num_classes=20)
  for m in net.modules():
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
      m.register_forward_hook(hook)

  y = net(Variable(torch.randn(1, 3, 300, 300)))
  pass
  # print(y.size())
