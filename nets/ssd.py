import torch
import torch.nn.functional as F
import torch.nn.init as init

from nets.anchors import *
from nets.vgg_base import *
from utils.bbox2target import *
from utils import old_detect

base = {'300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
                512, 512, 512],
        '512': [], }

# (num_channel, kernel_size, stride)
extras = {'300': [(256, 1, 1), (512, 3, 2),
                  (128, 1, 1), (256, 3, 2),
                  (128, 1, 1), (256, 3, 1),
                  (128, 1, 1), (256, 3, 1)],
          '512': []}

# (num_channel, number of boxes)
heads = {'300': [(512, 4), (1024, 6), (512, 6),
                 (256, 6), (256, 4), (256, 4)],
         '512': []}


class SSD(nn.Module):
  def __init__(self, extra_config=extras['300'], head_config=heads['300'], num_classes=20):
    super(SSD, self).__init__()
    self.num_clsses = num_classes + 1
    self.phase = 'train'

    self.base_network = vgg16base()

    self.extra_features = nn.ModuleList()
    self.reg_layers = nn.ModuleList()  # regression branch
    self.cls_layers = nn.ModuleList()  # classification branch
    self.register_buffer('anchors', torch.from_numpy(AnchorBox(v2)()).float())

    # extra feature maps
    C_in = self.base_network.out_channels
    for C, ksize, stride in extra_config:
      self.extra_features.append(nn.Conv2d(C_in, C, ksize, stride=stride, padding=stride - 1))
      C_in = C

    for C, num_box in head_config:
      self.reg_layers.append(nn.Conv2d(C, num_box * 4, 3, padding=1))
      self.cls_layers.append(nn.Conv2d(C, num_box * (num_classes + 1), 3, padding=1))

    # todo replace this after debug
    self.detect = old_detect.Detect(num_classes + 1,
                                    bkg_label=0, top_k=200,
                                    conf_thresh=0.01, nms_thresh=0.45)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
          m.bias.data.zero_()

  def forward(self, x):
    features = []
    out = self.base_network(x)
    features.append(out[0])  # output of conv4_3
    features.append(out[1])  # output of fc7

    out = out[1]
    for i, layer in enumerate(self.extra_features):
      out = F.relu(layer(out), inplace=True)
      if i % 2 == 1:
        features.append(out)

    reg_outputs = []
    cls_outputs = []
    for feature, reg_layer, conf_layer in zip(features, self.reg_layers, self.cls_layers):
      out = reg_layer(feature).permute(0, 2, 3, 1).contiguous()  # [B, H, W, num_box*4]
      reg_outputs.append(out.view(out.size(0), -1))
      out = conf_layer(feature).permute(0, 2, 3, 1).contiguous()  # [B, H, W, num_box*21]
      cls_outputs.append(out.view(out.size(0), -1))

    reg_outputs = torch.cat(reg_outputs, dim=1)  # [B, 8732*4]
    reg_outputs = reg_outputs.view(reg_outputs.size(0), -1, 4)  # [B, 8732, 4]
    cls_outputs = torch.cat(cls_outputs, dim=1)  # [B, 8732*21]
    cls_outputs = cls_outputs.view(cls_outputs.size(0), -1, self.num_clsses)  # [B, 8732, 21]

    if self.phase == 'eval':
      # todo replace this after debug
      # loc and priors are in xywh form
      decoded_boxes = \
        torch.cat((self.anchors[None, :, :2] +
                   reg_outputs[:, :, :2] * 0.1 * self.anchors[None, :, 2:],
                   self.anchors[None, :, 2:] * torch.exp(reg_outputs[:, :, 2:] * 0.2)), -1)
      # xywh -> xyxy
      decoded_boxes[:,:, :2] -= decoded_boxes [:,:, 2:] / 2
      decoded_boxes[:,:, 2:] += decoded_boxes [:,:, :2]

      # output = self.detect(reg_outputs, F.softmax(cls_outputs, dim=-1), self.anchors)
      return decoded_boxes, F.softmax(cls_outputs, dim=-1)
    else:
      return reg_outputs, cls_outputs

# if __name__ == '__main__':
#   features = []
#   from utils.io import *
#
#
#   def hook(self, input, output):
#     print(output.data.cpu().numpy().shape)
#     features.append(output.data.cpu().numpy())
#
#
#   net = SSD(extra_config=extras['300'], head_config=heads['300'], num_classes=20)
#   load_pretrain(net, '../model/vgg16/checkpoint.t7')
#
#   for m in net.modules():
#     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#       m.register_forward_hook(hook)
#
#   y = net(torch.randn(1, 3, 300, 300))
#   pass
#   print(y.size())
