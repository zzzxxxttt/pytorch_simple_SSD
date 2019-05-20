'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class Block(nn.Module):
  '''Depthwise conv + Pointwise conv'''

  def __init__(self, in_planes, out_planes, stride=1):
    super(Block, self).__init__()
    self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
    self.bn1 = nn.BatchNorm2d(in_planes)
    self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
    self.bn2 = nn.BatchNorm2d(out_planes)

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = F.relu(self.bn2(self.conv2(out)))
    return out


class MobileNet(nn.Module):
  # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
  cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]

  def __init__(self, num_classes=10):
    super(MobileNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(32)

    self.layers = nn.ModuleList()
    in_planes = 32
    for x in self.cfg:
      out_planes = x if isinstance(x, int) else x[0]
      stride = 1 if isinstance(x, int) else x[1]
      self.layers.append(Block(in_planes, out_planes, stride))
      in_planes = out_planes
      # self.linear = nn.Linear(1024, num_classes)

    self.register_parameter('scale', nn.Parameter(20 * torch.ones(512)))
    self.out_channels=1024

  def forward(self, x):
    x = F.relu(self.bn1(self.conv1(x)), inplace=True)

    for i, layer in enumerate(self.layers):
      x = layer(x)
      if i == 10:
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-10
        conv4_3 = self.scale[None, :, None, None] * torch.div(x, norm)
    return conv4_3, x

if __name__ == '__main__':
  from torch.autograd import Variable

  features = []


  def hook(self, input, output):
    print(output.data.cpu().numpy().shape)
    features.append(output.data.cpu().numpy())


  net = MobileNet()
  for m in net.modules():
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
      m.register_forward_hook(hook)

  y = net(Variable(torch.randn(1, 3, 300, 300)))
  pass
  # print(y.size())
