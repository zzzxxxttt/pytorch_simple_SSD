import torch
import torch.nn as nn
import math


class VGG(nn.Module):
  def __init__(self, conv_config):
    super(VGG, self).__init__()
    self.layers = nn.ModuleList()
    in_channels = 3

    for c in conv_config[:-2]:
      if c == 'M':
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
      else:
        self.layers.append(nn.Conv2d(in_channels, c, kernel_size=3, padding=1))
        self.layers.append(nn.ReLU(inplace=True))
        in_channels = c

    self.layers.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
    self.layers.append(nn.Conv2d(conv_config[-3], conv_config[-2], 3, padding=6, dilation=6))
    self.layers.append(nn.ReLU(inplace=True))
    self.layers.append(nn.Conv2d(conv_config[-2], conv_config[-1], 1))
    self.layers.append(nn.ReLU(inplace=True))

    # scale parameter for conv4_3
    self.scale = nn.Parameter(20 * torch.ones(1, conv_config[12], 1, 1))
    self.out_channels = conv_config[-1]

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
          m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def forward(self, x):
    conv4_3 = None
    for i, layer in enumerate(self.layers):
      x = layer(x)
      if i == 22:
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-10
        conv4_3 = self.scale * x / norm
    return conv4_3, x


def vgg16base():
  """VGG 16-layer model (configuration "D") with batch normalization"""
  return VGG([64, 64, 'M',
              128, 128, 'M',
              256, 256, 256, 'M',
              512, 512, 512, 'M',
              512, 512, 512, 1024, 1024])


if __name__ == '__main__':

  def hook(self, input, output):
    print(output.data.cpu().numpy().shape)


  net = vgg16base()
  for m in net.modules():
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
      m.register_forward_hook(hook)

  y = net(torch.randn(1, 3, 300, 300))
  pass
  # print(y.size())
