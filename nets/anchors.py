import numpy as np
from math import sqrt as sqrt
from itertools import product as product

v2 = {'feature_maps': [38, 19, 10, 5, 3, 1],
      'min_dim': 300,
      'steps': [8, 16, 32, 64, 100, 300],
      'min_sizes': [30, 60, 111, 162, 213, 264],
      'max_sizes': [60, 111, 162, 213, 264, 315],
      # 'aspect_ratios' : [[2, 1/2], [2, 1/2, 3, 1/3], [2, 1/2, 3, 1/3],
      #                    [2, 1/2, 3, 1/3], [2, 1/2], [2, 1/2]],
      'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
      'variance': [0.1, 0.2],
      'clip': True,
      'name': 'v2'}


class PriorBox(object):
  def __init__(self, config):
    super(PriorBox, self).__init__()
    # self.type = cfg.name
    self.image_size = config['min_dim']
    # number of priors for feature map location (either 4 or 6)
    # 使用的feature map 数量
    self.num_priors = len(config['aspect_ratios'])
    # feature map 大小
    self.feature_maps = config['feature_maps']
    # 最小的box大小
    self.min_sizes = config['min_sizes']
    # 最大的box大小
    self.max_sizes = config['max_sizes']
    # 相邻box之间距离
    self.steps = config['steps']
    # box比例
    self.aspect_ratios = config['aspect_ratios']
    # 如果box超出边界就截断
    self.clip = config['clip']
    self.version = config['name']

  def __call__(self):
    mean = []
    for k, f in enumerate(self.feature_maps):
      # 在当前feature map大小中循环
      # 00,01,...,0f,10,11,...,1f,...,f1,...,ff
      for i, j in product(range(f), repeat=2):
        f_k = self.image_size / self.steps[k]
        # unit center x,y
        cx = (j + 0.5) / f_k
        cy = (i + 0.5) / f_k

        # aspect_ratio: 1
        # rel size: min_size
        s_k = self.min_sizes[k] / self.image_size
        mean += [cx, cy, s_k, s_k]

        # aspect_ratio: 1
        # rel size: sqrt(s_k * s_(k+1))
        s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
        mean += [cx, cy, s_k_prime, s_k_prime]

        # rest of aspect ratios
        for ar in self.aspect_ratios[k]:
          mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
          mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]

    # 转换成[N,4]形式，共N个anchor box
    output = np.array(mean).reshape([-1, 4])
    if self.clip:
      output = np.clip(output, a_min=0, a_max=1.0)
    return output


if __name__ == '__main__':
  boxes = PriorBox(v2)()
  pass
