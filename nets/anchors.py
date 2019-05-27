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


class AnchorBox(object):
  def __init__(self, config):
    super(AnchorBox, self).__init__()
    # self.type = cfg.name
    self.image_size = config['min_dim']
    # number of anchors for each feature map
    self.num_anchors = len(config['aspect_ratios'])
    # size of each feature map
    self.feature_maps = config['feature_maps']
    # size of minimum anchor box in each feature map
    self.min_sizes = config['min_sizes']
    # size of maximum anchor box in each feature map
    self.max_sizes = config['max_sizes']
    # step size between anchors in each feature map
    self.steps = config['steps']
    # aspect ratio for anchor boxes in each feature map
    self.aspect_ratios = config['aspect_ratios']
    # clip anchor box if it is out of the image
    self.clip = config['clip']
    self.version = config['name']

  def __call__(self):
    boxes = []
    for k, f in enumerate(self.feature_maps):
      # 00,01,...,0f,10,11,...,1f,...,f1,...,ff
      for i, j in product(range(f), repeat=2):
        # the relative coordinate of an anchor box is (i + 0.5) * step_size / img_size
        cx = (j + 0.5) * self.steps[k] / self.image_size
        cy = (i + 0.5) * self.steps[k] / self.image_size

        # anchor boxes with size = min_size and aspect_ratio = 1
        s_k = self.min_sizes[k] / self.image_size
        boxes += [cx, cy, s_k, s_k]

        # anchor boxes with size = sqrt(min_size[k] * min_size[k+1]) and aspect_ratio = 1
        s_k_prime = sqrt(self.min_sizes[k] * self.max_sizes[k]) / self.image_size
        boxes += [cx, cy, s_k_prime, s_k_prime]

        # rest of aspect ratios
        for ratio in self.aspect_ratios[k]:
          # ratio * x * x = min_size ** 2
          # x = sqrt(min_size ** 2 / ratio) = min_size / sqrt(ratio)
          # x / img_szie = min_size / img_size / sqrt(ratio)
          boxes += [cx, cy, s_k * sqrt(ratio), s_k / sqrt(ratio)]
          boxes += [cx, cy, s_k / sqrt(ratio), s_k * sqrt(ratio)]

    output = np.array(boxes).reshape([-1, 4])  # shape: [8732, 4]
    if self.clip:
      output = np.clip(output, a_min=0, a_max=1.0)
    return output


# if __name__ == '__main__':
#   boxes = AnchorBox(v2)()
#   print(boxes)
