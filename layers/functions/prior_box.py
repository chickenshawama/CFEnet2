from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        #[38, 19, 10, 5, 3, 1]
        for k, f in enumerate(self.feature_maps):
            #feature_maps的边长,对每个点都迭代一遍，每个像素都有4-6个bbox
            for i, j in product(range(f), repeat=2):
                #f_k约等于feature map 的大小
                #image_size=300,step=[8, 16, 32, 64, 100, 300]
                #min_sizes=[30, 60, 111, 162, 213, 264]
                #max[60, 111, 162, 213, 264, 315]
                f_k = self.image_size / self.steps[k]

                # unit center x,y,归一化的
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                #30/300=0.1,60/300=0.2,111/300=0.37, 264/300=0.88,anchor和图片的比例
                s_k = self.min_sizes[k]/self.image_size
                #[归一化坐标,0.1,0.1]，方便在大图中使用
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1)),区别是一个是max一个是min
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                #[37.5,37.5 ,0.14,0.14]
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                #'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
