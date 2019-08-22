"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import BDD_ROOT, BDDAnnotationTransform, BDDDetection, BaseTransform
from data import BDD_CLASSES as labelmap
import torch.utils.data as data

from ssd import build_ssd

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2
cachefile='E:/work/ssd/CFEnet/ssd300_120000/val'
a=[]
for i, cls in enumerate(labelmap):
    filename=os.path.join(cachefile,cls+'_pr.pkl')
    with open(filename, 'rb') as f:
        recs = pickle.load(f)
        a.append(recs)
    print(cls)
    print(recs)
# if sys.version_info[0] == 2:
#     import xml.etree.cElementTree as ET
# else:
#     import xml.etree.ElementTree as ET
# detfile='E:/work/ssd/cvpr/cvpr2018 data/bdd100k/results/det_val_traffic sign.txt'
# with open(detfile, 'r') as f:
#     lines = f.readlines()
# if any(lines) == 1:
#     splitlines = [x.strip().split(' ') for x in lines]
#     print(splitlines)
#     image_ids = [x[0] for x in splitlines]
#     print(image_ids)
#     confidence = np.array([float(x[1]) for x in splitlines])
#     print(confidence)
#     BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
#     print(BB)
#     sorted_ind = np.argsort(-confidence)
#     print(sorted_ind)
#     sorted_scores = np.sort(-confidence)
#     BB = BB[sorted_ind, :]
#     print(BB)
#     image_ids = [image_ids[x] for x in sorted_ind]
#     print(image_ids)
    # # go down dets and mark TPs and FPs
    # # 对探测到的结果与annotation里的样本对比
    # # 循环次数等于探测到的结果数量
    # nd = len(image_ids)
    # print(nd)