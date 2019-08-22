import os
import numpy as np
from os.path import exists, splitext, isdir, isfile, join, split, dirname
import json
import matplotlib.image as mpimg
from collections import Iterable

def box2rect(box2d):
    """generate individual bounding box from label"""
    x1 = box2d['x1']
    y1 = box2d['y1']
    x2 = box2d['x2']
    y2 = box2d['y2']

    # Draw and add one box to the figure
    return [x1,y1,x2,y2]

# 读label
label_path = 'E:/work/ssd/cvpr/cvpr2018 data/bdd100k/labels/bdd100k_labels_images_train.json'
with open(label_path) as data_file:
    imgannos = json.load(data_file)
# 输出一个可迭代的，避免后面加上index的时候出错
if not isinstance(imgannos, Iterable):
    imgannos = [imgannos]

trainset=[]
com=[]
# 选择读label里第几张图
for i in imgannos:
    imginfo=[]
    for j in i['labels']:
        if j.__contains__('box2d'):
            oclass=j['category']
            bbox=box2rect(j['box2d'])
            imganno=[oclass,bbox]
            imginfo.append(imganno)
    # 读图
    image_dir = 'E:/work/ssd/cvpr/cvpr2018 data/bdd100k/images/100k/train'
    image_path = join(image_dir, i['name'])
    img = mpimg.imread(image_path)
    im = np.array(img, dtype=np.uint8)
    # read bbox
    com=[im,imginfo]
    trainset.append(com)
print(len(trainset))
#[img,[[class1,box1],[class2,box2],...]]