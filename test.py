from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import BDD_ROOT, BDD_CLASSES as labelmap
from PIL import Image
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
import torch.utils.data as data
from ssd import build_ssd
from data import *
import json

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/32000.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--save_folder2', default='eval/imgs', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=BDD_ROOT, help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
# torch.set_default_tensor_type('torch.FloatTensor')
# if torch.cuda.is_available():
#     map_location=lambda storage, loc: storage.cuda()
# else:
#     map_location='cpu'

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
if not os.path.exists(args.save_folder2):
    os.mkdir(args.save_folder2)

def test_net(save_folder, net, cuda, testset, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    filename = save_folder+'test1.txt'
    num_images = len(testset)
    images = dict()
    for i in range(num_images):
        print(i)
        if i==20:
            break
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        #img [255,255,3]
        img = testset.pull_image(i)
        imgname = testset.pull_name(i)
        img_id, annotation = testset.pull_anno(i)
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        # x = Variable(x.unsqueeze(0))
        x = x.unsqueeze(0)
        images[str(img_id)]=dict()
        images[str(img_id)]['ground truth']=[]
        # with open(filename, mode='a') as f:
        #     f.write('\nGROUND TRUTH FOR: '+img_id+'\n')
        #     for box in annotation:
        #         f.write('label: '+' || '.join(str(b)+':'+str(box[b]) for b in box)+'\n')
        for box in annotation:
            images[str(img_id)]['ground truth'].append([box['bbox'],labelmap[box['category_id']-1]])
            img=draw_boxes1(img, box['bbox'], labelmap[box['category_id']-1])
        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        #y loc[batch,8000,4], con[batch 8000 1], prior[8000,4]
        detections = y.data
        # scale each detection back up to the image,
        #huan yuan tu
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        pred_num = 0
        images[str(img_id)]['predections']=[]
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                # if pred_num == 0:
                #     with open(filename, mode='a') as f:
                #         f.write('PREDICTIONS: '+'\n')
                score = detections[0, i, j, 0]
                label_name = labelmap[i-1]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = [float(pt[0]), float(pt[1]), float(pt[2]), float(pt[3])]
                pred_num += 1
                images[str(img_id)]['predections'].append([coords,label_name,score.item()])
                img = draw_boxes2(img,coords,label_name,score.item())
                # with open(filename, mode='a') as f:
                #     f.write(str(pred_num)+' label: '+label_name+' score: ' +
                #             str(score) + ' '+' || '.join(str(c) for c in coords) + '\n')
                j += 1
        cv2.imwrite(os.path.join(save_folder, 'imgs', imgname), img)
    json_string = json.dumps(images)
    with open(filename, "w") as file:
        file.write(json_string)

# def test_net(save_folder, net, cuda, testset, transform, thresh):
#     # dump predictions and assoc. ground truth to text file for now
#     filename = save_folder+'test1.txt'
#     num_images = len(testset)
#     for i in range(num_images):
#         print('Testing image {:d}/{:d}....'.format(i+1, num_images))
#         img = testset.pull_image(i)
#         img_id, annotation = testset.pull_anno(i)
#         x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
#         x = Variable(x.unsqueeze(0))
#
#         with open(filename, mode='a') as f:
#             f.write('\nGROUND TRUTH FOR: '+img_id+'\n')
#             for box in annotation:
#                 f.write('label: '+' || '.join(str(b) for b in box)+'\n')
#         if cuda:
#             x = x.cuda()
#
#         y = net(x)      # forward pass
#         detections = y.data
#         # scale each detection back up to the image
#         scale = torch.Tensor([img.shape[1], img.shape[0],
#                              img.shape[1], img.shape[0]])
#         pred_num = 0
#         for i in range(detections.size(1)):
#             j = 0
#             while detections[0, i, j, 0] >= 0.6:
#                 if pred_num == 0:
#                     with open(filename, mode='a') as f:
#                         f.write('PREDICTIONS: '+'\n')
#                 score = detections[0, i, j, 0]
#                 label_name = labelmap[i-1]
#                 pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
#                 coords = (pt[0], pt[1], pt[2], pt[3])
#                 pred_num += 1
#                 with open(filename, mode='a') as f:
#                     f.write(str(pred_num)+' label: '+label_name+' score: ' +
#                             str(score) + ' '+' || '.join(str(c) for c in coords) + '\n')
#                 j += 1
def draw_boxes2(image, box, label, score):
    image_h, image_w, _ = image.shape
    xmin = int(box[0])
    ymin = int(box[1])
    xmax = int(box[2])
    ymax = int(box[3])
    cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0,255,0), 3)
    cv2.putText(image, label + ' ' + str(round(score,4)),(xmin, ymin - 13),cv2.FONT_HERSHEY_SIMPLEX,1e-3 * image_h,(0,255,0), 2)
    return image
def draw_boxes1(image, box, label):
    image_h, image_w, _ = image.shape
    xmin = int(box[0])
    ymin = int(box[1])
    xmax = int(box[0]+box[2])
    ymax = int(box[1]+box[3])
    cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (255,0,0), 3)
    cv2.putText(image, label,(xmin, ymin - 13),cv2.FONT_HERSHEY_SIMPLEX,1e-3 * image_h,(255,0,0), 2)
    return image

def test_voc():
    # load net
    num_classes = len(BDD_CLASSES) + 1 # +1 background
    net = build_ssd('test', 300, num_classes) # initialize SSD
    # net.load_state_dict(torch.load(args.trained_model,map_location=map_location))
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    testset = BDDDetection(args.voc_root, 'val', None, BDDAnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold)


if __name__ == '__main__':
    test_voc()
