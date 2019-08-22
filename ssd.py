import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os
import numpy as np

inpla=False

class CFE(nn.Module):
    def __init__(self,CFEl,CFEr,input_channel):
        super(CFE, self).__init__()
        self.cfel=nn.ModuleList(CFEl)
        self.cfer=nn.ModuleList(CFEr)
        self.conv_e=nn.Conv2d(input_channel, input_channel, kernel_size=1)
        self.act_e=nn.ReLU(inplace=inpla)
    def forward(self, x):
        y=x
        z=x
        for k, v in enumerate(self.cfel):
            z=v(z)
        for k, v in enumerate(self.cfer):
            y=v(y)
        y=torch.cat((z,y),1)
        y=self.conv_e(y)
        y=self.act_e(y)
        return x+y

class FFB(nn.Module):
    def __init__(self, c1,c2,size):
        super(FFB, self).__init__()
        self.conv_y1=nn.Conv2d(c1,c1, kernel_size=1)
        self.act1 = nn.ReLU(inplace=inpla)
        self.conv_y2=nn.Conv2d(c2,c1, kernel_size=1)
        self.act2 = nn.ReLU(inplace=inpla)
        self.up=nn.Upsample(size=size, mode='nearest')
    def forward(self, x,y):
        y1=self.conv_y1(x)
        y1=self.act1(y1)
        y2=self.conv_y2(y)
        y2=self.act2(y2)
        y2=self.up(y2)
        return y1+y2



class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes, cfelist, ffblist):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        self.cfe1 = cfelist[0]
        self.cfe2 = cfelist[1]

        self.ffb1 = ffblist[0]
        self.ffb2 = ffblist[1]
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)
        s1_1 = self.L2Norm(x)

        #cfe,32 512 38 38
        x=self.cfe1(x)
        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)

        s1_2 = x

        s1 = self.ffb1(s1_1,s1_2)
        s1 = self.cfe1(s1)
        sources.append(s1)
        x=self.cfe2(x)
        # apply extra layers and cache source layer outputs


        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=inpla)
            if k==0:
                s2=self.ffb2(s1_2,x)
                s2=self.cfe2(s2)
                sources.append(s2)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            #输出转换维度，permute，输出的还是特征图的格式，通道数是box数*loc或类数
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        #这边把每个特征图分别在x,y,z的维度上衔接了
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                #强行拉成需要的大小
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=inpla)]
            else:
                layers += [conv2d, nn.ReLU(inplace=inpla)]
            in_channels = v
    #19*19
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    #19*19
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    #10*10
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    #10*10
    layers += [pool5, conv6,
               nn.ReLU(inplace=inpla), conv7, nn.ReLU(inplace=inpla)]
    return layers

def add_CFE_l(cfg, i):
    layers=[]
    in_channels=i
    layers+=[nn.Conv2d(in_channels, cfg[1], kernel_size=(1,1))]
    layers+=[nn.ReLU(inplace=inpla)]
    layers += [nn.Conv2d(cfg[1], cfg[2], kernel_size=(1, 7), stride=1, padding=(0,3), groups=8)]
    layers += [nn.ReLU(inplace=inpla)]
    layers += [nn.Conv2d(cfg[2], cfg[3], kernel_size=(7, 1), stride=1, padding=(3,0), groups=8)]
    layers += [nn.ReLU(inplace=inpla)]
    layers += [nn.Conv2d(cfg[3], cfg[4], kernel_size=(1,1))]
    layers += [nn.ReLU(inplace=inpla)]
    return layers

def add_CFE_r(cfg, i):
    layers=[]
    in_channels=i
    layers+=[nn.Conv2d(in_channels, cfg[1],kernel_size=(1,1))]
    layers += [nn.ReLU(inplace=inpla)]
    layers += [nn.Conv2d(cfg[1], cfg[2], kernel_size=(7, 1), stride=1, padding=(3,0), groups=8)]
    layers += [nn.ReLU(inplace=inpla)]
    layers += [nn.Conv2d(cfg[2], cfg[3], kernel_size=(1, 7), stride=1, padding=(0,3), groups=8)]
    layers += [nn.ReLU(inplace=inpla)]
    layers += [nn.Conv2d(cfg[3], cfg[4], kernel_size=(1,1)),nn.ReLU(inplace=inpla)]
    layers += [nn.ReLU(inplace=inpla)]
    return layers



def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers

#多尺度特征层后的层，转化出bbox位置和21个类别的自信程度
def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}

cfe_config= {
    "300":[[512,256,256,256,256],[1024,512,512,512,512]],
    "500":[],
}



cfe_channellist=[512,1024]
#change size!
# scalefact=0
# for i in base['300']:
#     if isinstance(i,'str'):
#         scalefact+=1

def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    cfes = []
    fbbs = []
    ffb_sizelist = [[512, 1024, (np.ceil(size/8).astype(np.int8), np.ceil(size/8).astype(np.int8))], [1024, 256, (np.ceil(size/16).astype(np.int8),np.ceil(size/16).astype(np.int8))]]
    for i, j in enumerate(cfe_channellist):
        cfes.append(CFE(add_CFE_l(cfe_config[str(size)][i],j),add_CFE_r(cfe_config[str(size)][i],j),j))
    for i in ffb_sizelist:
        fbbs.append(FFB(i[0],i[1],i[2]))

    base_, extras_,  head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)],
                                     num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes,cfes,fbbs)
# phase, size, base, extras, head, num_classes, cfelist, ffblist
