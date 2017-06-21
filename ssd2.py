import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import v2, v1, v, v3, v600, v900, vXXX
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import os
from torch.nn.parameter import Parameter
import  gpustat
import gc
from pympler import tracker
from pympler import refbrowser
from network import *

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
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, base, extras, head, num_classes, size=300, ct=None):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        # TODO: implement __call__ in PriorBox
        self.ct = ct
        
        self.size = size


        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.conf = nn.ModuleList(head)
        self.enc = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.ReLU(inplace=True),            
        )



    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

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
        conf = list()
        # apply vgg up to conv4_3 relu
        #gpustat.print_gpustat(json=False)
        #gc.collect()
        #gpustat.print_gpustat(json=False)
        
        #self.tr.print_diff()
        for k in range(23):
            #print(k, x.size(), self.vgg[k])
            x = self.vgg[k](x)
        
        s = self.L2Norm(x)
        sources.append(x)
        print("conf1", x.size())
        #y = x

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            #print(k, x.size(), self.vgg[k])
            x = self.vgg[k](x)
        #z = self.enc(x)
        #yz = torch.cat([y,z],1)
        #sources.append(yz)
        x = self.enc(x)
        sources.append(x)
        print("conf2", x.size())
        
        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            #print(k, x.size(), v)
            if k % 2 == 1:  sources.append(x)
        # apply multibox head to source layers
        for (x, c) in zip(sources, self.conf):
            #print(x.size(),c)
            ret = c(x)
            #print(x.size(),ret.size(), c)
            conf.append(ret.permute(0, 2, 3, 1).contiguous())
            
        #for c in conf:
        #    print("conf", c.size())
        #conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        

        #output = conf.view(conf.size(0), -1, self.num_classes)
            
        return conf[1]

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

    def load_my_state_dict(self, state_dict):
 
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)



# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False, dilation=1):
    layers = []
    in_channels = i
    for j,v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            if j<8:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, dilation=dilation)
            if batch_norm:
                layers += [conv2d, nn.Sequential(nn.BatchNorm2d(v), nn.ReLU(inplace=True))]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
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


def multibox(vgg, extra_layers, cfg, num_classes):
    conf_layers = []
    vgg_source = [24, -2]
    #vgg_source = [-2]
    for k, v in enumerate(vgg_source):
        out = vgg[v].out_channels
        #if out==512: out = out * 2
        conf_layers += [nn.Conv2d(out,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers,  conf_layers


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '900': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
    '1200': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '1000': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '600': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],

    'XXX': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],

    'X': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    'XXX': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    'X': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '900': [256],
    '1200': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '600': [256],#, 'S', 512],
    #'300': [256],
    '512': [],
    '1000': [256],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    #'300': [1, 1],  # number of boxes per feature map location
    '1200': [4, 6, 6, 6, 4, 4], 
    '600': [2, 2, 2, 2, 2, 2], 
    'XXX': [2, 2, 2, 2, 2, 2], 
    'X': [1, 1, 1, 1, 1, 1], 
    '900': [2, 2, 2, 2, 2, 2], 
    '512': [],
    '1000': [1, 1], 
}


def build_ssd(phase, size=300, num_classes=21, scales=0, load=None, cuda=False, ct=None, dilation=1, batch_norm=False):
    torch.set_default_tensor_type('torch.FloatTensor')
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    xxx = size
    if scales is not None: 
        xxx = 'X'
        extras[xxx] = extras[xxx][:2*scales] 
    ssd = SSD(phase, *multibox(vgg(base[str(xxx)], 3, dilation=dilation, batch_norm=batch_norm),
                                add_extras(extras[str(xxx)], 1024, batch_norm=batch_norm),
                                mbox[str(xxx)], num_classes), num_classes, size=size, ct=ct)
    if cuda: ssd.cuda()
    if load is not None: 
        ssd.load_state_dict(torch.load(load))
    return ssd
