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
        if size==1000: v = v
        elif size==1200: v=v3
        elif size==600: v = v600
        elif size==900: v = v900
        self.ct = ct
        
        v = vXXX
        v['min_dim'] = size
        self.v = v
        self.v['sqrt'] = False
        self.priorbox = None
        #self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size


        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        
        self.softmax = nn.Softmax()
        self.detect = Detect(num_classes, 0, 2000, 0.01, 0.45)
        self.fmap = []
        self.tr = tracker.SummaryTracker()


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
        loc = list()
        conf = list()
        self.imgs = x
        self.ct.imgs = x
        # apply vgg up to conv4_3 relu
        #gpustat.print_gpustat(json=False)
        #gc.collect()
        #gpustat.print_gpustat(json=False)
        
        #self.tr.print_diff()
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)
        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)
        
        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
        self.fmap = []
        # apply multibox head to source layers
        #print(torch.zeros(1))
        for (x, l, c) in zip(sources, self.loc, self.conf):
            #print(x.size(), l(x).size(), c(x).size())
            
            self.fmap.append(l(x).size(2))
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        if self.priorbox is None:
            print("fmap", self.fmap)
            self.v['feature_maps'] = self.fmap
            #print(self.v)
            self.priorbox = PriorBox(self.v)
            self.priors = Variable(self.priorbox.forward(), volatile=True)
            
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        self.conf_output = conf

        if self.phase == "test":
            #print("test mode")
            tmp = self.softmax(conf.view(-1, self.num_classes))
            self.conf_output = tmp
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                tmp,  # conf preds
                self.priors                                     # default boxes
            )
            return output
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        #self.tr.print_diff()
        #ib = refbrowser.InteractiveBrowser(self)
        #ib.main()

        #torch.save(output, 'output.th')
        return [output]

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
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
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
    loc_layers = []
    conf_layers = []
    vgg_source = [24, -2]
    #vgg_source = [-2]
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


def build_ssd(phase, size=300, num_classes=21, scales=0, load=None, cuda=False, ct=None):
    torch.set_default_tensor_type('torch.FloatTensor')
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    xxx = size
    if scales is not None: 
        xxx = 'X'
        extras[xxx] = extras[xxx][:2*scales] 
    ssd = SSD(phase, *multibox(vgg(base[str(xxx)], 3),
                                add_extras(extras[str(xxx)], 1024),
                                mbox[str(xxx)], num_classes), num_classes, size=size, ct=ct)
    if cuda: ssd.cuda()
    if load is not None: 
        ssd.load_state_dict(torch.load(load))
    return ssd
