import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import v2 as cfg
from ..box_utils import match, log_sum_exp, point_form
from random import random
GPU = False
from PIL import Image
if torch.cuda.is_available():
    GPU = True
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
import numpy as np
from polarbear import *
import os

class HeatMapLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """


    def __init__(self, num_classes, dim=600, fmap=None,
                    neg=True, hneg=True, vor=False, 
                    dir="weights/logs_", ds=None):
        super(HeatMapLoss, self).__init__()
        self.num_classes = num_classes
        self.sf = nn.Softmax()
        self.dim = dim
        self.fmap = [100,56]
        self.neg = neg
        self.vor = vor
        self.hneg = hneg
        self.phase = "train"
        self.batch = 0
        self.dir = dir+"{phase}"
        os.makedirs(self.dir.format(phase="train"),exist_ok=True)
        os.makedirs(self.dir.format(phase="val"),exist_ok=True)
        self.aimg = None
        
    def set_phase(self, phase):
        self.phase = phase
        self.batch = 0

    def forward1(self, predictions, targets):
        try:
            return self.forward1(predictions, targets)
        except:
            return Variable(torch.Tensor([0])) 

    def forward(self, predictions, targets):
        """HeatMapLoss Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        self.batch += 1
        loc_data, conf_data, priors = predictions
        loc_data, conf_data, priors = loc_data.cpu(), conf_data.cpu(), priors.cpu()
        targets = targets.cpu()
        self.conf_data = conf_data
        
        
        i = self.fmap[0]
        conf_data = conf_data[0,:(i*i)].view((i,i,-1))
        
        fpath = self.fpath()
        aimg = self.aimg
        aimg.plot(save=fpath.format(type='aimg'))
        #torch.save(aimg, fpath.format(type='aimg')+".th")
        
        hmimg = self.hmimg(conf_data).wann(aimg.ann)
        hmimg.plot(save=fpath.format(type='hmimg'))
        torch.save(hmimg, fpath.format(type='hmimg')+".th")
        
        
        if self.neg:
            nimg = aimg.neg(ratio=1, minn=2)
            nimg.plot(save=fpath.format(type='nimg'))
            aimg = aimg.append(nimg.ann)
            
            
        if self.hneg:
            hnimg = hmimg.hneg(ratio=1, minn=2, th=True)
            hnimg.plot(save=fpath.format(type='hnimg'))
            aimg = aimg.append(hnimg.ann)
            
        if self.vor:
            vimg = aimg.ann.vor()
            vimg.plot(save=fpath.format(type='vimg'))
            aimg = aimg.append(vimg.ann)
        
        aimg = aimg.fcenter()    
        aimg.plot(save=fpath.format(type="pnimg"))
        
        nann = aimg.resize(i).ann
        pred = []
        for x,y in np.clip(nann.xy,0,i-1):
            pred.append(conf_data[y,x])
        pred = torch.stack(pred,0)
        truth = Variable(torch.from_numpy(nann.cl+1).long())
        loss = F.cross_entropy(pred, truth , size_average=False) 
        return loss/aimg.count
    
    def hmimg(self, conf):
        cd = self.sf(conf.view(-1,self.num_classes)).view(conf.size())
        npimg=(cd.data[:,:,0]*255).round().numpy()
        return AnnImg(npimg=npimg).resize(self.dim)
    
    def fpath(self):
        dir = self.dir.format(phase=self.phase)        
        fpath = dir+"/{phase}{batch}".format(batch=self.batch, phase=self.phase)
        fpath = fpath + '{type}.jpg'
        #print(fpath)
        return fpath
