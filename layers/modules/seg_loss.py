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
from PIL import Image
import logging
        
class SegLoss(nn.Module):
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


    def __init__(self, num_classes, dir, dim=600, phase="train"):
        super(SegLoss, self).__init__()
        self.num_classes = num_classes
        self.sf = nn.Softmax()
        self.dim = dim
        self.dir = dir
        self.batch = 0
        self.bc = nn.BCELoss()
        self.phase = phase
        self.dir = dir+"{phase}"
        os.makedirs(self.dir.format(phase="train"),exist_ok=False)
        os.makedirs(self.dir.format(phase="val"),exist_ok=False)
        self.iid,self.x,self.y = 0,0,0
            
    def set_phase(self, phase):
        self.phase = phase   
             
    def forward(self, pred, target):
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
        self.pred = pred[0][0].data.cpu().numpy()
        #print(pred.size(), target.size())
        mpred = pred[0][0] * (target[0]!=127).float()
        self.mpred = mpred.data.cpu().numpy()
        self.save()
        return self.bc(mpred, (target[0]==255).float())
    
    def save(self):
        fpath = self.fpath()
        img = img2Image(self.img)
        img.save(fpath.format(type='aaimg'))
        self.aimg.plotc().save(fpath.format(type="aimg"))
        self.aimg.unet_mask().save(fpath.format(type="mimg"))
        Image.fromarray((self.pred*255).astype('uint8')).save(fpath.format(type="pred"))
        Image.fromarray((self.mpred*255).astype('uint8')).save(fpath.format(type="mpred"))
        
    
    def fpath(self):
        dir = self.dir.format(phase=self.phase)        
        fpath = dir+"/{phase}_{batch}_{iid}_{x}_{y}_".format(
            batch=self.batch, phase=self.phase, iid=self.iid, x=self.x, y = self.y)
        fpath = fpath + '{type}.jpg'
        #print(fpath)
        return fpath
    
  
    
def img2Image(img):
    #print(img.size())
    C,H,W, = img.size()
    rgb_means = (104, 117, 123)
    new_img = img.clone().numpy().transpose((1,2,0))
    new_img += rgb_means
    return Image.fromarray(new_img.astype('uint8'))

      
