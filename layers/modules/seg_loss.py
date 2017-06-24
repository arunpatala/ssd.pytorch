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
        self.bce_loss = nn.BCELoss()
        self.log_loss=False
        self.dice_loss=True
        self.jaccard_loss=False
        self.pos = 0
        self.neg = 0
        self.do_cutoff = True
            
    def set_phase(self, phase):
        self.phase = phase
        self.batch = 0   
        self.pos = 0
        self.neg = 0
             
    def forward(self, pred, target, b=16):
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
        pred = pred[0,0,b:-b,b:-b]
        target = target[0,b:-b,b:-b]
        torch.save([pred,target],"tmp.th")
        
        mask = self.get_mask(pred, target)
        mask = Variable(mask.float()).cuda()
        mpred = pred * mask
        
        self.save(pred, target, mpred)
        return self.cls_loss((target==255).float(), mpred)
    
    def save(self, pred, target, mpred):
        #if self.batch % 300 == 0:
        #    print(self.batch, self.pos/(self.neg+1))
        self.pos += (target.data==255).sum()
        self.neg += ((target.data==0) & (mpred.data!=0)).sum()
        fpath = self.fpath()
        img = img2Image(self.img)
        img.save(fpath.format(type='aaimg'))
        self.aimg.plotc().save(fpath.format(type="aimg"))
        
        
        pred = pred.data.cpu().numpy()
        mpred = mpred.data.cpu().numpy()
        target = target.data.cpu().numpy()
        Image.fromarray((target).astype('uint8')).save(fpath.format(type="mimg"))
        Image.fromarray((pred*255).astype('uint8')).save(fpath.format(type="pred"))
        Image.fromarray((mpred*255).astype('uint8')).save(fpath.format(type="mpred"))
        
    
    def get_mask(self,p,t):
        mask1 = self.cutoff(p,t)
        mask2 = self.cutoff(p,t,True)
        mask3 = t.data==255
        mask = mask1 | mask2 | mask3
        return mask
 
    def cutoff(self, p,t, rand=False):
        tdata = t.data
        pdata = p.data
        npos = (tdata==255).int().sum()
        pred = pdata.clone()
        if rand: pred = torch.rand(pred.size()).cuda()
        pred[tdata!=0] = 0
        cutoff = pred.view(-1).sort(0,descending=True)[0][npos]
        mask = (pred>cutoff) 
        return mask
    
    def fpath(self):
        dir = self.dir.format(phase=self.phase)        
        fpath = dir+"/{phase}_{batch}_{iid}_{x}_{y}_".format(
            batch=self.batch, phase=self.phase, iid=self.iid, x=self.x, y = self.y)
        fpath = fpath + '{type}.jpg'
        #print(fpath)
        return fpath
    
    def cls_loss(self, y, y_pred):
        loss = 0.
        if self.log_loss:
            loss += self.bce_loss(y_pred, y)
        if self.dice_loss:
            intersection = (y_pred * y).sum()
            uwi = y_pred.sum() + y.sum() + 1 # without intersection union
            loss += (1 - intersection / (uwi))
        if self.jaccard_loss:
            intersection = (y_pred * y).sum()
            union = y_pred.sum() + y.sum() - intersection
            if union[0] != 0:
                loss += (1 - intersection / union)
        
        return loss

  
    
def img2Image(img):
    #print(img.size())
    C,H,W, = img.size()
    rgb_means = (104, 117, 123)
    new_img = img.clone().numpy().transpose((1,2,0))
    new_img += rgb_means
    return Image.fromarray(new_img.astype('uint8'))

      
