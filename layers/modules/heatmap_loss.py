import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import v2 as cfg
from ..box_utils import match, log_sum_exp, point_form
from random import random
GPU = False
if torch.cuda.is_available():
    GPU = True
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
import numpy as np
from polarbear import *

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


    def __init__(self, num_classes, overlap_thresh, prior_for_matching, 
                    bkg_label, neg_mining, neg_pos, neg_overlap, encode_target, 
                    neg_thresh=None, alpha=1, mids=None, dim=600, fmap=None):
        super(HeatMapLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching  = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']
        self.alpha = alpha
        if neg_thresh is not None: self.neg_thresh = neg_thresh
        else: self.neg_thresh = overlap_thresh
        self.mids = mids
        self.sf = nn.Softmax()
        self.dim = dim
        self.fmap = [100,56]
        self.neg = True
        self.vor = False
        self.hneg = True

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
        
        loc_data, conf_data, priors = predictions
        loc_data, conf_data, priors = loc_data.cpu(), conf_data.cpu(), priors.cpu()
        targets = targets.cpu()
        self.conf_data = conf_data
        
        i = self.fmap[0]
        conf_data = conf_data[0,:(i*i)].view((i,i,-1))
        aimg = self.aimg(conf_data, targets[0])
        
        if self.neg:
            nimg = aimg.append(aimg.negdist(d=50, ratio=1))
            nann = nimg.resize(i).ann
        
        if self.hneg:
            hnimg = aimg.resize(i).append(aimg.resize(i).hneg(d=10, ratio=1)).resize(self.dim)
            nann = nann.append(hnimg.resize(i).ann)
        #naimg.save('demo/tmp.jpg','demo/tmp.csv')
        if self.vor:
            nann = naimg.resize(i).ann
            vann = aimg.ann.vor()
            vimg =  AnnImg(aimg.img, vann)
            nann = vimg.resize(i).ann
        #print(nann.xy)
        
        #logging
        self.himg = aimg
        self.pnimg = hnimg
        
        pred = []
        for x,y in np.clip(nann.xy,0,i-1):
            pred.append(conf_data[y,x])
        pred = torch.stack(pred,0)
        truth = Variable(torch.from_numpy(nann.cl+1).long())
        loss = F.cross_entropy(pred, truth , size_average=False) 
        return loss/targets.size(1)
    
    def aimg(self, conf, targets):
        cd = self.sf(conf.view(-1,6)).view(conf.size())
        npimg=(cd.data[:,:,0]*255).round().numpy()
        ann = Ann(dets=targets.data.numpy(), dim=self.fmap[0])
        return AnnImg(npimg=npimg,ann=ann).resize(self.dim)
