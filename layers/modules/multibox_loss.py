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

class MultiBoxLoss(nn.Module):
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
                    bkg_label, neg_mining, neg_pos, neg_overlap, encode_target, neg_thresh=None, alpha=1, mids=None):
        super(MultiBoxLoss, self).__init__()
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
        self.softmax = nn.Softmax()

    def forward1(self, predictions, targets):
        try:
            return self.forward1(predictions, targets)
        except:
            return Variable(torch.Tensor([0])) 

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        #print("predictions", predictions)
        #print("targets", targets.size())
        #print(targets.size())
        #print(targets)
        loc_data, conf_data, priors = predictions
        loc_data, conf_data, priors = loc_data.cpu(), conf_data.cpu(), priors.cpu()
        targets = targets.cpu()
        num = loc_data.size(0)
        num_priors = (priors.size(0))
        num_classes = self.num_classes
        priors_point = point_form(priors)
        
        #torch.save(conf_data, "conf_data.th")
        self.conf_data = conf_data
        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4).cpu()
        conf_t = torch.LongTensor(num, num_priors).cpu()
        neg_thresh = self.neg_thresh
        if random() > 0.5: neg_thresh = 0
        for idx in range(num):
            #mask = torch.nonzero(targets.data[idx][:,-1]!=-1)
            #print(mask)
            #t = targets.index_select(0,mask)
            truths = targets[idx][:,:-1].data
            labels = targets[idx][:,-1].data
            defaults = priors.data
            mids = match(self.threshold,truths,defaults,self.variance,labels,loc_t,conf_t,idx,neg_thresh=neg_thresh)
        #if GPU:
        #    loc_t = loc_t.cuda()
        #    conf_t = conf_t.cuda()
        # wrap targets
        #print(loc_t)
        loc_t = Variable(loc_t, requires_grad=False).cpu()
        conf_t = Variable(conf_t,requires_grad=False).cpu()
        

        pos = conf_t > 0
        notneg = conf_t != 0
        neg = conf_t == 0
        mid = conf_t==-1
        conf_t[conf_t==-1] = 0
        
        num_pos = pos.sum()

        pos_nz = pos.data[0].nonzero().squeeze()
        #print(pos_nz)
        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        #print("loc_data", loc_data.size())
        #print("pos", pos.unsqueeze(pos.dim()).size())
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)

        loc_p = loc_data[pos_idx].view(-1,4)
        #print(loc_t)
        loc_t = loc_t[pos_idx].view(-1,4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1,self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1,1))
        
        # Hard Negative Mining
        loss_c[notneg] = 0 # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _,loss_idx = loss_c.sort(1, descending=True)
        idx_val, idx_rank = loss_idx.sort(1)
        #print("idx_rank",idx_rank.int())
        #idx_rank = idx_val
        num_pos = pos.long().sum(1)
        #print("num_pos", num_pos)
        #num_neg = torch.max((self.negpos_ratio//2)*num_pos, Variable(torch.LongTensor([[5]]).cuda()))
        num_neg = (self.negpos_ratio)*num_pos
        num_neg = torch.clamp(num_neg, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        
        rand_neg = idx_rank[0][num_neg.data[0][0]:-num_pos.data[0][0]]
        rand_neg = rand_neg.index_select(0, Variable(torch.randperm(rand_neg.size(0))))
        rand_neg = rand_neg[0:num_neg.data[0][0]]

        #print(rand_neg, mids)
        rand_neg = None
        if self.mids is not None and mids.nelement() > 0:
            if rand_neg is not None:
                rand_neg = (torch.cat([rand_neg, Variable(mids[0,:])],0))
            else:
                #print(mids.nelement()) 
                #print(mids)
                rand_neg = Variable(mids[:,1])
        #neg.mul_(0)
        
        #print("ones", ones)
        if rand_neg is not None:
            ones = torch.Tensor([[1]]).byte().cpu()
            ones = ones.expand_as(neg.index_select(1, rand_neg))
            neg.index_add_(1, rand_neg, Variable(ones) )
        
        
        neg_nz = neg.data[0].nonzero().squeeze()
        #if self.mids is not None and mids.nelement() > 0:
        #    print("mids", mids.nelement())
        #    neg_nz = mids[:,1].squeeze()
        #else: neg_nz = neg.data[0].nonzero().squeeze()[:2]
        #print("neg_nz", priors_point.data.index_select(0,neg_nz))
        #print("num_pos", num_pos, "num_neg", num_neg)
        self.neg_boxes = priors_point.data.index_select(0,neg_nz)
        self.pos_boxes = priors_point.data.index_select(0,pos_nz)
        self.targets = targets
        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        print("conf_p", conf_p.size())
        print("targets", targets_weighted.size())
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()
        loss_l/=N
        loss_c/=N
        #print(loss_l.data[0], loss_c.data[0])
        loss = self.alpha*loss_l+loss_c
        #print(loss)
        return loss
