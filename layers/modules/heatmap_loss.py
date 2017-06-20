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

import logging
        
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
                    dir="weights/logs_", ds=None, mode="ssd"):
        super(HeatMapLoss, self).__init__()
        self.mode = mode
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
        os.makedirs(self.dir.format(phase="train"),exist_ok=False)
        os.makedirs(self.dir.format(phase="val"),exist_ok=False)
        self.aimg = None
        self.iid = None
        self.x = None
        self.y = None
        self.loss = 0
        self.cnt = 0
        self.tp,self.fp,self.fn = 0,0,0
        self.npos = 0
        self.nneg = 0
        self.nrneg = 0
        self.nhneg = 0
        
        logger = logging.getLogger('heatmap')
        hdlr = logging.FileHandler(self.dir.format(phase="_log.log"))
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr) 
        logger.setLevel(logging.INFO)
        logger.error('error: STARTING LOGGING')
        logger.info('info: While this is just chatty')
        self.logger = logger
        self.anns = {}
        
    def set_phase(self, phase):
        print(self.phase, self.loss/(self.cnt+1))
        print("tp",self.tp,"fp",self.fp,"fn",self.fn)
        print("npos", self.npos, "nrneg", self.nrneg, "nhneg", self.nhneg)
        self.plot_all()
        self.tp,self.fp,self.fn = 0,0,0
        self.anns = {}
        self.loss = 0
        self.cnt = 0
        self.batch = 0
        self.npos = 0
        self.nneg = 0
        self.nrneg = 0
        self.nhneg = 0
        self.phase = phase
        self.prep = None

    def forward(self, predictions, targets):
        #print(predictions.size())
        if self.mode == "ssd":
            conf_data = predictions[0].cpu()
            targets = targets.cpu()
            return self.forward1(conf_data, targets)
        elif self.mode == "unet":
            targets = targets.cpu()
            conf_data = predictions[0].permute(1,2,0)
            #print(conf_data.size())
            return self.forward1(conf_data, targets)
        else: assert(self.mode == None) 


    def plot_all(self):
        tann = Ann(dets=None)
        path = self.dir.format(phase=self.phase+"_all") 
        os.makedirs(path,exist_ok=True)
        path = path + os.sep + "{iid}.jpg"
        ds = DataSource()
        aimgs = ds.dataset(dataset="all")
        for  iid in self.anns:
            tann = tann.append(self.anns[iid])
            aimg = aimgs.aimg(iid)[0]
            aimg = self.prep(aimg)
            tp,fp,fn = self.anns[iid].pn()
            self.logger.info(str(iid) +" : "+str(tp)+","+str(fp)+","+str(fn))
            aimg = aimg.wann(self.anns[iid])
            aimg.plot(save=path.format(iid=iid), rect=False)
            aimg.ann.save(path.format(iid=iid)+".csv")
        tann.save(self.dir.format(phase=self.phase+"_tann.csv"))
        print(tann.pn())
        
        
            
            
    def forward1(self, predictions, targets):
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
        #print(predictions.size())
        i = predictions.size(0)
        conf_data = predictions.cpu()
        #print(conf_data.size())
        #print(conf_data)
        fpath = self.fpath()
        img = img2Image(self.img)
        img.save(fpath.format(type='aaimg'))
        aimg = self.aimg
        #print("aimg",aimg.WH)
        
        aimg.plot(save=fpath.format(type='aimg'))
        #torch.save(aimg, fpath.format(type='aimg')+".th")
        
        hmimg = self.hmimg(conf_data).wann(aimg.ann)
        #print("hmimg",hmimg.WH)
        hmimg.plot(save=fpath.format(type='hmimg'))
        pnann, plot = hmimg.hm_pn(bg=aimg, p=75)
        self.anns[self.iid] = self.anns.get(self.iid,Ann(dets=None)).append(pnann.dxy(-self.x, -self.y)) 
        tp,fp,fn = pnann.pn()
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.logger.info(str(self.batch) + " tp " + str(self.tp) + " fp " + str(self.fp) +   " fn "  +str(self.fn)
                          + " prec " + str(self.tp/(self.tp+self.fp+1)) + " recall " + str(self.tp/(self.tp+self.fn+1)))
        plot.save(fpath.format(type='primg'))
        torch.save(hmimg, fpath.format(type='hmimg')+".th")
        
        
        if self.neg:
            nimg = aimg.neg(ratio=1, minn=1)
            nimg.plot(save=fpath.format(type='nimg'))
            aimg = aimg.append(nimg.ann)
            self.nrneg += nimg.count
            
            
        if self.hneg:
            hnimg = hmimg.hneg(ratio=1, minn=1, th=True)
            hnimg.plot(save=fpath.format(type='hnimg'))
            aimg = aimg.append(hnimg.ann)
            self.nhneg += hnimg.count
            
        if self.vor:
            vimg = aimg.ann.vor()
            vimg.plot(save=fpath.format(type='vimg'))
            aimg = aimg.append(vimg.ann)
        
        aimg = aimg.fcenter()    
        aimg.plot(save=fpath.format(type="pnimg"))
        
        nann = aimg.resize(i).ann
        pred = []
        self.npos += nann.fclass(0).count
        self.nneg += nann.fclass(-1).count
        self.logger.info(str(self.batch) + " pos " + str(self.npos) + " rneg " + str(self.nrneg) + " hneg " + str(self.nhneg) ) 
        self.logger.info(str(self.batch) +" tpos " + str(hmimg.count) + " rneg " + str(nimg.count) + " hneg " + str(hnimg.count))
        for x,y in np.clip(nann.xy,0,i-1):
            pred.append(conf_data[y,x])
        if len(pred) ==0: 
            pred = [conf_data[0,0]]
            pred = torch.stack(pred,0)
            truth = Variable(torch.from_numpy(np.array([0])).long()) 
        else:
            pred = torch.stack(pred,0)
            truth = Variable(torch.from_numpy(nann.cl+1).long())
        #print(pred, truth)
        loss = F.cross_entropy(pred, truth , size_average=False) 
        loss = loss/max(aimg.count,1)
        self.loss += loss.data[0]
        self.cnt += 1
        return loss
    
    def hmimg(self, conf):
        cd = self.sf(conf.view(-1,self.num_classes)).view(conf.size())
        #print(cd.size())
        npimg=(cd.data[:,:,0]*255).round().numpy()
        #print(npimg)
        return AnnImg(npimg=npimg).resize(self.dim)
    
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

