import cv2 
import os
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.serialization import load_lua
import numpy as np
from ssd import build_ssd

# from models import build_ssd as build_ssd_v1 # uncomment for older pool6 model

from layers.box_utils import decode, nms

from tqdm import tqdm, trange
from polarbear import *
from data import *
from torch.utils.data import DataLoader

import argparse

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--fpups', default=False, type=bool, help='filter pups')
parser.add_argument('--dataset', default="test", help='dataset')
parser.add_argument('--th', default=1.0, type=int, help='threshold')
parser.add_argument('--delete', default=False, type=bool, help='delete collate')
args = parser.parse_args()
print(args)

ds = DataSource()
test = ds.dataset(args.dataset)


def gen_xy(iid):
    #print("iid", iid)
    fpath = ds.path("th", iid=iid, x="x", y="y", dataset=args.dataset)
    #if not os.path.exists(fpath): return
    xy = torch.load(fpath)
    cnt = 0
    for i in range(1000):
        fpath = ds.path("th",iid=iid,x="batch",y=i, dataset=args.dataset)
        if not os.path.exists(fpath): break
        l,c,p = torch.load(fpath)
        if args.delete : os.remove(fpath)
        #l,c,p = l.data.cpu(), c.data.cpu(), p.data.cpu()
        l,c,p = l.data, c.data, p.data
        for j in range(l.size(0)):
            x,y = xy[cnt]
            cnt += 1
            yield x,y,(l[j],c[j],p)

def get_ann(dets, p=0.33, th=0.33, fpups=False):
    loc,conf,priors = dets
    decoded_boxes = decode(loc, priors, [0.1, 0.2])
    
    conf_scores = conf.t().contiguous()
    cl = 0
    c_mask = conf_scores[cl].lt(p)
    if fpups:
        p_mask = conf_scores[-1].lt(p)
        c_mask = c_mask & p_mask
    scores = conf_scores[cl][c_mask]
    l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
    boxes = decoded_boxes[l_mask].view(-1, 4)
    if(boxes.nelement()==0): return None
    ids, count = nms(boxes, 1-scores, th, 200)
    ids = ids.cpu()
    #print(boxes)
    ann_dets = (boxes[ids[:count]]*300).round().numpy()
    #print(ann_dets)
    ann = Ann(dets=ann_dets)
    return ann
softmax = nn.Softmax()

def get_ann2(dets, cut_off=0.5, th=0.33, fpups=False):
    loc, conf, priors = dets
    conf = softmax(Variable(conf)).data #.cpu()
    
    decoded_boxes = decode(loc, priors, [0.1, 0.2])
    conf_scores = conf.t().contiguous()
    conf_p, conf_cl = conf_scores[1:,:].max(0)
    
    cl = 0
    c_mask = conf_scores[cl]<0.5
    scores = conf_scores[cl][c_mask]
    l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
    boxes = decoded_boxes[l_mask].view(-1, 4)
    
    if(boxes.nelement()==0): return None
    boxes_cl = (conf_cl.squeeze()[c_mask]).unsqueeze(1).float()
    boxes_p = (conf_p.squeeze()[c_mask]).unsqueeze(1).float()
    dets = torch.cat([boxes_cl, boxes_p*100, boxes*300],1)
    #ids, count = nms(boxes, 1-scores, th, 200)
    #ids = ids.cpu()
    
    #ann_dets = (boxes[ids[:count]]*300).cpu().round().numpy()
    dets = dets.round().cpu().numpy().astype('int32')
    
    ann = Ann(dets=dets)
    
    return ann

def get_allann(iid, fpups=False):
    ret = []
    for x,y,(l,c,p) in gen_xy(iid):
        #print(x,y,l.size(),c.size(),p.size())
        ann = get_ann2((l,c,p),fpups=fpups)
        if ann is not None: 
            ann.dxy(-x,-y)
            ret.append(ann.dets)
    if len(ret)==0: return None
    ret = np.concatenate(ret)

    return Ann(dets=ret)

def plot_ann(iid, fpups=False):

    fpath = ds.path("th", iid=iid, x="x", y="y", dataset=args.dataset)
    if not os.path.exists(fpath): return
    
    ann = get_allann(iid, fpups=fpups)
    if ann is None: return
    annpath = ds.path("anns_test", iid=iid, dataset=args.dataset)
    ann.save(annpath)

    """aimg,_ = test.aimg(iid)
    pimg = AnnImg(aimg.img, ann).setScale(40).plotc()
    fpath = ds.path("plot", iid=iid, type="c", dataset=args.dataset)
    pimg.save(fpath)
    pimg = AnnImg(aimg.img, ann).setScale(40).plot(label=False).img
    fpath = ds.path("plot", iid=iid, type="r40", dataset=args.dataset)
    pimg.save(fpath)"""

ds.mkpath("plot", dataset=args.dataset)
ds.mkpath("anns_test", dataset=args.dataset)

for iid in tqdm(test.iids):
    plot_ann(iid, fpups=args.fpups)
