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
parser.add_argument('--iid', type=int, help='starting for testing')
parser.add_argument('--fpups', default=True, type=bool, help='filter pups')
parser.add_argument('--dataset', default="test", help='dataset')
args = parser.parse_args()


ds = DataSource()
test = ds.dataset(args.dataset)


def gen_xy(iid):
    print("iid", iid)
    xy = torch.load(ds.path("th", iid=iid, x="x", y="y", dataset=args.dataset))
    cnt = 0
    for i in range(1000):
        fpath = ds.path("th",iid=iid,x="batch",y=i, dataset=args.dataset)
        if not os.path.exists(fpath): break
        l,c,p = torch.load(fpath)
        l,c,p = l.data.cpu(), c.data.cpu(), p.data.cpu()
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

def get_allann(iid, fpups=False):
    ret = []
    for x,y,(l,c,p) in gen_xy(iid):
        #print(x,y,l.size(),c.size(),p.size())
        ann = get_ann((l,c,p),fpups=fpups)
        if ann is not None: 
            ann.dxy(-x,-y)
            ret.append(ann.dets)
    ret = np.concatenate(ret)

    return Ann(dets=ret)

def plot_ann(iid, fpups=False):
    aimg,_ = test.aimg(iid)
    ann = get_allann(iid, fpups=fpups)
    pimg = AnnImg(aimg.img, ann).setScale(40).plotc()
    fpath = ds.path("plot", iid=iid, type="c", dataset=args.dataset)
    pimg.save(fpath)
    pimg = AnnImg(aimg.img, ann).setScale(40).plot(label=False).img
    fpath = ds.path("plot", iid=iid, type="r40", dataset=args.dataset)
    pimg.save(fpath)

ds.mkpath("plot", dataset=args.dataset)
plot_ann(args.iid, fpups=args.fpups)