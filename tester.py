
import cv2 
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.serialization import load_lua
import numpy as np
from ssd import build_ssd
from IPython.display import display
# from models import build_ssd as build_ssd_v1 # uncomment for older pool6 model
from tqdm import tqdm, trange
from polarbear import *
from data import *
import argparse


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--start', default=0, type=int, help='starting for testing')
parser.add_argument('--batch_size', default=8, type=int, help='starting for testing')
args = parser.parse_args()



ds = DataSource()
test = ds.dataset("test")



net = build_ssd('train', 300, 6)    # initialize SSD
net.load_state_dict(torch.load('weights/sealions_95k.pth'))
net.cuda();



def gen_all(th, start=0):
    print("start",start)
    for iid in tqdm(test.iids[start:]):
        ai,_ = test.aimg(iid)
        mimg = ds.get_mimg(iid)
        train_dataset = SLTest(ai, mimg, th=th)
        train_loader = DataLoader(train_dataset, batch_size=1)
        ds.mkpath("th", dataset="test", iid=iid)
        #print("dataset",iid, len(train_dataset))
        for i in range(len(train_dataset)):
            dx,dy = train_dataset.xy[i]
            fpath = ds.path("th",iid=iid,x=dx,y=dy,dataset="test")
            if os.path.exists(fpath):continue
            #print(dx,dy)
            x,y = train_dataset[i]
            dets = net.forward(Variable(x.cuda().unsqueeze(0)))
            loc, conf, priors = dets[0]
            dets = (loc.data.cpu(),conf.data.cpu(),priors.data.cpu())
            torch.save(dets, fpath)           

def gen_dets(train_dataset, batch_size=16):
    train_loader = iter(DataLoader(train_dataset, batch_size=batch_size, shuffle=False))
    cnt = 0
    for X,Y in train_loader:
        loc, conf, priors = net.forward(Variable(X.cuda()))[0]
        for i in  range(X.size(0)):
            x,y = train_dataset.xy[cnt]
            cnt = cnt + 1
            yield x, y, (loc[i], conf[i], priors)
            
def gen_iid(iid, th, batch_size):
    ai,_ = test.aimg(iid)
    mimg = ds.get_mimg(iid)
    train_dataset = SLTest(ai, mimg, th=th)
    ds.mkpath("th", dataset="test", iid=iid)
    #print(iid, len(train_dataset))
    for x,y,(l,c,p) in gen_dets(train_dataset, batch_size):
        fpath = ds.path("th",iid=iid,x=x,y=y,dataset="test")
        if os.path.exists(fpath):continue
        #print(fpath)
        #torch.save((l,c,p), fpath)  


def gen_test(th, start=0, batch_size=8):
    print("start",start)
    for iid in tqdm(test.iids[start:]):
        gen_iid(iid, th, batch_size)


def save_iid(iid, th, batch_size):
    ai,_ = test.aimg(iid)
    mimg = ds.get_mimg(iid)
    train_dataset = SLTest(ai, mimg, th=th)
    train_loader = iter(DataLoader(train_dataset, batch_size=batch_size, shuffle=False))
    
    ds.mkpath("th", dataset="test", iid=iid)
    fpath = ds.path("th",iid=iid, x="x", y="y", dataset="test")
    torch.save(train_dataset.xy, fpath)
    for i,(X,Y) in enumerate(train_loader):
        fpath = ds.path("th", iid=iid, x="batch", y=i, dataset="test")
        dets = net.forward(Variable(X.cuda()))[0]
        torch.save(dets, fpath)
        


def save_test(th, start=0, batch_size=8):
    print("start",start)
    for iid in tqdm(test.iids[start:]):
        save_iid(iid, th, batch_size)

    
from torch.utils.data import DataLoader
#gen_test(0.9, start=args.start, batch_size=args.batch_size)
save_test(0.9, start=args.start, batch_size=args.batch_size)


