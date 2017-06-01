
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
from tqdm import tqdm, trange
from polarbear import *
from data import *

import argparse


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--start', default=900, type=int, help='starting for testing')
parser.add_argument('--batch_size', default=4, type=int, help='starting for testing')
parser.add_argument('--filter', default=False, type=bool, help='filter tested images')
parser.add_argument('--reverse', default=False, type=bool, help='reverse images')
parser.add_argument('--dataset', default="test", help='dataset')
parser.add_argument('--mask', default=True, type=bool, help='use masked')
parser.add_argument('--cuda', default=True, type=bool, help='use cuda')
parser.add_argument('--size', default=300, type=int, help='input size of network')
parser.add_argument('--load', default='weights/trainer_004_2.5533.pth.tar', help='input size of network')
parser.add_argument('--iid', default=None, help='single iid to test')
args = parser.parse_args()

print(args)

ds = DataSource()
test = ds.dataset(args.dataset)
if args.iid is not None: test.iids = [args.iid]
print(test.iids)


net = build_ssd('train', args.size, 6)    # initialize SSD

if args.cuda: 
    net.cuda();
    net.load_state_dict(torch.load(args.load))
else: 
    net.load_state_dict(torch.load( args.load, map_location=lambda storage, loc: storage))
net.eval();


def gen_all(th, start=0):
    print("start",start)
    for iid in tqdm(test.iids[start:]):
        ai,_ = test.aimg(iid)
        mimg = ds.get_mimg(iid)
        train_dataset = SLTest(ai, mimg, th=th)
        train_loader = DataLoader(train_dataset, batch_size=1)
        ds.mkpath("th", dataset=args.dataset, iid=iid)
        #print("dataset",iid, len(train_dataset))
        for i in range(len(train_dataset)):
            dx,dy = train_dataset.xy[i]
            fpath = ds.path("th",iid=iid,x=dx,y=dy,dataset=args.dataset)
            if os.path.exists(fpath):continue
            #print(dx,dy)
            x,y = train_dataset[i]
            if args.cuda: x = x.cuda()

            dets = net.forward(Variable(x.unsqueeze(0)))
            loc, conf, priors = dets[0]
            dets = (loc.data.cpu(),conf.data.cpu(),priors.data.cpu())
            torch.save(dets, fpath)           

def gen_dets(train_dataset, batch_size=16):
    train_loader = iter(DataLoader(train_dataset, batch_size=batch_size, shuffle=False))
    cnt = 0
    for X,Y in train_loader:
        loc, conf, priors = net.forward(Variable(X.cuda()))[0]
        for i in  range(X.size(0)):
            print(x,y)

            x,y = train_dataset.xy[cnt]
            cnt = cnt + 1
            yield x, y, (loc[i], conf[i], priors)
            
def gen_iid(iid, th, batch_size):
    ai,_ = test.aimg(iid)
    mimg = ds.get_mimg(iid)
    train_dataset = SLTest(ai, mimg, th=th)
    ds.mkpath("th", dataset=args.dataset, iid=iid)
    #print(iid, len(train_dataset))
    for x,y,(l,c,p) in gen_dets(train_dataset, batch_size):
        fpath = ds.path("th",iid=iid,x=x,y=y,dataset=args.dataset)
        if os.path.exists(fpath):continue
        #print(fpath)
        #torch.save((l,c,p), fpath)  


def gen_test(th, start=0, batch_size=8):
    print("start",start)
    for iid in tqdm(test.iids[start:]):
        gen_iid(iid, th, batch_size)


def save_iid(iid, th, batch_size):
    ai,_ = test.aimg(iid)
    #print(args.mask)
    if args.mask:   mimg = ds.get_mimg(iid)
    else: mimg = None
    train_dataset = SLTest(ai, mimg, th=th, tile=args.size)
    train_loader = iter(DataLoader(train_dataset, batch_size=batch_size, shuffle=False))
    
    ds.mkpath("th", dataset=args.dataset, iid=iid)
    fpath = ds.path("th",iid=iid, x="x", y="y", dataset=args.dataset)
    torch.save(train_dataset.xy, fpath)
    for i,(X,Y) in tqdm(enumerate(train_loader), total=len(train_dataset)//args.batch_size):
        #print(X.size())
        fpath = ds.path("th", iid=iid, x="batch", y=i, dataset=args.dataset)
        if args.cuda: X = X.cuda()
        dets = net.forward(Variable(X))[0]
        l,c,p = dets
        #print(l.size(), c.size(), p.size())
        torch.save(dets, fpath)
        


def save_test(th, start=0, batch_size=8, flt=False, reverse=False):
    print(test.iids)
    iids = test.iids #[ start:]
    print(iids)
    if flt: iids = list(filter(iids))
    if reverse: iids = iids[::-1]
    print("start", iids[0], len(iids), len(test.iids))
    for iid in tqdm(iids):
        save_iid(iid, th, batch_size)

def filter(iids):
    print("checking")
    for iid in tqdm(iids):
        fpath = ds.path("th", dataset=args.dataset, iid=iid, x="x", y="y")
        if not os.path.exists(fpath): yield iid

from torch.utils.data import DataLoader
#gen_test(0.9, start=args.start, batch_size=args.batch_size)
save_test(0.9, start=args.start, batch_size=args.batch_size, flt=args.filter, reverse=args.reverse)


