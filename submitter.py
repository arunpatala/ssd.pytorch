
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
parser.add_argument('--start', default=0, type=int, help='starting for testing')
parser.add_argument('--end', default=1000, type=int, help='starting for testing')
parser.add_argument('--dataset', default="test", help='dataset')
parser.add_argument('--size', default=900, type=int, help='size')
parser.add_argument('--classes', default=6, type=int, help='classes')
parser.add_argument('--cuda', default=True, type=bool, help='use cuda')
parser.add_argument('--load', default='weights/day1_001_2.4164.pth.tar', help='input size of network')
parser.add_argument('--iid', default=None, help='single iid to test')
parser.add_argument('--name', default="day1", help='experiment name')
parser.add_argument('--plot', default=True, type=bool, help='experiment name')
args = parser.parse_args()

datasource = DataSource()
test = datasource.dataset(args.dataset)
test.shuffle()
test.take(start=args.start, end=args.end)

model = build_ssd('test', args.size, args.classes, load=args.load, cuda=args.cuda)
datasource.mkpath('anns_test',test=args.name, dataset=args.dataset)
if args.plot: datasource.mkpath('plot_test',test=args.name, dataset=args.dataset)
iids = test.iids
if args.iid is not None: iids = [args.iid]
for iid in tqdm(iids):
    aimg,iid = test.aimg(iid)
    ann = aimg.test(model,args.size)
    fpath = datasource.path('anns_test',test=args.name, iid=iid, dataset=args.dataset)
    ann.save(fpath)
    if args.plot: 
        fpath = datasource.path('plot_test',test=args.name, iid=iid, dataset=args.dataset)
        #print(fpath)
        AnnImg(aimg.img, ann).plot().save(fpath)



