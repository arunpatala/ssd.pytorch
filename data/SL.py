import os
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from polarbear import *

class SLDetections(data.Dataset):

    def __init__(self, aimgs, tile=1000, st=500, fcount=10):
        self.ids = list()
        self.tile = tile       
        self.aimgs = aimgs
        print("loading dataset")
        for aimg, iid in tqdm(aimgs):
            for a,x,y in aimg.tile(tile,st):
                a = a.fpups()
                if a.ann.count >= fcount:
                    self.ids.append((iid,x,y))
        shuffle(self.ids)
                    

    def __getitem__(self, index):
        iid,x,y = self.ids[index]
        aimg,_ = self.aimgs.aimg(iid)
        aimg = aimg.cropd(x,y,self.tile).fpups()
        target = aimg.ann.ssd(self.tile).float()
        img = aimg.np(t=False)
        height, width, _ = img.shape

        #if self.transform is not None:
            #img = cv2.resize(np.array(img), (self.tile, self.tile)).astype(np.float32)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).squeeze().float()

        #if self.target_transform is not None:
        #    target = self.target_transform(target, width, height)
            # target = self.target_transform(target, width, height)

        return img, target

    def __len__(self):
        return len(self.ids)


class SLDetection(data.Dataset):

    def __init__(self, aimg, tile=1000, st=500, fcount=1):
        self.ids = list()
        self.tile = tile
        self.xy = []
        for aimg,x,y in aimg.tile(tile,st):
            #print(x,y)
            aimg = aimg.fpups()
            if aimg.ann.count >= fcount:
                self.ids.append(aimg.oneclass())
                self.xy.append((x,y))

    def __getitem__(self, index):
        aimg = self.ids[index]
        target = aimg.ann.ssd(self.tile).float()
        img = aimg.np(t=False)
        height, width, _ = img.shape

        #if self.transform is not None:
            #img = cv2.resize(np.array(img), (self.tile, self.tile)).astype(np.float32)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).squeeze().float()

        #if self.target_transform is not None:
        #    target = self.target_transform(target, width, height)
            # target = self.target_transform(target, width, height)

        return img, target

    def __len__(self):
        return len(self.ids)


class SLTest(data.Dataset):

    def __init__(self, aimg, mimg, tile=300, st=None, th=0.9):
        if st is None: st = tile - 100
        self.ids = list()
        W,H = aimg.WH
        if mimg is None: 
            #print("mimg is none")
            mimg = AnnImg(npimg=np.zeros((300,300)))
        self.tile = tile
        self.xy = []
        mimg = mimg.resize(W,H)
        for a,x,y in aimg.tile(tile, st):
            m = mimg.cropd(x,y,300)
            p = 1-(m.np().min()/255.0)
            if(p>th):
                self.ids.append(a)
                self.xy.append((x,y))
        

    def __getitem__(self, index):
        aimg = self.ids[index]
        #if aimg.ann is not None:
        #    target = aimg.ann.ssd(self.tile).float()
        #else: 
        target = torch.ones(1)
        img = aimg.np(t=False)
        height, width, _ = img.shape

        #if self.transform is not None:
            #img = cv2.resize(np.array(img), (self.tile, self.tile)).astype(np.float32)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).squeeze().float()

        #if self.target_transform is not None:
        #    target = self.target_transform(target, width, height)
            # target = self.target_transform(target, width, height)

        return img, target

    def __len__(self):
        return len(self.ids)
