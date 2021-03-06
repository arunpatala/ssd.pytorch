import os
import random
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
#module_path = os.path.abspath(os.path.join('/home/arun/github/'))
#if module_path not in sys.path:
#    sys.path.append(module_path)

import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from utils.augment import * 
from polarbear import *
import polarbear
from tqdm import tqdm

class SLDetections(data.Dataset):

    def __init__(self, aimgs, tile=1000, st=500, fcount=3, aug=False, limit=None):
        self.augs = None
        if aug: 
            self.augs = SSDAugmentation(tile)
            #tile = int(tile * 1.5)
        if limit is not None: aimgs.iids = aimgs.iids[1:limit+1]
        self.ids = list()
        self.tile = tile       
        self.aimgs = aimgs

        #print("loading dataset")
        th = []
        for aimg, iid in tqdm(aimgs):
            th.append((max(aimg.ann.max_th()),iid))
            for a,x,y in aimg.tile(tile,st):
                a = a.fpups()
                if a.ann.count >= fcount:
                    self.ids.append((iid,x,y))
        #print(max(th))
        #print(sorted(th))
        #shuffle(self.ids)
        self.wh = set()
                    

    def __getitem__(self, index):
        iid,x,y = self.ids[index]
        aimg,_ = self.aimgs.aimg(iid)
        aimg = aimg.cropd(x,y,self.tile)
        target = aimg.ann.ssd(self.tile).float()
        img = aimg.np(t=False)
        height, width, _ = img.shape
        
        if self.augs is not None:
            img = img.transpose(2, 0, 1)
            img, target = self.augs(img, target.numpy())
            img = img.transpose(1, 2, 0)
            img = img.copy()
        l = len(self.wh)
        height, width, _ = img.shape
        anno = (target[:,:4] * height/10).round() * 10
        w = anno[:,2] - anno[:,0]
        h = anno[:,3] - anno[:,1]

        wh = set(sorted(list(zip(w.tolist(), w.tolist()))))
        self.wh.update(wh)
        wh = set(sorted(list(zip(h.tolist(), h.tolist()))))
        self.wh.update(wh)
        #if(len(self.wh)!=l): print(self.wh)
        #print(img.shape, img.min(), img.max())
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

    def __init__(self, aimg, tile=1000, st=500, fcount=0, aug=False, ct=None):
        self.augs = None
        if aug: 
            self.augs = SSDAugmentation(tile)
            tile = int(tile * 1.5)

        self.ids = list()
        self.tile = tile
        self.xy = []
        for aimg,x,y in aimg.tile(tile,st):
            #print(x,y)
            aimg = aimg.fpups()
            if aimg.ann.count >= fcount:
                self.ids.append(aimg.oneclass())
                self.xy.append((x,y))

        self.wh = set()
        self.ct = ct

    def __getitem__(self, index):
        aimg = self.ids[index]
        if self.ct is not None: self.ct.aimg = aimg
        target = aimg.ann.ssd(self.tile).float()
        img = aimg.np(t=False)
        height, width, _ = img.shape
        img1 = img
        target1 = target


        if self.augs is not None:
            try:
                img = img.transpose(2, 0, 1)
                img, target = self.augs(img, target.numpy())
                img = img.transpose(1, 2, 0)
                img = img.copy()
            except: 
                print("ERROR")
                img = img1
                target = target1
        l = len(self.wh)
                


        #if self.transform is not None:
            #img = cv2.resize(np.array(img), (self.tile, self.tile)).astype(np.float32)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).squeeze().float()


        if target.nelement() == 0:
            #print("target empty")
            target = torch.FloatTensor([[-1,-1,-1,-1,0]]) 
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




class SLHalf(data.Dataset):

    def __init__(self, dataset="all", tile=800, st=700, fcount=0, aug=False, ct=None, half=0, limit=None, dbox=0, top=None, scale=1):
        
        if aug: self.augs = SSDAugmentation(tile)
        else: self.augs = None
        self.half = half
        self.ids = list()
        self.tile = tile
        self.dbox= dbox
        ds = DataSource()
        def prep(aimg):
            return aimg.rescale().scale(scale).dbox(-dbox).bcrop()[half]
        self.prep = prep
        aimgs = ds.dataset(dataset=dataset)
        if top is not None: 
            aimgs.iids = sorted(aimgs.counts(take=top)[:,0])
            
            print("top",aimgs.iids)
        self.aimgs = aimgs
        if limit is not None: aimgs.iids = aimgs.iids[:limit]
        for aimg, iid in tqdm(aimgs):
            #if aimg.ann.sc == 0: continue
            aimg = self.prep(aimg)        
            for aimg,x,y in aimg.tile(tile,st):
                aimg = aimg.fpups().fcenter()
                if aimg.count >= fcount:
                    self.ids.append((iid,x,y))
                    
        self.ct = ct
        self.half = half
        self.ct.prep = prep
        #shuffle(self.ids)

    def __getitem__(self, index):
        
        iid,x,y = self.ids[index]
        #print(iid,x,y)
        aimg = self.aimgs.aimg(iid)[0]
        aimg = self.prep(aimg)
        aimg = aimg.fpups().oneclass().cropd(x,y,self.tile)
        #aimg,_ = self.aimgs.aimg(iid)
        #aimg = self.ids[index]
        iid,x,y = self.ids[index]
        if self.ct is not None: 
            self.ct.aimg = aimg
            self.ct.iid = iid
            self.ct.x = x
            self.ct.y = y
            self.ct.prep = self.prep
            
        target = aimg.ann.ssd(self.tile).float()
        img = aimg.np(t=False)
        height, width, _ = img.shape
        img1 = img
        target1 = target



        if target.nelement() == 0:
            #print("target empty")
            target = torch.FloatTensor([[-1,-1,-1,-1,0]]) 
        if self.augs is not None:
                img = img.transpose(2, 0, 1)
                img, target = self.augs(img, target.numpy())
                img = img.transpose(1, 2, 0)
                img = img.copy()
                


        #if self.transform is not None:
            #img = cv2.resize(np.array(img), (self.tile, self.tile)).astype(np.float32)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).squeeze().float()
        if self.ct is not None: self.ct.img = img

        return img, target

    def __len__(self):
        return len(self.ids)
    
    

class SLSegmentation(data.Dataset):

    def __init__(self, dataset="all", tile=800, st=700, fcount=0, aug=False, ct=None, half=0, limit=None, 
                 top=None, scale=1, fpups=True, oneclass=True, vor=None):
        self.vor = vor
        if aug: self.augs = SSDAugmentation(tile)
        else: self.augs = None
        self.half = half
        self.ids = list()
        self.tile = tile
        ds = polarbear.ds.DataSource()
        def prep(aimg):
             aimg = aimg.rescale().scale(scale).bcrop()[half]
             if fpups: aimg=aimg.fpups()
             if oneclass: aimg = aimg.oneclass()
             return aimg
        self.prep = prep
        aimgs = ds.dataset(dataset=dataset)
        if top is not None: 
            aimgs.iids = sorted(aimgs.counts(take=top)[:,0])
            print("top",aimgs.iids)
        self.aimgs = aimgs
        if limit is not None: aimgs.iids = aimgs.iids[:limit]
        for aimg, iid in tqdm(aimgs):
            #if aimg.ann.sc == 0: continue
            aimg = self.prep(aimg)        
            for aimg,x,y in aimg.tile(tile,st):
                aimg = aimg.fpups().fcenter()
                if aimg.count >= fcount:
                    self.ids.append((iid,x,y))
                    
        
        self.half = half
        if ct is not None:
            self.ct = ct
            self.ct.prep = prep
        #shuffle(self.ids)

    def __getitem__(self, index):
        
        iid,x,y = self.ids[index]
        aimg = self.aimgs.aimg(iid)[0]
        aimg = self.prep(aimg)
        aimg = aimg.cropd(x,y,self.tile)
        aaimg = aimg
        mask = aimg.unet_mask(vor=self.vor)
        if self.augs is not None:
            aimg, mask = self.pilAugmentation(  aimg.img ,  aimg.unet_mask(vor=self.vor).img  )
            aimg = AnnImg(aimg)
            mask = AnnImg(mask)
        
        if self.ct is not None:
            self.ct.aimg = aimg
            self.ct.iid = iid
            self.ct.x = x
            self.ct.y = y
            
        img = aimg.np(t=False)
        target = aaimg.ann.ssd(self.tile).float()
        
        if self.augs is not None:
                img = img.transpose(2, 0, 1)
                img, target = self.augs(img, target.numpy())
                img = img.transpose(1, 2, 0)
                img = img.copy()
                
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).squeeze().float()
        target = mask.np()
        target = torch.from_numpy(target).float()
        if self.ct is not None: self.ct.img = img
        
        #if random.random() < 0.5:
        #    return img.transpose(Image.FLIP_LEFT_RIGHT) # img
        #    return img.transpose(Image.FLIP_LEFT_RIGHT) # mask
        #img = transforms.ToTensor()(img)
        #target = transforms.ToTensor()(target)
        
        # target = self.target_transform(target, width, height)
        # target = self.target_transform(target, width, height)

        #print(img.size(), target.size())
        
        return img, target

    def __len__(self):
        return len(self.ids)

    def pilAugmentation(self, img, target):
            w, h = img.size
            #Resize image
            osize =random.randint(int(1.0*w), int(1.5*w))
            img=img.resize((osize,osize))
            target=target.resize((osize,osize))
    
            #Random rotate
            rot=random.randint(0,360)
            img=img.rotate(rot)
            target=target.rotate(rot)
            
            #Random Crop
            tw, th = img.size
            if(tw > w and th > h):
                x1 = random.randint(0, tw - w)
                y1 = random.randint(0, th - h)
                img=img.crop((x1, y1, x1 + w, y1 + h))
                target=target.crop((x1, y1, x1 + w, y1 + h))
            #Random Flip
            if random.random() < 0.5:
                img= img.transpose(Image.FLIP_LEFT_RIGHT)       # img
                target= target.transpose(Image.FLIP_LEFT_RIGHT) # target
            if random.random() < 0.5:
                img= img.transpose(Image.FLIP_TOP_BOTTOM)       # img
                target= target.transpose(Image.FLIP_TOP_BOTTOM) # target
            return img,target
        
        
        
        
        