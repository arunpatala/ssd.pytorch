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

class SLDetection(data.Dataset):

    def __init__(self, aimg, tile=1000, st=500):
        self.ids = list()
        self.tile = tile
        for aimg,x,y in aimg.tile(tile,st):
            aimg = aimg.fpups()
            if aimg.ann.count > 10:
                self.ids.append(aimg.oneclass())

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
