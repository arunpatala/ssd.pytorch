import cv2
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset

from torchsample.modules import ModuleTrainer
from torchsample.callbacks import *
from torchsample.regularizers import *
from torchsample.constraints import *
from torchsample.initializers import *
from torchsample.metrics import *
import torch.nn.init as init
from network import *

import torch.optim as optim

import os
from torchvision import datasets

from data import VOCroot, v2, v1, AnnotationTransform, VOCDetection, detection_collate, BaseTransform, AnnTensorTransform
from data import SLHalf
from layers import MultiBoxLoss, HeatMapLoss
from ssd import build_ssd
from hm import PosNeg

import argparse

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--version', default='v2', help='conv11_2(v2) or pool6(v1) as last layer')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--iterations', default=120000, type=int, help='Number of training epochs')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
parser.add_argument('--visdom', default=False, type=bool, help='Use visdom to for loss visualization')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--num_classes', default=6, help='num of classes')
parser.add_argument('--dim', default=512,type=int, help='dimension of input to model')
parser.add_argument('--alpha', default=0, type=int, help='multibox alpha for loss')
parser.add_argument('--th', default=0.33, type=float, help='threshold')
parser.add_argument('--neg_th', default=0.30, type=float, help='neg threshold')
parser.add_argument('--neg_pos', default=1, type=float, help='ratio')
parser.add_argument('--load', default=None, help='trained model')
parser.add_argument('--iid', default=0, type=int, help='trained model')
parser.add_argument('--dataset', default='all',  help='dataset to use')
parser.add_argument('--mids', default=False,  type=bool, help='use mids or not')
parser.add_argument('--aug', default=True,  type=bool, help='use augmentation')
parser.add_argument('--epochs', default=10,  type=int, help='num of epochs')
parser.add_argument('--scale', default=1,  type=float, help='use scaled image')
parser.add_argument('--neg', default=True,  type=bool, help='negative random')
parser.add_argument('--hneg', default=True,  type=bool, help='hard negetive mining')
parser.add_argument('--vor', default=False,  type=bool, help='use voronoi neg samples')
parser.add_argument('--fcount', default=3,  type=int, help='min count of sealions')
parser.add_argument('--dbox', default=25,  type=int, help='diff box')
parser.add_argument('--limit', default=None,  type=int, help='limit of dataloader')
parser.add_argument('--top', default=None,  type=int, help='take images by count')
parser.add_argument('--name', help='name to store current logs(required)')
parser.add_argument('--tshuffle', default=True, type=bool, help='shuffle train data')
args = parser.parse_args()
print(args)
#"""weights/sealions_95k.pth"""
cfg = (v1, v2)[args.version == 'v2']



#criterion = MultiBoxLoss(args.num_classes, args.th, True, 0, True, args.neg_pos, 0.5, False, alpha=args.alpha, neg_thresh=args.neg_th, mids=args.mids)
criterion = HeatMapLoss(args.num_classes, dim=args.dim, ds=None,
                        neg=args.neg, hneg=args.hneg, vor=args.vor, 
                        dir='weights/logshm_{name}_'.format(name=args.name), mode="unet")

#model = Network()
print("building unet")
model = UNet(args.num_classes)

if args.cuda:
    model.cuda()
if args.load is not None:
    if args.cuda: 
        model.load_my_state_dict(torch.load(args.load))
    else: 
        model.load_my_state_dict(torch.load(args.load, map_location=lambda storage, loc: storage))
        


from polarbear import *


train_dataset = SLHalf(half=0, dataset=args.dataset, tile=args.dim, st=args.dim-100, fcount=args.fcount, 
                       aug=args.aug, ct=criterion, limit=args.limit, dbox=args.dbox, top=args.top)
val_dataset = SLHalf(half=1, dataset=args.dataset, tile=args.dim, st=args.dim-100, fcount=args.fcount, 
                     aug=args.aug,ct=criterion, limit=args.limit, dbox=args.dbox, top=args.top)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.tshuffle)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

trainer = ModuleTrainer(model)

chk = ModelCheckpoint(directory="weights", monitor="loss", filename='hmtrainer_'+args.name+'_{epoch}_train_{loss}.pth.tar', verbose=1)

optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
trainer.compile(loss=criterion,
                optimizer=optimizer,
                #optimizer='adadelta',
                #regularizers=regularizers,
                #constraints=constraints,
                #initializers=initializers,
                #metrics=metrics, 
                callbacks=[chk, PosNeg(model, criterion, val_dataset)])

print("trainer compilation done")


#val_dataset = VOCDetection(VOCroot, [('2007', 'test')], BaseTransform(
#        ssd_dim, rgb_means), AnnTensorTransform())
#val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

if args.cuda:
    trainer.fit_loader(train_loader, val_loader, nb_epoch=args.epochs, verbose=1, cuda_device=0)
else: trainer.fit_loader(train_loader, val_loader, nb_epoch=args.epochs, verbose=1)

