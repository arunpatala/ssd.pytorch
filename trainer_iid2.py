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


import torch.optim as optim

import os
from torchvision import datasets

from data import VOCroot, v2, v1, AnnotationTransform, VOCDetection, detection_collate, BaseTransform, AnnTensorTransform
from data import SLDetection
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
parser.add_argument('--dim', default=411,type=int, help='dimension of input to model')
parser.add_argument('--alpha', default=0, type=int, help='multibox alpha for loss')
parser.add_argument('--th', default=0.33, type=float, help='threshold')
parser.add_argument('--neg_th', default=0.30, type=float, help='neg threshold')
parser.add_argument('--neg_pos', default=1, type=float, help='ratio')
parser.add_argument('--load', default=None, help='trained model')
parser.add_argument('--iid', default=411, type=int, help='trained model')
parser.add_argument('--dataset', default='all',  help='dataset to use')
parser.add_argument('--mids', default=False,  type=bool, help='use mids or not')
parser.add_argument('--aug', default=False,  type=bool, help='use augmentation')
parser.add_argument('--epochs', default=80,  type=int, help='num of epochs')
args = parser.parse_args()
print(args)
#"""weights/sealions_95k.pth"""
cfg = (v1, v2)[args.version == 'v2']

#model = Network()
print("building ssd")
model = build_ssd('train', args.dim, args.num_classes)
print("building ssd done")
vgg_weights = torch.load(args.save_folder + args.basenet)
print('Loading base network...')
model.vgg.load_state_dict(vgg_weights)

def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


print('Initializing weights...')
# initialize newly added layers' weights with xavier method
model.extras.apply(weights_init)
model.loc.apply(weights_init)
model.conf.apply(weights_init)  

if args.cuda:
    model.cuda()
if args.load is not None:
    if args.cuda: 
        model.load_my_state_dict(torch.load(args.load))
    else: 
        model.load_my_state_dict(torch.load(args.load, map_location=lambda storage, loc: storage))
        
#criterion = MultiBoxLoss(args.num_classes, args.th, True, 0, True, args.neg_pos, 0.5, False, alpha=args.alpha, neg_thresh=args.neg_th, mids=args.mids)
criterion = HeatMapLoss(args.num_classes, args.th, True, 0, True, args.neg_pos, 
                        0.5, False, alpha=args.alpha, neg_thresh=args.neg_th, 
                        mids=args.mids, dim=args.dim, fmap=model.fmap)


from polarbear import *

ds = DataSource()
all = ds.dataset(args.dataset)
aimg,_ = all.aimg(args.iid)
aimg = aimg.fpups().oneclass().scale(3).setbox(40)
#aimg.ann.dbox(5)
print("unique", aimg.ann.unique())
print("max_th", aimg.ann.max_th().max())
aimg, vimg = aimg.bcrop()

print("train count",aimg.count)
print("val count",vimg.count)
aimg.plot(save="weights/trainer.png")
vimg.plot(save="weights/validator.png")
train_dataset = SLDetection(aimg, tile=args.dim, st=args.dim-300, fcount=5, aug=args.aug)
val_dataset = SLDetection(vimg, tile=args.dim, st=args.dim-300, fcount=5, aug=args.aug)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

trainer = ModuleTrainer(model)

chk = ModelCheckpoint(directory="weights", monitor="val_loss", filename='trainer_'+str(args.iid)+'_{epoch}_{loss}.pth.tar', verbose=1)

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

