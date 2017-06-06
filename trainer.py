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
from data import SLDetection, SLDetections
from layers.modules import MultiBoxLoss
from ssd import build_ssd
from cb import PosNeg

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
parser.add_argument('--dim', default=900, type=int, help='dimension of input to model')
parser.add_argument('--alpha', default=1, type=float, help='multibox alpha for loss')
parser.add_argument('--th', default=0.5, type=float, help='threshold')
parser.add_argument('--neg_th', default=0.2, type=float, help='neg threshold')
parser.add_argument('--neg_pos', default=2, type=float, help='ratio')
parser.add_argument('--load', default=None, help='trained model')
parser.add_argument('--iid', default=None, type=int, help='iid to train on')
parser.add_argument('--mids', default=True,  type=bool, help='use mids or not')
parser.add_argument('--exp', default="day1", help='exp name')
parser.add_argument('--epochs', default=80, type=int, help='exp name')
parser.add_argument('--fcount', default=1, type=int, help='filter count')
parser.add_argument('--aug', default=True, type=bool, help='use augmentations')
parser.add_argument('--limit', default=None, type=int, help='limit the dataset')
args = parser.parse_args()
print(args)
cfg = (v1, v2)[args.version == 'v2']
torch.set_default_tensor_type('torch.FloatTensor')
#print(torch.zeros(1))
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
        model.load_state_dict(torch.load(args.load))
    else: 
        model.load_state_dict(torch.load(args.load, map_location=lambda storage, loc: storage))

criterion = MultiBoxLoss(args.num_classes, args.th, True, 0, True, args.neg_pos, 0.5, False, alpha=args.alpha, neg_thresh=args.neg_th, mids=args.mids)


from polarbear import *

ds = DataSource()

train_dataset = SLDetections(ds.train, tile=args.dim, st=args.dim-100, fcount=args.fcount, aug=args.aug, limit=args.limit)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

val_dataset = SLDetections(ds.val, tile=args.dim, st=args.dim-100, fcount=args.fcount, limit=args.limit)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

trainer = ModuleTrainer(model)

chk = ModelCheckpoint(directory="weights", monitor="loss", filename=args.exp+'_{epoch}_{loss}.pth.tar', verbose=1)
chkt = ModelCheckpoint(directory="weights", monitor="loss", filename='trainer.pth.tar', verbose=1)

optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
trainer.compile(loss=criterion,
                optimizer=optimizer,
                #optimizer='adadelta',
                #regularizers=regularizers,
                #constraints=constraints,
                #initializers=initializers,
                #metrics=metrics, 
                callbacks=[chk, PosNeg(model, criterion, val_dataset), chkt])

print("trainer compilation done")
#print(torch.zeros(1))
if args.cuda:
    trainer.fit_loader(train_loader, nb_epoch=args.epochs, verbose=1, cuda_device=0)
else: trainer.fit_loader(train_loade, nb_epoch=args.epochs, verbose=1)

