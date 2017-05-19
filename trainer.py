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

import os
from torchvision import datasets

from data import VOCroot, v2, v1, AnnotationTransform, VOCDetection, detection_collate, BaseTransform, AnnTensorTransform
from layers.modules import MultiBoxLoss
from ssd import build_ssd

import argparse

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--version', default='v2', help='conv11_2(v2) or pool6(v1) as last layer')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--iterations', default=120000, type=int, help='Number of training epochs')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
parser.add_argument('--visdom', default=False, type=bool, help='Use visdom to for loss visualization')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--num_classes', default=21, help='num of classes')
parser.add_argument('--dim', default=300, help='dimension of input to model')
args = parser.parse_args()

cfg = (v1, v2)[args.version == 'v2']

ssd_dim = 300  # only support 300 now
rgb_means = (104, 117, 123)  # only support voc now

#model = Network()

model = build_ssd('train', args.dim, args.num_classes)
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

criterion = MultiBoxLoss(args.num_classes, 0.5, True, 0, True, 3, 0.5, False)

trainer = ModuleTrainer(model)


trainer.compile(loss=criterion,
                optimizer='adadelta')
                #regularizers=regularizers,
                #constraints=constraints,
                #initializers=initializers,
                #metrics=metrics, 
                #callbacks=callbacks)



cfg = (v1, v2)[args.version == 'v2']
train_sets = [('2007', 'trainval'), ('2012', 'trainval')]

train_dataset = VOCDetection(VOCroot, train_sets,  AnnTensorTransform())
#train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

val_dataset = VOCDetection(VOCroot, [('2007', 'test')], BaseTransform(
        ssd_dim, rgb_means), AnnTensorTransform())
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)


trainer.fit_loader(train_loader, nb_epoch=20, verbose=1)

