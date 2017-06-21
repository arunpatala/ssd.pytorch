import numpy as np
import torch

from PIL import Image
from argparse import ArgumentParser

from torch.optim import SGD, Adam, SGD
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import os

class Trainer(object):
    
    def __init__(self, model, criterion, dpath, cuda=False, lr=0.0001, momentum=0.9, weight_decay=5e-4):
        self.model = model
        self.criterion = criterion
        self.cuda = cuda
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dpath = dpath
        self.filename = dpath + os.sep + 'model_epoch_{epoch}_{step}_{phase}_loss_{loss}.pth'

        
    def train_val(self, train_loader, val_loader, epochs=1):
        for epoch in range(1,epochs+1):
            if train_loader is not None:
                self.train_epoch(train_loader, True,epoch=epoch)
            if val_loader is not None:
                self.train_epoch(val_loader, False, epoch=epoch)
            
    
    def train_epoch(self, loader, train=True, epoch=1):
        model = self.model
        criterion = self.criterion
        pbar = tqdm(enumerate(loader),total=len(loader))
        lr = 0
        mode = "train"
        if train: 
            model.train()
            lr = self.lr
            pbar.set_description('TRAIN %i' % epoch)

        else: 
            mode = "val"
            model.eval()
            pbar.set_description('VAL %i' % epoch)
            lr = 0
        self.criterion.set_phase(mode)
            
        epoch_loss = []
        
        
        optimizer = SGD(model.parameters(), lr=lr,
                      momentum=self.momentum, weight_decay=self.weight_decay)

        for step, (images, labels) in pbar:
            if self.cuda:
                images = images.cuda()
                labels = labels.cuda()

            inputs = Variable(images)
            targets = Variable(labels)
            outputs = model(inputs)
            
            optimizer.zero_grad()
            loss = criterion(outputs, targets[:, 0])
            loss.backward()
            if train:
                optimizer.step()

            epoch_loss.append(loss.cpu().data[0])
            average = sum(epoch_loss) / len(epoch_loss)
            pbar.set_postfix(loss=average)
            if step % 300 == 0: 
                self.save(model, epoch, mode, average, step)

            #pbar.update(average)
            
        
        average = sum(epoch_loss) / len(epoch_loss)
        #print(epoch_loss)
        print('loss: {average} (epoch: {epoch}, phase: {phase})'.format(average=average, epoch=epoch, phase=mode))
        self.save(model, epoch, mode, average, "END")
        
    def save(self, model, epoch, mode, average, step):
        f = self.filename.format(epoch=epoch, phase=mode, loss= "%.3f" % average, step =step)
        torch.save(model.state_dict(), f)
        print("model saved to "+f)
        #model.save_state_dict(f+".tar")
        

