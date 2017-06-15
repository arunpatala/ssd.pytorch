from torchsample.callbacks import Callback
import torch
from polarbear import *
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import gpustat
import gc


class PosNeg(Callback):
    """
    Abstract base class used to build new callbacks.
    """

    def __init__(self, net, ct, val):
        self.net = net
        self.ct = ct
        self.softmax = nn.Softmax()
        self.val = val
        print("ct",ct)

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        self.ct.set_phase("train")

    def on_epoch_end(self, epoch, logs=None):
        #val_detect(self.net, self.ct, self.val)
        self.ct.set_phase("val")

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end2(self, batch, logs=None):
        gpustat.print_gpustat(json=False)
    
    def on_batch_end(self, batch, logs=None):
        pass
        
    def on_train_begin(self, logs=None):
        self.ct.set_phase('train')

    def on_train_end(self, logs=None):
        print("training end")
        self.ct.set_phase('val')
        
def img2Image(img):
    _,C,H,W, = img.size()
    rgb_means = (104, 117, 123)
    new_img = img[0].cpu().clone().numpy().transpose((1,2,0))
    new_img += rgb_means
    return Image.fromarray(new_img.astype('uint8'))
