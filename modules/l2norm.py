import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.init as init

class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.scale = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels,1))
        init.constant(self.weight, scale)
        # self.bias = nn.Parameter(torch.Tensor(n_channels))
        #self.gradWeight = nn.Parameter(torch.Tensor(self.n_channels))

    def forward(self, input):
        norm = input.pow(2).sum(1).sqrt()
        out = self.scale * input / (norm.expand_as(input) + self.eps)
        return out
