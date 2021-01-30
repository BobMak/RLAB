import numpy as np
from scipy.constants import k
from math import log, log2, log10
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.flatten(x)


class MyFun(torch.autograd.Function):
    def forward(self, inp):
        return inp

    def backward(self, grad_out):
        grad_input = grad_out.clone()
        print('Custom backward called!')
        return grad_input


class NormalOutput(nn.Module):
    def __init__(self, inp, out, activation=nn.Sigmoid):
        super().__init__()
        self.m = nn.Linear(inp, out)
        self.v = nn.Linear(inp, out)
        # self.parallel = nn.Parallel(self.w1, self.w2)
        self.act1 = nn.Sigmoid()
        self.act2 = nn.ReLU()  # no negative outputs for variance

    def forward(self, inputs):
        # res1 = self.act1(self.m(inputs))  # self.act1(
        mout = self.m(inputs)
        vout = torch.clamp(self.v(inputs), min=0.001)
        return mout, vout


class ActorCriticOutput(nn.Module):
    def __init__(self, inp, out, isContinuous=False, activation=nn.Sigmoid):
        super().__init__()
        self.critic = nn.Linear(inp, 1)
        self.isContinuous = isContinuous
        if isContinuous:
            self.actor = NormalOutput(inp, out, activation=activation)
        else:
            self.actor = nn.Sequential(
                nn.Linear(inp, out),
                activation())

    def forward(self, inputs):
        value  = self.critic(inputs)
        action = self.actor (inputs)
        return action, value
