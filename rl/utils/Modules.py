import numpy as np
from scipy.constants import k
from math import log, log2, log10
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


def oneHot( x:int, maximum:int):
    assert x <= maximum, "int to onehot transform error: x > max"
    out = np.zeros(maximum)
    out[x-1] = 1
    return out


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
    def __init__(self, inp, out, activation=nn.Tanh):
        super().__init__()
        self.m = nn.Linear(inp, out)
        log_std = -0.5 * np.ones(out, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))  # , requires_grad=True
        # self.parallel = nn.Parallel(self.w1, self.w2)
        self.act1 = activation

    def forward(self, inputs):
        # res1 = self.act1(self.m(inputs))  # self.act1(
        mout = self.m(inputs)
        vout = torch.exp(self.log_std)
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
