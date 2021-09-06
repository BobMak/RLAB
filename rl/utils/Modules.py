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


# encode: gets image and direction data from the raw observation, turns it into a flat tensor
# decode: turns the flat tensor into a tuple of image and direction tensors inside the forward pass
class MultiDataEncoderDecoder:
    def __init__(self, sample:[torch.tensor]):
        self.shapes = [s.shape for s in sample]

    def encode(self, inputs: []):
        return torch.cat([s.flatten() for s in inputs])

    def decode(self, x: torch.tensor):
        original = []
        start_idx = 0
        for s in self.shapes:
            s_len = 1
            for dim in s:
                s_len *= dim
            original.append( x[start_idx : start_idx + s_len] )
            start_idx += s_len
        return original


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
