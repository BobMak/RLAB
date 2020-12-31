import numpy as np
from scipy.constants import k
from math import log, log2, log10
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

import pickle


class NormalOutput(nn.Module):
    def __init__(self, inp, out, activation=nn.Tanh):
        super().__init__()
        self.w1 = nn.Linear(inp, out)
        self.w2 = nn.Linear(inp, out)
        self.act1 = activation()
        self.act2 = activation()

    def forward(self, inputs):
        res1 = self.act1(self.w1(inputs))
        res2 = self.act2(self.w2(inputs))
        return res1, res2


class ActorCriticOutput(nn.Module):
    def __init__(self, inp, out, isContinuous=False, activation=nn.Tanh):
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


class EnvNormalizer():
    def __init__(self, env):
        self.mean = (env.observation_space.high + env.observation_space.low) / 2
        self.var  = (env.observation_space.high - env.observation_space.low) / 2

    def __call__(self, inp):
        return (inp - self.mean) * self.var


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.flatten(x)


def save_model(model):
    with open(str(model)+".pkl", 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


def load_model(model):
    with open(str(model)+".pkl", 'rb') as f:
        model = pickle.load(f)
    return model

def entropy( probabilities:[], base=None ):
    logFu = { None:log, 2: log2, 10:log10 }[base]
    return -sum( [ p * logFu(p) for p in probabilities ] )


