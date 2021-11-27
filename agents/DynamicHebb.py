import gym
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import copy
import numpy as np


class Neuron():
    def __init__(self, parents):
        self.parents = parents
        self.children = []

    def fire(self):
        do shit


"""
learn feature filters with dynamic hebbian networks, adding neurons until
 
"""
class DynaHebb():
    def __init__(self, inputDim):
        self.hebb = []

