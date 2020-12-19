import gym
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import copy
import numpy as np


class Layer(nn.Module):
    def __init__(self, inpDim, outLayers, initialDensity=1, dropoff=0.5):
        """
        a sparse layer with connections to further layers in the network
        number or connections drops off with each next layer, with each next
        being dropoff * current number
        :param inpDim:
        :param outLayers: all remaining layers in the network that will be connected
        :param initialDensity:
        :param dropoff:
        """
        connections = nn.Linear(inpDim, outLayers[0].inpDim)
        prune.random_unstructured(connections, name="weight", amount=1-initialDensity)
        self.toLayers = [
            connections,
            nn.ReLU()
        ]
        self.inpDim = inpDim
        for n, layer in enumerate( outLayers[1:]):
            connections = nn.Linear(inpDim, layer.inpDim)
            prune.random_unstructured(connections, name="weight", amount=1-dropoff*n)
            self.toLayers.append( connections )
            self.toLayers.append( nn.ReLU() )


class PLSN(nn.Module):
    def __init__(self, inp, out):
        # starting state
        self.layers = [
            nn.Linear(inp, 10),
            nn.ReLU(),
            nn.Linear(10, out),
            nn.ReLU(),
        ]