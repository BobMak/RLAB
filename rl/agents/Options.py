import gym
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import copy
import numpy as np

from agents.PPO import PPO


class Options(nn.Module):
    def __init__(self, inpDim,
                 stateDim,
                 outDim,
                 pickerDim,
                 pickerDepth,
                 nOptions,
                 optDim,
                 optDepth,
                 device="gpu"):

        self.lState = nn.LSTM(inpDim+outDim, stateDim)
        self.aState = nn.ReLU()

        self.options = []

        # init our options
        for _ in range(nOptions):
            policy = PPO(stateDim, optDim, outDim,
                         clip_ratio=0.2,
                         isContinuous=False,
                         useLSTM=False,
                         nLayers=optDepth,
                         usewandb=False,
                         device=device)
            self.options.append(policy)

        # init our option picker
        layers = [nn.Linear(stateDim, pickerDim),
                  nn.ReLU()]
        for _ in range(max(0, pickerDepth-2)):
            layers.append(nn.Linear(pickerDim, pickerDim))
            layers.append(nn.ReLU())
        self.optionPicker = nn.Sequential(
            *layers,
            nn.Linear(pickerDim, nOptions),
            nn.Sigmoid()
        )
        # keep history of used options to update on backward pass
        self.polIdxs = []
        
    def forward(self, x):
        pass

