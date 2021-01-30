import gym
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import copy
import numpy as np


class PoliPoli(nn.Module):
    def __init__(self, inpDim, stateDim, outDim, nPolcies, polDim, polDepth):

        self.lState = nn.LSTM(inpDim+outDim, stateDim)
        self.aState = nn.ReLU()

        self.policies = nn.ModuleList()
        # init n policies
        for n in range(nPolcies):
            policy = []
            policy.append(nn.Linear(stateDim, polDim))
            policy.append(nn.ReLU())
            for l in range(max(1, polDepth-2)):
                policy.append(nn.Linear(polDim, polDim))
                policy.append(nn.ReLU())
            policy.append(nn.Linear(polDim, outDim))
            policy.append(nn.ReLU())
            self.policies.append(nn.Sequential(*policy))
        # model picker assigns value of using a model in current state
        self.lPolicyPicker = nn.Linear(stateDim, nPolcies)
        self.aPolicyPicker = nn.ReLU()
        # keep history of used policies to update on backward pass
        self.polIdxs = []
        
    def forward(self, x):


