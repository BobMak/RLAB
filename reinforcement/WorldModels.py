import gym
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class WorldModels(nn.Module):
    def __init__(self, inpdim, hiddim, outdim):
        self.vision = nn.Sequential(
            nn.Linear(inpdim, hiddim),
            nn.ReLU(),
            nn.Linear(hiddim, hiddim),
            nn.Sigmoid()
        )
        self.memory = nn.Sequential(
            nn.LSTM(hiddim, hiddim),
            nn.Sigmoid()
        )
        self.controller = nn.Sequential(
            nn.Linear(hiddim, hiddim),
            nn.ReLU(),
            nn.Linear(hiddim, outdim),
            nn.Sigmoid()
        )



    def forward(self, inpt):
        vis = self.vision.forward(inpt)
        mem = self.memory(vis)
        act = self.controller()