import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random as r


class Agent:
    def __init__(self, inp):
        self.hypernet = np.ndarray((inp, 1))
        self.subnets = []

    # Unseen experience. Create a new subnet for that, and grow a new neuron
    # that will activate the said subnet
    def accommodate(self):
        oldOut = self.hypernet[:-1]
        newOut = oldOut.append(r.random())

    def assimilate(self):
        pass

    # Try to find inconsistencies in models by trying to predict an imaginable
    # input with different subnets that are supposed to get the same output
    def imagine(self):
        pass

    def forward(self, inpVec):
        np.dot(inpVec, self.hypernet)
        pass

# off-policy vs on-policy
# stochastic vs deterministic policies
# categorical vs gaussian-diagonal (multivariate normal distribution)
# 

# Model for world-understanding and the policy for decision making.
# Self-supervised continuous pre-training. Pre-train the model using the input from
# the policy. Pre-train new network using updated policy for better representation search

# Model. Tries to minimize the state prediction error
class selfSuperModel(nn.Module):
    def __init__(self, env):
        self.affine1 = nn.Linear(len(env.observation_space.sample()), 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, len(env.observation_space.sample()))
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


# Policy. Tries to maximize the reward using model
class selfSuperPloicy(nn.Module):
    def __init__(self, env):
        self.affine1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, len(env.action_space.sample()))
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


# Use encoder ouput as critic's input
class AutoencoderModel:
    def __init__(self, obs_dim, act_dim):
        # has to be less than output and less than input
        # How deep? How to determine the depth of the encoder-decoder thingy?
        # What is input dimention? Only current obs or memory as well?
        encoder_out = None
        self.encoder = nn.Sequential( nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim))

        self.critic = nn.Sequential( nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim))



