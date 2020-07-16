import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class NNAPolicy:
    def __init__(self, action_space, observations):
        self.action_space = action_space
        self.model = nn.Sequential(
            nn.Linear(observations.shape[0]+action_space.n, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
            nn.Sigmoid())
        self.model.cuda()
        self.expReward = 0
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    def forward(self, obs):
        # Predict future based on current action and observation
        expRewards, actions = [], []
        for n in range(self.action_space.n):
            act = [0 for x in range(self.action_space.n)]
            act[n] = 1
            expRewards.append(self.model(torch.tensor(act.extend(obs))))
            actions.append(act)
        # Q value
        self.expReward = max(expRewards)
        action = actions[np.argmax(expRewards)]
        return action
    def backward(self, newReward):
        self.optimizer.zero_grad()
        pred = torch.tensor([self.expReward])
        real = torch.tensor([newReward])
        print(pred.dim(), real.dim())
        self.criterion(pred, real)
        self.optimizer.step()
    def act(self, ob, rew, done):
        act = self.forward(ob)
        self.backward(rew)
        return act