import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.DNNAgent import DNNAgent


class DQLearn(DNNAgent):
    def __init__(self, inp, hid, action_space, useLSTM=False, nLayers=1, usewandb=False, epsilon=0.9, buffLenght=1):
        super(DQLearn, self).__init__(inp + action_space, hid, 1, useLSTM, nLayers, usewandb)
        self.action_space = action_space
        self.expRewards  = []
        self.epsilon     = epsilon
        self.criterion   = nn.MSELoss()
        self.buffLenght  = buffLenght
        self.buffCount   = 0
        self.trainStateActions = []

    def forward(self, obs):
        # Predict state value based on current action and observation
        expRewards, actions = [], []
        for n in range(self.action_space):
            act = [0 for x in range(self.action_space)]
            act[n] = 1
            stateAction = torch.cat( [torch.tensor(act), obs] )
            expRewards.append(self.model( stateAction ))
            self.trainStateActions.append(stateAction)
            actions.append(act)
        # pick action
        if np.random.rand() < self.epsilon:
            # use max Q value
            actionIdx = np.argmax(expRewards)
        else:
            # explore
            actionIdx = np.random.randint(0, len(actions))
        self.expRewards.append(expRewards[actionIdx])
        # action = actions[actionIdx]
        return actionIdx

    def getAction(self, obs):
        return self.forward(obs)

    def backward(self):
        pred = torch.stack(self.expRewards)
        real = self.trainRewards
        # print(pred.dim(), real.dim())
        grad = ((pred - real)**2).mean()
        grad.backward()
        #self.criterion(pred, real)
        self.p_optimizer.step()
        self.p_optimizer.zero_grad()
        if self.use_wandb:
            wandb.log ({ "awgReward": real.mean() } )
        print("train reward", self.trainRewards.mean())
        # self.avgRewards = self.trainRewards.mean()
        self.buffCount += 1
        if self.buffCount >= self.buffLenght:
            self.buffCount = 0
            # Reset episode buffer
            self.trainRewards      = torch.tensor([]).to(self.device)
            self.trainStateActions = []
            self.expRewards        = []
        if self.use_lstm:
            self.clearLSTMState()

    def __str__(self):
        return f"DQN_h{super().__str__()}"