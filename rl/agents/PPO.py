"""
PPO implementation
"""

import torch
import wandb

from agents.PolicyOptimization import PolicyGradients


class PPO(PolicyGradients):
    def __init__(self, inp, hid, out, epsilon=0.01, isContinuous=False, useLSTM=False, nLayers=1, usewandb=False):
        super().__init__(inp, hid, out, isContinuous=isContinuous, useLSTM=useLSTM, nLayers=nLayers, usewandb=usewandb)
        self.epsilon = epsilon

    def clip(self):
        # Compute an advantage
        r = self.trainRewards  # - self.avgRewards
        if self.usewandb:
            wandb.log({"awgReward": r.mean()})
        logProbs = torch.stack(self.logProbs)
        # clip it
        torch.min( logProbs, torch.clip(logProbs, 1-self.epsilon, 1+self.epsilon) )
        grad = -(logProbs * r).mean()
        grad.backward()

    # gradient of one trajectory
    def backward(self):

        self.optimizer.step()
        self.optimizer.zero_grad()
        print("train reward", self.trainRewards.mean(), "grad", grad, "advtg", r)
        self.avgRewards = self.trainRewards.mean()
        # Reset episode buffer
        self.trainRewards = torch.tensor([]).to(self.device)
        self.trainActions = torch.tensor([]).to(self.device)
        self.trainStates  = torch.tensor([]).to(self.device)
        self.logProbs = []
        if self.useLSTM:
            self.clearLSTMState()

    def __str__(self):
        return f"PPO_h + {super().__str__()[4:]}"