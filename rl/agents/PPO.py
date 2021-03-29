"""
PPO implementation
"""

import torch
import wandb

from agents.PolicyOptimization import PolicyGradients


class PPO(PolicyGradients):
    def __init__(self, inp, hid, out, epsilon=0.01, isContinuous=False, useLSTM=False, nLayers=1, usewandb=False, env=None):
        super().__init__(inp, hid, out, isContinuous, useLSTM, nLayers, usewandb, env)
        self.epsilon = epsilon

    # gradient of one trajectory
    def backward(self):
        self.optimizer.zero_grad()
        # Compute an advantage
        r = self.trainRewards  # - self.avgRewards
        if self.usewandb:
            wandb.log({"awgReward": r.mean()})
        logProbs = torch.stack(self.logProbs)
        # clip it
        grad = -(logProbs * r).mean()
        grad = grad.clamp(min=1 - self.epsilon)
        grad.backward()
        self.optimizer.step()
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
        return f"PPO_h{super().__str__()[4:]}"