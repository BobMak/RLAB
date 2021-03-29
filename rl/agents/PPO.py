"""
PPO implementation
"""

import torch
import wandb
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from agents.PolicyOptimization import PolicyGradients


class PPO(PolicyGradients):
    def __init__(self, inp, hid, out, epsilon=0.01, isContinuous=False, useLSTM=False, nLayers=1, usewandb=False, env=None):
        super().__init__(inp, hid, out, isContinuous, useLSTM, nLayers, usewandb, env)
        self.epsilon = epsilon
        self.oldLogProbs = []

    def getAction(self, x):
        action = self.forward(x)
        if self.isContinuous:
            action_distribution = Normal(*action)
            sampled_action = action_distribution.sample()
        else:
            action_distribution = Categorical(logits=action)
            sampled_action = action_distribution.sample()  # .item()
        # save the log likelihood of taking that action for backprop
        logProb = action_distribution.log_prob(sampled_action)
        self.log_probs.append(logProb)
        if not self.isContinuous:
            sampled_action = sampled_action.item()
        return sampled_action

    # gradient of one trajectory
    def backward(self):
        self.optimizer.zero_grad()
        # Compute an advantage
        r = self.train_rewards - self.avg_rewards
        if self.use_wandb:
            wandb.log({"awgReward": r.mean()})
        logProbs = torch.stack(self.log_probs)
        # clip it
        grad = -(logProbs * r).mean()
        if r.mean() > 0:
            grad = grad.clamp(max=1 + self.epsilon)
        else:
            grad = grad.clamp(min=1 - self.epsilon)
        grad.backward()
        self.optimizer.step()
        print("train reward", self.train_rewards.mean(), "grad", grad, "advtg", r.mean())
        self.avg_rewards = self.train_rewards.mean()
        # Reset episode buffer
        self.train_rewards = torch.tensor([]).to(self.device)
        self.trainActions = torch.tensor([]).to(self.device)
        self.trainStates  = torch.tensor([]).to(self.device)
        self.oldLogProbs = []
        self.log_probs    = []
        if self.use_lstm:
            self.clearLSTMState()

    def __str__(self):
        return f"PPO_{super().__str__()[4:]}"