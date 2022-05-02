"""
Policy Gradients implementation
for continuous or discreet action spaces

"""
import numpy as np
import wandb
import torch.nn as nn
import torch.optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from utils.Modules import NormalOutput
from agents.Agent import Agent


class PolicyGradients(Agent):
    def __init__(self, inp, hid, out, isContinuous=False, useLSTM=False, nLayers=1, usewandb=False, env=None, dev="cpu", action_arr=False):
        super(PolicyGradients, self).__init__(inp, hid, out, useLSTM, nLayers, usewandb, env, dev)
        self.isContinuous = isContinuous
        # replace the discreet output with a continuous Gaussian output
        if isContinuous:
            policy = [*self.model[:-1]]
            self.model = nn.Sequential(
                *policy,
                NormalOutput(hid, out, activation=nn.Tanh)
            ).to(self.device)
        else:
            policy = [*self.model]
            self.model = nn.Sequential(
                *policy,
                # nn.Sigmoid()
            ).to(self.device)

        critic = [nn.Linear(inp + out * (2 if isContinuous else 1), hid)]
        for layer in range(nLayers):
            critic.extend([nn.Linear(hid, hid), nn.ReLU()])
        self.critic = nn.Sequential(
            *critic[:-2],
            nn.Linear(hid, 1),
            nn.Tanh()
        )
        self.p_optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.c_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.log_probs     = []
        self.avg_rewards   = torch.tensor([0.0])

        self.action_arr = action_arr

    def getAction(self, x):
        action_distribution = self.getActionDistribution(x)
        sampled_action = action_distribution.sample()  # .item()
        # save the log likelihood of taking that action for backprop
        logProb = action_distribution.log_prob(sampled_action)
        self.log_probs.append(logProb)
        self.train_actions.append(sampled_action)
        if not self.isContinuous:
            sampled_action = sampled_action.item()
        if self.action_arr:
            return np.array([sampled_action])
        else:
            return sampled_action


    def getActionDistribution(self, x):
        distribution_params = self.forward(x)
        if self.isContinuous:
            action_distribution = Normal(*distribution_params)
        else:
            action_distribution = Categorical(logits=distribution_params)
        return action_distribution

    def getExpectedValues(self, x):
        actions = self.forward(x)
        if self.isContinuous:
            means = actions[0]
            std = actions[1].repeat(*means.shape[:-1], 1)
            actions = torch.cat([ means, std ], dim=len(means.shape) - 1)
        state_action = torch.cat([x, actions],  dim=len(actions.shape)-1)
        value = self.critic.forward(state_action)  #, dim=1
        return value*2

    # gradient of one trajectory
    def backward(self):
        self.p_optimizer.zero_grad()
        logProbs = torch.stack(self.log_probs)
        # Compute an advantage
        r = self.train_rewards #- self.avgRewards
        if self.use_wandb:
            wandb.log ({ "awgReward": r.mean() } )
        grad = -(logProbs * r).mean()
        grad.backward()
        self.p_optimizer.step()
        print("train reward", self.train_rewards.mean(), "grad", grad, "advtg", r.mean())
        self.avg_rewards = self.train_rewards.mean()
        # Reset episode buffer
        self.train_rewards = torch.tensor([]).to(self.device)
        self.train_states  = torch.tensor([]).to(self.device)
        self.train_actions = []
        self.log_probs     = []
        if self.use_lstm:
            self.clearLSTMState()

    def __str__(self):
        return f"PG_{super().__str__()}" + ("C" if self.isContinuous else "D")


