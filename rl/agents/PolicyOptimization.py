"""
Policy Gradients implementation
for continuous or discreet action spaces

"""
import wandb
import torch.nn as nn
import torch.optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from utils.Modules import NormalOutput
from agents.Agent import Agent


class PolicyGradients(Agent):
    def __init__(self, inp, hid, out, isContinuous=False, useLSTM=False, nLayers=1, usewandb=False, env=None):
        super(PolicyGradients, self).__init__(inp, hid, out, useLSTM, nLayers, usewandb, env)
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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)

        self.log_probs     = []
        self.avg_rewards   = 0

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

    def getActionDistribution(self, x):
        distribution_params = self.forward(x)
        if self.isContinuous:
            action_distribution = Normal(*distribution_params)
        else:
            action_distribution = Categorical(logits=distribution_params)
        return action_distribution

    # gradient of one trajectory
    def backward(self):
        self.optimizer.zero_grad()
        logProbs = torch.stack(self.log_probs)
        # Compute an advantage
        r = self.train_rewards #- self.avgRewards
        if self.use_wandb:
            wandb.log ({ "awgReward": r.mean() } )
        grad = -(logProbs * r).mean()
        grad.backward()
        self.optimizer.step()
        print("train reward", self.train_rewards.mean(), "grad", grad, "advtg", r.mean())
        self.avg_rewards = self.train_rewards.mean()
        # Reset episode buffer
        self.train_rewards = torch.tensor([]).to(self.device)
        self.trainActions = torch.tensor([]).to(self.device)
        self.trainStates  = torch.tensor([]).to(self.device)
        self.log_probs     = []
        if self.use_lstm:
            self.clearLSTMState()

    def __str__(self):
        return f"PG_{super().__str__()}" + ("C" if self.isContinuous else "D")


