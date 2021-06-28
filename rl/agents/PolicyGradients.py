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
from agents.DNNAgent import DNNAgent


class PolicyGradients(DNNAgent):
    def __init__(self, inp, hid, out,
                 isContinuous=False,
                 useLSTM=False,
                 nLayers=1,
                 usewandb=False,
                 env=None,
                 device="cpu"):
        super(PolicyGradients, self).__init__(inp, hid, out, useLSTM, nLayers, usewandb, env, device)
        self.isContinuous = isContinuous
        # replace the discreet output with a continuous Gaussian output
        if isContinuous:
            policy = [*self.model[:-2]]
            self.model = nn.Sequential(
                *policy,
                NormalOutput(hid, out, activation=nn.Tanh)
            ).to(self.device)
            self._actionSample =        self._actionSampleCont
            self._actionDistribution =  self._actionDistributionCont
        else:
            policy = [*self.model]
            self.model = nn.Sequential(
                *policy,
                nn.Sigmoid()
            ).to(self.device)
            self._actionSample =        self._actionSampleDisc
            self._actionDistribution =  self._actionDistributionDisc

        critic = [nn.Linear(inp + out, hid)]
        for layer in range(nLayers):
            critic.extend([nn.Linear(hid, hid), nn.ReLU()])
        self.critic = nn.Sequential(
            *critic[:-2],
            nn.Linear(hid, 1),
            nn.Tanh()
        )
        self.p_optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        self.c_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.log_probs     = []
        self.neg_log_probs = []
        self.avg_reward   = torch.tensor([0.0])

    # evaluate on action, called only when on environment interaction
    def getAction(self, x):
        action_distribution = self.getActionDistribution(x.squeeze())
        sampled_action = action_distribution.sample()  # .item()
        # save the log likelihood of taking that action for backprop
        logProb = action_distribution.log_prob(sampled_action)
        self.log_probs.append(logProb)
        # is this right?
        self.neg_log_probs.append(torch.log(torch.ones_like(logProb) - torch.exp(logProb)))
        self.train_actions.append(sampled_action)
        return self._actionSample(sampled_action)

    def _actionSample(self, x):
        raise NotImplemented()  # Is assigned in init, either a _getActionCont or _getActionDisc

    def _actionSampleCont(self, sampled_action):
        return np.array(sampled_action)

    def _actionSampleDisc(self, sampled_action):
        return np.array(sampled_action.item())

    def getActionDistribution(self, x):
        distribution_params = self.forward(x)
        return self._actionDistribution(distribution_params)

    # this must be called during the backprop phase instead of getActionDistribution to ensure distributions are
    # computed in order when using LSTM
    def getAllActionDistributions(self, x):
        if self.use_lstm:
            episode_start_idx = 0
            distributions = []
            for episode_end_idx in self.episode_breaks:
                for i in range(episode_start_idx, episode_end_idx):
                    distributions.append(self.forward(x[i]))
                self.clearLSTMState()
                episode_start_idx = episode_end_idx
            distribution_params = torch.stack(distributions)
        else:
            distribution_params = self.forward(x)
        return self._actionDistribution(distribution_params)

    def _actionDistribution(self, x):
        raise NotImplemented()  # Is assigned in init, either a _actionDistributionDisc or _actionDistributionCont

    def _actionDistributionCont(self, distribution_params):
        return Normal(*distribution_params)

    def _actionDistributionDisc(self, distribution_params):
        return Categorical(logits=distribution_params)

    def getAllExpectedvalues(self, x):
        actions = self.train_actions
        state_action = torch.cat([x, torch.stack(actions)],  dim=1)
        value = self.critic.forward(state_action)  #, dim=1
        return value

    # one gradient ascent step
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
        self.avg_reward = self.train_rewards.mean()
        # Reset episode buffer
        self.train_rewards = torch.tensor([]).to(self.device)
        self.train_states  = torch.tensor([]).to(self.device)
        self.train_actions = []
        self.log_probs     = []
        if self.use_lstm:
            self.clearLSTMState()
        return self.avg_reward

    def __str__(self):
        return f"PG_{super().__str__()}" + ("C" if self.isContinuous else "D")


