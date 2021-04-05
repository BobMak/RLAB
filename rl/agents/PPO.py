"""
PPO implementation
"""

import torch
import wandb
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from agents.PolicyOptimization import PolicyGradients


class PPO(PolicyGradients):
    def __init__(self, inp, hid, out, clip_ratio=0.2, isContinuous=False, useLSTM=False, nLayers=1, usewandb=False, env=None):
        super().__init__(inp, hid, out, isContinuous, useLSTM, nLayers, usewandb, env)
        self.clip_ratio = clip_ratio

    # gradient of one trajectory
    def backward(self):
        actions = torch.stack(self.train_actions)
        # Compute an advantage
        pred_values = self.getExpectedvalues(self.train_states).detach()
        critic_loss = torch.nn.MSELoss()(pred_values, self.train_rewards)
        r = self.train_rewards - pred_values
        r = (r - r.mean()) / (r.std() + 1e-10).detach()
        if self.use_wandb:
            wandb.log({"avgReward": self.train_rewards.mean()})
            wandb.log({"avgAdvantage": r.mean()})
            wandb.log({"criticLoss": critic_loss.mean()})
        old_log_probs = torch.stack(self.log_probs).detach()
        # update actor
        for _ in range(80):
            self.p_optimizer.zero_grad()
            # compute log_probs after the update
            action_distributions = self.getActionDistribution(self.train_states)
            log_probs = action_distributions.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            # clip it
            surr1 = torch.clamp(ratio, min=1 - self.clip_ratio, max=1 + self.clip_ratio) * r
            surr2 = ratio * r
            grad = torch.min( surr1, surr2 )
            grad = -grad.mean()
            grad.backward()  # retain_graph=True
            self.p_optimizer.step()
        # update critic
        for _ in range(80):
            self.c_optimizer.zero_grad()
            pred_values = self.getExpectedvalues(self.train_states)
            critic_loss = torch.nn.MSELoss()(pred_values, self.train_rewards)
            critic_loss.backward()
            self.c_optimizer.step()

        pred_values = self.getExpectedvalues(self.train_states)
        critic_loss = torch.nn.MSELoss()(pred_values, self.train_rewards.flatten())
        print("\ntrain reward", self.train_rewards.mean(), "advtg", r.mean(), "critic loss", critic_loss)
        self.avg_rewards = self.train_rewards.mean()
        # Reset episode buffer
        self.train_rewards = torch.tensor([]).to(self.device)
        self.train_states  = torch.tensor([]).to(self.device)
        self.log_probs     = []
        self.train_actions = []
        if self.use_lstm:
            self.clearLSTMState()

    def __str__(self):
        return f"PPO_{super().__str__()[3:]}"