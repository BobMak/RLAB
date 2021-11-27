"""
PPO implementation
"""

import torch
import wandb
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from agents.PolicyOptimization import PolicyGradients


class PPO(PolicyGradients):
    def __init__(self, inp, hid, out, clip_ratio=0.2, isContinuous=False, useLSTM=False, nLayers=1, usewandb=False, env=None, dev="cpu"):
        super().__init__(inp, hid, out, isContinuous, useLSTM, nLayers, usewandb, env, dev)
        self.clip_ratio = clip_ratio

    # gradient of one trajectory
    def backward(self):
        actions = torch.stack(self.train_actions)
        pred_values = self.getExpectedvalues(self.train_states).detach()
        r = self.train_rewards
        r = (r - r.mean()) / (r.std() + 1e-10).detach()
        critic_loss = torch.nn.MSELoss()(pred_values, r)
        print("critic loss", critic_loss)
        # todo add critic loss
        adv = torch.sub(r, pred_values.flatten())
        if self.use_wandb:
            wandb.log({"avgReward": r.mean()})
        old_log_probs = torch.stack(self.log_probs).detach()
        if self.isContinuous:
            # r = r.unsqueeze(1)
            adv = adv.unsqueeze(1)
        # update actor
        for _ in range(80):
            self.p_optimizer.zero_grad()
            # compute log_probs after the update
            action_distributions = self.getActionDistribution(self.train_states)
            log_probs = action_distributions.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            # clip it
            surr1 = torch.clamp(ratio, min=1 - self.clip_ratio, max=1 + self.clip_ratio) * adv
            surr2 = ratio * adv
            grad = torch.min( surr1, surr2 )
            grad = -grad.mean()
            grad.backward()
            self.p_optimizer.step()
        # update critic
        for _ in range(40):
            self.c_optimizer.zero_grad()
            pred_values = self.getExpectedvalues(self.train_states)
            critic_loss = torch.nn.MSELoss()(pred_values.flatten(), r)
            critic_loss.backward()
            self.c_optimizer.step()

        print("\ntrain reward", self.train_rewards.mean())
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