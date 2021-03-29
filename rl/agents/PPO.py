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

    def getAction(self, x):
        action = self.model.forward(x)
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
        # Compute an advantage
        pred_values = self.getExpectedvalue(self.train_states)
        r = self.train_rewards - pred_values.detach()
        r = (r - r.mean()) / (r.std() + 1e-10)
        if self.use_wandb:
            wandb.log({"awgReward": r.mean()})
        old_log_probs = torch.stack(self.log_probs)
        # update actor
        for _ in range(80):
            self.p_optimizer.zero_grad()
            # compute log_probs after the update
            action_distributions = self.getActionDistribution(self.train_states)
            log_probs = action_distributions.log_prob(self.train_actions)
            ratio = torch.exp(log_probs - old_log_probs)
            # clip it
            surr1 = ratio.clamp(1 - self.clip_ratio, 1 + self.clip_ratio) * r
            surr2 = ratio * r
            grad = torch.min( surr1, surr2 )
            grad = -(grad).mean()
            # print(grad)
            grad.backward(retain_graph=True)
            self.p_optimizer.step()
        # update critic
        for _ in range(80):
            self.c_optimizer.zero_grad()
            pred_values = self.getExpectedvalue(self.train_states)
            critic_loss = torch.nn.MSELoss()(pred_values, self.train_rewards)
            critic_loss.backward(retain_graph=True)
            self.c_optimizer.step()

        pred_values = self.getExpectedvalue(self.train_states)
        critic_loss = torch.nn.MSELoss()(pred_values, self.train_rewards.flatten())
        print("\ntrain reward", self.train_rewards.mean(), "advtg", r.mean(), "critic loss", critic_loss)
        self.avg_rewards = self.train_rewards.mean()
        # Reset episode buffer
        self.train_rewards = torch.tensor([]).to(self.device)
        self.train_actions = torch.tensor([]).to(self.device)
        self.train_states  = torch.tensor([]).to(self.device)
        self.log_probs    = []
        if self.use_lstm:
            self.clearLSTMState()

    def __str__(self):
        return f"PPO_{super().__str__()[3:]}"