"""
PPO implementation
"""

import torch
import wandb
import gc

from torch import nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from agents.PolicyOptimization import PolicyGradients
from agents.PPO import PPO


class OptionEntropy(PolicyGradients):
    def __init__(self, inp, hid, out,
                 clip_ratio=0.2,
                 isContinuous=False,
                 useLSTM=False,
                 nLayersTop=1,
                 nLayersOpt=1,
                 usewandb=False,
                 env=None,
                 device="cuda:0"):
        self.n_options = 10
        super().__init__(inp, hid, self.n_options, False, False, nLayersTop, usewandb, env, device)
        self.clip_ratio = clip_ratio
        self.representation_states = torch.tensor([]).to(self.device)
        self.options = []
        for o in range(self.n_options):
            self.options.append(PPO(hid, hid, out, clip_ratio, isContinuous, useLSTM, nLayersOpt, False, env, device))

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

    # gradient of one trajectory
    def backward(self):
        actions = torch.stack(self.train_actions)
        # Compute an advantage
        pred_values = self.getAllExpectedValueBatches(self.train_states).detach()
        critic_loss = torch.nn.MSELoss()(pred_values, self.train_rewards)
        r = torch.sub(self.train_rewards.unsqueeze(1), pred_values)
        r = (r - r.mean()) / (r.std() + 1e-10).detach()
        if self.use_wandb:
            wandb.log({"avgReward": self.train_rewards.mean()})
            wandb.log({"avgAdvantage": r.mean()})
            wandb.log({"criticLoss": critic_loss.mean()})
        old_log_probs = torch.stack(self.log_probs).detach()
        # old_neg_log_probs = torch.stack(self.neg_log_probs).detach()
        # lg_list = []
        # for pos, neg, ra in zip(old_log_probs, old_neg_log_probs, r):
        #     lg_list.append(pos) if ra > 0 else lg_list.append(neg)
        # old_log_probs = torch.stack(lg_list)
        # PPO update actor
        for _ in range(80):
            self.p_optimizer.zero_grad()
            # compute log_probs after the update
            action_distributions = self.getAllActionDistributions(self.train_states)
            log_probs = action_distributions.log_prob(actions)
            # lg_pos = action_distributions.log_prob(actions)
            # lg_neg = action_distributions.log_prob(1-actions)
            # lg_list = []
            # for pos, neg, ra in zip(lg_pos, lg_neg, r):
            #     lg_list.append(pos) if ra>0 else lg_list.append(neg)
            # log_probs = torch.stack(lg_list)
            ratio = torch.exp(log_probs.squeeze() - old_log_probs.squeeze())
            # clip it
            surr1 = torch.clamp(ratio, min=1 - self.clip_ratio, max=1 + self.clip_ratio) * r
            surr2 = ratio * r
            grad = torch.min( surr1, surr2 )
            grad = -grad.mean()
            grad.backward()  # retain_graph=True
            self.p_optimizer.step()
            if self.use_lstm:
                self.clearLSTMState()
        # PPO update critic
        for _ in range(80):
            self.c_optimizer.zero_grad()
            pred_values = self.getAllExpectedValueBatches(self.train_states)
            critic_loss = torch.nn.MSELoss()(pred_values, self.train_rewards)
            critic_loss.backward()
            self.c_optimizer.step()
            if self.use_lstm:
                self.clearLSTMState()

        # pred_values = self.getAllExpectedvalues(self.train_states)
        # critic_loss = torch.nn.MSELoss()(pred_values, self.train_rewards.flatten())
        self.avg_reward = self.train_rewards.mean()
        print("\ntrain reward", self.avg_reward, "advtg", r.mean(), "critic loss", critic_loss)
        # Reset episode buffer
        self.train_rewards = torch.tensor([]).to(self.device)
        self.train_states  = torch.tensor([]).to(self.device)
        self.log_probs     = []
        self.train_actions = []
        if self.use_lstm:
            self.clearLSTMState()
        return self.avg_reward

    def __str__(self):
        return f"PPO_{super().__str__()[3:]}"