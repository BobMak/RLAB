"""
PPO implementation
"""
import numpy as np
import torch
import wandb

from agents.PolicyOptimization import PolicyGradients


class PPOCuriosityCount(PolicyGradients):
    def __init__(self, inp, hid, out, envshape, curiosityDrop=0.001, clip_ratio=0.2, isContinuous=False, useLSTM=False, nLayers=1, usewandb=False, env=None, dev="cpu"):
        super().__init__(inp, hid, out, isContinuous, useLSTM, nLayers, usewandb, env, dev)
        self.clip_ratio = clip_ratio
        self.state_curiosity = np.ones([ 10, 10])
        self.curiosity_drop = curiosityDrop
        self.curiosity_rewards = np.zeros(1200)
        self.idx = 0

    def getAction(self, x):
        idx = np.array(((x+1)*10 -1).numpy(), dtype=np.int32)
        self.state_curiosity[idx[0]][idx[1]] -= self.curiosity_drop
        self.curiosity_rewards[self.idx] = self.state_curiosity[idx[0]][idx[1]]
        self.idx += 1
        return super().getAction(x)

    # gradient of one trajectory
    def backward(self):
        actions = torch.stack(self.train_actions)
        pred_values = self.getExpectedvalues(self.train_states).detach()
        r = self.train_rewards
        r = (r - r.mean()) / (r.std() + 1e-10).detach()
        critic_loss = torch.nn.MSELoss()(pred_values, r)
        print("critic loss", critic_loss)
        r = torch.add(r, torch.tensor(self.curiosity_rewards, dtype=torch.float32).to(self.device))
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
        self.curiosity_rewards = np.zeros(1200)
        self.idx = 0
        self.log_probs     = []
        self.train_actions = []
        if self.use_lstm:
            self.clearLSTMState()

    def __str__(self):
        return f"PPO_{super().__str__()[3:]}"