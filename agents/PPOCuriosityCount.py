"""
PPO implementation
"""
import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt

from agents.PolicyOptimization import PolicyGradients


class PPOCuriosityCount(PolicyGradients):
    def __init__(self, inp, hid, out, envshape, batch_size, curiosityStateGranularity=30, curiosityDrop=0.001, clip_ratio=0.2, isContinuous=False, useLSTM=False, nLayers=1, usewandb=False, env=None, dev="cpu"):
        super().__init__(inp, hid, out, isContinuous, useLSTM, nLayers, usewandb, env, dev)
        self.clip_ratio = clip_ratio
        self.curiosityMultiplier = 1.0
        self.state_curiosity = np.ones([ curiosityStateGranularity, curiosityStateGranularity]) * self.curiosityMultiplier
        self.curiosity_drop = curiosityDrop
        self.curiosity_rewards = np.zeros(batch_size)
        self.curiosityStateGranularity = curiosityStateGranularity
        self.batch_size = batch_size
        self.idx = 0
        # init a curiosity plot
        self.fig, (self.ax, self.rx) = plt.subplots(2)
        self.ax.set_title("Curiosity")
        # average rewards
        self.rx.set_title("Rewards")
        self.fig.tight_layout()
        self.fig.show()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def getAction(self, x):
        if self.training:
            idx = np.array(((x+1)/2*self.curiosityStateGranularity).numpy(), dtype=np.int32)
            # reduce curiosity for that state, but don't drop below zero
            self.state_curiosity[idx[0]][idx[1]] = \
                max(0, self.state_curiosity[idx[0]][idx[1]] - self.curiosity_drop)
            self.curiosity_rewards[self.idx] = self.state_curiosity[idx[0]][idx[1]]
            self.idx += 1
            # update curiosity plot every 100 steps
            if self.idx % 100 == 0:
                self.ax.clear()
                self.ax.imshow(self.state_curiosity, cmap='hot', interpolation='nearest', vmin=0, vmax=self.curiosityMultiplier)
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
        return super().getAction(x)

    # def resetCuriosity(self):


    # gradient of one trajectory
    def backward(self):
        actions = torch.stack(self.train_actions)
        pred_values = self.getExpectedValues(self.train_states).detach()
        r = torch.add(self.train_rewards, torch.tensor(self.curiosity_rewards[:len(self.train_states)], dtype=torch.float32).to(self.device))
        r = (r - r.mean()) / (r.std() + 1e-10).detach()
        critic_loss = torch.nn.MSELoss()(pred_values, r)
        print("critic loss", critic_loss)
        r += torch.tensor(self.curiosity_rewards[:len(r)], dtype=torch.float32).to(self.device)
        adv = torch.sub(r, pred_values.flatten())
        if self.use_wandb:
            wandb.log({"avgReward": r.mean()})
        old_log_probs = torch.stack(self.log_probs).detach()
        if self.isContinuous:
            # adv = r.unsqueeze(1)
            adv = adv.unsqueeze(1)
        self.rx.clear()
        self.rx.plot(adv)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
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
        for _ in range(80):
            self.c_optimizer.zero_grad()
            pred_values = self.getExpectedValues(self.train_states)
            critic_loss = torch.nn.MSELoss()(pred_values.flatten(), r)
            critic_loss.backward()
            self.c_optimizer.step()

        print("\ntrain reward", self.train_rewards.mean())
        self.avg_rewards = self.train_rewards.mean()
        # Reset episode buffer
        self.train_rewards = torch.tensor([]).to(self.device)
        self.train_states  = torch.tensor([]).to(self.device)
        self.curiosity_rewards = np.zeros(self.batch_size)
        # self.state_curiosity = np.ones([self.curiosityStateGranularity, self.curiosityStateGranularity]) * 2
        self.idx = 0
        self.log_probs     = []
        self.train_actions = []
        if self.use_lstm:
            self.clearLSTMState()

    def __str__(self):
        return f"PPO_{super().__str__()[3:]}"