"""
PPO implementation
"""

import torch
import wandb
import gc
import heapq

from torch import nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from agents.PPO import PPO


class Mu(PPO):
    def __init__(self, inp, hid, world_hid, out,
                 mcts=1000,
                 clip_ratio=0.2,
                 isContinuous=False,
                 useLSTM=False,
                 nLayers=1,
                 nWorldModelLayers=2,
                 usewandb=False,
                 env=None,
                 device="cpu"):
        """:param mcts: Number of MonteCarlo Tree Search calls"""
        super().__init__(inp, hid, out, clip_ratio, isContinuous, useLSTM, nLayers, usewandb, env, device)
        world_model = [nn.Linear(inp+out, world_hid)]
        for layer in range(nWorldModelLayers):
            world_model.extend([nn.Linear(world_hid, world_hid), nn.ReLU()])
        self.world_model = nn.Sequential(
            *world_model[:-2],
            nn.Linear(world_hid, world_hid),
            nn.Tanh()
        )
        self.w_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=3e-4)
        self.mcts = mcts

    def getAction(self, x):
        max_value = -99999999
        best_action = None

        action_distribution = self.getActionDistribution(x.squeeze())

        if len(self.train_states)>0:
            state_value = self.train_states[-1]
        else:
            # todo what
            pass
        state_queue = []
        for _ in range(self.mcts):
            sampled_action = action_distribution.sample()  # .item()
            state_action = torch.cat([x, sampled_action])
            value = self.critic.forward(state_action)
            if value > max_value:
                max_value = value
                best_action = sampled_action
        # save the log likelihood of taking that action for backprop
        logProb = action_distribution.log_prob(best_action)
        self.log_probs.append(logProb)
        self.train_actions.append(best_action)
        return self._actionSample(best_action)

    def __str__(self):
        return f"CriticalPPO_{super().__str__()[4:]}"