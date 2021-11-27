import numpy as np
import torch
import torch.nn as nn


class NormalOutput(nn.Module):
    def __init__(self, inp, out, activation=nn.Tanh):
        super().__init__()
        self.m = nn.Linear(inp, out)
        log_std = -0.5 * np.ones(out, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.act1 = activation

    def forward(self, inputs):
        mout = self.m(inputs)
        vout = torch.exp(self.log_std)
        return mout, vout


class ActorCriticOutput(nn.Module):
    def __init__(self, inp, out, isContinuous=False, activation=nn.Sigmoid):
        super().__init__()
        self.critic = nn.Linear(inp, 1)
        self.isContinuous = isContinuous
        if isContinuous:
            self.actor = NormalOutput(inp, out, activation=activation)
        else:
            self.actor = nn.Sequential(
                nn.Linear(inp, out),
                activation())

    def forward(self, inputs):
        value  = self.critic(inputs)
        action = self.actor (inputs)
        return action, value
