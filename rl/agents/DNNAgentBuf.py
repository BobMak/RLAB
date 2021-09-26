"""
Policy Gradients implementation
for continuous or discreet action spaces

"""
import os
import wandb
import torch.nn as nn
import torch.optim

from ExpBuffer import Buffer


class DNNAgent:
    def __init__(self, inp, hid, out,
                 bufferLen, nLayers=1, env=None, device="cpu"):
        self.inp      = inp
        self.hid      = hid
        self.out      = out
        self.nls      = nLayers
        self.device   = torch.device(device)  # cpu
        policy = []
        policy.append(nn.Linear(inp, hid))
        policy.append(nn.ReLU())

        for n in range(nLayers-1):
            policy.append(nn.Linear(hid, hid))
            policy.append(nn.ReLU())

        policy.append(nn.Linear(hid, hid))
        policy.append(nn.ReLU())

        policy.append(nn.Linear(hid, out))
        self.model = nn.Sequential(*policy).to(self.device)

        learning_rate = 3e-4
        self.p_optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        if env==None:
            self.env = f"env(obs={inp},act={out})"
        else:
            self.env = env

        self.buf = Buffer(inp, out, bufferLen)
        self.episode_breaks = []

    # forward function for a single state input
    def forward(self, x):
        return self.model(x)

    def setEnv(self, env):
        self.env = env

    def getAction(self, x):
        raise NotImplemented()

    def getActionDistribution(self, x):
        raise NotImplemented()

    # Save episode's rewards and state-actions
    def saveEpisode(self, states, rewards):


    # gradient of one trajectory
    def backward(self):
        raise NotImplemented()

    def setInputModule(self, module):
        withInput = [module]
        withInput.extend(self.model)
        if self.use_lstm:
            self.lstm_idx += 1
        self.model = nn.Sequential(*withInput).to(self.device)
        learning_rate = 1e-2
        self.p_optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def save(self, path="."):
        if not os.path.exists(path):
            os.mkdir(path)
        if self.use_wandb:
            wandb.save(path + "/" + str(self))
        else:
            torch.save(self.model, path + "/" + str(self))

    def load(self, path="."):
        if self.use_wandb:
            wandb.restore(path + "/" + str(self))
        else:
            self.model = torch.load(path + "/" + str(self))

    def __str__(self):
        return f"{self.env}_h{self.hid}l{self.nls}_" + ("L" if self.use_lstm else "") \
                                         + ("w" if self.use_wandb else "")
