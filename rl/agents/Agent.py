"""
Policy Gradients implementation
for continuous or discreet action spaces

"""
import os
import wandb
import torch.nn as nn
import torch.optim


class Agent:
    def __init__(self, inp, hid, out,
                 useLSTM=False, nLayers=1, usewandb=False, env=None):
        self.hid      = hid
        self.nls      = nLayers
        self.use_lstm = useLSTM
        self.hidden   = hid
        self.device   = torch.device("cpu")  # cpu
        self.use_wandb= usewandb
        policy = []
        policy.append(nn.Linear(inp, hid))
        policy.append(nn.ReLU())
        if useLSTM:
            policy.append(nn.LSTM(hid, hid))
            policy.append(nn.ReLU())
            self.hidden_lstm = (torch.randn(1, 1, hid),
                                torch.randn(1, 1, hid))
            self.lstm_idx = len(policy) - 2
        else:
            policy.append(nn.Linear(hid, hid))
            policy.append(nn.ReLU())

        for n in range(nLayers-1):
            policy.append(nn.Linear(hid, hid))
            policy.append(nn.ReLU())

        policy.append(nn.Linear(hid, out))
        self.model = nn.Sequential(*policy).to(self.device)

        learning_rate = 1e-2
        self.p_optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        if env==None:
            self.env = f"env(obs={inp},act={out})"
        else:
            self.env = env

        self.train_states  = torch.tensor([]).to(self.device)
        self.train_rewards = torch.tensor([]).to(self.device)
        self.train_actions = []

    def forward(self, x):
        if self.use_lstm:
            out = x
            for layer in self.model[:self.lstm_idx]:
                out = layer(out)
            # LSTM requires hid vector from the previous pass
            # ensure correct format for the backward pass
            out = out.view(out.numel() // self.hidden, 1, -1)
            out, self.hidden_lstm = self.model[self.lstm_idx](out, self.hidden_lstm)
            for layer in self.model[self.lstm_idx + 1:]:
                out = layer(out)
            return out
        else:
            return self.model(x)

    def setEnv(self, env):
        self.env = env

    def getAction(self, x):
        raise NotImplemented()

    def getActionDistribution(self, x):
        raise NotImplemented()

    # Save episode's rewards and state-actions
    def saveEpisode(self, states, rewards):
        self.train_states = torch.cat([self.train_states,
                                       torch.as_tensor(states, dtype=torch.float32, device=self.device)])
        self.train_rewards= torch.cat([self.train_rewards,
                                       torch.as_tensor(rewards, dtype=torch.float32, device=self.device)])

    # gradient of one trajectory
    def backward(self):
        raise NotImplemented()

    def clearLSTMState(self):
        self.hidden_lstm = (torch.randn(1, 1, self.hidden),
                            torch.randn(1, 1, self.hidden))

    def setInputModule(self, module):
        withInput = [module]
        withInput.extend(self.model)
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


