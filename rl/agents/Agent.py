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
                 useLSTM=False, nLayers=1, usewandb=False, device="cpu"):
        self.hid      = hid
        self.nls      = nLayers
        self.useLSTM  = useLSTM
        self.hidden   = hid
        self.device   = torch.device(device)  # cpu
        self.usewandb = usewandb
        policy = []
        policy.append(nn.Linear(inp, hid))
        policy.append(nn.ReLU())
        if useLSTM:
            policy.append(nn.LSTM(hid, hid))
            policy.append(nn.ReLU())
            self.hiddenLSTM = (torch.randn(1, 1, hid),
                               torch.randn(1, 1, hid))
            self.lstmIdx = len(policy) - 2
        else:
            policy.append(nn.Linear(hid, hid))
            policy.append(nn.ReLU())

        for n in range(nLayers-1):
            policy.append(nn.Linear(hid, hid))
            policy.append(nn.ReLU())

        policy.append(nn.Linear(hid, out))
        self.model = nn.Sequential(*policy).to(self.device)

        learning_rate = 1e-2
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.trainStates  = torch.tensor([]).to(self.device)
        self.trainActions = torch.tensor([]).to(self.device)
        self.trainRewards = torch.tensor([]).to(self.device)

    def forward(self, x):
        if self.useLSTM:
            out = x
            for layer in self.model[:self.lstmIdx]:
                out = layer(out)
            # LSTM requires hid vector from the previous pass
            # ensure correct format for the backward pass
            out = out.view(out.numel() // self.hidden, 1, -1)
            out, self.hiddenLSTM = self.model[self.lstmIdx](out, self.hiddenLSTM)
            for layer in self.model[self.lstmIdx + 1:]:
                out = layer(out)
            return out
        else:
            return self.model(x)

    def getAction(self, x):
        raise NotImplemented()

    # Save episode's rewards and state-actions
    def saveEpisode(self, states, actions, rewards):
        self.trainStates = torch.cat([self.trainStates,
                                      torch.as_tensor(states, dtype=torch.float32, device=self.device)])
        self.trainActions= torch.cat([self.trainActions,
                                      torch.as_tensor(actions, dtype=torch.float32, device=self.device)])
        self.trainRewards= torch.cat([self.trainRewards,
                                     torch.as_tensor(rewards, dtype=torch.float32, device=self.device)])

    # gradient of one trajectory
    def backward(self):
        raise NotImplemented()

    def clearLSTMState(self):
        self.hiddenLSTM = (torch.randn(1, 1, self.hidden),
                           torch.randn(1, 1, self.hidden))

    def setInputModule(self, module):
        withInput = [module]
        withInput.extend(self.model)
        self.lstmIdx += 1
        self.model = nn.Sequential(*withInput).to(self.device)
        learning_rate = 1e-2
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def save(self, path="."):
        if not os.path.exists(path):
            os.mkdir(path)
        if self.usewandb:
            wandb.save(path + "/" + str(self))
        else:
            torch.save(self.model, path + "/" + str(self))

    def load(self, path="."):
        if self.usewandb:
            wandb.restore(path + "/" + str(self))
        else:
            self.model = torch.load(path + "/" + str(self))

    def __str__(self):
        return f"{self.hid}l{self.nls}_" + ("L" if self.useLSTM else "") \
                                         + ("w" if self.usewandb else "")


