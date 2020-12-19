"""
Policy Gradients implementation
for continuous or discreet action spaces

"""

import torch.nn as nn
import torch.optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from agents import Utils


class PolicyGradients(nn.Module):
    def __init__(self, inp, hidden, out,
                 isImage=False, isContinuous=False, useLSTM=False,
                 nLayers=1):
        super(PolicyGradients, self).__init__()
        self.isContinuous = isContinuous
        self.useLSTM = useLSTM
        self.hidden = hidden
        self.device = torch.device("cpu")  # cpu
        policy = []
        if isImage:
            policy.append(nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1))
            policy.append(nn.BatchNorm2d(4))

        policy.append(nn.Linear(inp, hidden))
        policy.append(nn.ReLU())
        # policy.append(nn.Dropout())
        if useLSTM:
            policy.append(nn.LSTM(hidden, hidden))
            policy.append(nn.ReLU())
            self.hiddenLSTM = (torch.randn(1, 1, hidden),
                               torch.randn(1, 1, hidden))
            self.hiddenIdx = len(policy)-2
        else:
            policy.append(nn.Linear(hidden, hidden))
            policy.append(nn.ReLU())

        for n in range(nLayers):
            policy.append(nn.Linear(hidden, hidden))
            policy.append(nn.ReLU())

        if isContinuous:
            policy.append(Utils.NormalOutput(hidden, out, activation=nn.Sigmoid))
        else:
            policy.append(nn.Linear(hidden, out))
            policy.append(nn.Softmax())
        self.policy = nn.Sequential(*policy).to(self.device)

        learning_rate = 1e-2
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.trainStates = torch.tensor([]).to(self.device)
        self.trainActions = torch.tensor([]).to(self.device)
        self.trainRewards = torch.tensor([]).to(self.device)

        self.avgRewards = 0

    def forward(self, inp):
        if self.useLSTM:
            out = inp
            for layer in self.policy[:self.hiddenIdx]:
                out = layer(out)
            # LSTM requires hidden vector from the previous pass
            # ensure correct format for the backward pass
            out = out.view(out.numel() // self.hidden, 1, -1)
            out, self.hiddenLSTM = self.policy[self.hiddenIdx](out, self.hiddenLSTM)
            for layer in self.policy[self.hiddenIdx+1:]:
                out = layer(out)
            return out
        else:
            return self.policy(inp)

    def getAction(self, inp):
        if self.isContinuous:
            action = Normal(*self.forward(inp)).sample()
        else:
            action = Categorical(self.forward(inp)).sample().item()
        return action

    # Save episode's rewards and state-actions
    def saveEpisode(self, states, actions, rewards):
        self.trainStates = torch.cat([self.trainStates,
                                      torch.as_tensor(states, dtype=torch.float32, device=self.device)])
        self.trainActions= torch.cat([self.trainActions,
                                      torch.as_tensor(actions, dtype=torch.float32, device=self.device)])
        self.trainRewards= torch.cat([self.trainRewards,
                                     torch.as_tensor(rewards, dtype=torch.float32, device=self.device)])

    # gradient of one trajectory
    def backprop(self):
        self.policy.zero_grad()
        if self.isContinuous:
            out = Normal(*self.forward(self.trainStates)).log_prob(self.trainActions)
            out[out!=out] = 0  # replace NaNs with 0s
        else:
            out = Categorical(self.forward(self.trainStates)).log_prob(self.trainActions)

        # Compute an advantage
        r = self.trainRewards - self.avgRewards
        grad = -(out * r).mean()
        grad.backward()
        self.optimizer.step()
        print("train reward", self.trainRewards.mean(), "grad", grad, "advtg", r)
        self.avgRewards = self.trainRewards.mean()
        # Reset episode buffer
        self.trainRewards = torch.tensor([]).to(self.device)
        self.trainActions = torch.tensor([]).to(self.device)
        self.trainStates  = torch.tensor([]).to(self.device)
        if self.useLSTM:
            self.clearLSTMState()

    def clearLSTMState(self):
        self.hiddenLSTM = (torch.randn(1, 1, self.hidden),
                           torch.randn(1, 1, self.hidden))

    def __str__(self):
        return f"PolicyGradients" + ("Cont" if self.isContinuous else "Disc")
