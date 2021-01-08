"""
Policy Gradients implementation
for continuous or discreet action spaces

"""

import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from utils.Modules import NormalOutput


class PolicyGradients(nn.Module):
    def __init__(self, inp, hid, out,
                 isContinuous=False, useLSTM=False, nLayers=1):
        super(PolicyGradients, self).__init__()
        self.hid          = hid
        self.nls          = nLayers
        self.isContinuous = isContinuous
        self.useLSTM      = useLSTM
        self.hidden       = hid
        self.device       = torch.device("cpu")  # cpu
        policy = []
        policy.append(nn.Linear(inp, hid))
        policy.append(nn.ReLU())
        if useLSTM:
            policy.append(nn.LSTM(hid, hid))
            policy.append(nn.ReLU())
            self.hiddenLSTM = (torch.randn(1, 1, hid),
                               torch.randn(1, 1, hid))
            self.hiddenIdx = len(policy)-2
        else:
            policy.append(nn.Linear(hid, hid))
            policy.append(nn.ReLU())

        for n in range(nLayers):
            policy.append(nn.Linear(hid, hid))
            policy.append(nn.ReLU())

        if isContinuous:
            policy.append(NormalOutput(hid, out, activation=nn.Sigmoid))
        else:
            policy.append(nn.Linear(hid, out))
        self.policy = nn.Sequential(*policy).to(self.device)

        learning_rate = 1e-2
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.trainStates  = torch.tensor([]).to(self.device)
        self.trainActions = torch.tensor([]).to(self.device)
        self.trainRewards = torch.tensor([]).to(self.device)
        self.logProbs     = []
        self.avgRewards = 0

    def forward(self, x):
        if self.useLSTM:
            out = x
            for layer in self.policy[:self.hiddenIdx]:
                out = layer(out)
            # LSTM requires hid vector from the previous pass
            # ensure correct format for the backward pass
            out = out.view(out.numel() // self.hidden, 1, -1)
            out, self.hiddenLSTM = self.policy[self.hiddenIdx](out, self.hiddenLSTM)
            for layer in self.policy[self.hiddenIdx+1:]:
                out = layer(out)
            return out
        else:
            return self.policy(x)

    def getAction(self, x):
        action = self.forward(x)
        if self.isContinuous:
            action_distribution = Normal(*action)
            sampled_action = action_distribution.sample()
        else:
            action_distribution = Categorical(F.softmax(action))
            sampled_action = action_distribution.sample()  # .item()
        # save the log likelihood of taking that action for backprop
        logProb = action_distribution.log_prob(sampled_action)
        self.logProbs.append(logProb)
        if not self.isContinuous:
            sampled_action = sampled_action.item()
        return sampled_action

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
        # compute one outputs sequentially if using LSTM
        # if self.useLSTM:
        #     out = []
        #     self.clearLSTMState()
        #     for idx in range(len(self.trainStates)):
        #         out.append(self.forward(self.trainStates[idx]))
        # else:
        #     out = self.forward(self.trainStates)
        #
        # if self.isContinuous:
        #     out = Normal(*out).log_prob(self.trainActions)
        #     out[out != out] = 0  # replace NaNs with 0s
        # else:
        #     out = Categorical(out).log_prob(self.trainActions)
        logProbs = torch.stack(self.logProbs)
        # Compute an advantage
        r = self.trainRewards - self.avgRewards
        grad = -(logProbs * r).mean()
        grad.backward()
        self.optimizer.step()
        self.policy.zero_grad()
        print("train reward", self.trainRewards.mean(), "grad", grad, "advtg", r)
        self.avgRewards = self.trainRewards.mean()
        # Reset episode buffer
        self.trainRewards = torch.tensor([]).to(self.device)
        self.trainActions = torch.tensor([]).to(self.device)
        self.trainStates  = torch.tensor([]).to(self.device)
        self.logProbs     = []
        if self.useLSTM:
            self.clearLSTMState()

    def clearLSTMState(self):
        self.hiddenLSTM = (torch.randn(1, 1, self.hidden),
                           torch.randn(1, 1, self.hidden))

    def setInputModule(self, module):
        withInput = [module]
        withInput.extend(self.policy)
        self.hiddenIdx += 1
        self.policy = nn.Sequential(*withInput).to(self.device)

    def __str__(self):
        return f"PG_h{self.hid}l{self.nls}_" + ("C" if self.isContinuous else "D") \
                                             + ("L" if self.useLSTM else "_")


