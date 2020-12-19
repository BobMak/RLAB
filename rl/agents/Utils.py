import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

import pickle


class NormalOutput(nn.Module):
    def __init__(self, inp, out, activation=nn.Tanh):
        super().__init__()
        self.w1 = nn.Linear(inp, out)
        self.w2 = nn.Linear(inp, out)
        self.act1 = activation()
        self.act2 = activation()

    def forward(self, inputs):
        res1 = self.act1(self.w1(inputs))
        res2 = self.act2(self.w2(inputs))
        return res1, res2


class ActorCriticOutput(nn.Module):
    def __init__(self, inp, out, isContinuous=False, activation=nn.Tanh):
        super().__init__()
        self.critic = nn.Linear(inp, 1)
        self.isContinuous = isContinuous
        if isContinuous:
            self.actor = NormalOutput(inp, out, activation=activation)
        else:
            self.actor = nn.Sequential(
                nn.Linear(inp, out),
                activation()
            )

    def forward(self, inputs):
        value  = self.critic(inputs)
        action = self.actor (inputs)
        return action, value


def save_model(model):
    with open(str(model)+".pkl", 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


def load_model(model_name):
    with open(model_name+".pkl", 'rb') as f:
        model = pickle.load(f)
    return model
