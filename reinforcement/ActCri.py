import torch
import torch.nn as nn
import torch.nn.functional as F


# for discreet actions
class ActorCritic:
    def __init__(self, input_size, hidden_size, output_size, action_size, training):

        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(input_size, output_size),
            nn.Sigmoid(),
        )

        self.critic = self.actor = nn.Sequential(
            nn.Linear(input_size+action_size, 1)
        )

        self.buffer_values  = []
        self.buffer_actions = []
        self.buffer_states  = []
        self.last_act = [0]*action_size

        self.training = training

    def forward(self, obs):
        act = torch.as_tensor(self.last_act, dtype=torch.float32)
        obs = torch.as_tensor(obs, dtype=torch.float32)
        action = self.actor(obs)
        value = self.critic(torch.cat([act, obs], dim=0))

        return action, value

    def get_action(self, obs):
        act, val = self.forward(obs)
        act = torch.distributions.Categorical(F.softmax(act)).sample().item()
        self.last_act = act
        if self.training:
            self.buffer_values.append(val)
            self.buffer_actions.append(act)
            self.buffer_states.append(obs)
        return act

    def get_value(self, obs):
        _, val = self.forward(obs)
        return val