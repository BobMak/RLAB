import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic:
    def __init__(self, input_size, hidden_size, output_size):

        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(input_size, output_size),
            nn.Sigmoid(),
        )

        self.critic = self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size)
        )

        self.buffer_values        = []
        self.buffer_state_actions = []

    def forward(self, obs):
        action = self.actor(obs)
        value = self.critic(action + obs)

        return action, value

    def get_action(self, obs):
        act, _ = self.forward(obs)
        # action_probs = torch.distributions.Categorical(act)

        return F.softmax(act)

    def get_value(self, obs):
        _, val = self.forward(obs)
        