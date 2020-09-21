import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

import wandb

import gym


# Predict the q-value of the state/action
class DQN(nn.Module):
    def __init__(self, d_in, d_h, d_out):
        super(DQN, self).__init__()
        self.lin1 = nn.Linear(d_in, d_h)
        self.act1 = nn.Tanh()
        self.lin2 = nn.Linear(d_h, d_h)
        self.act2 = nn.ReLU()
        self.lin3 = nn.Linear(d_h, d_h)
        self.act3 = nn.ReLU()
        self.lin4 = nn.Linear(d_h, d_out)
        self.act4 = nn.Sigmoid()
        
        self.replayBuffer = []
        self.optim = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()  # reduction="sum"

    def forward(self, x):
        """:arg x: state + action
        :return q value"""
        out = self.lin1(x)
        out = self.act1(out)
        out = self.lin2(out)
        out = self.act2(out)
        out = self.lin3(out)
        out = self.act3(out)
        out = self.lin4(out)
        out = self.act4(out)
        return out
    
    def clearReplayBuffer(self):
        self.replayBuffer.clear()
        
    def addToReplayBuffer(self, transition):
        self.replayBuffer.append(transition)

    def train(self):
        self.optim.zero_grad()
        for x in self.replayBuffer:
            pass


if __name__ == "__main__":
    wandb.login()
    env = gym.make('CartPole-v0')

    inp = env.observation_space.shape[0]
    hidden = 20
    out = env.action_space.n

    policy = DQN()

    env.action_space.sample()
    obs = env.reset()
    inps = torch.as_tensor(obs, dtype=torch.float32)
    act = policy.forward(inps)
    done = False

    for x in range(10):
        while not done:
            obs, rew, done, _ = env.step(act)
            act = policy.forward(act)
        policy.backward()
        policy.zero_grad()



