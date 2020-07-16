import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

import gym


env = gym.make('CartPole-v0')

inp = env.observation_space.shape[0]
hidden = 20
out = env.action_space.n

policy = nn.Sequential(
    nn.Linear(inp, hidden),
    nn.ReLU(),
    nn.Linear(hidden, hidden),
    nn.ReLU(),
    nn.Linear(hidden, out),
    nn.Sigmoid()
)

optimizer = optim.Adam(policy.parameters(), lr=0.001)
policy.optim = optimizer


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



