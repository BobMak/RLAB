"""
Following https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
Implementing simple policy optimization algo.

prerequisits:
how gaussian sampling works

infinite horizon vs finite horizon
"""

import gym
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


env = gym.make('CartPole-v0')
env.reset()

batch_size = 5000
epochs = 50

input_size = 4
hidden_size = 32
output_size = 2

device = torch.device("cuda")

policy = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    # nn.ReLU(),
    nn.Tanh(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size),
    # nn.ReLU(),
    nn.Sigmoid()
).to(device)

tobs = env.observation_space.sample()
print("Action space", env.action_space.n)
print("test policy out", Categorical(policy(torch.as_tensor(tobs, dtype=torch.float32, device=device))).sample().item())
learning_rate = 1e-2
optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
print("action space", env.action_space.sample())
print("env observation", env.observation_space.sample())


# gradient of one trajectory
def getGrad(policy, states, rew, actions):
    expRewrads = [] # [sum(rew)] * len(rew)
    # sum of the following rewards
    for i in range(len(rew)):
        expRewrads.append(sum(rew[i:]))
    # actions.append(policy.forward(sa[0]))
    _st = torch.as_tensor(states,  dtype=torch.float32, device=device)
    _ac = torch.as_tensor(actions, dtype=torch.int32, device=device)
    out = Categorical(policy(_st)).log_prob(_ac)
    r = torch.as_tensor(expRewrads, dtype=torch.float32, device=device)
    grad = -(out * r).mean()
    return grad


for n in range(epochs):
    obs = env.reset()
    _obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    trajectories = []
    sa_count = 0
    states  = []
    actions = []
    rewards = []
    while True:
        sa_count += 1
        action = Categorical(policy(_obs)).sample().item()
        states.append(obs.copy())
        actions.append(action)
        obs, reward, done, info = env.step(action)
        _obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        rewards.append(reward)
        if done:
            trajectories.append((states, rewards, actions))
            states  = []
            actions = []
            rewards = []
            obs = env.reset()
            _obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
            if sa_count > batch_size:
                sa_count = 0
                break

    for trj in trajectories:
        policy.zero_grad()
        grad = getGrad(policy, trj[0], trj[1], trj[2])
        grad.backward()
        optimizer.step()
    print("avg rewards:", sum(sum(trj[1]) for trj in trajectories)/len(trajectories))
    trajectories = []


obs = env.reset()
env.render()
obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
print("test")
for _ in range(10):
    rewards = []
    done = False
    while not done:
        env.render()
        # print(obs)
        action = torch.argmax(policy(obs)).item()
        obs, reward, done, info = env.step(action)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        rewards.append(reward)
        if done:
            print("test rewards", sum(rewards))
            rewards = []
            obs = env.reset()
            obs = torch.as_tensor(obs, dtype=torch.float32, device=device)


