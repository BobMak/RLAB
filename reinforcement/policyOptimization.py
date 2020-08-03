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


env = gym.make('CartPole-v0')

env.reset()

batch_size = 1000
epochs = 10

input_size = 4
hidden_size = 10
output_size = 2

policy = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size),
    nn.ReLU(),
)

tobs = env.observation_space.sample()
print("Action space", env.action_space.n)
print("test policy out", torch.argmax(F.softmax(policy(torch.as_tensor(tobs, dtype=torch.float32)))).item())
learning_rate = 1e-4
optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
print("action space", env.action_space.sample())
print("env observation", env.observation_space.sample())


# gradient of one trajectory
def getGrad(policy, trj, rew):
    expRewrads = []
    actions    = []
    for i, sa in enumerate(trj):
        expRewrads.append(sum(rew[i:]))
        actions.append(policy.forward(sa[0]))
    grad = torch.dot(torch.as_tensor(expRewrads, dtype=torch.float32), torch.as_tensor(actions, dtype=torch.float32))
    return grad


for n in range(epochs):
    obs = env.reset()
    obs = torch.as_tensor(obs, dtype=torch.float32)
    trajectories = []
    sa_count = 0
    state_actions = []
    rewards = []
    while True:
        sa_count += 1
        action = torch.argmax(policy.forward(obs)).item()
        state_actions.append((obs, action))
        obs, reward, done, info = env.step(action)
        obs = torch.as_tensor(obs, dtype=torch.float32)
        rewards.append(reward)
        if done:
            trajectories.append((state_actions, rewards))
            rewards = []
            state_actions = []
            obs = env.reset()
            obs = torch.as_tensor(obs, dtype=torch.float32)
            if sa_count > batch_size:
                sa_count = 0
                break
    print("epoch", n)
    for trj in trajectories:
        policy.zero_grad()
        grad = getGrad(policy, trj[0], trj[1])
        grad.backward()
        optimizer.step()
        trajectories = []

obs = env.reset()
obs = torch.as_tensor(obs, dtype=torch.float32)
print("testing")
for _ in range(3):
    rewards = []
    done = True
    while done:
        # print(obs)
        action = torch.argmax(policy(obs)).item()
        obs, reward, done, info = env.step(action)
        obs = torch.as_tensor(obs, dtype=torch.float32)
        rewards.append(reward)
    print("test rewards", sum(rewards)/len(rewards))


