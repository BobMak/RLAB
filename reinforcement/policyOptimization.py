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
    nn.Softmax(dim=1)
)

tobs = env.observation_space.sample()
print("Action space", env.action_space.n)
print("test policy out", policy(torch.as_tensor(tobs, dtype=torch.float32)))
learning_rate = 1e-4
optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
print(env.action_space)
print(env.observation_space.sample())


def getGrad(policy, trj, rew):
    expRewrads = []
    actions    = []
    for i, sa in enumerate(trj):
        expRewrads.append(sum(rew[i:]))
        actions.append(policy.forward(sa[0]))
    grad = torch.dot(torch.tensor(expRewrads), torch.tensor(actions))
    return grad


for n in range(epochs):
    obs = env.reset()
    obs = torch.as_tensor(obs, dtype=torch.float32)
    trajectories = []
    state_actions = []
    rewards = []
    while True:
        print(obs)
        action = torch.argmax(policy.forward(obs))
        state_actions.append((obs, action))
        obs, reward, done, info = env.step(action)
        obs = torch.as_tensor(obs, dtype=torch.float32)
        rewards.append(reward)
        if done:
            trajectories.append((state_actions, rewards))
            if len(state_actions) > batch_size:
                break

    for trj in trajectories:
        policy.zero_grad()
        grad = getGrad(policy, trj, rewards)
        grad.backward()
        optimizer.step()

obs = env.reset()
obs = torch.as_tensor(obs, dtype=torch.float32)
for _ in range(3):
    rewards = []
    done = True
    while done:
        print(obs)
        action = torch.argmax(policy(obs))
        obs, reward, done, info = env.step(action)
        obs = torch.as_tensor(obs, dtype=torch.float32)
        rewards.append(reward)
    print("test rewards", sum(rewards)/len(rewards))


