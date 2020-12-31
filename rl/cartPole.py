import gym
import torch
import torch.nn as nn
import torch.functional as F
from torch.distributions.categorical import Categorical


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


env = gym.make("CartPole-v0")
value = env.reset()

batch_size = 5000
epochs = 20

input_size = 4
hidden_size = 32
output_size = 2

policy = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size),
    nn.Sigmoid()
).to(device)

policy(value)