
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid, OBJECT_TO_IDX

import torch
import numpy as np
from torch import nn

import wandb

from agents.ActorCritic import ActorCritic
from agents.PolicyGradients import PolicyGradients
from agents.PPO import PPO
from agents.DQLearn import DQLearn
from utils.Cache import load_model, save_model
from utils.EnvHelper import EnvHelper
from utils.Modules import oneHot


class GridWorldInput:
    def __call__(self, inp):
        image = inp["image"]
        direction = oneHot(inp["direction"], 4)
        return inp


class World3DInput(nn.Module):
    IMG_CHANNELS = 3
    IMG_WIDTH    = 80
    IMG_HEIGHT   = 60
    INPUT_MEAN   = 128

    def __init__(self, out):
        """Process 3d worlds env input. see 3d worlds env for input format"""
        super().__init__()
        self.l1 = nn.Conv2d(3, 12, (3,3))
        self.a1 = nn.ReLU()
        self.l2 = nn.Conv2d(12, 1, (5, 5))
        self.a2 = nn.ReLU()
        self.l3 = nn.Linear(54*74, out)
        self.a3 = nn.ReLU()

    def forward(self, x):
        x = x/self.INPUT_MEAN
        # realtime one input per forward vs batch
        if len(x.shape)==3:
            x = x.reshape(1, self.IMG_CHANNELS, self.IMG_WIDTH, self.IMG_HEIGHT)
        else:
            x = x.reshape(x.shape[0], self.IMG_CHANNELS, self.IMG_WIDTH, self.IMG_HEIGHT)
        x = self.l1(x)
        x = self.a1(x)
        x = self.l2(x)
        x = self.a2(x)
        # Flatten
        if len(x.shape) == 3:
            x = torch.flatten(x)
        else:
            x = torch.flatten(x, 1)
        x = self.l3(x)
        x = self.a3(x)
        return x

if __name__ == "__main__":
    use_cached = False
    use_lstm = False
    use_wandb = False
    env_name = "MiniGrid-FourRooms-v0"
    batch_size = 500
    epochs= 10
    success_reward = 200
    normalize = False

    env = gym.make(env_name)
    env.reset()

    if use_wandb:
        wandb.init()

    print("action sample", env.action_space.sample())
    print("observation sample", env.observation_space.sample())
    print("env observation", env.observation_space["image"].shape[0])

    input_size = env.observation_space["image"].shape[0]
    hidden_size = 32
    n_layers = 2
    # if is_continuous:
    #     output_size = env.action_space.shape[0]
    # else:
    output_size = env.action_space.n

    policy = PPO(input_size,
                hidden_size,
                output_size,
                clip_ratio=1.0,
                isContinuous=False,
                useLSTM=use_lstm,
                nLayers=n_layers,
                usewandb=use_wandb)
    # model = ActorCritic(input_size,
    #                      hidden_size,
    #                      output_size,
    #                      isContinuous=is_continuous,
    #                      useLSTM=use_lstm,
    #                      nLayers=2)
    # policy = DQLearn(input_size,
    #                  hidden_size,
    #                  output_size,
    #                  useLSTM=use_lstm,
    #                  nLayers=5,
    #                  usewandb=use_wandb)
    policy.setEnv(env_name)
    if use_cached:
        policy.load("cachedModels")
        envHelper = EnvHelper(policy, env)
    else:
        if use_wandb:
            wandb.watch(policy.model, log="all")
        envHelper = EnvHelper(policy, env, batch_size=batch_size, epochs=epochs, normalize=normalize, success_reward=success_reward)
        envHelper.trainPolicy()
        policy.save("cachedModels")

    envHelper.evalueatePolicy()
    env.close()



