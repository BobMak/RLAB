
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
from utils.Modules import oneHot, MultiDataEncoderDecoder


# encode: gets image and direction data from the raw observation, turns it into a flat tensor
# decode: turns the flat tensor into a tuple of image and direction tensors inside the forward pass
class GridWorldInputHandler(MultiDataEncoderDecoder):
    N_CLASSES  = 11
    IMG_WIDTH  = 7
    IMG_HEIGHT = 7
    
    def __init__(self, sample):
        super(GridWorldInputHandler, self).__init__(self.gridextract(sample))
        
    def gridextract(self, inputs):
        img = inputs['image']
        # 11 classes of objects
        img = img[::, ::, 0]
        img = torch.tensor((np.arange(self.N_CLASSES) == img[..., None] - 1), dtype=torch.float32,
                           device='cuda')  # neat oneliner from Divakar
        img = img.reshape(1, self.N_CLASSES, self.IMG_WIDTH, self.IMG_HEIGHT)
        dir = torch.tensor(oneHot(inputs['direction'], 3), dtype=torch.float32, device='cuda')
        return img, dir
        
    def encode(self, inputs:{}):
        return super().encode([*self.gridextract(inputs)])


# handles the image + direction data
class GridWorldInput(nn.Module):
    N_CLASSES = 11
    IMG_WIDTH    = 7
    IMG_HEIGHT   = 7
    INPUT_MEAN   = 128

    def __init__(self, out, decoder):
        """Process 3d worlds env input. see 3d worlds env for input format"""
        super().__init__()
        self.l1 = nn.Conv2d(self.N_CLASSES, 1, (3,3))
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(23, out-4)
        self.a2 = nn.ReLU()
        self.decoder = decoder

    def forward(self, x):
        # the 2d observation part
        x = self.decoder(x)
        img = x[0]

        out = self.l1(img)
        out = self.a1(out)
        out = torch.flatten(out)
        out = self.l2(out)
        out = self.a2(out)
        # the direction part

        out = torch.cat([out, x[1]])
        return out


if __name__ == "__main__":
    use_cached = False
    use_lstm = False
    use_wandb = False
    env_name = "MiniGrid-FourRooms-v0"
    batch_size = 100
    epochs= 10
    success_reward = 200
    normalize = False
    device = 'cuda'

    env = gym.make(env_name)
    smp = env.reset()

    if use_wandb:
        wandb.init()

    print("action sample", env.action_space.sample())
    print("observation sample", env.observation_space.sample())
    print("env observation", env.observation_space["image"].shape[0])

    input_size = 24
    hidden_size = 32
    n_layers = 2
    # if is_continuous:
    #     output_size = env.action_space.shape[0]
    # else:
    output_size = env.action_space.n

    policy = PPO(input_size,
                hidden_size,
                output_size,
                clip_ratio=0.2,
                isContinuous=False,
                useLSTM=use_lstm,
                nLayers=n_layers,
                usewandb=use_wandb,
                device=device)

    policy.setEnv(env_name)

    # state encoder/decoder for the grid world
    gridworld_encdec = GridWorldInputHandler(env.reset())
    # gridworld-specific input module with a parallel cnn and linear module to merge visual and direction signals
    policy.setInputModule(GridWorldInput(input_size, gridworld_encdec.decode))
    if use_cached:
        policy.load("cachedModels")
        envHelper = EnvHelper(policy, env)
    else:
        if use_wandb:
            wandb.watch(policy.model, log="all")
        envHelper = EnvHelper(policy, env, batch_size=batch_size, epochs=epochs, normalize=normalize, success_reward=success_reward)

        envHelper.setInputHandler(gridworld_encdec.encode)
        envHelper.trainPolicy()
        policy.save("cachedModels")

    envHelper.evalueatePolicy()
    env.close()

