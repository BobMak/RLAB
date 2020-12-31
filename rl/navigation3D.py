import gym
import torch
import torch.nn as nn
import numpy as np

import gym_miniworld

import Utils
from agents.ActorCritic import ActorCritic


def trainMountainCarPG(policy, env, batch_size=5000, epochs=10):
    states = []
    actions = []
    rewards = []
    for n in range(epochs):
        print(f"---{n+1}/{epochs}---")
        obs_raw = env.reset()
        obs = torch.from_numpy(obs_raw)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=policy.device)
        sa_count = 0
        while True:
            sa_count += 1
            action = policy.getAction(obs)
            states.append(obs_raw)
            actions.append(action)
            obs_raw, reward, done, info = env.step(action)
            obs = np.array(obs_raw, dtype=float)
            obs = torch.from_numpy(obs)
            obs = torch.as_tensor(obs, dtype=torch.float32, device=policy.device)
            rewards.append(reward)

            if done:
                expRewrads = []
                for i in range(len(rewards)):
                    # sliding window finite time horizon
                    # window = 50
                    # idx_min = min(i, len(rewards) - window - 1)
                    # idx_max = min(len(rewards) - 1, i + window)
                    # expRewrads.append(torch.as_tensor([sum(rewards[idx_min:idx_max])],
                    #                                   dtype=torch.float32,
                    #                                   device=policy.device))
                    # reward to go
                    # expRewrads.append(torch.as_tensor([sum(rewards[i:])],
                    #                                   dtype=torch.float32,
                    #                                   device=policy.device))
                    rewards = (rewards - np.mean(rewards))/(np.std(rewards)+1)
                    expRewrads = torch.as_tensor(rewards,
                                                 dtype=torch.float32,
                                                 device=policy.device)
                    # reward sum
                    # expRewrads.append(torch.as_tensor([sum(rewards)],
                    #                                   dtype=torch.float32,
                    #                                   device=policy.device))
                policy.saveEpisode(states, actions, expRewrads)
                states = []
                actions = []
                rewards = []
                obs_raw = env.reset()
                obs = torch.from_numpy(obs_raw)
                obs = torch.as_tensor(obs, dtype=torch.float32, device=policy.device)
                if sa_count > batch_size:
                    break
        policy.backprop()
    return policy


def evalueateMountainCar(policy, env):
    obs = env.reset()
    obs = torch.from_numpy(obs)
    obs = torch.as_tensor(obs, dtype=torch.float32, device=policy.device)
    print("testing")
    for _ in range(10):
        rewards = []
        done = False
        while not done:
            env.render()
            action = policy.getAction(obs)  # .cpu().numpy()
            obs, reward, done, info = env.step(action)
            obs = np.array(obs, dtype=float)
            obs = torch.from_numpy(obs)
            obs = torch.as_tensor(obs, dtype=torch.float32, device=policy.device)
            rewards.append(reward)
            if done:
                print("test rewards", sum(rewards))
                rewards = []
                obs = env.reset()
                obs = torch.from_numpy(obs)
                obs = torch.as_tensor(obs, dtype=torch.float32, device=policy.device)


class World3DInput(nn.Module):
    IMG_CHANNELS = 3
    IMG_WIDTH    = 80
    IMG_HEIGHT   = 60
    INPUT_MEAN   = 128
    def __init__(self, out):
        """Process 3d worlds env input. see 3d worlds env for input format"""
        super().__init__()
        self.l1 = nn.Conv2d(3, 1, (3,3))
        self.a1 = nn.ReLU()
        self.l2 = nn.Conv2d(1, 1, (5, 5))
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
    use_cached    = False
    is_continuous = False
    use_lstm      = True

    env = gym.make('MiniWorld-Hallway-v0')
    env.reset()

    print("action sample", env.action_space.sample())
    print("observation sample", env.observation_space.sample())
    print("env observation", env.observation_space.shape[0])

    input_size  = 32
    hidden_size = input_size
    if is_continuous:
        output_size = env.action_space.shape[0]
    else:
        output_size = env.action_space.n
    layers_number = 3
    # policy = PolicyOptimization.PolicyGradients(input_size,
    #                                             hidden_size,
    #                                             output_size,
    #                                             isContinuous=is_continuous,
    #                                             useLSTM=use_lstm,
    #                                             nLayers=2)
    policy = ActorCritic(input_size,
                         hidden_size,
                         output_size,
                         isContinuous=is_continuous,
                         useLSTM=use_lstm,
                         nLayers=layers_number)

    if use_cached:
        policy = Utils.load_model(policy)

    policy.setInputModule(World3DInput(hidden_size))
    policy = trainMountainCarPG(policy, env, batch_size=500, epochs=25)
    Utils.save_model(policy)

    evalueateMountainCar(policy, env)
    env.close()

