from math import log, log2, log10
import torch
import numpy as np


class EnvVanillaInput:
    def __call__(self, inp):
        return inp


class EnvNormalizer:
    def __init__(self, env):
        self.mean = (env.observation_space.high + env.observation_space.low) / 2
        self.var  = (env.observation_space.high - env.observation_space.low) / 2
        # self.mean[self.mean==np.inf] = env.observation_space.high

    def __call__(self, inp):
        return (inp - self.mean) / self.var


class EnvHelper:
    def __init__(self, policy, env, batch_size=5000, epochs=10, normalize=False):
        self.policy = policy
        self.epochs = epochs
        self.env = env
        self.batch_size = batch_size
        if normalize:
            self.inputHandler = EnvNormalizer(self.env)
        else:
            self.inputHandler = EnvVanillaInput()

    def trainPolicy(self):
        states = []
        actions = []
        rewards = []
        for n in range(self.epochs):
            print(f"---{n+1}/{self.epochs}---")
            obs_raw = self.inputHandler(self.env.reset())
            obs = torch.from_numpy(obs_raw)
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.policy.device)
            sa_count = 0
            while True:
                sa_count += 1
                action = self.policy.getAction(obs)
                states.append(obs_raw)
                actions.append(action)
                obs_raw, reward, done, info = self.env.step(action)
                obs = np.array(self.inputHandler(obs_raw), dtype=float)
                obs = torch.from_numpy(obs)
                obs = torch.as_tensor(obs, dtype=torch.float32, device=self.policy.device)
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
                        # rewards = (rewards - np.mean(rewards))/(np.std(rewards)+1)
                        # expRewrads = torch.as_tensor(rewards,
                        #                                   dtype=torch.float32,
                        #                                   device=self.policy.device)
                        # reward sum
                        expRewrads.append(torch.as_tensor([sum(rewards)],
                                                          dtype=torch.float32,
                                                          device=self.policy.device))
                    self.policy.saveEpisode(states, actions, expRewrads)
                    states = []
                    actions = []
                    rewards = []
                    obs_raw = self.inputHandler(self.env.reset())
                    obs = torch.from_numpy(obs_raw)
                    obs = torch.as_tensor(obs, dtype=torch.float32, device=self.policy.device)
                    if sa_count > self.batch_size:
                        break
            self.policy.backprop()
        return self.policy

    def evalueatePolicy(self):
        inpNorm = EnvNormalizer(self.env)
        obs = inpNorm(self.env.reset())
        obs = torch.from_numpy(obs)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.policy.device)
        print("testing")
        for _ in range(10):
            rewards = []
            done = False
            while not done:
                self.env.render()
                action = self.policy.getAction(obs)  # .cpu().numpy()
                obs, reward, done, info = self.env.step(action)
                obs = np.array(inpNorm(obs), dtype=float)
                obs = torch.from_numpy(obs)
                obs = torch.as_tensor(obs, dtype=torch.float32, device=self.policy.device)
                rewards.append(reward)
            print("test rewards", sum(rewards))
            obs = inpNorm(self.env.reset())
            obs = torch.from_numpy(obs)
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.policy.device)


def entropy( probabilities:[], base=None ):
    logFu = { None:log, 2: log2, 10:log10 }[base]
    return -sum( [ p * logFu(p) for p in probabilities ] )



