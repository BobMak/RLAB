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
    def __init__(self, policy, env, batch_size=5000, epochs=10, normalize=False, batch_is_episode=False):
        self.policy = policy
        self.epochs = epochs
        self.env = env
        self.batch_size = batch_size
        self.batch_is_episode = batch_is_episode
        if normalize:
            self.inputHandler = EnvNormalizer(self.env)
        else:
            self.inputHandler = EnvVanillaInput()

        self._window = 50     # sliding window reward assignment
        self._gamma  = 0.995  # discounted future reward gamma 
        self.setComputeRewardsStrategy("rewardToGoDiscounted")

    def computeRewards(self, rewards) -> [torch.Tensor]:
        return None

    def rewardToGo(self, rewards):
        expRewrads = []
        for i in range(len(rewards)):
            expRewrads.append(torch.as_tensor([sum(rewards[i:])],
                                          dtype=torch.float32,
                                          device=self.policy.device))
        return expRewrads

    def rewardToGoDiscounted(self, rewards):
        expRewrads = []
        for i in range(len(rewards)):
            r = 0
            for idx, _r in enumerate(rewards[i:]):
                r += _r * ( self._gamma**idx )
            expRewrads.append(torch.as_tensor([r],
                                          dtype=torch.float32,
                                          device=self.policy.device))
        return expRewrads

    def rewardSum(self, rewards):
        expRewrads = []
        for i in range(len(rewards)):
            expRewrads.append(torch.as_tensor([sum(rewards)],
                                          dtype=torch.float32,
                                          device=self.policy.device))
        return expRewrads

    def rewardSlidingWindow(self, rewards):
        expRewrads = []
        for i in range(len(rewards)):
            idx_min = min(i, len(rewards) - self._window - 1)
            idx_max = min(len(rewards) - 1, i + self._window)
            expRewrads.append(torch.as_tensor([sum(rewards[idx_min:idx_max])],
                                              dtype=torch.float32,
                                              device=self.policy.device))
        return expRewrads

    def setComputeRewardsStrategy(self, strategy:str, gamma=0.995, window=50):
        self._gamma  = gamma
        self._window = window
        self.computeRewards = {
            "rewardToGo": self.rewardToGo,
            "rewardToGoDiscounted": self.rewardToGoDiscounted,
            "rewardSum": self.rewardSum,
            "rewardSlidingWindow": self.rewardSlidingWindow
        }[strategy]

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
                    self.policy.saveEpisode(states, actions, self.computeRewards(rewards))
                    states = []
                    actions = []
                    rewards = []
                    obs_raw = self.inputHandler(self.env.reset())
                    obs = torch.from_numpy(obs_raw)
                    obs = torch.as_tensor(obs, dtype=torch.float32, device=self.policy.device)
                    if sa_count > self.batch_size or self.batch_is_episode:
                        break
            self.policy.backward()
        return self.policy

    def evalueatePolicy(self):
        obs = self.inputHandler(self.env.reset())
        obs = torch.from_numpy(obs)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.policy.device)
        print("testing")
        for _ in range(10):
            rewards = []
            done = False
            while not done:
                self.env.render()
                action = self.policy.getAction(obs)  #.detach()  # .cpu().numpy()
                obs, reward, done, info = self.env.step(action)
                obs = np.array(self.inputHandler(obs), dtype=float)
                obs = torch.from_numpy(obs)
                obs = torch.as_tensor(obs, dtype=torch.float32, device=self.policy.device)
                rewards.append(reward)
            print("test rewards", sum(rewards))
            obs = self.inputHandler(self.env.reset())
            obs = torch.from_numpy(obs)
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.policy.device)


def entropy( probabilities:[], base=None ):
    logFu = { None:log, 2: log2, 10:log10 }[base]
    return -sum( [ p * logFu(p) for p in probabilities ] )



