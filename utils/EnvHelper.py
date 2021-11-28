from math import log, log2, log10

import scipy.signal
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


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class EnvHelper:
    def __init__(self, policy, env, batch_size=5000, epochs=10, normalize=False, batch_is_episode=False, success_reward=None):
        self.policy = policy
        self.best_policy = None
        self.best_policy_reward = -1e10
        self.epochs = epochs
        self.env = env
        self.batch_size = batch_size
        self.batch_is_episode = batch_is_episode
        if normalize:
            self.inputHandler = EnvNormalizer(self.env)
        else:
            self.inputHandler = EnvVanillaInput()

        self._gamma  = 0.995  # discounted future reward gamma
        self.success_reward = success_reward

    def rewardToGo(self, rewards):
        expRewrads = []
        for i in range(len(rewards)):
            expRewrads.append(torch.as_tensor([sum(rewards[i:])],
                                          dtype=torch.float32,
                                          device=self.policy.device))
        return expRewrads

    def rewardToGoDiscounted(self, rewards, gamma=0.995):
        expRewrads = []
        for i in range(len(rewards)):
            r = 0
            for idx, _r in enumerate(rewards[i:]):
                r += _r * ( gamma**idx )
            expRewrads.append(torch.as_tensor([r],
                                          dtype=torch.float32,
                                          device=self.policy.device))
        return expRewrads

    def rewardToGoDiscExpectation(self, rewards, obs, failed=True, gamma=0.995):
        """discounted to go reward, with added expectation of future rewards
        if the episode finished without termination"""
        expectedAfter = self.policy.getExpectedValues(obs)
        return discount_cumsum(rewards, gamma)+discount_cumsum([expectedAfter[0].item()]*len(rewards), gamma)[::-1]


    def rewardSum(self, rewards):
        expRewrads = []
        for i in range(len(rewards)):
            expRewrads.append(torch.as_tensor([sum(rewards)],
                                          dtype=torch.float32,
                                          device=self.policy.device))
        return expRewrads

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
                if done or sa_count >= self.batch_size:
                    break
            if sum(rewards) > self.best_policy_reward:
                self.best_policy = self.policy
                self.best_policy_reward = sum(rewards)
            self.policy.saveEpisode(states, self.rewardSum(rewards))  # self.rewardToGoDiscExpectation(rewards, obs)
            # self.policy.saveEpisode(states, self.rewardToGoDiscExpectation(rewards, obs))
            # self.policy.saveEpisode(states, self.rewardToGoDiscounted(rewards))
            obs_raw = self.inputHandler(self.env.reset())
            obs = torch.from_numpy(obs_raw)
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.policy.device)
            states  = []
            actions = []
            rewards = []
            if self.success_reward and self.success_reward <= sum(rewards):
                print(f"policy has reached optimal performance with avg score {sum(rewards)}")
                return self.policy
            self.policy.backward()
        return self.policy

    def evalueatePolicy(self):
        obs = self.inputHandler(self.env.reset())
        obs = torch.from_numpy(obs)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.policy.device)
        print("testing best policy")
        self.best_policy.setTraining(False)
        for _ in range(10):
            rewards = []
            done = False
            while not done:
                self.env.render()
                action = self.best_policy.getAction(obs)  #.detach()  # .cpu().numpy()
                obs, reward, done, info = self.env.step(action)
                obs = np.array(self.inputHandler(obs), dtype=float)
                obs = torch.from_numpy(obs)
                obs = torch.as_tensor(obs, dtype=torch.float32, device=self.best_policy.device)
                rewards.append(reward)
            print("test rewards", sum(rewards))
            obs = self.inputHandler(self.env.reset())
            obs = torch.from_numpy(obs)
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.best_policy.device)


def entropy( probabilities:[], base=None ):
    logFu = { None:log, 2: log2, 10:log10 }[base]
    return -sum( [ p * logFu(p) for p in probabilities ] )



