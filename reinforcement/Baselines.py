import gym
from gym import logger

import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


class NNAPolicy:
    def __init__(self, action_space, observations):
        self.action_space = action_space
        self.model = nn.Sequential(
            nn.Linear(observations.shape[0]+action_space.n, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
            nn.Sigmoid())
        self.model.cuda()
        self.expReward = 0
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    def forward(self, obs):
        # Predict future based on current action and obs
        expRewards, actions = [], []
        for n in range(self.action_space.n):
            act = [0 for x in range(self.action_space.n)]
            act[n] = 1
            expRewards.append(self.model(torch.tensor(act.extend(obs))))
            actions.append(act)
        self.expReward = max(expRewards)
        # perform the action giving max expected reward
        action = actions[expRewards.index(max(expRewards))]
        return action
    def backward(self, newReward):
        self.optimizer.zero_grad()
        pred = torch.tensor([self.expReward])
        real = torch.tensor([newReward])
        print(pred.dim(), real.dim())
        self.criterion(pred, real)
        self.optimizer.step()
    def act(self, ob, rew, done):
        act = self.forward(ob)
        return act


if __name__ == '__main__':
    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)
    env = gym.make("CartPole-v0")
    # env = gym.make("Pong-v0")
    env.seed(0)
    # agent = RandomAgent(env.action_space)
    agent = NNAPolicy(env.action_space, env.observation_space)
    episode_count = 100
    reward = 0
    done = False

    for i in range(episode_count):
        batch_rew = []
        batch_act = []
        batch_obs = []
        batch_pred = []
        ob = env.reset()
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            batch_rew.append(reward)
            batch_act.append(action)
            batch_obs.append(ob)
            batch_pred.append()
            if done:
                agent.backward()
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.close()