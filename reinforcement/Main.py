import numpy as np
import gym

# from reinforcement.Agent import selfSuperModel, selfSuperPloicy
from reinforcement import DQLearn


env = gym.make('CartPole-v0')

# 
prev_observation = None
action = env.action_space.sample()


for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()
