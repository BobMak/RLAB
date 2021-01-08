import gym
import torch
import numpy as np

from agents import ActorCritic
from utils.Cache import load_model, save_model
from utils.EnvHelper import EnvHelper


if __name__ == "__main__":
    use_cached = False
    is_continuous = False
    use_lstm = True

    if is_continuous:
        env = gym.make("MountainCarContinuous-v0")
    else:
        env = gym.make("MountainCar-v0")
    env.reset()

    print("action sample", env.action_space.sample())
    print("observation sample", env.observation_space.sample())
    print("env observation", env.observation_space.shape[0])

    input_size = env.observation_space.shape[0]
    hidden_size = 32
    if is_continuous:
        output_size = env.action_space.shape[0]
    else:
        output_size = env.action_space.n

    # policy = PolicyOptimization.PolicyGradients(input_size,
    #                                             hidden_size,
    #                                             output_size,
    #                                             isContinuous=is_continuous,
    #                                             useLSTM=use_lstm,
    #                                             nLayers=2)
    policy = ActorCritic.ActorCritic(input_size,
                                     hidden_size,
                                     output_size,
                                     isContinuous=is_continuous,
                                     useLSTM=use_lstm,
                                     nLayers=5)
    if use_cached:
        policy = load_model(policy)
        envHelper = EnvHelper(policy, env)
    else:
        envHelper = EnvHelper(policy, env, batch_size=5000, epochs=20)
        envHelper.trainPolicy()
        save_model(policy)

    envHelper.evalueatePolicy()
    env.close()



