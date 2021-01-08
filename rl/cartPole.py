import gym
import torch
import numpy as np
import wandb

from agents.ActorCritic import ActorCritic
from agents.PolicyOptimization import PolicyGradients
from utils.Cache import load_model, save_model
from utils.EnvHelper import EnvHelper


if __name__ == "__main__":
    # wandb.init()

    use_cached = False
    use_lstm = False

    env = gym.make("CartPole-v0")
    env.reset()

    print("action sample", env.action_space.sample())
    print("observation sample", env.observation_space.sample())
    print("env observation", env.observation_space.shape[0])

    input_size = env.observation_space.shape[0]
    hidden_size = 32
    # if is_continuous:
    #     output_size = env.action_space.shape[0]
    # else:
    output_size = env.action_space.n

    policy = PolicyGradients(input_size,
                            hidden_size,
                            output_size,
                            isContinuous=False,
                            useLSTM=use_lstm,
                            nLayers=2)
    # policy = ActorCritic(input_size,
    #                      hidden_size,
    #                      output_size,
    #                      isContinuous=is_continuous,
    #                      useLSTM=use_lstm,
    #                      nLayers=5)
    if use_cached:
        policy = load_model(policy)
        envHelper = EnvHelper(policy, env)
    else:
        # wandb.watch(policy, log="all")
        envHelper = EnvHelper(policy, env, batch_size=5000, epochs=20, normalize=True)
        envHelper.trainPolicy()
        save_model(policy)

    envHelper.evalueatePolicy()
    env.close()



