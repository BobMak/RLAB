import gym
import torch
import numpy as np
import wandb

from agents.ActorCritic import ActorCritic
from agents.PolicyOptimization import PolicyGradients
from agents.PPO import PPO
from agents.DQLearn import DQLearn
from utils.Cache import load_model, save_model
from utils.EnvHelper import EnvHelper


if __name__ == "__main__":
    use_cached = False
    use_lstm = False
    use_wandb = False
    env_name = "CartPole-v0"

    env = gym.make(env_name)
    env.reset()

    if use_wandb:
        wandb.init()

    print("action sample", env.action_space.sample())
    print("observation sample", env.observation_space.sample())
    print("env observation", env.observation_space.shape[0])

    input_size = env.observation_space.shape[0]
    hidden_size = 32
    # if is_continuous:
    #     output_size = env.action_space.shape[0]
    # else:
    output_size = env.action_space.n

    # policy = PolicyGradients(input_size,
    #                         hidden_size,
    #                         output_size,
    #                         isContinuous=False,
    #                         useLSTM=use_lstm,
    #                         nLayers=2,
    #                         usewandb=use_wandb)

    policy = PPO(input_size,
                hidden_size,
                output_size,
                isContinuous=False,
                useLSTM=use_lstm,
                nLayers=2,
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
        envHelper = EnvHelper(policy, env, batch_size=500, epochs=30, normalize=False, success_reward=200)
        envHelper.trainPolicy()
        policy.save("cachedModels")

    envHelper.evalueatePolicy()
    env.close()



