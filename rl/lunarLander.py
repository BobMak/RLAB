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
    is_continuous = True

    batch_size = 2000
    epochs= 50
    success_reward = 200
    normalize = False

    if is_continuous:
        env_name = "LunarLanderContinuous-v2"
    else:
        env_name = "LunarLander-v2"

    env = gym.make(env_name)
    env.reset()

    if use_wandb:
        wandb.init()

    print("action sample", env.action_space.sample())
    print("observation sample", env.observation_space.sample())
    print("env observation", env.observation_space.shape[0])

    input_size = env.observation_space.shape[0]
    if is_continuous:
        output_size = env.action_space.shape[0]
    else:
        output_size = env.action_space.n
    hidden_size = 32
    n_layers = 2

    policy = PPO(input_size,
                hidden_size,
                output_size,
                clip_ratio=0.4,
                isContinuous=is_continuous,
                useLSTM=use_lstm,
                nLayers=n_layers,
                usewandb=use_wandb)

    policy.setEnv(env_name)
    if use_cached:
        policy.load("cachedModels")
        envHelper = EnvHelper(policy, env)
    else:
        if use_wandb:
            wandb.watch(policy.model, log="all")
        envHelper = EnvHelper(policy, env, batch_size=batch_size, epochs=epochs, normalize=normalize, success_reward=success_reward)
        envHelper.trainPolicy()
        policy.save("cachedModels")

    envHelper.evalueatePolicy()
    env.close()



