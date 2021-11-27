import gym
import wandb

from utils.EnvHelper import EnvHelper
from agents.PPO import PPO


if __name__ == "__main__":
    use_cached = False
    is_continuous = False
    use_lstm = False
    number_of_layers = 5
    hidden_size = 128
    batch_size = 1000
    batch_is_episode = False
    epochs = 200
    use_wandb = False

    if is_continuous:
        env_name = "MountainCarContinuous-v0"
    else:
        env_name = "MountainCar-v0"
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

    policy = PPO(input_size,
                 hidden_size,
                 output_size,
                 clip_ratio=0.4,
                 isContinuous=is_continuous,
                 useLSTM=use_lstm,
                 nLayers=number_of_layers,
                 usewandb=use_wandb)

    policy.setEnv(env_name)
    if use_cached:
        policy.load("cachedModels")
        envHelper = EnvHelper(policy, env, batch_is_episode=batch_is_episode)
    else:
        if use_wandb:
            wandb.watch(policy.model, log="all")
        envHelper = EnvHelper(policy, env, batch_size=batch_size, epochs=epochs, normalize=False)
        envHelper.setComputeRewardsStrategy("rewardToGoDiscounted", gamma=0.98)
        envHelper.trainPolicy()
        policy.save("cachedModels")

    envHelper.evalueatePolicy()
    env.close()



