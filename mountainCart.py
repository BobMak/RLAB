import gym
import wandb

from utils.EnvHelper import EnvHelper
from agents.PPO import PPO
from agents.PPOCuriosityCount import PPOCuriosityCount


if __name__ == "__main__":
    use_cached = False
    is_continuous = True
    use_lstm = False
    number_of_layers = 4
    hidden_size = 32
    batch_size = 8000
    batch_is_episode = False
    epochs = 15
    use_wandb = False

    curiosityStateGranularity = 30
    curiosityDrop = 0.001
    clip_ratio = 0.4

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

    # policy = PPOCuriosityCount(input_size,
    #             hidden_size,
    #             output_size,
    #             env.observation_space.shape,
    #             batch_size,
    #             curiosityStateGranularity=curiosityStateGranularity,
    #             curiosityDrop=curiosityDrop,
    #             clip_ratio=clip_ratio,
    #             isContinuous=is_continuous,
    #             useLSTM=use_lstm,
    #             nLayers=number_of_layers,
    #             usewandb=use_wandb)

    policy.setEnv(env_name)
    if use_cached:
        policy.load("cachedModels")
        envHelper = EnvHelper(policy, env, batch_is_episode=batch_is_episode)
    else:
        if use_wandb:
            wandb.watch(policy.model, log="all")
        envHelper = EnvHelper(policy, env, batch_size=batch_size, epochs=epochs, normalize=False)
        envHelper.trainPolicy()
        policy.save("cachedModels")

    envHelper.evalueatePolicy()
    env.close()



