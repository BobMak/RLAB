import gym
import wandb

from agents import ActorCritic, PolicyOptimization
from utils.Cache import load_model, save_model
from utils.EnvHelper import EnvHelper


if __name__ == "__main__":
    use_cached = False
    is_continuous = True
    use_lstm = False
    number_of_layers = 6
    hidden_size = 32
    batch_size = 5000
    epochs = 50
    use_wandb = True

    if is_continuous:
        env = gym.make("MountainCarContinuous-v0")
    else:
        env = gym.make("MountainCar-v0")
    env.reset()

    print("action sample", env.action_space.sample())
    print("observation sample", env.observation_space.sample())
    print("env observation", env.observation_space.shape[0])

    input_size = env.observation_space.shape[0]
    if is_continuous:
        output_size = env.action_space.shape[0]
    else:
        output_size = env.action_space.n

    # policy = PolicyOptimization.PolicyGradients(input_size,
    #                                             hidden_size,
    #                                             output_size,
    #                                             isContinuous=is_continuous,
    #                                             useLSTM=use_lstm,
    #                                             nLayers=5)
    policy = ActorCritic.ActorCritic(input_size,
                                     hidden_size,
                                     output_size,
                                     isContinuous=is_continuous,
                                     useLSTM=use_lstm,
                                     nLayers=number_of_layers)
    if use_cached:
        policy = load_model(policy)
        envHelper = EnvHelper(policy, env)
    else:
        if use_wandb:
            wandb.init()
            wandb.watch(policy.policy, log="all")
        envHelper = EnvHelper(policy, env, batch_size=batch_size, epochs=epochs, normalize=False)
        envHelper.trainPolicy()
        save_model(policy)

    envHelper.evalueatePolicy()
    env.close()



