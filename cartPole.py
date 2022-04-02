import gym
import wandb

from agents.PPO import PPO
from utils.EnvHelper import EnvHelper


if __name__ == "__main__":
    use_cached = False
    use_lstm = False
    use_wandb = False
    env_name = "Pendulum-v0"
    batch_size = 20000
    epochs= 30
    success_reward = 200
    normalize = False

    env = gym.make(env_name)
    env.reset()

    if use_wandb:
        wandb.init()

    print("action sample", env.action_space.sample())
    print("observation sample", env.observation_space.sample())
    print("env observation", env.observation_space.shape[0])

    input_size = env.observation_space.shape[0]
    hidden_size = 8
    n_layers = 2

    output_size = env.action_space.shape[0]

    policy = PPO(input_size,
                hidden_size,
                output_size,
                clip_ratio=0.2,
                isContinuous=True,
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
        envHelper = EnvHelper(policy,
                              env,
                              batch_size=batch_size,
                              epochs=epochs,
                              normalize=normalize,
                              success_reward=success_reward)
        envHelper.trainPolicy()
        policy.save("cachedModels")

    envHelper.evalueatePolicy()
    env.close()



