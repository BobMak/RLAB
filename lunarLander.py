import gym
import wandb
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger()

from agents.PPO import PPO


from utils.EnvHelper import EnvHelper


if __name__ == "__main__":
    use_cached = False
    use_lstm = False
    use_wandb = False
    is_continuous = True

    batch_size = 8000
    epochs= 20
    success_reward = 200
    normalize = False
    hidden_size = 32
    n_layers = 4
    clip_ratio = 0.2

    if is_continuous:
        env_name = "LunarLanderContinuous-v2"
    else:
        env_name = "LunarLander-v2"

    env = gym.make(env_name)
    env.reset()

    if use_wandb:
        wandb.init(project="lunar-lander")

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
                clip_ratio=clip_ratio,
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



