import math

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


def wandb_train():
    with wandb.init(project="LunarLander-v2"):
        config = wandb.config
        train_lander(
            config.batch_size,
            config.epochs,
            200,                # success reward
            config.hidden_size,
            config.n_layers,
            config.clip_ratio,
            use_wandb=True,
            use_cached=False,
            use_lstm=False,
            eval_render=False,
            eval_episodes=10
        )


def sweep(count=10):
    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'avg_reward',
            'goal': 'maximize'
        },
        'parameters': {
            'batch_size': {
                'distribution': 'q_log_uniform',
                'q': 1.0,
                'min': math.log(800),
                'max': math.log(33000)
            },
            'epochs': {'values': [30, 40, 50, 80, 100]},
            'n_layers': {'values': [1, 2, 3, 5]},
            'hidden_size': {
                'distribution': 'q_log_uniform',
                'q': 1.0,
                'min': math.log(4),
                'max': math.log(128)
            },
            'clip_ratio': {'values': [0.02, 0.06, 0.1, 0.2, 0.3, 0.4]}
        }
    }
    sweep_id = wandb.sweep(sweep_config, project="lunar-lander")
    wandb.agent(sweep_id, function=wandb_train, count=count)


def train_lander(batch_size, epochs, success_reward, hidden_size, n_layers, clip_ratio,
                 use_wandb=False,
                 use_cached=False,
                 use_lstm=False,
                 eval_render=False,
                 eval_episodes=10):
    is_continuous = True

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
        envHelper = EnvHelper(policy, env, batch_size=batch_size, epochs=epochs, normalize=False, success_reward=success_reward)
        envHelper.trainPolicy()
        policy.save("cachedModels")

    envHelper.evalueatePolicy(n_runs=eval_episodes, render=eval_render, use_wandb=use_wandb)
    env.close()


if __name__ == "__main__":
    # manual
    batch_size = 29000
    epochs = 80
    success_reward = 200
    hidden_size = 19
    n_layers = 2
    clip_ratio = 0.3
    train_lander(
        batch_size,
        epochs,
        success_reward,
        hidden_size,
        n_layers,
        clip_ratio,
        use_wandb=False,
        use_cached=True,
        use_lstm=False,
        eval_render=True,
        eval_episodes=5
    )

    # sweep
    # sweep(100)






