import gym
import torch
import numpy as np

from agents import ActorCritic, Utils


# defines how reward collection from the environment is handled
# defines rl environment hyperparamenters
def trainMountainCarPG(policy, env, batch_size=5000, epochs=10):

    states = []
    actions = []
    rewards = []

    for n in range(epochs):
        print(f"---{n+1}/{epochs}---")
        obs_raw = env.reset()
        obs = torch.from_numpy(obs_raw)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=policy.device)
        sa_count = 0
        while True:
            sa_count += 1
            action = policy.getAction(obs)
            states.append(obs_raw)
            actions.append(action)
            obs_raw, reward, done, info = env.step(action)
            obs = np.array(obs_raw, dtype=float)
            obs = torch.from_numpy(obs)
            obs = torch.as_tensor(obs, dtype=torch.float32, device=policy.device)
            rewards.append(reward)

            if done:
                expRewrads = []
                for i in range(len(rewards)):
                    # sliding window finite time horizon
                    # window = 50
                    # idx_min = min(i, len(rewards) - window - 1)
                    # idx_max = min(len(rewards) - 1, i + window)
                    # expRewrads.append(torch.as_tensor([sum(rewards[idx_min:idx_max])],
                    #                                   dtype=torch.float32,
                    #                                   device=policy.device))
                    # reward to go
                    # expRewrads.append(torch.as_tensor([sum(rewards[i:])],
                    #                                   dtype=torch.float32,
                    #                                   device=policy.device))
                    rewards = (rewards - np.mean(rewards))/(np.std(rewards)+1)
                    expRewrads = torch.as_tensor(rewards,
                                                      dtype=torch.float32,
                                                      device=policy.device)
                    # reward sum
                    # expRewrads.append(torch.as_tensor([sum(rewards)],
                    #                                   dtype=torch.float32,
                    #                                   device=policy.device))
                policy.saveEpisode(states, actions, expRewrads)
                states = []
                actions = []
                rewards = []
                obs_raw = env.reset()
                obs = torch.from_numpy(obs_raw)
                obs = torch.as_tensor(obs, dtype=torch.float32, device=policy.device)
                if sa_count > batch_size:
                    sa_count = 0
                    break
        policy.backprop()
    return policy


def evalueateMountainCar(policy, env):
    obs = env.reset()
    obs = torch.from_numpy(obs)
    obs = torch.as_tensor(obs, dtype=torch.float32, device=policy.device)
    print("testing")
    for _ in range(10):
        rewards = []
        done = False
        while not done:
            env.render()
            action = policy.getAction(obs)  # .cpu().numpy()
            obs, reward, done, info = env.step(action)
            obs = np.array(obs, dtype=float)
            obs = torch.from_numpy(obs)
            obs = torch.as_tensor(obs, dtype=torch.float32, device=policy.device)
            rewards.append(reward)
            if done:
                print("test rewards", sum(rewards))
                rewards = []
                obs = env.reset()
                obs = torch.from_numpy(obs)
                obs = torch.as_tensor(obs, dtype=torch.float32, device=policy.device)


if __name__ == "__main__":
    use_cached = False
    is_continuous = False
    use_lstm = True
    name_cached = "ActorCritic" + ("Cont" if is_continuous else "Disc")
    # name_cached = "PolicyGradients" + ("Cont" if is_continuous else "Disc")

    if is_continuous:
        env = gym.make("MountainCarContinuous-v0")
    else:
        env = gym.make("MountainCar-v0")
    env.reset()

    print("action sample", env.action_space.sample())
    print("observation sample", env.observation_space.sample())
    print("env observation", env.observation_space.shape[0])

    if use_cached:
        policy = Utils.load_model(name_cached)
    else:
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
        policy  = ActorCritic.ActorCritic(input_size,
                                          hidden_size,
                                          output_size,
                                          isContinuous=is_continuous,
                                          useLSTM=use_lstm,
                                          nLayers=5)
        policy = trainMountainCarPG(policy, env, batch_size=500, epochs=500)
        Utils.save_model(policy)

    evalueateMountainCar(policy, env)
    env.close()



