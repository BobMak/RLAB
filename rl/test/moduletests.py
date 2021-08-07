import unittest
import torch
from torch.distributions.normal import Normal
import torch.nn as nn

# import spinningup.spinup as spinup
import spinup.algos.pytorch.ppo.core as spinnUpCore

import gym

from agents.PPO import PPO
from utils.EnvHelper import EnvHelper
from utils.Modules import NormalOutput


class TestLinear(unittest.TestCase):
    def test_learning(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(1, 1)
        )
        learning_rate = 1e-1
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()
        trueDistibution = Normal(1.5, 0.1)
        pred_y = []
        true_y = []
        for x in range(10):
            pred_y.append(model( torch.tensor([1], dtype=torch.float32)))
            true_y.append(trueDistibution.sample())
            # r = criterion(torch.stack(pred_y), torch.stack(true_y))
            loss = criterion( torch.stack(pred_y), torch.stack(true_y))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred_y = []
            true_y = []
        res = model(torch.tensor([1], dtype=torch.float32))
        self.assertAlmostEqual(1.5, float(model(torch.tensor([1], dtype=torch.float32))), delta=.2)


class TestNormalModule(unittest.TestCase):
    def test_baseline(self):
        normal = spinnUpCore.MLPGaussianActor(1, 1, [], nn.Tanh)
        learning_rate = 1e-2
        optimizer = torch.optim.SGD(normal.parameters(), lr=learning_rate)
        y = 1.5
        std = 0.1
        trueDistibution = Normal(y, std)
        inpt = 1.0
        pred_y = []
        true_y = []
        for _ in range(1000):
            logprobs = []
            # for x in range(10):
            action_distribution, logp = normal(torch.tensor([inpt], dtype=torch.float32))
            sampled_action = action_distribution.sample()
            pred_y.append(sampled_action)
            # action_distribution, logp = normal(torch.tensor([inpt], dtype=torch.float32), sampled_action)
            logprobs.append(action_distribution.log_prob(sampled_action))
            # logprobs.append(logp)
            logProbs = torch.stack(logprobs)
            true_y.append(trueDistibution.sample())
            r = torch.max(torch.tensor([-10.0]),-(true_y[0] - pred_y[0])**2)
            # r = criterion(torch.stack(pred_y), torch.stack(true_y))
            grad = -(logProbs * r).mean()
            optimizer.zero_grad()
            grad.backward()
            optimizer.step()
            # print("avg val", torch.stack(pred_y).mean(), "avg, true", torch.stack(true_y).mean(), end=", ")
            # print("train reward", r.mean())
            pred_y = []
            true_y = []
            if abs(action_distribution.mean - trueDistibution.mean) < 0.19\
                    and abs(action_distribution.stddev - trueDistibution.stddev) < 0.19:
                print(f'spinningUp gaussian is fit after {_} iterations')
                break
        action_distribution, logp = normal(torch.tensor([inpt], dtype=torch.float32))
        # sampled_action = action_distribution.sample()
        # pred_y.append(sampled_action)
        # print("spinning up normal after training", sampled_action)
        self.assertAlmostEqual(action_distribution.mean, trueDistibution.mean, delta=.2)
        self.assertAlmostEqual(action_distribution.stddev, trueDistibution.stddev, delta=.2)
        # self.assertAlmostEqual(y, sampled_action, delta=.2)

    def test_learning(self):
        normal = NormalOutput(1, 1)
        learning_rate = 1e-2
        optimizer = torch.optim.SGD(normal.parameters(), lr=learning_rate)
        y = 1.5
        delta = 0.1
        trueDistibution = Normal(y, delta)
        inpt = 1.0
        pred_y = []
        true_y = []
        for _ in range(1000):
            logprobs = []
            # for x in range(10):
            action_distribution = Normal(*normal(torch.tensor([inpt], dtype=torch.float32)))
            sampled_action = action_distribution.sample()
            pred_y.append(sampled_action)
            true_y.append(trueDistibution.sample())
            logprobs.append(action_distribution.log_prob(sampled_action))
            logProbs = torch.stack(logprobs)
            r = torch.max(torch.tensor([-10.0]),-(true_y[0] - pred_y[0])**2)
            # r = criterion(torch.stack(pred_y), torch.stack(true_y))
            grad = -(logProbs*r).mean()
            optimizer.zero_grad()
            grad.backward()
            optimizer.step()
            # print("avg val", torch.stack(pred_y).mean(), "avg, true", torch.stack(true_y).mean(), end=", ")
            # print("train reward", r.mean())
            pred_y = []
            true_y = []
            if abs(action_distribution.mean - trueDistibution.mean) < 0.19\
                    and abs(action_distribution.stddev - trueDistibution.stddev) < 0.19:
                print(f'custom gaussian is fit after {_} iterations')
                break
        action_distribution = Normal(*normal(torch.tensor([inpt], dtype=torch.float32)))
        self.assertAlmostEqual(action_distribution.mean, trueDistibution.mean, delta=.2)
        self.assertAlmostEqual(action_distribution.stddev, trueDistibution.stddev, delta=.2)
        # self.assertAlmostEqual(y, sampled_action, delta=.2)
    # Reset episode buffer

    def test_PPO(self):
        use_lstm = False
        number_of_layers = 1
        hidden_size = 6
        batch_size = 10
        epochs = 1
        use_wandb = False
        for is_continuous in [True, False]:
            if is_continuous:
                env_name = "MountainCarContinuous-v0"
            else:
                env_name = "MountainCar-v0"
            env = gym.make(env_name)
            env.reset()

            input_size = env.observation_space.shape[0]
            if is_continuous:
                output_size = env.action_space.shape[0]
            else:
                output_size = env.action_space.n
            policy = PPO(input_size,
                         hidden_size,
                         output_size,
                         clip_ratio=0.2,
                         isContinuous=is_continuous,
                         useLSTM=use_lstm,
                         nLayers=number_of_layers,
                         usewandb=use_wandb,
                         device="cpu")
            policy.setEnv(env_name)
            envHelper = EnvHelper(policy, env, batch_size=batch_size, epochs=epochs, normalize=False, success_reward=5)
            envHelper.setComputeRewardsStrategy("rewardToGoDiscounted", gamma=0.98)
            envHelper.trainPolicy()

            envHelper.evalueatePolicy(vis=False)
            env.close()


if __name__ == '__main__':
    unittest.main()
