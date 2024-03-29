import unittest
import torch
from torch.distributions.normal import Normal
import torch.nn as nn

# import spinningup.spinup as spinup
import spinup.algos.pytorch.ppo.core as spinnUpCore

import gym
from utils.EnvHelper import EnvNormalizer
from utils.Modules import NormalOutput

class TestNormalizer(unittest.TestCase):
    def setUp(self):
        self.env = gym.make("CartPole-v0")
        self.env.reset()
        self.envNorm = EnvNormalizer(self.env)

    def test_varience(self):
        normzlizedMax = self.envNorm(self.env.observation_space.high)
        for x in normzlizedMax:
            self.assertLessEqual(x, 2, f"{x} is > 2")
        normzlizedMin = self.envNorm(self.env.observation_space.low)
        for x in normzlizedMin:
            self.assertGreaterEqual(x, -2, f"{x} is < -2")

    def tearDown(self):
        self.env.close()


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


if __name__ == '__main__':
    unittest.main()
