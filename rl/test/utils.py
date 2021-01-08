import unittest

import gym
from utils.EnvHelper import EnvNormalizer

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


if __name__ == '__main__':
    unittest.main()