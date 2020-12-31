"""
PPO implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PPO():
    def __init__(self, inp, hid, out, inputModule=None):
