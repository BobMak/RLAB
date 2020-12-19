"""
to do:
architecture with stable state representation whatever that means
"""

import torch
import torch.nn as nn
import torch.functional as F


# state vector 512
# obs   -> state += upd( obs )
# state -> state+n

class Generator:
    def __init__(self, in_features, hid_features):
        self.seed = 1
        self.device = torch.device("gpu") if torch.cuda.is_available else torch.device("cpu")
        self.generator = nn.Sequential(
            nn.Linear(in_features, hid_features),
            nn.Tanh(),
            nn.Linear(in_features, hid_features),
            nn.Tanh(),
            nn.Linear(in_features, hid_features),
            nn.Tanh(),
            nn.Linear(in_features, hid_features),
            nn.Tanh(),
        )

    def toStateSpace(self, data):
        pass

    def generage(self):
        res = []

        return res