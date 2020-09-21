import torch
import torch.nn as nn


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


    def generage(self):
        res = []

        return res