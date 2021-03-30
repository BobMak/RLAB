import wandb
import torch.nn as nn
import torch.optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from utils.Modules import NormalOutput
from agents.PolicyOptimization import PolicyGradients


class HierarchicalPG(PolicyGradients):
    def __init__(self, inp, hid, out, isContinuous=False, useLSTM=False, nLayers=1, usewandb=False):
        super(PolicyGradients, self).__init__(inp, hid, out, isContinuous, useLSTM, nLayers, usewandb)
        