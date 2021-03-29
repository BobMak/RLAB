import wandb
import torch.nn as nn
import torch.optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from agents.PolicyOptimization import PolicyGradients
from utils.Modules import ActorCriticOutput


class ActorCritic(PolicyGradients):
    def __init__(self, inp, hid, out,
                 isContinuous=False, useLSTM=False, nLayers=1, usewandb=True):
        super().__init__(inp, hid, out,
                         isContinuous=isContinuous, useLSTM=useLSTM,
                         nLayers=nLayers, usewandb=usewandb)
        # replace the actor layer with an actorCritic layer
        if isContinuous:
            policy = [*self.model[:-1]]
        else:
            policy = [*self.model[:-1]]
        self.model = nn.Sequential(
            *policy,
            ActorCriticOutput(hid, out, isContinuous)
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)

        self.criticValues = []
        self.logProbs     = []

    def getAction(self, inputs):
        action, value = self.forward(inputs)
        self.criticValues.append(value)
        if self.isContinuous:
            action_distribution = Normal(*action)
            sampled_action = action_distribution.sample()
        else:
            action_distribution = Categorical(logits=action)
            sampled_action = action_distribution.sample()  #.item()
        # save the log likelihood of taking that action for backprop
        logProb = action_distribution.log_prob(sampled_action)
        self.logProbs.append(logProb)
        if not self.isContinuous:
            sampled_action = sampled_action.item()
        return sampled_action

    # gradient of one trajectory
    def backward(self):
        #  dtype=torch.float32, device=self.device)
        logProbs = torch.stack(self.logProbs)
        criticValues = torch.stack(self.criticValues)
        logProbs[logProbs != logProbs] = 1  # replace all nans
        # critic update
        valueGrad = (criticValues - self.trainRewards).mean()
        valueGrad.backward(retain_graph=True)
        # actor update
        r = self.trainRewards - criticValues.detach()  # advantage  self.trainRewards-criticValues
        grad = -(logProbs * r).mean()  #-(value * verr).mean()
        grad.backward()
        rewardMean = self.trainRewards.mean().item()
        print("avg train reward", rewardMean, "grad", grad.item())
        self.avgRewards = self.trainRewards.mean()
        self.optimizer.step()
        self.optimizer.zero_grad()
        criticMean = criticValues.mean().item()
        print("avg estimated reward", criticMean, "grad", valueGrad.item())
        if self.use_wandb:
            wandb.log({"awgReward": rewardMean})
            wandb.log({"awgCritic": criticMean})
        # Reset episode buffer
        self.trainRewards = torch.tensor([]).to(self.device)
        self.trainActions = torch.tensor([]).to(self.device)
        self.trainStates  = torch.tensor([]).to(self.device)
        self.criticValues = []
        self.logProbs     = []
        if self.use_lstm:
            self.clearLSTMState()

    def __str__(self):
        return f"AC_h{super().__str__()[4:]}"
