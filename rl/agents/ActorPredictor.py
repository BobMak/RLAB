import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from agents.PolicyOptimization import PolicyGradients


class ActorCritic(PolicyGradients):
    def __init__(self, inp, hid, out, isImage=False, isContinuous=False, useLSTM=False):
        super().__init__(inp, hid, out, isImage=isImage, isContinuous=isContinuous, useLSTM=useLSTM)
        # replace the actor layer with an actorCritic layer
        if isContinuous:
            policy = [*self.policy[:-1]]
        else:
            policy = [*self.policy[:-2]]
        self.policy = nn.Sequential(
            *policy,
            
        ).to(self.device)

        self.criticValues = torch.tensor([]).to(self.device)

    def getAction(self, inputs):
        # action = self.forward(inputs)

        if self.isContinuous:
            sampled_action = Normal(*action).sample()
        else:
            sampled_action = Categorical(F.softmax(action)).sample().item()
        return sampled_action

    # gradient of one trajectory
    def backward(self):
        self.policy.zero_grad()
        action = self.forward(self.trainStates)

        if self.isContinuous:
            outAction = Normal(*action).log_prob(self.trainActions)
            outAction[outAction != outAction] = 0  # replace NaNs with 0s
        else:
            outAction = Categorical(action).log_prob(self.trainActions)

        # actor update
        r = self.criticValues  # self.trainRewards -
        # verr = self.criticValues - self.trainRewards//200
        grad = -(outAction * r).mean()  # -(value * verr).mean()
        grad.backward()
        print("train reward", self.trainRewards.mean(), "grad", grad, "advtg", r)
        self.avgRewards = self.trainRewards.mean()
        self.optimizer.step()
        print("train value", self.criticValues.mean())
        # Reset episode buffer
        self.trainRewards = torch.tensor([]).to(self.device)
        self.trainActions = torch.tensor([]).to(self.device)
        self.trainStates = torch.tensor([]).to(self.device)
        if self.use_lstm:
            self.clearLSTMState()

    def __str__(self):
        return "ActorPredictor" + ("Cont" if self.isContinuous else "Disc")
