import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from agents.PolicyOptimization import PolicyGradients
import Utils


class ActorCritic(PolicyGradients):
    def __init__(self, inp, hid, out,
                 isContinuous=False, useLSTM=False, nLayers=1):
        super().__init__(inp, hid, out,
                         isContinuous=isContinuous, useLSTM=useLSTM, nLayers=nLayers)
        # replace the actor layer with an actorCritic layer
        if isContinuous:
            policy = [*self.policy[:-1]]
        else:
            policy = [*self.policy[:-2]]
        self.policy = nn.Sequential(
            *policy,
            Utils.ActorCriticOutput(hid, out, isContinuous)
        ).to(self.device)
        
        self.criticValues = torch.tensor([]).to(self.device)

    def getAction(self, inputs):
        action, value = self.forward(inputs)
        self.criticValues = torch.cat([self.criticValues,
                                       torch.as_tensor(value, dtype=torch.float32, device=self.device)])
        if self.isContinuous:
            sampled_action = Normal(*action).sample()
        else:
            sampled_action = Categorical(F.softmax(action)).sample().item()
        return sampled_action

    # gradient of one trajectory
    def backprop(self):
        self.policy.zero_grad()
        action, value = self.forward(self.trainStates)

        if self.isContinuous:
            outAction = Normal(*action).log_prob(self.trainActions)
            outAction[outAction != outAction] = 0  # replace NaNs with 0s
        else:
            outAction = Categorical(action).log_prob(self.trainActions)

        # critic update
        valueError = self.criticValues - self.trainRewards
        valueGrad = (valueError**2).mean()
        valueGrad.backward(retain_graph=True)
        # actor update
        r = self.criticValues  # self.trainRewards -
        # verr = self.criticValues - self.trainRewards//200
        grad = -(outAction * r).mean() #-(value * verr).mean()
        grad.backward()
        print("train reward", self.trainRewards.mean(), "grad", grad)
        self.avgRewards = self.trainRewards.mean()
        self.optimizer.step()
        print("train value", self.criticValues.mean())
        # Reset episode buffer
        self.trainRewards = torch.tensor([]).to(self.device)
        self.trainActions = torch.tensor([]).to(self.device)
        self.trainStates  = torch.tensor([]).to(self.device)
        self.criticValues = torch.tensor([]).to(self.device)
        if self.useLSTM:
            self.clearLSTMState()

    def __str__(self):
        return f"AC_h{self.hid}l{self.nls}_" + ("C" if self.isContinuous else "D") \
                                             + ("L" if self.useLSTM else "_")
