import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from agents.PolicyOptimization import PolicyGradients
from utils.Modules import ActorCriticOutput


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
            ActorCriticOutput(hid, out, isContinuous)
        ).to(self.device)
        
        self.criticValues = torch.tensor([]).to(self.device)
        self.logProbs     = torch.tensor([]).to(self.device)

    def getAction(self, inputs):
        action, value = self.forward(inputs)
        self.criticValues = torch.cat([self.criticValues,
                                       torch.as_tensor(value, dtype=torch.float32, device=self.device)])
        if self.isContinuous:
            action_distribution = Normal(*action)
            sampled_action = action_distribution.sample()
        else:
            action_distribution = Categorical(F.softmax(action))
            sampled_action = action_distribution.sample()  #.item()
        # save the log likelihood of taking that action for backprop
        logProb = action_distribution.log_prob(sampled_action)
        self.logProbs = torch.cat([self.logProbs, logProb])
        if not self.isContinuous:
            sampled_action = sampled_action.item()
        return sampled_action

    # gradient of one trajectory
    def backprop(self):
        # self.policy.zero_grad()
        # compute one outputs sequentially if using LSTM
        # if self.useLSTM:
        #     out = torch.tensor([]).to(self.device)
        #     self.clearLSTMState()
        #     for idx in range(len(self.trainStates)):
        #         action, _ = self.forward(self.trainStates[idx])
        #         out = torch.cat([out, *action])
        #         if self.isContinuous:
        #             out = Normal(*out).log_prob(self.trainActions)
        #             out[out != out] = 0  # replace NaNs with 0s
        #         else:
        #             out = Categorical(out).log_prob(self.trainActions)
        # else:
        #     out, _ = self.forward(self.trainStates)
        #     if self.isContinuous:
        #         out = Normal(*out).log_prob(self.trainActions)
        #         out[out!=out] = 0  # replace NaNs with 0s
        #     else:
        #         out = Categorical(out).log_prob(self.trainActions)
        self.logProbs[self.logProbs != self.logProbs] = 0  # replace all nans with 0s
        # critic update
        valueError = self.criticValues - self.trainRewards
        valueGrad = (valueError**2).mean()
        valueGrad.backward(retain_graph=True)
        # actor update
        r = self.criticValues  # advantage  self.trainRewards # -
        # verr = self.criticValues - self.trainRewards//200
        grad = -(self.logProbs * r).mean() #-(value * verr).mean()
        grad.backward()
        print("avg train reward", self.trainRewards.mean(), "grad", grad)
        self.avgRewards = self.trainRewards.mean()
        self.optimizer.step()
        self.policy.zero_grad()
        print("avg estimated reward", self.criticValues.mean())
        # Reset episode buffer
        self.trainRewards = torch.tensor([]).to(self.device)
        self.trainActions = torch.tensor([]).to(self.device)
        self.trainStates  = torch.tensor([]).to(self.device)
        self.criticValues = torch.tensor([]).to(self.device)
        self.logProbs     = torch.tensor([]).to(self.device)
        if self.useLSTM:
            self.clearLSTMState()

    def __str__(self):
        return f"AC_h{self.hid}l{self.nls}_" + ("C" if self.isContinuous else "D") \
                                             + ("L" if self.useLSTM else "_")
