from agents.DQLearn import DQLearn


class DDPG(DQLearn):
    def __init__(self, inp, hid, action_space, useLSTM=False, nLayers=1, usewandb=False, epsilon=0.9, buffLenght=1):
        super().__init__(inp, hid, action_space, useLSTM=False, nLayers=1, usewandb=False, epsilon=0.9, buffLenght=1)
