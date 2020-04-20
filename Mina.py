"""

"""

import numpy as np


class Agent:
    def __init__(self):
        self.state = []
        self.select_input = []  # Input vector that is considered

    def predict(self, input):
        pass

    def add_vector(self, layer_n):
        """Add new distinct vector"""
        self.state[layer_n]

    def node(self, layer_n):
        self.state[layer_n] = np.vstack((self.state[layer_n], ))

    def connect_vectors(self):
        """Connect vectors associating them together"""

    def collapse_vectors(self):
        """Replace several vectors with one"""
