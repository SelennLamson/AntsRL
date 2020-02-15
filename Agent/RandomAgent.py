import numpy as np
import random

class RandomAgent:

    def __init__(self, n_action):
        self.n_action = n_action

    def choose_action(self, s):

        actions = [None, None, None, None]

        index_a = np.random.randint(2, self.n_action, size=1)[0]

        if index_a>1:
            value = random.uniform(-1, 1)
        else:
            value = False
        actions[index_a] = value
        return actions
