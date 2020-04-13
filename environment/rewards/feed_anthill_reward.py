import numpy as np

from environment.ants import Ants
from environment.rewards.reward import Reward

class FeedAnthillReward(Reward):
    def __init__(self, factor):
        super(FeedAnthillReward, self).__init__(factor)
        self.ants_holding = None

    def setup(self, ants: Ants):
        super(FeedAnthillReward, self).setup(ants)
        self.rewards = np.zeros(ants.n_ants)
        self.ants_holding = self.ants.holding

    def observation(self, obs_coords, perception, agent_state):
        self.rewards = (agent_state[:, 0] < self.ants_holding) * 10.0
        self.ants_holding = agent_state[:, 0]
