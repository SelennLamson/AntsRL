import numpy as np

from environment.ants import Ants
from environment.rewards.reward import Reward

class PickUpFoodReward(Reward):
    def __init__(self, discount):
        super(PickUpFoodReward, self).__init__(discount)
        self.ants_holding = None

    def setup(self, ants: Ants):
        super(PickUpFoodReward, self).setup(ants)
        self.rewards = np.zeros(ants.n_ants)
        self.ants_holding = self.ants.holding

    def observation(self, obs_coords, perception, agent_state):
        self.rewards = (agent_state[:, 0] > self.ants_holding) * 1.0
        self.ants_holding = agent_state[:, 0].copy()
