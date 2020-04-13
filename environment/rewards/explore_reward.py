import numpy as np

from environment.ants import Ants
from environment.rewards.reward import Reward

class ExploreReward(Reward):
    def __init__(self, discount):
        super(ExploreReward, self).__init__(discount)
        self.explored_map = None

    def setup(self, ants: Ants):
        super(ExploreReward, self).setup(ants)
        self.explored_map = np.zeros((self.environment.w, self.environment.h), dtype=bool)

    def observation(self, obs_coords, perception, agent_state):
        # Computing how many new blocks were explored by each ant
        self.rewards = np.sum(1 - self.explored_map[obs_coords[:, :, :, 0], obs_coords[:, :, :, 1]], axis=(1, 2)) * 0.1

        # Writing exploration to exploration map
        self.explored_map[obs_coords[:, :, :, 0], obs_coords[:, :, :, 1]] = True

    def should_visualize(self):
        return True

    def visualization(self):
        return self.explored_map.copy()
