import numpy as np

from environment.ants import Ants
from environment.rewards.reward import Reward
from environment.anthill import Anthill

class AnthillHeadingReward(Reward):
    def __init__(self, factor):
        super(AnthillHeadingReward, self).__init__(factor)
        self.previous_dist = None
        self.anthill_x = 0
        self.anthill_y = 0
        self.ants = None

    def compute_distance(self, x, y):
        return ((x - self.anthill_x)**2 + (y - self.anthill_y)**2)**0.5

    def setup(self, ants: Ants):
        super(AnthillHeadingReward, self).setup(ants)
        self.rewards = np.zeros(ants.n_ants)
        self.ants = ants

        for obj in ants.environment.objects:
            if isinstance(obj, Anthill):
                self.anthill_x = obj.x
                self.anthill_y = obj.y

        self.previous_dist = self.compute_distance(ants.x, ants.y)

    def observation(self, obs_coords, perception, agent_state):
        new_dist = self.compute_distance(self.ants.x, self.ants.y)
        self.rewards = (self.previous_dist > new_dist) * (self.ants.holding > 0) * 0.1
        self.previous_dist = new_dist
