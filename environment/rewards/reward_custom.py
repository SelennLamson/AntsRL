import numpy as np

from environment.ants import Ants
from environment.rewards.reward import Reward


class All_Rewards(Reward):
    def __init__(self, fct_explore=1, fct_food=1, fct_anthill=5, fct_explore_holding=1):
        super(All_Rewards, self).__init__()
        self.explored_map = None
        self.fct_explore = fct_explore
        self.fct_food = fct_food
        self.fct_anthill = fct_anthill
        self.fct_explore_holding = fct_explore_holding

        self.ants_holding = None

    def setup(self, ants: Ants):
        super(All_Rewards, self).setup(ants)
        self.rewards = self.ants.holding
        self.ants_holding = self.ants.holding
        self.explored_map = np.zeros((self.environment.w, self.environment.h), dtype=bool)

    def observation(self, obs_coords, perception, agent_state):
        self.rewards = np.zeros_like(self.rewards)

        rewards_food = agent_state[:, 0] - self.ants_holding
        rewards_anthill = rewards_food.copy()
        rewards_food[rewards_food < 0] = 0
        self.ants_holding = agent_state[:, 0]

        if self.fct_explore != 0 or self.fct_explore_holding != 0:
            # Computing how many new blocks were explored by each ant
            rewards_explore = np.sum(1 - self.explored_map[obs_coords[:, :, :, 0], obs_coords[:, :, :, 1]],
                                          axis=(1, 2)) / 10
            rewards_explore = np.array([r * self.fct_explore if h == 0 else r * self.fct_explore_holding for r, h in zip(rewards_explore, self.ants_holding)])
            # Writing exploration to exploration map
            self.explored_map[obs_coords[:, :, :, 0], obs_coords[:, :, :, 1]] = True
            self.rewards += rewards_explore

        # Reward anthill
        rewards_anthill[rewards_anthill > 0] = 0
        rewards_anthill[rewards_anthill < 0] = 1

        self.rewards += rewards_food * self.fct_food + rewards_anthill * self.fct_anthill

    def visualization(self):
        return self.explored_map.copy()
