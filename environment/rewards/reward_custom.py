import numpy as np

from environment.ants import Ants
from environment.anthill import Anthill
from environment.rewards.reward import Reward


class ExplorationReward(Reward):
    def __init__(self, ):
        super(ExplorationReward, self).__init__()
        self.explored_map = None

    def setup(self, ants: Ants):
        super(ExplorationReward, self).setup(ants)
        self.explored_map = np.zeros((self.environment.w, self.environment.h), dtype=bool)

    def observation(self, obs_coords, perception, agent_state):
        # Computing how many new blocks were explored by each ant
        self.rewards = np.sum(1 - self.explored_map[obs_coords[:, :, :, 0], obs_coords[:, :, :, 1]], axis=(1, 2)) / 10

        # Writing exploration to exploration map
        self.explored_map[obs_coords[:, :, :, 0], obs_coords[:, :, :, 1]] = True

    def visualization(self):
        return self.explored_map.copy()


class Food_Reward(Reward):
    def __init__(self, ):
        super(Food_Reward, self).__init__()

    def setup(self, ants: Ants):
        super(Food_Reward, self).setup(ants)
        self.rewards = self.ants.holding
        self.ants_holding = self.ants.holding

    def observation(self, obs_coords, perception, agent_state):
        self.rewards = agent_state[:, 0] - self.ants_holding
        self.rewards[self.rewards < 0] = 10  # Deals with negative rewards when ants drops the food
        self.ants_holding = agent_state[:, 0]


class All_Rewards(Reward):
    def __init__(self, fct_explore=1, fct_food=1, fct_anthill=5, fct_explore_holding=0, fct_headinganthill=1):
        super(All_Rewards, self).__init__()
        self.explored_map = None
        self.fct_explore = fct_explore
        self.fct_food = fct_food
        self.fct_anthill = fct_anthill
        self.fct_explore_holding = fct_explore_holding
        self.fct_headinganthill = fct_headinganthill

        self.previous_dist = None
        self.anthill_x = 0
        self.anthill_y = 0

        self.ants_holding = None

    def compute_distance(self, x, y):
        return ((x - self.anthill_x) ** 2 + (y - self.anthill_y) ** 2) ** 0.5



    def setup(self, ants: Ants):
        super(All_Rewards, self).setup(ants)
        self.rewards = self.ants.holding
        self.ants_holding = self.ants.holding
        self.explored_map = np.zeros((self.environment.w, self.environment.h), dtype=bool)
        self.rewards_anthillheading = np.zeros(ants.n_ants)
        self.ants = ants

        for obj in ants.environment.objects:
            if isinstance(obj, Anthill):
                self.anthill_x = obj.x
                self.anthill_y = obj.y

        self.previous_dist = self.compute_distance(ants.x, ants.y)

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

        #Heading anthill

        new_dist = self.compute_distance(self.ants.x, self.ants.y)
        self.rewards_anthillheading = (self.previous_dist > new_dist) * (self.ants.holding > 0) * 0.1
        self.previous_dist = new_dist

        self.rewards += rewards_food * self.fct_food + rewards_anthill * self.fct_anthill + self.rewards_anthillheading * self.fct_headinganthill

    def visualization(self):
        return self.explored_map.copy()
