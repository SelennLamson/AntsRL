import numpy as np

from environment.ants import Ants
from environment.rewards.reward import Reward

class ExplorationReward (Reward):
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
	def __init__(self, fct_explore=1, fct_food=1, fct_anthill=5):
		super(All_Rewards, self).__init__()
		self.explored_map = None
		self.fct_explore = fct_explore
		self.fct_food = fct_food
		self.fct_anthill = fct_anthill

	def setup(self, ants: Ants):
		super(All_Rewards, self).setup(ants)
		self.rewards = self.ants.holding
		self.ants_holding = self.ants.holding
		self.explored_map = np.zeros((self.environment.w, self.environment.h), dtype=bool)

	def observation(self, obs_coords, perception, agent_state):
		self.rewards_food = agent_state[:, 0] - self.ants_holding
		self.rewards_anthill = self.rewards_food.copy()
		self.rewards_food[self.rewards_food < 0] = 0
		self.ants_holding = agent_state[:, 0]

		# Computing how many new blocks were explored by each ant
		self.rewards_explore = np.sum(1 - self.explored_map[obs_coords[:, :, :, 0], obs_coords[:, :, :, 1]], axis=(1, 2)) / 10
		self.rewards_explore = np.array([r if i==0 else -r for r, i in zip(self.rewards_explore, self.ants_holding)])
		# Writing exploration to exploration map
		self.explored_map[obs_coords[:, :, :, 0], obs_coords[:, :, :, 1]] = True

		# Reward anthill
		self.rewards_anthill[self.rewards_anthill > 0] = 0
		self.rewards_anthill[self.rewards_anthill < 0] = 1

		self.rewards = self.rewards_explore * self.fct_explore + self.rewards_food * self.fct_food + self.rewards_anthill * self.fct_anthill

	def visualization(self):
		return self.explored_map.copy()
