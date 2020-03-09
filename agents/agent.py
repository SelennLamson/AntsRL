import numpy as np
from numpy import ndarray
from typing import Tuple, Optional
from abc import ABC

from environment.RL_api import RLApi

class Agent(ABC):
	def __init__(self, name: str):
		self.name = name
		self.observation_space = None
		self.n_ants = 0

	def setup(self, rl_api: RLApi, trained_model: Optional[str] = None):
		"""
		Creates the model and sets it up.
		:param rl_api: the RL Api through which ants are controlled
		:param trained_model: an optional file name of a model to load
		"""
		self.observation_space = (rl_api.perception_coords.shape[0], rl_api.perception_coords.shape[1], len(rl_api.perceived_objects))
		self.n_ants = rl_api.ants.n_ants

	def initialize(self, rl_api: RLApi):
		"""
		Initializes the agents on a new environment (called at each new episode).
		:param rl_api: the RL Api to access the environment
		"""
		pass

	def train(self, done: bool, step: int) -> float:
		"""
		Trains the network after a step.
		:param done: Is the episode finished?
		:param step: Number of the step
		:return: mean loss over that training session
		"""
		return 0

	def update_replay_memory(self, states: ndarray, actions: ndarray, rewards: ndarray, new_states: ndarray, done: bool):
		"""
		Registers a new transition into the replay memory.
		:param states: the initial states of each ant before action
		:param actions: the action taken by each ant
		:param rewards: the reward earned by each ant
		:param new_states: the states reached by each ant after the action
		:param done: is the episode finished?
		"""
		pass

	def get_action(self, state: ndarray, training: bool) -> Tuple[Optional[ndarray], Optional[ndarray], Optional[ndarray]]:
		"""
		Computes the action to perform in a certain observed state. The returned arrays can be None if the model works with default action.
		:param state: the observed state of each ant
		:param training: are we in a training phase?
		:return: the actions of each ant, which is a tuple containing: the rotation action, the mandibles state and the pheromones activation
		"""
		return None, None, None

	def save_model(self, file_name: str):
		"""
		Saves the model weights.
		"""
		pass

	def load_model(self, file_name: str):
		"""
		Loads the model weights.
		"""
		pass
