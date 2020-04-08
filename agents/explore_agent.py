from collections import deque
from typing import Optional, Tuple
import random
import numpy as np
from numpy import ndarray
import time

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, Reshape
from keras.optimizers import Adam
from ModifiedTensorBoard import ModifiedTensorBoard

from agents.agent import Agent
from environment.pheromone import Pheromone
from environment.RL_api import RLApi


MODEL_NAME = 'Explore_Agent'

REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 256
UPDATE_TARGET_EVERY = 1


class ReplayMemoryDataset:
	def __init__(self, max_len, observation_space):
		self.max_len = max_len
		self.observation_space = observation_space

		self.states = np.zeros([max_len] + list(observation_space), dtype=float)
		self.actions = np.zeros(max_len, dtype=int)
		self.rewards = np.zeros(max_len, dtype=float)
		self.new_states = np.zeros([max_len] + list(observation_space), dtype=float)
		self.dones = np.zeros(max_len, dtype=bool)

		self.head = 0
		self.fill = 0

	def __len__(self):
		return self.fill

	def __getitem__(self, idx):
		return self.states[idx], self.actions[idx], self.rewards[idx], self.new_states[idx], self.dones[idx]

	def random_access(self, n):
		indices = random.sample(range(len(self)), n)
		return self[indices]

	def add_safe(self, states, actions, rewards, new_states, done, add):
		begin = self.head
		end = begin + add
		self.states[begin:end] = states[:add]
		self.actions[begin:end] = actions[:add]
		self.rewards[begin:end] = rewards[:add]
		self.new_states[begin:end] = new_states[:add]
		self.dones[begin:end] = np.ones(add) * done

	def append(self, states, actions, rewards, new_states, done):
		add = min(self.max_len - self.head, len(actions))
		self.add_safe(states, actions, rewards, new_states, done, add)

		self.fill = min(self.max_len, self.head + add)
		self.head = (self.head + add) % self.max_len

		if add != len(actions):
			self.append(states[add:], actions[add:], rewards[add:], new_states[add:], done)


class ExploreAgent(Agent):
	def __init__(self, epsilon=0.1, discount=0.5, rotations=3, pheromones=3):
		super(ExploreAgent, self).__init__("explore_agent")

		self.epsilon = epsilon
		self.discount = discount
		self.rotations = rotations

		self.model = None
		self.target_model = None

		# An array with last n steps for training
		self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
		# self.replay_memory = None

		# Used to count when to update target network with main network's weights
		self.target_update_counter = 0

	def setup(self, rl_api: RLApi, trained_model: Optional[str] = None):
		super(ExploreAgent, self).setup(rl_api, trained_model)

		# self.replay_memory = ReplayMemoryDataset(REPLAY_MEMORY_SIZE, self.observation_space)

		# Main model
		self.model = self.create_model()
		self.target_model = self.create_model()

		if trained_model is not None:
			self.load_model(trained_model)

		self.target_model.set_weights(self.model.get_weights())

	def create_model(self):
		model = Sequential()
		model.add(Flatten(input_shape=self.observation_space))
		model.add(Dense(32))
		model.add(Dense(self.rotations, activation='linear'))
		model.compile(loss="mse", optimizer=Adam(lr=0.0001), metrics=['accuracy'])
		return model

	def initialize(self, rl_api: RLApi):
		rl_api.ants.activate_all_pheromones(np.ones((self.n_ants, len([obj for obj in rl_api.perceived_objects if isinstance(obj, Pheromone)]))) * 10)

	def train(self, done: bool, step: int) -> float:
		# Start training only if certain number of samples is already saved
		if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
			return 0


		minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
		# mem_states, mem_actions, mem_rewards, mem_new_states, mem_done = self.replay_memory.random_access(MINIBATCH_SIZE)
		mem_states = np.array([s[0] for s in minibatch], dtype=float)
		mem_actions = np.array([s[1] for s in minibatch], dtype=int)
		mem_rewards = np.array([s[2] for s in minibatch], dtype=float)
		mem_new_states = np.array([s[3] for s in minibatch], dtype=float)
		mem_done = np.array([s[4] for s in minibatch], dtype=bool)

		target_qs = self.model.predict(mem_states)
		future_qs = self.target_model.predict(mem_new_states)

		max_future_qs = np.max(future_qs, axis=1)

		new_q = mem_rewards + self.discount * max_future_qs * mem_done

		target_qs[np.arange(len(target_qs)), mem_actions] = new_q[np.arange(len(target_qs))]

		history = self.model.fit(mem_states, target_qs, batch_size=MINIBATCH_SIZE,
								 verbose=0,
								 shuffle=False)

		# Update target network counter every episode
		if done:
			self.target_update_counter += 1

		# If counter reaches set value, update target network with weights of main network
		if self.target_update_counter >= UPDATE_TARGET_EVERY:
			self.target_model.set_weights(self.model.get_weights())
			self.target_update_counter = 0

		return history.history['loss'][0]

	def update_replay_memory(self, states: ndarray, agent_state: ndarray,
							 actions: Tuple[Optional[ndarray], Optional[ndarray]], rewards: ndarray,
							 new_states: ndarray, new_agent_states: ndarray, done: bool):
		# self.replay_memory.append(states, actions[0] + self.rotations // 2, rewards, new_states, done)
		for i in range(self.n_ants):
			self.replay_memory.append((states[i], actions[0][i] + self.rotations // 2, rewards[i], new_states[i], done))

	def get_action(self, state: ndarray, training: bool) -> Tuple[Optional[ndarray], Optional[ndarray]]:
		if random.random() > self.epsilon or not training:
			# Ask network for next action
			predict = self.target_model.predict(state)
			rotation = np.argmax(predict, axis=1) - self.rotations // 2
		else:
			# Random turn
			rotation = np.random.randint(low=0, high=self.rotations, size=self.n_ants) - self.rotations // 2

		return rotation, None

	def save_model(self, file_name: str):
		self.model.save('./agents/models/' + file_name)

	def load_model(self, file_name: str):
		self.model = load_model('./agents/models/' + file_name)
		self.target_model = load_model('./agents/models/' + file_name)
