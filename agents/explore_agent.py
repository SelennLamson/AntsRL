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


MODEL_NAME = 'CNN'

REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 256
UPDATE_TARGET_EVERY = 1


class ExploreAgent (Agent):
	def __init__(self, epsilon=0.1, discount=0.5, rotations=3):
		super(ExploreAgent, self).__init__("explore_agent")

		self.epsilon = epsilon
		self.discount = discount
		self.rotations = rotations

		self.model = None
		self.target_model = None

		# An array with last n steps for training
		self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

		# Used to count when to update target network with main network's weights
		self.target_update_counter = 0

		# Custom tensorboard object
		self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

	def setup(self, rl_api: RLApi, trained_model: Optional[str] = None):
		super(ExploreAgent, self).setup(rl_api, trained_model)

		# Main model
		self.model = self.create_model()
		self.target_model = self.create_model()

		if trained_model is not None:
			self.load_model(trained_model)

		self.target_model.set_weights(self.model.get_weights())

	def create_model(self):
		model = Sequential()
		#model.add(Reshape((5, 5, 6), input_shape=self.observation_space))

		#model.add(Conv2D(256, (3, 3)))
		#model.add(Activation('relu'))

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
		current_states = np.array([transition[0] for transition in minibatch])
		current_qs_list = self.model.predict(current_states)

		new_current_states = np.array([transition[3] for transition in minibatch])
		future_qs_list = self.target_model.predict(new_current_states)

		X = []
		y = []

		for index, (current_state, action, reward, new_current_state, done_mb) in enumerate(minibatch):

			# If not a terminal state, get new q from future states, otherwise set it to 0
			# almost like with Q Learning, but we use just part of equation here
			if not done_mb:
				max_future_q = np.max(future_qs_list[index])
				new_q = reward + self.discount * max_future_q
			else:
				new_q = reward

			# Update Q value for given state
			current_qs = current_qs_list[index]
			current_qs[action] = new_q

			# And append to our training data
			X.append(current_state)
			y.append(current_qs)
			# Fit on all samples as one batch, log only on terminal state

		history = self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE,
								 verbose=0,
								 callbacks=[self.tensorboard] if done else None,
								 shuffle=False)

		# Update target network counter every episode
		if done:
			self.target_update_counter += 1

		# If counter reaches set value, update target network with weights of main network
		if self.target_update_counter >= UPDATE_TARGET_EVERY:
			self.target_model.set_weights(self.model.get_weights())
			self.target_update_counter = 0

		return history.history['loss'][0]

	def update_replay_memory(self, states: ndarray, actions: Tuple[Optional[ndarray], Optional[ndarray], Optional[ndarray]], rewards: ndarray, new_states: ndarray, done: bool):
		for i in range(self.n_ants):
			self.replay_memory.append((states[i], actions[0][i] + self.rotations // 2, rewards[i], new_states[i], done))

	def get_action(self, state: ndarray, training: bool) -> Tuple[Optional[ndarray], Optional[ndarray], Optional[ndarray]]:
		if random.random() > self.epsilon or not training:
			# Ask network for next action
			predict = self.target_model.predict(state)
			rotation = np.argmax(predict, axis=1) - self.rotations // 2
		else:
			# Random turn
			rotation = np.random.randint(low=0, high=self.rotations, size=self.n_ants) - self.rotations // 2

		return rotation, None, None

	def save_model(self, file_name: str):
		self.model.save('./agents/models/' + file_name)

	def load_model(self, file_name: str):
		self.model = load_model('./agents/models/' + file_name)
		self.target_model = load_model('./agents/models/' + file_name)
