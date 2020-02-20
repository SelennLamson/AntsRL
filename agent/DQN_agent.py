from collections import deque
import random
import numpy as np
import time

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, Reshape
from keras.optimizers import Adam
from ModifiedTensorBoard import ModifiedTensorBoard

MODEL_NAME = 'CNN'

REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY_SIZE = 300
MINIBATCH_SIZE = 128
DISCOUNT = 0.2
UPDATE_TARGET_EVERY = 2


class DQNAgent:
    def __init__(self, n_ants):
        self.observation_space = (n_ants, 5, 5, 6)

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))


    def create_model(self):
        model = Sequential()
        model.add(Reshape((5, 5, 6), input_shape=self.observation_space))
        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(32))
        model.add(Dense(3, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def train(self, terminal_state, step):
        """
        Function to train our agent. Huge thanks to @sentdex for the code and the inspiration.
        :param terminal_state:
        :param step:
        """

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)
            # Fit on all samples as one batch, log only on terminal state

        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE,
                       verbose=0,
                       callbacks=[self.tensorboard] if terminal_state else None,
                       shuffle=False)
        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1
        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

