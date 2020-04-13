import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random

class ReplayMemory(Dataset):
    """Rolling replay memory: arrays are initialized at full length but elements are inserted at current head cursor.
    Writing is rolling, meaning that when head reaches the maximum length of arrays, it cycles back to the beginning
    and overwrites old elements."""

    def __init__(self, max_len, n_rewards, observation_space, agent_space, action_space):
        self.max_len = max_len
        self.n_rewards = n_rewards
        self.observation_space = observation_space
        self.agent_space = agent_space
        self.action_space = action_space

        # Initializing zero-arrays of full length
        self.states = torch.zeros([max_len] + list(observation_space), dtype=torch.float32)
        self.agent_states = torch.zeros([max_len] + list(agent_space), dtype=torch.float32)
        self.actions = torch.zeros([max_len] + list(action_space), dtype=int)
        self.rewards = torch.zeros([max_len, n_rewards], dtype=torch.float32)
        self.new_states = torch.zeros([max_len] + list(observation_space), dtype=torch.float32)
        self.new_agent_states = torch.zeros([max_len] + list(agent_space), dtype=torch.float32)
        self.dones = torch.zeros(max_len, dtype=bool)

        # None of them require gradient computation, save some resources:
        self.states.requires_grad = False
        self.agent_states.requires_grad = False
        self.actions.requires_grad = False
        self.rewards.requires_grad = False
        self.new_states.requires_grad = False
        self.new_agent_states.requires_grad = False
        self.dones.requires_grad = False

        # Current writing head in memory
        self.head = 0

        # Current quantity of elements added to the memory ; will saturate at 'max_len'
        self.fill = 0

    def __len__(self):
        """Retrieves the length of the replay memory (only available entries)"""
        return self.fill

    def __getitem__(self, idx):
        """Retrieves item at given index or list of indices.
        :param idx: integer, list of integers or integer tensor
        :return tuple containing the slices of replay memory arrays at given indices"""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.states[idx], self.agent_states[idx], self.actions[idx], self.rewards[idx], self.new_states[idx], self.new_agent_states[idx], self.dones[idx]

    def random_access(self, n):
        """Retrieves n random elements from replay memory.
        :param n: number of elements to retrieve
        :return tuple : states, agent_states, actions, rewards, new_states, new_agent_states, dones ; random corresponding slices of replay memory arrays"""
        indices = random.sample(range(len(self)), n)
        return self[indices]

    def _extend_unsafe(self, states, agent_states, actions, rewards, new_states, new_agent_states, done, add):
        """PRIVATE | Extends current memory with given arrays, up to 'add' elements.
        Unsafe: no length check. This function shouldn't be called from outside this class.
        :param states: world observations
        :param agent_states: agent observations
        :param actions: taken actions
        :param rewards: received rewards
        :param new_states: world observations after taking action
        :param new_agent_states: agent observations after taking action
        :param done: is episode done?
        :param add: number of elements to add, from the beginning of passed arrays
        """
        begin = self.head
        end = begin + add
        self.states[begin:end] = torch.from_numpy(states[:add])
        self.agent_states[begin:end] = torch.from_numpy(agent_states[:add])
        self.actions[begin:end] = torch.from_numpy(actions[:add])
        self.rewards[begin:end] = torch.from_numpy(rewards[:add])
        self.new_states[begin:end] = torch.from_numpy(new_states[:add])
        self.new_agent_states[begin:end] = torch.from_numpy(new_agent_states[:add])
        self.dones[begin:end] = torch.ones(add) * done

    def extend(self, states, agent_states, actions, rewards, new_states, new_agent_states, done):
        """ Extends the replay memory with all given entries. Memory writing is rolling: when reaching
        saturation, old elements are overwritten automatically.
        :param states: 				ndarray[ _ , observation_space]		world observations
        :param agent_states: 		ndarray[ _ , agent_space]			agent observations
        :param actions: 			ndarray[ _ , action_space]			taken actions
        :param rewards: 			ndarray[ _ ]						received rewards
        :param new_states: 			ndarray[ _ , observation_space] 	world observations after taking action
        :param new_agent_states: 	ndarray[ _ , agent_space]			agent observations after taking action
        :param done: 				bool								is episode done?
        """

        # How many elements can we add at current head position, without overflowing?
        add = min(self.max_len - self.head, len(actions[0]))

        # Concatenate and reshape actions to have a ndarray of shape (n_ants, 2)
        if actions[1] is not None:
            actions = np.stack((actions[0], actions[1]), axis=-1)
        else:
            actions = np.stack((actions[0], np.ones(actions[0].shape)), axis=-1)

        # Add those elements
        self._extend_unsafe(states, agent_states, actions, rewards, new_states, new_agent_states, done, add)

        # Updating fill (how much space is left in the replay memory, before saturation)
        self.fill = max(self.fill, min(self.max_len, self.head + add))

        # Updating head position, putting it back to 0 if reached max length
        self.head = (self.head + add) % self.max_len

        # If not all elements where added, recursively call 'extend', cycling back to beginning of array
        if add != len(actions):
            self.extend(states[add:], agent_states[add:], actions[add:], rewards[add:], new_states[add:], new_agent_states[add:], done)
