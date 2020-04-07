from collections import deque
from typing import Optional, Tuple
import random
import numpy as np
from numpy import ndarray
import time
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from agents.agent import Agent
from environment.pheromone import Pheromone
from environment.RL_api import RLApi
from agents.explore_agent_pytorch import ExploreAgentPytorch

MODEL_NAME = 'Collect_Agent'

REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 5
UPDATE_TARGET_EVERY = 1


class ReplayMemoryDataset(Dataset):
    def __init__(self, max_len, observation_space):
        self.max_len = max_len
        self.observation_space = observation_space

        self.states = torch.zeros([max_len] + list(observation_space), dtype=torch.float32)
        self.actions = torch.zeros(max_len, dtype=int)
        self.rewards = torch.zeros(max_len, dtype=torch.float32)
        self.new_states = torch.zeros([max_len] + list(observation_space), dtype=torch.float32)
        self.dones = torch.zeros(max_len, dtype=bool)

        self.states.requires_grad = False
        self.actions.requires_grad = False
        self.rewards.requires_grad = False
        self.new_states.requires_grad = False
        self.dones.requires_grad = False

        self.head = 0
        self.fill = 0

    def __len__(self):
        return self.fill

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.states[idx], self.actions[idx], self.rewards[idx], self.new_states[idx], self.dones[idx]

    def random_access(self, n):
        indices = random.sample(range(len(self)), n)
        return self[indices]

    def add_safe(self, states, actions, rewards, new_states, done, add):
        begin = self.head
        end = begin + add
        self.states[begin:end] = torch.from_numpy(states[:add])
        self.actions[begin:end] = torch.from_numpy(actions[:add])
        self.rewards[begin:end] = torch.from_numpy(rewards[:add])
        self.new_states[begin:end] = torch.from_numpy(new_states[:add])
        self.dones[begin:end] = torch.ones(add) * done

    def append(self, states, actions, rewards, new_states, done):
        add = min(self.max_len - self.head, len(actions))
        self.add_safe(states, actions, rewards, new_states, done, add)

        self.fill = min(self.max_len, self.head + add)
        self.head = (self.head + add) % self.max_len

        if add != len(actions):
            self.append(states[add:], actions[add:], rewards[add:], new_states[add:], done)



class CollectModel(nn.Module):
    def __init__(self, observation_space, rotations, pheromones, explore_model):
        super(CollectModel, self).__init__()

        self.explore_model = explore_model

        # Freeze model weights for first layer
        for name, p in self.explore_model.named_parameters():
            if "layer1" in name:
                p.requires_grad = False

        input_size = 1
        for dim in observation_space:
            input_size *= dim

        self.layer3 = nn.Linear(32, pheromones)
        self.input_size = input_size

    def forward(self, state):
        out = self.explore_model.layer1(state.view(-1, self.input_size))
        rotations = self.explore_model.layer2(out)
        pheromones = self.layer3(out)
        return rotations, pheromones


class CollectAgent(Agent):
    def __init__(self, epsilon=0.1, discount=0.5, rotations=3, pheromones=2):
        super(CollectAgent, self).__init__("explore_agent_pytorch")

        self.epsilon = epsilon
        self.discount = discount
        self.rotations = rotations
        self.pheromones = pheromones

        self.model = None
        self.target_model = None
        self.criterion = None
        self.optimizer = None

        # An array with last n steps for training
        self.replay_memory = None

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0
        self.state = None

    def setup(self, rl_api: RLApi, trained_model: Optional[str] = None):
        super(CollectAgent, self).setup(rl_api, trained_model)

        self.replay_memory = ReplayMemoryDataset(REPLAY_MEMORY_SIZE, self.observation_space)
        self.state = torch.zeros([rl_api.ants.n_ants] + list(self.observation_space), dtype=torch.float32)

        self.explore_agent = ExploreAgentPytorch(pretrained=True)
        self.explore_agent.setup(rl_api, '6_4_17_explore_agent_pytorch.h5')

        # Main model
        self.model = CollectModel(self.observation_space, self.rotations, self.pheromones, self.explore_agent.model)
        self.target_model = CollectModel(self.observation_space, self.rotations, self.pheromones, self.explore_agent.model)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        if trained_model is not None:
            self.load_model(trained_model)

        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def initialize(self, rl_api: RLApi):
        rl_api.ants.activate_all_pheromones(
            np.ones((self.n_ants, len([obj for obj in rl_api.perceived_objects if isinstance(obj, Pheromone)]))) * 10)


    def train(self, done: bool, step: int) -> float:
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return 0

        # Get a minibatch from replay memory
        mem_states, mem_actions, mem_rewards, mem_new_states, mem_done = self.replay_memory.random_access(
            MINIBATCH_SIZE)
        print("Mem_actions: ", mem_actions)

        with torch.no_grad():
            future_qs_rotation, future_qs_pheromones = self.target_model(mem_new_states)
            target_qs_rotation, target_qs_pheromones = self.model(mem_states)



            # Update q_value for rotation
            max_future_qs = torch.max(future_qs_rotation, dim=1).values
            new_qs = mem_rewards + self.discount * max_future_qs * ~mem_done
            target_qs_rotation[np.arange(len(target_qs_rotation)), mem_actions[0].tolist()] = new_qs[np.arange(len(target_qs_rotation))]

            # Update Q_value for pheromones
            max_future_qs = torch.max(future_qs_pheromones, dim=1).values
            new_qs = mem_rewards + self.discount * max_future_qs * ~mem_done
            future_qs_pheromones[np.arange(len(future_qs_pheromones)), mem_actions[1].tolist()] = new_qs[np.arange(len(future_qs_pheromones))]



        # loss = self.criterion(self.model(mem_states), target_qs)
        loss_rotation = self.criterion(self.model(mem_states)[0], target_qs_rotation)
        loss_pheromones = self.criterion(self.model(mem_states)[1], future_qs_pheromones)

        self.optimizer.zero_grad()
        loss_rotation.backward()
        loss_pheromones.backward()
        self.optimizer.step()

        # Update target network counter every episode
        if done:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter >= UPDATE_TARGET_EVERY:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()
            self.target_update_counter = 0

        return loss_rotation.item(), loss_pheromones.item()

    def update_replay_memory(self, states: ndarray,
                             actions: Tuple[Optional[ndarray], Optional[ndarray]], rewards: ndarray,
                             new_states: ndarray, done: bool):
        self.replay_memory.append(states, (actions[0] + self.rotations // 2, actions[1]), rewards, new_states, done)

    def get_action(self, state: ndarray, training: bool) -> Tuple[
        Optional[ndarray], Optional[ndarray]]:
        if random.random() > self.epsilon or not training:
            # Ask network for next action
            with torch.no_grad():
                #predict = torch.max(self.target_model(torch.Tensor(state)), dim=1).indices.numpy()
                qs_rotation, qs_pheromones = self.target_model(torch.Tensor(state))
                action_rot = torch.max(qs_rotation, dim=1).indices.numpy()
                action_phero = torch.max(qs_pheromones, dim=1).indices.numpy()
            rotation = action_rot - self.rotations // 2
            pheromone = action_phero
        else:
            # Random turn
            rotation = np.random.randint(low=0, high=self.rotations, size=self.n_ants) - self.rotations // 2
            # Random pheromones
            pheromone = np.random.randint(low=0, high=self.pheromones, size=self.n_ants)

        return rotation, None, pheromone

    def save_model(self, file_name: str):
        torch.save(self.model.state_dict(), './agents/models/' + file_name)

    def load_model(self, file_name: str):
        self.model.load_state_dict(torch.load('./agents/models/' + file_name))
        self.target_model.load_state_dict(torch.load('./agents/models/' + file_name))
