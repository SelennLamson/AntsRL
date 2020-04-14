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
from agents.replay_memory import ReplayMemory

MODEL_NAME = 'Collect_Agent_Memory'

REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 264
UPDATE_TARGET_EVERY = 1

class CollectModelMemory(nn.Module):
    def __init__(self, observation_space, agent_space, mem_size, rotations, pheromones):
        super(CollectModelMemory, self).__init__()

        self.input_size = 1
        for dim in observation_space:
            self.input_size *= dim

        self.agent_input_size = 1
        for dim in agent_space:
            self.agent_input_size *= dim

        self.mem_size = mem_size

        self.layer1 = nn.Linear(self.input_size + self.agent_input_size + self.mem_size, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 32)
        self.layer4 = nn.Linear(32, self.input_size + self.agent_input_size + self.mem_size)

        self.rotation_layer1 = nn.Linear(self.input_size + self.agent_input_size + self.mem_size, 64)
        self.rotation_layer2 = nn.Linear(64, 128)
        self.rotation_layer3 = nn.Linear(128, rotations)

        self.pheromone_layer1 = nn.Linear(self.input_size + self.agent_input_size + self.mem_size, 32)
        self.pheromone_layer2 = nn.Linear(32, pheromones)

        self.memory_layer1 = nn.Linear(self.input_size + self.agent_input_size + self.mem_size, 64)
        self.memory_layer2 = nn.Linear(64, 64)
        self.memory_layer3 = nn.Linear(64, self.mem_size)
        self.forget_layer = nn.Linear(64, self.mem_size)

    def forward(self, state, agent_state):
        old_memory = agent_state[:, self.agent_input_size:]
        all_input = torch.cat([state.view(-1, self.input_size), agent_state.view(-1, self.agent_input_size + self.mem_size)], dim=1)

        general = torch.relu(self.layer1(all_input))
        general = torch.relu(self.layer2(general))
        general = torch.relu(self.layer3(general))
        general = self.layer4(general)

        rotation = self.rotation_layer1(general + all_input)
        rotation = self.rotation_layer2(rotation)
        rotation = self.rotation_layer3(rotation)

        pheromone = self.pheromone_layer1(general + all_input)
        pheromone = self.pheromone_layer2(pheromone)

        memory = self.memory_layer1(general + all_input)
        memory = self.memory_layer2(memory)
        new_memory = torch.tanh(self.memory_layer3(memory))
        forget_fact = torch.sigmoid(self.forget_layer(memory))
        new_memory = new_memory * forget_fact + old_memory * (1 - forget_fact)

        return rotation, pheromone, new_memory


class CollectAgentMemory(Agent):
    def __init__(self, epsilon=0.1, discount=0.5, rotations=3, pheromones=3, learning_rate=1e-4):
        super(CollectAgentMemory, self).__init__("collect_agent_memory")

        self.learning_rate = learning_rate

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

        self.mem_size = 10
        self.agent_and_mem_space = None
        self.previous_memory = None

    def setup(self, rl_api: RLApi, trained_model: Optional[str] = None):
        super(CollectAgentMemory, self).setup(rl_api, trained_model)

        self.previous_memory = torch.zeros((rl_api.ants.n_ants, self.mem_size))
        self.agent_and_mem_space = [2 + self.mem_size]

        self.replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE, self.observation_space, self.agent_and_mem_space, self.action_space)
        self.state = torch.zeros([rl_api.ants.n_ants] + list(self.observation_space), dtype=torch.float32)

        # Main model
        self.model = CollectModelMemory(self.observation_space, self.agent_space, self.mem_size, self.rotations, self.pheromones)
        self.target_model = CollectModelMemory(self.observation_space, self.agent_space, self.mem_size, self.rotations, self.pheromones)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

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
        mem_states, mem_agent_state, mem_actions, mem_rewards, mem_new_states, mem_new_agent_state, mem_done = self.replay_memory.random_access(
            MINIBATCH_SIZE)

        with torch.no_grad():
            # Predicting actions (we don't use agent's memory)
            future_qs_rotation, future_qs_pheromones, _ = self.target_model(mem_new_states, mem_new_agent_state)
            target_qs_rotation, target_qs_pheromones, _ = self.model(mem_states, mem_agent_state)

            # Update Q value for rotation
            max_future_qs = torch.max(future_qs_rotation, dim=1).values
            new_qs = mem_rewards + self.discount * max_future_qs * ~mem_done
            target_qs_rotation[np.arange(len(target_qs_rotation)), mem_actions[:, 0].tolist()] = new_qs[np.arange(len(target_qs_rotation))]

            # Update Q value for pheromones
            max_future_qs = torch.max(future_qs_pheromones, dim=1).values
            new_qs = mem_rewards + self.discount * max_future_qs * ~mem_done
            target_qs_pheromones[np.arange(len(target_qs_pheromones)), mem_actions[:, 1].tolist()] = new_qs[np.arange(len(target_qs_pheromones))]

        output = self.model(mem_states, mem_agent_state)
        loss_rotation = self.criterion(output[0], target_qs_rotation)
        loss_pheromones = self.criterion(output[1], target_qs_pheromones)
        loss = loss_rotation + loss_pheromones

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network counter every episode
        if done:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter >= UPDATE_TARGET_EVERY:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()
            self.target_update_counter = 0

        return loss.item()

    def update_replay_memory(self, states: ndarray, agent_state: ndarray,
                             actions: Tuple[Optional[ndarray], Optional[ndarray]], rewards: ndarray,
                             new_states: ndarray, new_agent_states: ndarray, done: bool):
        self.replay_memory.extend(states,
                                  np.hstack([agent_state, self.previous_memory]),
                                  (actions[0] + self.rotations // 2, actions[1]),
                                  rewards,
                                  new_states,
                                  np.hstack([new_agent_states, actions[2]]),
                                  done)

    def get_action(self, state: ndarray, agent_state: ndarray, training: bool) -> Tuple[Optional[ndarray], Optional[ndarray]]:
        if random.random() > self.epsilon or not training:
            # Ask network for next action
            with torch.no_grad():
                #predict = torch.max(self.target_model(torch.Tensor(state)), dim=1).indices.numpy()
                qs_rotation, qs_pheromones, self.previous_memory = self.target_model(torch.Tensor(state), torch.cat([torch.Tensor(agent_state), self.previous_memory], dim=1))
                action_rot = torch.max(qs_rotation, dim=1).indices.numpy()
                action_phero = torch.max(qs_pheromones, dim=1).indices.numpy()
            rotation = action_rot - self.rotations // 2
            pheromone = action_phero
        else:
            # Random turn
            rotation = np.random.randint(low=0, high=self.rotations, size=self.n_ants) - self.rotations // 2
            # Random pheromones
            pheromone = np.random.randint(low=0, high=self.pheromones, size=self.n_ants)
            # We don't reset memory to zero, we keep previous value

        return rotation, pheromone, self.previous_memory.numpy()

    def save_model(self, file_name: str):
        torch.save(self.model.state_dict(), './agents/models/' + file_name)

    def load_model(self, file_name: str):
        self.model.load_state_dict(torch.load('./agents/models/' + file_name))
        self.target_model.load_state_dict(torch.load('./agents/models/' + file_name))
