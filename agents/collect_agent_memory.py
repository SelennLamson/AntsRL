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
MIN_REPLAY_MEMORY_SIZE = 2000
MINIBATCH_SIZE = 1024
UPDATE_TARGET_EVERY = 1

class CollectModelMemory(nn.Module):
    def __init__(self, observation_space, agent_space, mem_size, rotations, pheromones, rewards):
        super(CollectModelMemory, self).__init__()

        self.input_size = 1
        for dim in observation_space:
            self.input_size *= dim
        self.observation_space = observation_space

        self.agent_input_size = 1
        for dim in agent_space:
            self.agent_input_size *= dim

        self.mem_size = mem_size

        self.rotations = rotations
        self.pheromones = pheromones
        self.rewards = rewards

        power = 5 # was 4 before

        def p2(x):
            return 2**(x + power)

        self.vision_out = p2(3)
        self.vision3_in = (observation_space[0] - 4) * (observation_space[1] - 4) * p2(1)

        self.vision1 = nn.Conv2d(observation_space[2], p2(1), (3, 3))
        self.vision2 = nn.Conv2d(p2(1), p2(1), (3, 3))
        self.vision3 = nn.Linear(self.vision3_in, self.vision_out)

        self.layer1 = nn.Linear(self.vision_out + self.agent_input_size + self.mem_size, p2(2))
        self.layer2 = nn.Linear(p2(2), p2(2))
        self.layer3 = nn.Linear(p2(2), p2(2))
        self.layer4 = nn.Linear(p2(2), self.vision_out + self.agent_input_size + self.mem_size)

        self.rotation_layer1 = nn.Linear(self.vision_out + self.agent_input_size + self.mem_size, p2(2))
        self.rotation_layer2 = nn.Linear(p2(2), p2(3))
        self.rotation_layer3 = nn.Linear(p2(3), p2(3))
        self.rotation_layer4 = nn.Linear(p2(3), self.rotations * self.rewards)

        self.pheromone_layer1 = nn.Linear(self.vision_out + self.agent_input_size + self.mem_size, p2(3))
        self.pheromone_layer2 = nn.Linear(p2(3), self.pheromones * self.rewards)

        self.memory_layer1 = nn.Linear(self.vision_out + self.agent_input_size + self.mem_size, p2(2))
        self.memory_layer2 = nn.Linear(p2(2), p2(3))
        self.memory_layer3 = nn.Linear(p2(3), self.mem_size)
        self.forget_layer = nn.Linear(p2(3), self.mem_size)

    def forward(self, state, agent_state):
        vision = torch.relu(self.vision1(state.transpose(1, 3)))
        vision = torch.relu(self.vision2(vision))
        vision = self.vision3(vision.view(-1, self.vision3_in))

        old_memory = agent_state[:, self.agent_input_size:]
        all_input = torch.cat([vision, agent_state.view(-1, self.agent_input_size + self.mem_size)], dim=1)

        general = torch.relu(self.layer1(all_input))
        general = torch.relu(self.layer2(general))
        general = torch.relu(self.layer3(general))
        general = self.layer4(general)

        rotation = self.rotation_layer1(general + all_input)
        rotation = self.rotation_layer2(rotation)
        rotation = self.rotation_layer3(rotation)
        rotation = self.rotation_layer4(rotation).view(-1, self.rotations, self.rewards)

        pheromone = self.pheromone_layer1(general + all_input)
        pheromone = self.pheromone_layer2(pheromone).view(-1, self.pheromones, self.rewards)

        memory = self.memory_layer1(general + all_input)
        memory = self.memory_layer2(memory)
        new_memory = torch.tanh(self.memory_layer3(memory))
        forget_fact = torch.sigmoid(self.forget_layer(memory))
        new_memory = new_memory * forget_fact + old_memory * (1 - forget_fact)

        return rotation, pheromone, new_memory


class CollectAgentMemory(Agent):
    def __init__(self, rewards, epsilon=0.1, learning_rate=1e-5, rotations=3, pheromones=3):
        super(CollectAgentMemory, self).__init__("collect_agent_memory")

        self.rewards = rewards

        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount = torch.Tensor([r.discount for r in rewards])
        if torch.cuda.is_available():
            self.discount = self.discount.cuda()
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

        self.mem_size = 20
        self.agent_and_mem_space = None
        self.previous_memory = None

    def setup(self, rl_api: RLApi, trained_model: Optional[str] = None):
        super(CollectAgentMemory, self).setup(rl_api, trained_model)

        self.previous_memory = torch.zeros((rl_api.ants.n_ants, self.mem_size))
        self.agent_and_mem_space = [self.agent_space[0] + self.mem_size]

        self.replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE, len(self.rewards), self.observation_space, self.agent_and_mem_space, self.action_space)
        self.state = torch.zeros([rl_api.ants.n_ants] + list(self.observation_space), dtype=torch.float32)

        # Main model
        self.model = CollectModelMemory(self.observation_space, self.agent_space, self.mem_size, self.rotations, self.pheromones, len(self.rewards))
        self.target_model = CollectModelMemory(self.observation_space, self.agent_space, self.mem_size, self.rotations, self.pheromones, len(self.rewards))
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if trained_model is not None:
            self.load_model(trained_model)

        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        if torch.cuda.is_available():
            self.model.cuda()
            self.target_model.cuda()

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
            # seeds = mem_agent_state[:, 1]

            # Predicting actions (we don't use agent's memory)
            future_qs_rotation, future_qs_pheromones, _ = self.target_model(mem_new_states, mem_new_agent_state)
            target_qs_rotation, target_qs_pheromones, _ = self.model(mem_states, mem_agent_state)

            # Computing total Q value to determine max action
            future_qs_rot = future_qs_rotation.clone()
            future_qs_phe = future_qs_pheromones.clone()
            for i, r in enumerate(self.rewards):
                future_qs_rot[:, :, i] *= r.factor
                future_qs_phe[:, :, i] *= r.factor
                # specificity = 0.2 + 0.8 * (seeds >= 1 / len(self.rewards) * i) * (seeds <= 1 / len(self.rewards) * (i + 1))
                # future_qs_rot[:, :, i] *= r.factor * specificity.view(MINIBATCH_SIZE, -1)
                # future_qs_phe[:, :, i] *= r.factor * specificity.view(MINIBATCH_SIZE, -1)
            future_qs_rot = torch.sum(future_qs_rot, axis=2)
            future_qs_phe = torch.sum(future_qs_phe, axis=2)
            max_rot_action = torch.argmax(future_qs_rot, axis=1)
            max_phe_action = torch.argmax(future_qs_phe, axis=1)

            # Update Q value for rotation
            new_qs_rot = mem_rewards + self.discount.view(-1, len(self.rewards)) * future_qs_rotation[np.arange(MINIBATCH_SIZE), max_rot_action.tolist(), :] * ~mem_done.view(MINIBATCH_SIZE, -1)
            target_qs_rotation[np.arange(MINIBATCH_SIZE), mem_actions[:, 0].tolist()] = new_qs_rot[np.arange(MINIBATCH_SIZE)]

            # Update Q value for pheromones
            new_qs_phe = mem_rewards + self.discount.view(-1, len(self.rewards)) * future_qs_pheromones[np.arange(MINIBATCH_SIZE), max_phe_action.tolist(), :] * ~mem_done.view(MINIBATCH_SIZE, -1)
            target_qs_pheromones[np.arange(MINIBATCH_SIZE), mem_actions[:, 1].tolist()] = new_qs_phe[np.arange(MINIBATCH_SIZE)]

        pred_qs_rotation, pred_qs_pheromones, _ = self.model(mem_states, mem_agent_state)

        loss_rotation = self.criterion(pred_qs_rotation, target_qs_rotation)
        loss_pheromones = self.criterion(pred_qs_pheromones, target_qs_pheromones)

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
                                  np.clip(rewards, -1, 1),
                                  new_states,
                                  np.hstack([new_agent_states, actions[2]]),
                                  done)

    def get_action(self, state: ndarray, agent_state: ndarray, training: bool) -> Tuple[Optional[ndarray], Optional[ndarray]]:
        if random.random() > self.epsilon or not training:
            # Ask network for next action
            with torch.no_grad():
                qs_rotation, qs_pheromones, self.previous_memory = self.target_model(torch.Tensor(state).cuda(), torch.cat([torch.Tensor(agent_state), self.previous_memory], dim=1).cuda())
                self.previous_memory = self.previous_memory.cpu()

                # seeds = agent_state[:, 1]

                # Computing total Q value to determine max action
                qs_rot = qs_rotation.cpu().numpy()
                qs_phe = qs_pheromones.cpu().numpy()
                for i, r in enumerate(self.rewards):
                    qs_rot[:, :, i] *= r.factor
                    qs_phe[:, :, i] *= r.factor

                    # specificity = 0.2 + 0.8 * (seeds >= 1 / len(self.rewards) * i) * (seeds <= 1 / len(self.rewards) * (i + 1))

                    # qs_rot[:, :, i] *= r.factor * specificity[:, np.newaxis].repeat(self.rotations, 1)
                    # qs_phe[:, :, i] *= r.factor * specificity[:, np.newaxis].repeat(self.pheromones, 1)
                qs_rot = np.sum(qs_rot, axis=2)
                qs_phe = np.sum(qs_phe, axis=2)

                action_rot = np.argmax(qs_rot, axis=1).astype(int)
                action_phero = np.argmax(qs_phe, axis=1).astype(int)

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
