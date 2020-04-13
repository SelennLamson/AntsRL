from abc import ABC
import numpy as np

from environment.ants import Ants

class Reward (ABC):
    def __init__(self, factor=1):
        self.factor = factor

        self.ants = None
        self.environment = None
        self.rewards = None

    def setup(self, ants: Ants):
        """
        Setups the reward on a new ants group
        :param ants: the new ant group
        """
        self.ants = ants
        self.environment = ants.environment
        self.rewards = np.zeros(self.ants.n_ants, dtype=float)

    def observation(self, obs_coords, perception, agent_state):
        """
        Called by the RL Api when performing an observation.
        :param obs_coords: the coordinates observed by each individual ant
        :param perception: the perception states of the ants
        """
        pass

    def step(self, done, turn_index, open_close_mandibles, on_off_pheromones):
        """
        Computes the reward of each ant at a certain step.
        :param done: is this the end of the game?
        :param turn_index: turning action of all ants
        :param open_close_mandibles: boolean state of ants' mandibles
        :param on_off_pheromones: boolean state of ants' pheromone activations
        :return: a numpy array of shape (n_ants) containing the per-ants reward
        """
        return self.rewards

    def should_visualize(self):
        """
        Returns True if the reward can and should be visualized.
        :return: boolean
        """
        return False

    def visualization(self):
        """
        Returns a visualization heatmap that will be visible in the GUI. Returning None for no heatmap (saves disk space and performance).
        :return: None or a numpy array of shape (env.w, env.h)
        """
        return None
