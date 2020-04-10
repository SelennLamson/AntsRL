import numpy as np
from typing import List

from .environment import Environment, EnvObject
from .pheromone import Pheromone
from .food import Food

class AntsVisualization(EnvObject):
    def __init__(self, env, ants_xyt, mandibles, holding, reward_state):
        super().__init__(env)
        self.ants = ants_xyt.copy()
        self.mandibles = mandibles.copy()
        self.holding = holding.copy()
        self.reward_state = reward_state.copy()


class Ants (EnvObject):
    def __init__(self, environment: Environment, n_ants: int, max_hold, xyt=None):
        super().__init__(environment)

        self.n_ants = n_ants
        self.max_hold = max_hold

        # Column 1: X coord (0 ; w)
        # Column 2: Y coord (0 ; h)
        # Column 3: Theta (-1 ; 1)
        self.ants = xyt.copy()
        self.warp_xy()

        self.prev_ants = self.ants.copy()

        self.phero_activation = np.zeros((n_ants, 0))
        self.pheromones: List[Pheromone] = []

        # True when mandibles are closed (active)
        self.mandibles = np.zeros(n_ants, dtype=bool)
        self.holding = np.zeros(n_ants)
        self.reward_state = np.zeros(n_ants, dtype=np.uint8)

        # Random seed specific to each ant:
        self.seed = np.random.random(n_ants)

    def visualize_copy(self, newenv):
        return AntsVisualization(newenv, self.ants, self.mandibles, self.holding, self.reward_state)

    @property
    def x(self):
        return self.ants[:, 0]

    @property
    def y(self):
        return self.ants[:, 1]

    @property
    def xy(self):
        return self.ants[:, 0:2]

    @property
    def theta(self):
        return self.ants[:, 2]

    def warp_theta(self):
        self.ants[:, 2] = np.mod(self.theta, 2*np.pi)

    def rotate_ants(self, add_theta):
        self.ants[:, 2] += add_theta
        self.warp_theta()

    def warp_xy(self):
        self.ants[:, 0] = np.mod(self.ants[:, 0], self.environment.w)
        self.ants[:, 1] = np.mod(self.ants[:, 1], self.environment.h)

    def translate_ants(self, add_xy):
        self.ants[:, 0:2] += add_xy
        self.warp_xy()

    def forward_ants(self, add_fwd):
        add_x = np.cos(self.theta) * add_fwd
        add_y = np.sin(self.theta) * add_fwd
        self.translate_ants(np.vstack([add_x, add_y]).T)

    def register_pheromone(self, pheromone: Pheromone):
        self.phero_activation = np.hstack([self.phero_activation, np.zeros((self.n_ants, 1))]).astype(np.bool)
        self.pheromones.append(pheromone)

    def activate_all_pheromones(self, new_activations):
        self.phero_activation = new_activations.copy()

    def activate_pheromone(self, phero_index):
        for i, phero in enumerate(phero_index):
            if phero == 0:
                self.phero_activation[i] = [0, 0]
            elif phero == 1:
                self.phero_activation[i] = [256, 0]
            else:
                self.phero_activation[i] = [0, 256]

    def emit_pheromones(self, phero_index):
        phero = self.pheromones[phero_index]
        phero.add_pheromones(self.xy.astype(int), self.phero_activation[:, phero_index])

    def update_mandibles(self, new_mandible):
        closing = np.bitwise_and(new_mandible, 1 - self.mandibles)
        opening = np.bitwise_and(1 - new_mandible, self.mandibles)
        xy = self.prev_ants[:, 0:2].astype(int)

        self.mandibles = new_mandible.copy()
        for obj in self.environment.objects:
            if isinstance(obj, Food):
                # Ants closing their mandibles are taking food
                taken = np.minimum(self.max_hold, np.maximum(0, obj.qte[xy[:, 0], xy[:, 1]])) * closing

                # Ants opening their mandibles are dropping food
                dropped = self.holding.copy() * opening

                obj.qte[xy[:, 0], xy[:, 1]] += dropped - taken
                self.holding += taken - dropped

    def give_reward(self, added_rewards):
        add = (added_rewards > 0) * 255
        self.reward_state = np.minimum(self.reward_state + add, 255)

    def update(self):
        self.prev_ants = self.ants.copy()
        for obj in self.environment.objects:
            if isinstance(obj, Pheromone):
                if obj in self.pheromones:
                    phero_i = self.pheromones.index(obj)
                    self.emit_pheromones(phero_i)
        self.reward_state = (self.reward_state * 0.9).astype(np.uint8)

    def update_step(self):
        return 999

    def apply_func(self, func):
        for i in range(self.n_ants):
            x, y, t = self.ants[i]
            ps = self.phero_activation[i]
            x, y, t, ps = func(x, y, t, ps)
            self.ants[i] = x, y, t
            self.phero_activation[i] = ps
        self.warp_theta()
        self.warp_xy()

