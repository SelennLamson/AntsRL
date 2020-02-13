import numpy as np
from typing import List

from .environment import Environment, EnvObject
from .pheromone import Pheromone
from .food import Food


class AntsVisualization(EnvObject):
	def __init__(self, env, ants_xyt, mandibles, holding):
		super().__init__(env)
		self.ants = ants_xyt.copy()
		self.mandibles = mandibles.copy()
		self.holding = holding.copy()


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

	def visualize_copy(self, newenv):
		return AntsVisualization(newenv, self.ants, self.mandibles, self.holding)

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

	def activate_pheromone(self, phero_index, new_activation):
		self.phero_activation[:, phero_index] = new_activation

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
				taken = np.minimum(self.max_hold, np.maximum(0, obj.qte[xy[:, 0], xy[:, 1]]))
				taken[1 - closing] = 0

				# Ants opening their mandibles are dropping food
				dropped = self.holding.copy()
				dropped[1 - opening] = 0

				obj.qte[xy[:, 0], xy[:, 1]] += dropped - taken
				self.holding += taken - dropped



	def update(self):
		self.prev_ants = self.ants.copy()
		for obj in self.environment.objects:
			if isinstance(obj, Pheromone):
				if obj in self.pheromones:
					phero_i = self.pheromones.index(obj)
					self.emit_pheromones(phero_i)

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

