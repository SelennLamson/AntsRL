import numpy as np
from typing import List

from .environment import Environment, EnvObject
from .pheromone import Pheromone


class Ants (EnvObject):
	def __init__(self, environment: Environment, n_ants: int):
		super().__init__(environment)

		self.n_ants = n_ants

		# Column 1: X coord (0 ; w)
		# Column 2: Y coord (0 ; h)
		# Column 3: Theta (-1 ; 1)
		self.ants = np.random.random((n_ants, 3))
		self.ants[:, 0] *= self.environment.w
		self.ants[:, 1] *= self.environment.h
		self.ants[:, 2] *= np.pi * 2

		self.prev_ants = self.ants.copy()

		self.phero_activation = np.zeros((n_ants, 0), dtype=np.bool)
		self.pheromones: List[Pheromone] = []

	def visualize_copy(self, newenv):
		newants = Ants(newenv, self.n_ants)
		newants.prev_ants = self.prev_ants.copy()
		return newants

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

	def activate_pheromone(self, phero_index, add_mask, del_mask):
		self.phero_activation[:, phero_index] |= add_mask
		self.phero_activation[:, phero_index] &= np.bitwise_not(del_mask)

	def emit_pheromones(self, phero_index, strength):
		phero = self.pheromones[phero_index]
		phero.add_pheromones(self.xy[self.phero_activation[:, phero_index], :].astype(int), strength)

	def update(self):
		self.prev_ants = self.ants.copy()

	def update_step(self):
		return 1

	def apply_func(self, func):
		for i in range(self.n_ants):
			x, y, t = self.ants[i]
			ps = self.phero_activation[i]
			x, y, t, ps = func(x, y, t, ps)
			self.ants[i] = x, y, t
			self.phero_activation[i] = ps
		self.warp_theta()
		self.warp_xy()

