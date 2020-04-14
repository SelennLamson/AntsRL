import numpy as np
from scipy.signal import convolve2d
from .environment import Environment, EnvObject

DIFFUSE_FACTOR = 0
EVAP_FACTOR = 0.001

DIFFUSE_FILTER = np.ones((3, 3)) * DIFFUSE_FACTOR
DIFFUSE_FILTER[1, 1] = 1 - 8 * DIFFUSE_FACTOR
DIFFUSE_FILTER *= 1 - EVAP_FACTOR

class PheromoneVisualization (EnvObject):
	def __init__(self, environment: Environment, color, max_val, phero):
		super().__init__(environment)
		self.color = color
		self.max_val = max_val
		self.phero = phero.astype(np.uint8)


class Pheromone (EnvObject):
	def __init__(self, environment: Environment, color=(64, 64, 64), max_val=None, phero=None):
		super().__init__(environment)

		self.color = color
		self.max_val = max_val
		self.w = environment.w
		self.h = environment.h
		if phero is None:
			self.phero = np.zeros((self.w, self.h))
		else:
			self.phero = phero.copy()

	def visualize_copy(self, newenv):
		return PheromoneVisualization(newenv, self.color, self.max_val, self.phero)

	def add_pheromones(self, add_xy, phero_strength):
		if len(add_xy) == 0:
			return
		self.phero[add_xy[:, 0], add_xy[:, 1]] += phero_strength
		if self.max_val is not None:
			self.phero = np.minimum(self.phero, self.max_val)

	def update(self):
		self.phero = convolve2d(self.phero, DIFFUSE_FILTER, 'same', 'fill', 0)
		self.phero[self.phero < 0.01] = 0
