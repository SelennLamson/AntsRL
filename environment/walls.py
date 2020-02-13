import numpy as np

from .environment import Environment, EnvObject
from .ants import Ants
from .pheromone import Pheromone

class Walls (EnvObject):
	def __init__(self, environment: Environment, map_in):
		super().__init__(environment)

		self.w = environment.w
		self.h = environment.h

		self.map = map_in.astype(bool)

	def visualize_copy(self, newenv):
		return self

	def update_step(self):
		return -1

	def update(self):
		for obj in self.environment.objects:
			if isinstance(obj, Ants):
				xy = obj.xy.astype(int)
				colliding_ants = self.map[xy[:, 0], xy[:, 1]]
				obj.ants[colliding_ants, 0:2] = obj.prev_ants[colliding_ants, 0:2]
				obj.ants[colliding_ants, 2] += np.random.random(np.sum(colliding_ants)) - 0.5
			elif isinstance(obj, Pheromone):
				obj.phero[self.map] = 0
