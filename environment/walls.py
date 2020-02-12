import numpy as np
import noise
import matplotlib.pyplot as plt

from .environment import Environment, EnvObject
from .ants import Ants
from .pheromone import Pheromone

class Walls (EnvObject):
	def __init__(self, environment: Environment, map_in=None):
		super().__init__(environment)

		self.w = environment.w
		self.h = environment.h

		if map_in is None:
			scale = 22.0
			octaves = 2
			persistence = 0.5
			lacunarity = 2.0

			shape = (self.w, self.h)
			self.map = np.zeros(shape)
			for i in range(shape[0]):
				for j in range(shape[1]):
					self.map[i][j] = noise.pnoise2(i / scale,
												   j / scale - 0.3,
												   octaves=octaves,
												   persistence=persistence,
												   lacunarity=lacunarity,
												   base=0)
			self.map = self.map > 0.05
		else:
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
