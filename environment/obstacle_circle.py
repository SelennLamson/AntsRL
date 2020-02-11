import numpy as np
from .environment import Environment, EnvObject
from .ants import Ants
from typing import List, Tuple

class ObstacleCircle (EnvObject):
	def __init__(self, environment: Environment, center: Tuple[float, float], radius: float, weight: float):
		super().__init__(environment)
		self.w = environment.w
		self.h = environment.h

		self.center = np.array(center, dtype=float).reshape((1, 2))
		self.radius = radius
		self.weight = weight

	def visualize_copy(self, newenv):
		newobst = ObstacleCircle(newenv, self.center, self.radius, self.weight)
		return newobst

	def update(self):
		total_force = np.zeros((1, 2))
		for obj in self.environment.objects:
			if isinstance(obj, Ants):
				vecs_from_center = self.center - obj.xy
				dist_to_center = np.sum(vecs_from_center**2, axis=1)**0.5
				norm_vecs = vecs_from_center / (dist_to_center[:, np.newaxis] + 0.001)
				vecs_from_radius = vecs_from_center - norm_vecs * self.radius
				vecs_from_radius[dist_to_center > self.radius, :] = 0
				total_force += np.sum(vecs_from_radius, axis=0)
			elif isinstance(obj, ObstacleCircle):
				vec = self.center - obj.center
				dist = np.sum(vec**2)**0.5
				if dist < obj.radius + self.radius:
					self.center += vec / (dist + 0.001) * (obj.radius + self.radius - dist)

		self.center -= total_force / self.weight

		for obj in self.environment.objects:
			if isinstance(obj, Ants):
				vecs_from_center = self.center - obj.xy
				dist_to_center = np.sum(vecs_from_center ** 2, axis=1) ** 0.5
				norm_vecs = vecs_from_center / (dist_to_center[:, np.newaxis] + 0.001)
				vecs_from_radius = vecs_from_center - norm_vecs * self.radius
				vecs_from_radius[dist_to_center > self.radius, :] = 0
				obj.translate_ants(vecs_from_radius)

	def update_step(self):
		return 0
