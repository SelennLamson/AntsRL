import numpy as np
from .environment import Environment, EnvObject
from .ants import Ants
from typing import List, Tuple
from utils import *


class CircleObstaclesVisualization (EnvObject):
	def __init__(self, env, centers, radiuses, weights):
		super().__init__(env)
		self.centers = centers.copy()
		self.radiuses = radiuses.copy()
		self.weights = weights.copy()

class CircleObstacles (EnvObject):
	def __init__(self, environment: Environment, centers, radiuses, weights):
		super().__init__(environment)
		self.w = environment.w
		self.h = environment.h

		self.n_obst = len(radiuses)
		self.centers = centers
		self.radiuses = radiuses
		self.weights = weights

		self.crossed_radiuses = self.radiuses[AX, :] + self.radiuses[:, AX]
		self.crossed_weights = self.weights[AX, :] / (self.weights[:, AX] + self.weights[AX, :])

	def visualize_copy(self, newenv):
		return CircleObstaclesVisualization(newenv, self.centers, self.radiuses, self.weights)

	def update(self):
		for obj in self.environment.objects:
			if isinstance(obj, Ants):
				vecs_from_centers = self.centers[AX, :, :] - obj.xy[:, AX, :]
				dist_to_centers = np.sum(vecs_from_centers**2, axis=2)**0.5
				vecs_from_radiuses = vecs_from_centers * (1 - self.radiuses[AX, :] / (dist_to_centers + 0.001))[:, :, AX]
				vecs_from_radiuses[dist_to_centers > self.radiuses[AX, :], :] = 0

				self.centers -= np.sum(vecs_from_radiuses, axis=0) / self.weights[:, AX]

		# vecs_center_to_center = self.centers[AX, :, :] - self.centers[:, AX, :]
		# dist_center_to_center = np.sum(vecs_center_to_center**2, axis=2)**0.5
		# vecs_norm = vecs_center_to_center / (dist_center_to_center[:, :, AX] + 0.001)
		# dist_center_to_center -= self.crossed_radiuses
		# dist_center_to_center[dist_center_to_center > 0] = 0
		# dist_center_to_center *= self.crossed_weights * 0.25
		# # dist_center_to_center = np.tril(dist_center_to_center, -1)
		# self.centers += np.sum(vecs_norm * dist_center_to_center[:, :, AX], axis=1)

		for obj in self.environment.objects:
			if isinstance(obj, Ants):
				vecs_from_centers = self.centers[AX, :, :] - obj.xy[:, AX, :]
				dist_to_centers = np.sum(vecs_from_centers ** 2, axis=2) ** 0.5
				vecs_from_radiuses = vecs_from_centers * (1 - self.radiuses[AX, :] / (dist_to_centers + 0.001))[
														 :, :, AX]
				vecs_from_radiuses[dist_to_centers > self.radiuses[AX, :], :] = 0
				obj.translate_ants(np.sum(vecs_from_radiuses, axis=1))

	def update_step(self):
		return 0
