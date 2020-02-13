import numpy as np
import noise
import matplotlib.pyplot as plt

from .environment import Environment, EnvObject
from .food import Food

class AnthillVisualization (EnvObject):
	def __init__(self, env, x, y, radius, food):
		super().__init__(env)
		self.x = x
		self.y = y
		self.radius = radius
		self.food = food

class Anthill (EnvObject):
	def __init__(self, environment: Environment, x, y, radius):
		super().__init__(environment)

		self.w = environment.w
		self.h = environment.h

		self.x = x
		self.y = y
		self.radius = radius

		self.food = 0
		self.area = np.zeros((self.w, self.h), dtype=bool)
		for x in range(self.w):
			for y in range(self.h):
				dist = ((self.x - x) ** 2 + (self.y - y) ** 2) ** 0.5
				if dist <= self.radius:
					self.area[x, y] = True

	def visualize_copy(self, newenv):
		return AnthillVisualization(newenv, self.x, self.y, self.radius, self.food)

	def update_step(self):
		return 1000

	def update(self):
		for obj in self.environment.objects:
			if isinstance(obj, Food):
				gain = obj.qte * self.area
				obj.qte -= gain
				self.food += np.sum(gain)
