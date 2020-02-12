import numpy as np
import noise
import matplotlib.pyplot as plt

from .environment import Environment, EnvObject

class Food (EnvObject):
	def __init__(self, environment: Environment, qte=None):
		super().__init__(environment)

		self.w = environment.w
		self.h = environment.h

		if qte is None:
			scale = 10.0
			octaves = 6
			persistence = 0.5
			lacunarity = 2.0

			shape = (self.w, self.h)
			self.qte = np.zeros(shape)
			for i in range(shape[0]):
				for j in range(shape[1]):
					self.qte[i][j] = noise.pnoise2(i / scale,
												   j / scale,
												   octaves=octaves,
												   persistence=persistence,
												   lacunarity=lacunarity,
												   base=0)
			self.qte -= 0.1
			self.qte[self.qte < 0] = 0
			self.qte[self.qte > 0] = 10
		else:
			self.qte = qte.copy()

	def visualize_copy(self, newenv):
		return Food(newenv, self.qte)

