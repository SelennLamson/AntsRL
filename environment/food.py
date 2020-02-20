import numpy as np
import noise
import matplotlib.pyplot as plt

from .environment import Environment, EnvObject

class FoodVisualization(EnvObject):
	def __init__(self, environment: Environment, qte):
		super().__init__(environment)
		self.qte = qte.astype(np.uint8)

class Food (EnvObject):
	def __init__(self, environment: Environment, qte):
		super().__init__(environment)
		self.qte = qte.astype(float)

	def visualize_copy(self, newenv):
		return FoodVisualization(newenv, self.qte)

