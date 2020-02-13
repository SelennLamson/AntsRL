import numpy as np
import noise
import matplotlib.pyplot as plt

from .environment import Environment, EnvObject

class Food (EnvObject):
	def __init__(self, environment: Environment, qte):
		super().__init__(environment)
		self.qte = qte.copy()

	def visualize_copy(self, newenv):
		return Food(newenv, self.qte)

