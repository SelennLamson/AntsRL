import numpy as np
from typing import List


class EnvObject:
	def __init__(self, environment: 'Environment'):
		self.environment = environment
		if self.environment is not None:
			self.environment.add_object(self)

	def visualize_copy(self, newenv: 'Environment'):
		return EnvObject(newenv)

	def update(self):
		pass

	def update_step(self):
		return 0


class Environment:
	def __init__(self, w, h):
		self.w = w
		self.h = h
		self.objects: List[EnvObject] = []

	def add_object(self, obj: EnvObject):
		self.objects.append(obj)

	def detach_object(self, obj: EnvObject):
		if obj in self.objects:
			self.objects.remove(obj)

	def save_state(self):
		newenv = Environment(self.w, self.h)
		for obj in self.objects:
			newenv.add_object(obj.visualize_copy(newenv))
		return newenv

	def update(self):
		steps = [(obj.update_step(), obj) for obj in self.objects]
		steps.sort(key=lambda x: x[0])

		for step, obj in steps:
			obj.update()
