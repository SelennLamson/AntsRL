import random
from utils import *

#
# class AnthillPlacer:
# 	def __init__(self, center_x, center_y, min_radius, max_radius):
#

class PerlinGenerator:
	def __init__(self, scale=22.0, density=0.05, octaves=2, persistence=0.5, lacunarity=2.0):
		self.scale = scale
		self.density = density
		self.octaves = octaves
		self.persistence = persistence
		self.lacunarity = lacunarity

	def generate(self, w, h):
		return perlin_noise_generator(w, h,
									  offset_x=random.randint(-10000, 10000),
									  offset_y=random.randint(-10000, 10000),
									  scale=self.scale,
									  octaves=self.octaves,
									  persistence=self.persistence,
									  lacunarity=self.lacunarity
									  ) > self.density


class CirclesGenerator:
	def __init__(self, n_circles, min_radius, max_radius):
		self.n_circles = n_circles
		self.min_radius = min_radius
		self.max_radius = max_radius

	def generate(self, w, h):
		gen = np.zeros((w, h), dtype=bool)
		for i in range(self.n_circles):
			radius = int(random.random() * (self.max_radius - self.min_radius) + self.min_radius)
			xc = int(random.random() * (w - 2 * radius) + radius)
			yc = int(random.random() * (h - 2 * radius) + radius)

			for x in range(xc - radius, xc + radius + 1):
				for y in range(yc - radius, yc + radius + 1):
					dist = ((xc - x) ** 2 + (yc - y) ** 2) ** 0.5
					if dist <= radius:
						gen[x, y] = True
		return gen
