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

class OneCircleGenerator:
    def __init__(self, opt_x, opt_y, min_radius, max_radius):
        self.opt_x = opt_x
        self.opt_y = opt_y
        self.min_radius = min_radius
        self.max_radius = max_radius

    def generate(self, w, h):
        gen = np.zeros((w, h), dtype=bool)

        radius = int(random.random() * (self.max_radius - self.min_radius) + self.min_radius)
        xc = int(random.random() * (w - 2 * radius) + radius) if self.opt_x is None else self.opt_x
        yc = int(random.random() * (h - 2 * radius) + radius) if self.opt_y is None else self.opt_y

        for x in range(xc - radius, xc + radius + 1):
            for y in range(yc - radius, yc + radius + 1):
                dist = ((xc - x) ** 2 + (yc - y) ** 2) ** 0.5
                if dist <= radius:
                    gen[x, y] = True
        return gen

class AvoidingCirclesGenerator:
    def __init__(self, n_circles, min_radius, max_radius, x_avoid, y_avoid, avoid_radius):
        self.n_circles = n_circles
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.x_avoid = x_avoid
        self.y_avoid = y_avoid
        self.avoid_radius = avoid_radius

    def generate(self, w, h):
        gen = np.zeros((w, h), dtype=bool)
        for i in range(self.n_circles):
            radius = xc = yc = 0

            correct = False
            while not correct:
                radius = int(random.random() * (self.max_radius - self.min_radius) + self.min_radius)
                xc = int(random.random() * (w - 2 * radius) + radius)
                yc = int(random.random() * (h - 2 * radius) + radius)

                if ((xc - self.x_avoid)**2 + (yc - self.y_avoid)**2)**0.5 >= self.avoid_radius + radius:
                    correct = True

            for x in range(xc - radius, xc + radius + 1):
                for y in range(yc - radius, yc + radius + 1):
                    dist = ((xc - x) ** 2 + (yc - y) ** 2) ** 0.5
                    if dist <= radius:
                        gen[x, y] = True
        return gen
