import random
from utils import *

from environment.environment import Environment
from environment.anthill import Anthill
from environment.ants import Ants
from environment.pheromone import Pheromone
from environment.circle_obstacles import CircleObstacles
from environment.walls import Walls
from environment.food import Food
from environment.RL_api import RLApi

PHERO_COLORS = [
    (255, 64, 0),
    (64, 64, 255),
    (100, 255, 100)
]

class EnvironmentGenerator:
    def __init__(self, w, h, n_ants, n_pheromones, n_rocks, food_generator, walls_generator, max_steps, seed=None):
        self.w = w
        self.h = h
        self.n_ants = n_ants
        self.n_pheromones = n_pheromones
        self.n_rocks = n_rocks
        self.food_generator = food_generator
        self.walls_generator = walls_generator

        # Complex defaults:
        # self.perception_mask = np.array([[0, 1, 1, 1, 0],
        #                                  [1, 1, 1, 1, 1],
        #                                  [1, 1, 1, 1, 1],
        #                                  [0, 1, 1, 1, 0],
        #                                  [0, 0, 1, 0, 0]], dtype=bool)
        self.perception_mask = np.array([[0, 0, 1, 1, 1, 0, 0],
                                         [0, 1, 1, 1, 1, 1, 0],
                                         [1, 1, 1, 1, 1, 1, 1],
                                         [1, 1, 1, 1, 1, 1, 1],
                                         [1, 1, 1, 1, 1, 1, 1],
                                         [0, 1, 1, 1, 1, 1, 0],
                                         [0, 0, 1, 1, 1, 0, 0]], dtype=bool)

        self.perception_shift = 4

        self.max_steps = max_steps
        self.seed = seed

    def setup_perception(self, new_mask, new_shift):
        self.perception_mask = new_mask
        self.perception_shift = new_shift

    def generate(self, rl_api: RLApi):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed * 5)

        env = Environment(self.w, self.h, self.max_steps)
        perceived_objects = []

        anthill = Anthill(env,
                          int(random.random() * self.w * 0.5 + self.w * 0.25),
                          int(random.random() * self.h * 0.5 + self.h * 0.25),
                          int(random.random() * min(self.w, self.h) * 0.1 + min(self.w, self.h) * 0.1))
        perceived_objects.append(anthill)

        world_walls = self.walls_generator.generate(self.w, self.h)
        world_walls[anthill.area] = False
        walls = Walls(env, world_walls)
        perceived_objects.append(walls)

        food = Food(env, self.food_generator.generate(self.w, self.h))
        food.qte *= (1 - walls.map)
        perceived_objects.append(food)


        if self.n_rocks > 0:
            rock_centers = np.random.random((self.n_rocks, 2))
            rock_centers[:, 0] *= self.w * 0.75
            rock_centers[:, 1] *= self.h * 0.25
            rock_centers[:, 0] += self.w * 0.25
            rock_centers[:, 1] += self.h * 0.25
            rocks = CircleObstacles(env, centers=rock_centers,
                                    radiuses=np.random.random(n_rocks) * 5 + 5,
                                    weights=np.random.random(n_rocks) * 50 + 50)
            perceived_objects.append(rocks)

        ants_angle = np.random.random(self.n_ants) * 2 * np.pi
        ants_dist = np.random.random(self.n_ants) * anthill.radius * 0.8
        ants_x = np.cos(ants_angle) * ants_dist + anthill.x
        ants_y = np.sin(ants_angle) * ants_dist + anthill.y
        ants_t = np.random.random(self.n_ants) * 2 * np.pi

        ants = Ants(env, self.n_ants, 5, xyt=np.array([ants_x, ants_y, ants_t]).T)
        perceived_objects.insert(0, ants)

        for p in range(self.n_pheromones):
            phero = Pheromone(env, color=PHERO_COLORS[p % len(PHERO_COLORS)], max_val=255)
            ants.register_pheromone(phero)
            perceived_objects.insert(p + 1, phero)

        rl_api.register_ants(ants)
        rl_api.setup_perception(self.perception_mask.shape[0] // 2,
                                perceived_objects,
                                self.perception_mask,
                                self.perception_shift)
        return env
