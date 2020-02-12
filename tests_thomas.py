import time
import pickle
import random
import math

from environment.ants import *
from environment.pheromone import *
from environment.circle_obstacles import *
from environment.walls import *
from environment.food import *

def ant_func_jumpx(x, y, t, ps):
	return x + 1, y, t, ps


def follow_pheromone(x, y, t, ps):
	best = 0
	best_angle = 0
	for a in range(-1, 2):
		angle = a * 0.25 * np.pi
		posx = int(round(x + 3 * math.cos(angle + t)))
		posy = int(round(y + 3 * math.sin(angle + t)))
		if 0 <= posx < env.w and 0 <= posy < env.h:
			if phero1.phero[posx, posy] > best:
				best = phero1.phero[posx, posy]
				best_angle = angle * 0.1
	return x, y, best_angle + t, ps


env = Environment(200, 100)

walls = Walls(env)
food = Food(env)

n_rocks = 10
rock_centers = np.random.random((n_rocks, 2))
rock_centers[:, 0] *= 150
rock_centers[:, 1] *= 75
rock_centers[:, 0] += 50
rock_centers[:, 1] += 25
rocks = CircleObstacles(env, centers=rock_centers, radiuses=np.random.random(n_rocks) * 5 + 5, weights=np.random.random(n_rocks) * 5 + 2)

phero1 = Pheromone(env, color=(255, 64, 0), max_val=20)
phero2 = Pheromone(env, color=(64, 64, 255), max_val=20)

n = 300
ants = Ants(env, n, 100)
ants.register_pheromone(phero1)
ants.register_pheromone(phero2)


fwd = np.ones(n) * 0.5
rot = np.ones(n) * 0.01

steps = 500
states = []

print("Starting simulation...")
start_time = time.time()
for s in range(steps):
	states.append(env.save_state())

	if s % 10 == 0:
		ants.activate_pheromone(0, np.random.randint(0, 10, n))
		ants.activate_pheromone(1, np.random.randint(0, 10, n))
		ants.update_mandibles(np.random.randint(0, 2, n))

	# ants.rotate_ants(np.random.random(n) * 0.1 - 0.05)
	ants.forward_ants(fwd)
	# ants.emit_pheromones(0, 10)
	# ants.emit_pheromones(1, 10)
	ants.apply_func(follow_pheromone)
	# phero.add_pheromones(np.floor(ants.xy).astype(int), 10)
	env.update()
print("Took {:.2f}s to simulate {} steps.".format(time.time() - start_time, steps))

pickle.dump(states, open("saved/save-test.arl", "wb"))

# VISUALIZE THE EPISODE
import visualize

