import sys
import os
import pygame
import pickle
import numpy as np
from scipy.signal import convolve2d

from environment.ants import Ants
from environment.pheromone import Pheromone
from environment.circle_obstacles import CircleObstacles
from environment.walls import Walls
from environment.food import Food


base_path = "saved/"
saved = os.listdir(base_path)
while True:
    for i, s in enumerate(saved):
        print('- [' + str(i) + '] ' + s)
    ans = input("Chose a saved file: ")
    try:
        ans = int(ans)
        assert (0 <= ans < len(saved))
        break
    except (ValueError, AssertionError):
        pass
file_name = saved[ans]
states = pickle.load(open(base_path + file_name, "rb"))


pygame.init()

big_dim = 1600
ew = states[0].w
eh = states[0].h
if ew > eh:
    sw = big_dim
    sh = big_dim // ew * eh
else:
    sw = big_dim
    sh = big_dim // eh * ew
size = sw, sh
e_max = max(ew, eh)
zoom_factor = big_dim / e_max

# ant = pygame.transform.smoothscale(pygame.image.load("assets/ant_32.png"), (24, 24))
# holding_ant = pygame.transform.smoothscale(pygame.image.load("assets/holding_ant_32.png"), (24, 24))
ant = pygame.image.load("assets/ant_32.png")
holding_ant = pygame.image.load("assets/holding_ant_32.png")
dirt = pygame.image.load("assets/dirt.jpg")
rock = pygame.image.load("assets/rock.png")
rock_tile = pygame.image.load("assets/rock_tile.jpg")

screen = pygame.display.set_mode(size)
clock = pygame.time.Clock()

background = pygame.Surface(size)
for xd in range(0, sw, dirt.get_width()):
    for yd in range(0, sh, dirt.get_height()):
        background.blit(dirt, (xd, yd))

rock_background = pygame.Surface(size)
for xd in range(0, sw, rock_tile.get_width()):
    for yd in range(0, sh, rock_tile.get_height()):
        rock_background.blit(rock_tile, (xd, yd))

ants_layer = pygame.Surface(size, pygame.SRCALPHA)
obstacle_layer = pygame.Surface(size, pygame.SRCALPHA)
walls_layer = pygame.Surface(size, pygame.SRCALPHA)
food_layer = pygame.Surface(size)
phero_layer = pygame.Surface(size)

for obj in states[0].objects:
    if isinstance(obj, Walls):
        img = np.zeros((obj.map.shape[0], obj.map.shape[1]))
        img[obj.map] = 255
        # f = np.array([[0.05, 0.05, 0.05],
        #               [0.05, 0.60, 0.05],
        #               [0.05, 0.05, 0.05]])
        # img = convolve2d(img, f, 'same')
        img = img[:, :, np.newaxis].repeat(3, axis=2)
        img = pygame.surfarray.pixels_red(pygame.transform.smoothscale(pygame.surfarray.make_surface(img), size))[:, :]

        walls_layer.blit(rock_background, (0, 0))
        alpha = pygame.surfarray.pixels_alpha(walls_layer)
        alpha[:, :] = img[:, :]
        del alpha

step = 0
while 1:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()

    e = states[step % len(states)]

    ants_layer.fill((0, 0, 0, 0))
    obstacle_layer.fill((0, 0, 0, 0))
    phero_layer.blit(background, (0, 0))

    for obj in e.objects:
        if isinstance(obj, Ants):
            xyt = obj.ants.copy()
            xyt[:, 0] *= sw / ew
            xyt[:, 1] *= sh / eh
            xyt[:, 2] *= -1

            for i, (x, y, t) in enumerate(xyt):
                img = ant if obj.holding[i] == 0 else holding_ant
                rotated = pygame.transform.rotate(img, t / np.pi * 180 - 90)
                ants_layer.blit(rotated, (x - rotated.get_width()/2, y - rotated.get_height()/2))
        elif isinstance(obj, CircleObstacles):
            for rock_i in range(obj.n_obst):
                rock_scaled = pygame.transform.scale(rock, (int(obj.radiuses[rock_i] * zoom_factor * 2), int(obj.radiuses[rock_i] * zoom_factor * 2)))
                posx, posy = (obj.centers[rock_i] * zoom_factor).astype(int)
                obstacle_layer.blit(rock_scaled, (posx - rock_scaled.get_width() / 2, posy - rock_scaled.get_height() / 2))
        elif isinstance(obj, Pheromone):
            img = ((obj.phero / (np.max(obj.phero) + 0.0001))[:, :, np.newaxis] * np.array(obj.color)[np.newaxis, np.newaxis, :]).astype(np.uint8) * 0.5
            phero_layer.blit(pygame.transform.scale(pygame.surfarray.make_surface(img), size), (0, 0), special_flags=pygame.BLEND_RGB_ADD)
        elif isinstance(obj, Food):
            norm = obj.qte / (np.max(obj.qte) + 0.0001)
            img = np.array((255, 255, 255))[np.newaxis, np.newaxis, :] * (1 - norm)[:, :, np.newaxis] + np.array((255, 0, 0))[np.newaxis, np.newaxis, :] * norm[:, :, np.newaxis]
            phero_layer.blit(pygame.transform.scale(pygame.surfarray.make_surface(img), size), (0, 0), special_flags=pygame.BLEND_RGB_MULT)

    screen.blit(phero_layer, (0, 0))
    screen.blit(walls_layer, (0, 0))
    screen.blit(obstacle_layer, (0, 0))
    screen.blit(ants_layer, (0, 0))
    pygame.display.flip()
    clock.tick(30)
    step += 1
