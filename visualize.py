import sys
import os
import pygame
import pickle
import numpy as np

from environment.ants import Ants
from environment.pheromone import Pheromone
from environment.obstacle_circle import ObstacleCircle


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

ant = pygame.transform.smoothscale(pygame.image.load("assets/ant_32_white.png"), (24, 24))
antrect = ant.get_rect()

screen = pygame.display.set_mode(size)
clock = pygame.time.Clock()

step = 0
while 1:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()

    e = states[step % len(states)]

    screen.fill((0, 0, 0))

    ants_layer = pygame.Surface(size, pygame.SRCALPHA)
    obstacle_layer = pygame.Surface(size, pygame.SRCALPHA)
    phero_layer = pygame.Surface(size)

    for obj in e.objects:
        if isinstance(obj, Ants):
            xyt = obj.prev_ants.copy()
            xyt[:, 0] *= sw / ew
            xyt[:, 1] *= sh / eh
            xyt[:, 2] *= -1

            for x, y, t in xyt:
                rotated = pygame.transform.rotate(ant, t / np.pi * 180 - 90)
                ants_layer.blit(rotated, (x - rotated.get_width()/2, y - rotated.get_height()/2))
        elif isinstance(obj, ObstacleCircle):
            pygame.draw.circle(obstacle_layer, (200, 128, 64), (obj.center * zoom_factor)[0].astype(int), int(obj.radius * zoom_factor))
        elif isinstance(obj, Pheromone):
            img = ((obj.phero / (np.max(obj.phero) + 0.0001))[:, :, np.newaxis] * np.array(obj.color)[np.newaxis, np.newaxis, :]).astype(np.uint8)
            phero_layer.blit(pygame.transform.scale(pygame.surfarray.make_surface(img), size), (0, 0), special_flags=pygame.BLEND_RGB_ADD)


    screen.blit(phero_layer, (0, 0))
    screen.blit(obstacle_layer, (0, 0))
    screen.blit(ants_layer, (0, 0))
    pygame.display.flip()
    clock.tick(30)
    step += 1
