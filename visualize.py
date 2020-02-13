import sys
import os
import pygame
import pickle
from scipy.signal import convolve2d
import numpy as np
AX = np.newaxis

from environment.ants import AntsVisualization
from environment.pheromone import Pheromone
from environment.circle_obstacles import CircleObstaclesVisualization
from environment.walls import Walls
from environment.food import Food
from environment.RL_api import RLPerceptiveFieldVisualization
from environment.anthill import AnthillVisualization

FOOD_COLOR = (64, 255, 64)
PERCEPTIVE_FIELD_COLOR = (255, 255, 0)
ANTHILL_COLOR = (0, 0, 255, 128)
ANTS_SIZE = (24, 24) # Original image size: (32, 32)


def mix_alpha(rgb1, alpha1, rgb2, alpha2):
    mixed_alpha = alpha1 + alpha2 * (1 - alpha1)
    mixed_rgb = (rgb1 * alpha1[:, :, AX] + rgb2 * alpha2[:, :, AX] * (1 - alpha1[:, :, AX])) / (mixed_alpha[:, :, AX] + 0.001)
    return mixed_rgb.astype(np.uint8), mixed_alpha

def toggle_view(index, _shift, _view):
    if _shift:
        _view[:] = [False for elt in _view]
        _view[index] = True
    else:
        _view[index] = not _view[index]


# --- LOADING A SAVED FILE ---
base_path = "saved/"
saved = os.listdir(base_path)
if len(saved) == 1:
    ans = 0
else:
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


# --- INITIALIZING PYGAME ---
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
margin_top = 25
margin_bot = 50
screen_size = sw, sh + margin_bot + margin_top
e_max = max(ew, eh)
zoom_factor = big_dim / e_max

screen = pygame.display.set_mode(screen_size)
clock = pygame.time.Clock()
font = pygame.font.SysFont('consolas', 24, True)
small_font = pygame.font.SysFont('consolas', 16, False)

# --- LOADING ASSETS ---
ant = pygame.transform.smoothscale(pygame.image.load("assets/ant_32.png"), ANTS_SIZE)
dirt = pygame.image.load("assets/dirt.png")
rock = pygame.image.load("assets/rock.png")
rock_tile = pygame.image.load("assets/rock_tile.jpg")

holding_ant = pygame.transform.smoothscale(pygame.image.load("assets/holding_ant_32.png"), ANTS_SIZE)
color = pygame.Surface((100, 100))
color.fill(FOOD_COLOR)
holding_ant.blit(color, (0, 0), special_flags=pygame.BLEND_RGB_MULT)


# --- CREATING BACKGROUND IMAGE AND PRECOMPUTED ARRAYS ---
background = pygame.Surface(size)
for xd in range(0, sw, dirt.get_width()):
    for yd in range(0, sh, dirt.get_height()):
        background.blit(dirt, (xd, yd))

rock_background = pygame.Surface(size, pygame.SRCALPHA)
for xd in range(0, sw, rock_tile.get_width()):
    for yd in range(0, sh, rock_tile.get_height()):
        rock_background.blit(rock_tile, (xd, yd))

anthill = pygame.Surface((ew, eh), pygame.SRCALPHA)
anthill.fill(ANTHILL_COLOR)

FOOD_FILL = np.array(FOOD_COLOR)[AX, AX, :].repeat(ew, axis=0).repeat(eh, axis=1)
PERCEPTIVE_FIELD_FILL = np.array(PERCEPTIVE_FIELD_COLOR)[AX, AX, :].repeat(ew, axis=0).repeat(eh, axis=1)
PHEROMONES_FILL = []

for obj in states[0].objects:
    if isinstance(obj, Walls):
        img = np.zeros((obj.map.shape[0], obj.map.shape[1]))
        img[obj.map] = 255
        img = img[:, :, AX].repeat(3, axis=2)
        img = pygame.surfarray.pixels_red(pygame.transform.scale(pygame.surfarray.make_surface(img), size))[:, :]
        alpha = pygame.surfarray.pixels_alpha(rock_background)
        alpha[:, :] = img[:, :]
        del alpha
    elif isinstance(obj, Pheromone):
        PHEROMONES_FILL.append(np.array(obj.color)[AX, AX, :].repeat(ew, axis=0).repeat(eh, axis=1))
    elif isinstance(obj, AnthillVisualization):
        alpha = pygame.surfarray.pixels_alpha(anthill)
        for x in range(ew):
            for y in range(eh):
                dist = ((obj.x - x) ** 2 + (obj.y - y) ** 2) ** 0.5
                if dist > obj.radius:
                    alpha[x, y] = 0
        del alpha
background.blit(pygame.transform.scale(anthill, size), (0, 0))


# --- CREATING OTHER LAYERS ---
ants_layer = pygame.Surface(size, pygame.SRCALPHA)
obstacle_layer = pygame.Surface(size, pygame.SRCALPHA)
color_repr_layer = pygame.Surface((states[0].w, states[0].h), pygame.SRCALPHA)
color_repr_img = np.zeros((states[0].w, states[0].h, 3), dtype=np.uint8)
color_repr_alpha = np.zeros((states[0].w, states[0].h))


# --- ITERATING OVER SAVED STEPS ---
step = 0
speed = 1
paused = False
view = [True] * 10
while 1:
    # Getting step's environment
    e = states[step % len(states)]

    # Reinitializing layers
    screen.fill((0, 0, 0))
    ants_layer.fill((0, 0, 0, 0))
    obstacle_layer.fill((0, 0, 0, 0))
    color_repr_img[:, :, :] = 255
    color_repr_alpha[:, :] = 0

    # Displaying every environment object
    pi = 0
    for obj in e.objects:
        if isinstance(obj, AntsVisualization):
            xyt = obj.ants.copy()
            xyt[:, 0] *= sw / ew
            xyt[:, 1] *= sh / eh
            xyt[:, 2] *= -1

            # Display one ant at each location, with holding food icon if it's the case
            for i, (x, y, t) in enumerate(xyt):
                img = ant if obj.holding[i] == 0 else holding_ant
                rotated = pygame.transform.rotate(img, t / np.pi * 180 - 90)
                ants_layer.blit(rotated, (x - rotated.get_width()/2, y - rotated.get_height()/2))
        elif isinstance(obj, CircleObstaclesVisualization):
            # Display a scaled rock image at each location
            for rock_i in range(len(obj.centers)):
                rock_scaled = pygame.transform.scale(rock, (int(obj.radiuses[rock_i] * zoom_factor * 2), int(obj.radiuses[rock_i] * zoom_factor * 2)))
                posx, posy = (obj.centers[rock_i] * zoom_factor).astype(int)
                obstacle_layer.blit(rock_scaled, (posx - rock_scaled.get_width() / 2, posy - rock_scaled.get_height() / 2))
        elif isinstance(obj, Pheromone) and view[3]:
            # Display the quantity of pheromones on the color representation layer (color indicated at initialization of pheromones)
            alph = (obj.phero / (obj.max_val + 0.0001)) * 0.6
            color_repr_img, color_repr_alpha = mix_alpha(color_repr_img, color_repr_alpha, PHEROMONES_FILL[pi], alph)
            pi += 1
        elif isinstance(obj, Food) and view[4]:
            # Display the quantity of food on the color representation layer (color as constant at beginning of this script)
            alph = obj.qte / (np.max(obj.qte) + 0.0001) * 0.3
            color_repr_img, color_repr_alpha = mix_alpha(color_repr_img, color_repr_alpha, FOOD_FILL, alph)
        elif isinstance(obj, RLPerceptiveFieldVisualization) and view[2]:
            # Display the perceptive field on the color representation layer (color as constant at beginning of this script)
            if obj.perceptive_field is not None:
                alph = obj.perceptive_field * 0.3
                color_repr_img, color_repr_alpha = mix_alpha(color_repr_img, color_repr_alpha, PERCEPTIVE_FIELD_FILL, alph)
        elif isinstance(obj, AnthillVisualization):
            txt = font.render("FOOD IN ANTHILL: " + str(int(obj.food)), True, (255, 255, 255))
            screen.blit(txt, (sw - 30 - txt.get_width(), sh + 13 + margin_top))
    step_str = str(step)
    if step < 1000:
        step_str = " " + step_str
    if step < 100:
        step_str = " " + step_str
    if step < 10:
        step_str = " " + step_str
    screen.blit(font.render("STEP " + step_str + " / " + str(len(states)) + "  - " + ("SPEED " + str(speed) if not paused else "PAUSED"), True, (255, 255, 255)), (30, sh + 13 + margin_top))
    help_txt = small_font.render("0. BACKGROUND - 1. ANTS - 2. PERCEPTIVE FIELDS - 3. PHEROMONES - 4. FOOD - 5. OBSTACLES", True, (255, 255, 255))
    screen.blit(help_txt, (sw // 2 - help_txt.get_width() // 2, 6))

    # Converting the color representation layer from numpy to pygame
    color_repr_layer.blit(pygame.surfarray.make_surface(color_repr_img), (0, 0))
    alpha = pygame.surfarray.pixels_alpha(color_repr_layer)
    alpha[:, :] = (color_repr_alpha * 255).astype(np.uint8)
    del alpha

    # Blitting every layer on screen in right order
    if view[0]:
        screen.blit(background, (0, margin_top))
    screen.blit(pygame.transform.scale(color_repr_layer, size), (0, margin_top))
    if view[0]:
        screen.blit(rock_background, (0, margin_top))
    if view[5]:
        screen.blit(obstacle_layer, (0, margin_top))
    if view[1]:
        screen.blit(ants_layer, (0, margin_top))

    shift = pygame.key.get_pressed()[pygame.K_LSHIFT] or pygame.key.get_pressed()[pygame.K_RSHIFT]
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_KP_ENTER, pygame.K_RETURN]:
                paused = not paused
            elif event.key == pygame.K_UP:
                speed = min(20, speed + 1)
                paused = False
            elif event.key == pygame.K_DOWN:
                speed = max(-20, speed - 1)
                paused = False
            elif event.key in [pygame.K_KP0, pygame.K_0]:   # Background
                toggle_view(0, shift, view)
            elif event.key in [pygame.K_KP1, pygame.K_1]:   # Ants
                toggle_view(1, shift, view)
            elif event.key in [pygame.K_KP2, pygame.K_2]:   # Perceptive fields
                toggle_view(2, shift, view)
            elif event.key in [pygame.K_KP3, pygame.K_3]:   # Pheromones
                toggle_view(3, shift, view)
            elif event.key in [pygame.K_KP4, pygame.K_4]:   # Food
                toggle_view(4, shift, view)
            elif event.key in [pygame.K_KP5, pygame.K_5]:   # Obstacles
                toggle_view(5, shift, view)

    if pygame.key.get_pressed()[pygame.K_LEFT]:
        if shift:
            step -= 20
        else:
            step -= 1
    if pygame.key.get_pressed()[pygame.K_RIGHT]:
        if shift:
            step += 20
        else:
            step += 1

    # Next step, next tick
    pygame.display.flip()
    clock.tick(30)
    if not paused:
        step += speed
    step = step % len(states)
