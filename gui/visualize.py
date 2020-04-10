import sys
import os
import pygame
import pickle
from utils import *

from environment.ants import AntsVisualization
from environment.pheromone import PheromoneVisualization, Pheromone
from environment.circle_obstacles import CircleObstaclesVisualization
from environment.walls import Walls
from environment.food import FoodVisualization
from environment.RL_api import RLVisualization
from environment.anthill import AnthillVisualization

REWARD_ANTS_COLOR = np.array([255, 255, 128, 255])
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


class Visualizer:
    def __init__(self):
        self.background = None
        self.rock_background = None
        self.FOOD_FILL = None
        self.RL_FILL = None
        self.PHEROMONES_FILL = None
        self.ants_layer = None
        self.obstacle_layer = None
        self.color_repr_layer = None
        self.color_repr_img = None
        self.color_repr_alpha = None
        self.size = None
        self.ant = None
        self.dirt = None
        self.rock = None
        self.rock_tile = None
        self.holding_ant = None
        self.sw = self.sh = self.ew = self.eh = 0
        self.big_dim = 1600

    def setup_environment(self, env):
        # --- CREATING BACKGROUND IMAGE AND PRECOMPUTED ARRAYS ---
        self.background = pygame.Surface(self.size)
        for xd in range(0, self.sw, self.dirt.get_width()):
            for yd in range(0, self.sh, self.dirt.get_height()):
                self.background.blit(self.dirt, (xd, yd))

        self.rock_background = pygame.Surface(self.size, pygame.SRCALPHA)
        for xd in range(0, self.sw, self.rock_tile.get_width()):
            for yd in range(0, self.sh, self.rock_tile.get_height()):
                self.rock_background.blit(self.rock_tile, (xd, yd))

        anthill = pygame.Surface((self.ew, self.eh), pygame.SRCALPHA)
        anthill.fill(ANTHILL_COLOR)

        self.FOOD_FILL = np.array(FOOD_COLOR)[AX, AX, :].repeat(self.ew, axis=0).repeat(self.eh, axis=1)
        self.RL_FILL = np.array(PERCEPTIVE_FIELD_COLOR)[AX, AX, :].repeat(self.ew, axis=0).repeat(self.eh, axis=1)
        self.PHEROMONES_FILL = []

        for obj in env.objects:
            if isinstance(obj, Walls):
                img = np.zeros((obj.map.shape[0], obj.map.shape[1]))
                img[obj.map] = 255
                img = img[:, :, AX].repeat(3, axis=2)
                img = pygame.surfarray.pixels_red(pygame.transform.scale(pygame.surfarray.make_surface(img), self.size))[:,
                      :]
                alpha = pygame.surfarray.pixels_alpha(self.rock_background)
                alpha[:, :] = img[:, :]
                del alpha
            elif isinstance(obj, PheromoneVisualization) or isinstance(obj, Pheromone):
                self.PHEROMONES_FILL.append(np.array(obj.color)[AX, AX, :].repeat(self.ew, axis=0).repeat(self.eh, axis=1))
            elif isinstance(obj, AnthillVisualization):
                alpha = pygame.surfarray.pixels_alpha(anthill)
                for x in range(self.ew):
                    for y in range(self.eh):
                        dist = ((obj.x - x) ** 2 + (obj.y - y) ** 2) ** 0.5
                        if dist > obj.radius:
                            alpha[x, y] = 0
                del alpha
        self.background.blit(pygame.transform.scale(anthill, self.size), (0, 0))

        # --- CREATING OTHER LAYERS ---
        self.ants_layer = pygame.Surface(self.size, pygame.SRCALPHA)
        self.obstacle_layer = pygame.Surface(self.size, pygame.SRCALPHA)
        self.color_repr_layer = pygame.Surface((env.w, env.h), pygame.SRCALPHA)
        self.color_repr_img = np.zeros((env.w, env.h, 3), dtype=np.uint8)
        self.color_repr_alpha = np.zeros((env.w, env.h))

    def visualize(self, file_name=None):
        # --- LOADING A SAVED FILE ---
        base_path = "saved/"
        if file_name is None or not os.path.exists(base_path + file_name):
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

        big_dim = self.big_dim
        self.ew = states[0].w
        self.eh = states[0].h
        if self.ew > self.eh:
            self.sw = big_dim
            self.sh = big_dim // self.ew * self.eh
        else:
            self.sw = big_dim
            self.sh = big_dim // self.eh * self.ew
        self.size = self.sw, self.sh
        margin_top = 25
        margin_bot = 50
        screen_size = self.sw, self.sh + margin_bot + margin_top
        e_max = max(self.ew, self.eh)
        zoom_factor = big_dim / e_max

        screen = pygame.display.set_mode(screen_size)
        clock = pygame.time.Clock()
        font = pygame.font.SysFont('consolas', 24, True)
        small_font = pygame.font.SysFont('consolas', 16, False)

        # --- LOADING ASSETS ---
        self.ant = pygame.transform.smoothscale(pygame.image.load("assets/ant_32_white.png"), ANTS_SIZE)
        self.dirt = pygame.image.load("assets/dirt.jpg")
        self.rock = pygame.image.load("assets/rock.png")
        self.rock_tile = pygame.image.load("assets/rock_tile.jpg")

        self.holding_ant = pygame.transform.smoothscale(pygame.image.load("assets/carried_food_32.png"), ANTS_SIZE)
        color = pygame.Surface((100, 100))
        color.fill(FOOD_COLOR)
        self.holding_ant.blit(color, (0, 0), special_flags=pygame.BLEND_RGB_MULT)

        self.setup_environment(states[0])
        walls = [obj for obj in states[0].objects if isinstance(obj, Walls)][0]

        # --- ITERATING OVER SAVED STEPS ---
        step = 0
        speed = 1
        paused = False
        view = [True] * 10
        while 1:
            # Getting step's environment
            e = states[step % len(states)]

            new_walls = [obj for obj in e.objects if isinstance(obj, Walls)][0]
            if new_walls != walls:
                walls = new_walls
                self.setup_environment(e)

            # Reinitializing layers
            screen.fill((0, 0, 0))
            self.ants_layer.fill((0, 0, 0, 0))
            self.obstacle_layer.fill((0, 0, 0, 0))
            self.color_repr_img[:, :, :] = 255
            self.color_repr_alpha[:, :] = 0

            # Displaying every environment object
            pi = 0
            for obj in e.objects:
                if isinstance(obj, AntsVisualization):
                    xyt = obj.ants.copy()
                    xyt[:, 0] *= self.sw / self.ew
                    xyt[:, 1] *= self.sh / self.eh
                    xyt[:, 2] *= -1

                    # Display one ant at each location, with holding food icon if it's the case
                    for i, (x, y, t) in enumerate(xyt):
                        img = self.ant.copy()
                        color = pygame.Surface((100, 100))
                        color.fill((REWARD_ANTS_COLOR * obj.reward_state[i] / 255).astype(np.uint8))
                        img.blit(color, (0, 0), special_flags=pygame.BLEND_RGB_MULT)

                        if obj.holding[i] > 0:
                            img.blit(self.holding_ant, (0, 0))

                        # img = self.ant if not obj.mandibles[i] else self.holding_ant
                        # img = self.ant if obj.holding[i] == 0 else self.holding_ant
                        rotated = pygame.transform.rotate(img, t / np.pi * 180 - 90)
                        self.ants_layer.blit(rotated, (x - rotated.get_width()/2, y - rotated.get_height()/2))

                elif isinstance(obj, CircleObstaclesVisualization):
                    # Display a scaled rock image at each location
                    for rock_i in range(len(obj.centers)):
                        rock_scaled = pygame.transform.scale(self.rock,
                                                             (int(obj.radiuses[rock_i] * zoom_factor * 2),
                                                              int(obj.radiuses[rock_i] * zoom_factor * 2)))
                        posx, posy = (obj.centers[rock_i] * zoom_factor).astype(int)
                        self.obstacle_layer.blit(rock_scaled, (posx - rock_scaled.get_width() / 2, posy - rock_scaled.get_height() / 2))

                elif isinstance(obj, PheromoneVisualization) and view[3]:
                    # Display the quantity of pheromones on the color representation layer (color indicated at initialization of pheromones)
                    alph = (obj.phero / (obj.max_val + 0.0001)) * 0.6
                    self.color_repr_img, self.color_repr_alpha = mix_alpha(self.color_repr_img,
                                                                           self.color_repr_alpha,
                                                                           self.PHEROMONES_FILL[pi],
                                                                           alph)
                    pi += 1

                elif isinstance(obj, FoodVisualization) and view[4]:
                    # Display the quantity of food on the color representation layer (color as constant at beginning of this script)
                    alph = obj.qte / (np.max(obj.qte) + 0.0001) * 0.3
                    self.color_repr_img, self.color_repr_alpha = mix_alpha(self.color_repr_img,
                                                                           self.color_repr_alpha,
                                                                           self.FOOD_FILL,
                                                                           alph)

                elif isinstance(obj, RLVisualization) and view[2]:
                    # Display the perceptive field on the color representation layer (color as constant at beginning of this script)
                    if obj.heatmap is not None:
                        alph = (obj.heatmap / np.max(obj.heatmap)) * 0.3
                        self.color_repr_img, self.color_repr_alpha = mix_alpha(self.color_repr_img,
                                                                               self.color_repr_alpha,
                                                                               self.RL_FILL,
                                                                               alph)

                elif isinstance(obj, AnthillVisualization):
                    txt = font.render("FOOD IN ANTHILL: " + str(int(obj.food)), True, (255, 255, 255))
                    screen.blit(txt, (self.sw - 30 - txt.get_width(), self.sh + 13 + margin_top))

            step_str = str(step)
            if step < 1000:
                step_str = " " + step_str
            if step < 100:
                step_str = " " + step_str
            if step < 10:
                step_str = " " + step_str
            screen.blit(font.render("STEP " + step_str + " / " + str(len(states)) + "  - " + ("SPEED " + str(speed) if not paused else "PAUSED"),
                                    True, (255, 255, 255)),
                        (30, self.sh + 13 + margin_top))
            help_txt = small_font.render("0. BACKGROUND - 1. ANTS - 2. PERCEPTIVE FIELDS - 3. PHEROMONES - 4. FOOD - 5. OBSTACLES",
                                         True, (255, 255, 255))
            screen.blit(help_txt, (self.sw // 2 - help_txt.get_width() // 2, 6))

            # Converting the color representation layer from numpy to pygame
            self.color_repr_layer.blit(pygame.surfarray.make_surface(self.color_repr_img), (0, 0))
            alpha = pygame.surfarray.pixels_alpha(self.color_repr_layer)
            alpha[:, :] = (self.color_repr_alpha * 255).astype(np.uint8)
            del alpha

            # Blitting every layer on screen in right order
            if view[0]:
                screen.blit(self.background, (0, margin_top))
            screen.blit(pygame.transform.scale(self.color_repr_layer, self.size), (0, margin_top))
            if view[0]:
                screen.blit(self.rock_background, (0, margin_top))
            if view[5]:
                screen.blit(self.obstacle_layer, (0, margin_top))
            if view[1]:
                screen.blit(self.ants_layer, (0, margin_top))

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
