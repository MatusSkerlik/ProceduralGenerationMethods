"""
This script makes cave system using Cellular Automata
"""

import random
import time
from typing import List

import pygame

# global params
width, height = 50, 50


def encode_coords(x: int, y: int):
    """ Encode coordinates for array access """
    return x + (width * y)


def moore_nbs(x: int, y: int, Map: List[int]):
    """ Return values of neighbours """
    top, top_right, right, bottom_right, bottom, bottom_left, left, top_left = 0, 0, 0, 0, 0, 0, 0, 0

    if y > 0:
        top = Map[encode_coords(x, y - 1)]
        if x > 0:
            top_left = Map[encode_coords(x - 1, y - 1)]
        if x < width - 1:
            top_right = Map[encode_coords(x + 1, y - 1)]

    if y < height - 1:
        bottom = Map[encode_coords(x, y + 1)]
        if x > 0:
            bottom_right = Map[encode_coords(x - 1, y + 1)]
        if x < width - 1:
            bottom_left = Map[encode_coords(x + 1, y + 1)]

    if x > 0:
        left = Map[encode_coords(x - 1, y)]
    if x < width - 1:
        right = Map[encode_coords(x + 1, y)]

    return top, top_right, right, bottom_right, bottom, bottom_left, left, top_left


def neumann_nbs(x: int, y: int, Map: List[int]):
    top, right, bottom, left = 0, 0, 0, 0

    if x > 0:
        left = Map[encode_coords(x - 1, y)]
    if x < width - 1:
        right = Map[encode_coords(x + 1, y)]
    if y > 0:
        top = Map[encode_coords(x, y - 1)]
    if y < height - 1:
        bottom = Map[encode_coords(x, y + 1)]

    return top, right, bottom, left


def count_alive(nbs):
    """ Count alive neighbours """
    return len([1 for _ in nbs if _ == 1])


def step(survive_ratio: int, reborn_ratio: int, Map: List[int]):
    """ Step function in cellular automata """

    new_map = [0 for _ in range(width * height)]
    for x in range(width):
        for y in range(height):
            alive = count_alive(moore_nbs(x, y, Map))
            if Map[encode_coords(x, y)] == 1:  # it is alive
                if alive >= survive_ratio:  # it has minimum amount of alive nbs to survive
                    new_map[encode_coords(x, y)] = 1
            else:  # it is dead
                if alive > reborn_ratio:  # it has minimum amout of alive to give birth
                    new_map[encode_coords(x, y)] = 1

    return new_map


if __name__ == '__main__':
    # local params
    scale = 6
    font_size = 16
    running = True
    render = True
    make_step = False
    birth_probability = 0.7
    Map = [1 if random.random() > birth_probability else 0 for _ in range(width * height)]

    # initialization of pygame environment
    pygame.init()
    screen = pygame.display.set_mode([width * scale, height * scale])

    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    render = True
                    make_step = True
                elif event.key == pygame.K_r:
                    Map = [1 if random.random() > birth_probability else 0 for _ in range(width * height)]
                    render = True
                elif event.key == pygame.K_s:
                    pygame.image.save(screen, "img/%d.png" % time.time())

        if make_step:
            Map = step(5, 1, Map)
            make_step = False

        if render:
            screen.fill((0, 0, 0))
            for x in range(width):
                for y in range(height):
                    if Map[encode_coords(x, y)] == 1:
                        pygame.draw.rect(screen, color=(255, 255, 255),
                                         rect=(x * scale, y * scale, scale, scale))
            pygame.display.flip()
            render = False

    pygame.quit()
