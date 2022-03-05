"""
This script makes chasm using Cellular Automata
"""

import random
import time
import math

from copy import copy
from typing import List

import pygame

# global params
width, height = 80, 50

class Cell:
    def __init__(self, v = 0, c = 0):
        self.v = v
        self.c = c
        self.visited_points = set()


class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

def encode_coords(x: int, y: int):
    """ Encode coordinates for array access """
    return x + (width * y)


def moore_nbs(x: int, y: int, Map: List[Cell]):
    """ Return values of neighbours """
    top, top_right, right, bottom_right, bottom, bottom_left, left, top_left = None, None, None, None, None, None, None, None 

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


def neumann_nbs(x: int, y: int, Map: List[Cell]):
    top, right, bottom, left = None, None, None, None

    if x > 0:
        left = copy(Map[encode_coords(x - 1, y)])
    if x < width - 1:
        right = copy(Map[encode_coords(x + 1, y)])
    if y > 0:
        top = copy(Map[encode_coords(x, y - 1)])
    if y < height - 1:
        bottom = copy(Map[encode_coords(x, y + 1)])

    return top, right, bottom, left

def step(Map: List[Cell], Points: List[Point]):
    Map_copy = [Cell() for _ in range(width * height)]
    for x in range(width):
        for y in range(height):
            cell = copy(Map[encode_coords(x, y)])
            if cell.v == 0:
                continue
            # find closest point
            p_vector = Point(0, 0)
            min_d = int(width * height)
            for p in Points:
                if cell.v == 2 and p.x == x and p.y == y:
                    cell.visited_points.add(p)
                if p not in cell.visited_points:
                    p_v = Point(x - p.x, y - p.y) 
                    d = math.sqrt((p_v.x ** 2) + (p_v.y ** 2))
                    if cell.v == 2 and d < 3:
                        cell.visited_points.add(p)
                    if d < min_d:
                        min_d = d
                        p_vector = p_v

            if cell.v == 2:
                if len(cell.visited_points) == len(Points):
                    cell.c = 0
                    cell.v = 0

                if cell.c > 0:
                    # 0 = up, 1 = left, 2 = down, 3 = right
                    dirs = []
                    if (p_vector.x > 0):
                        dirs += [1 for _ in range(int(p_vector.x))]
                    else:
                        dirs += [3 for _ in range(int(-p_vector.x))]

                    if (p_vector.y > 0):
                        dirs += [0 for _ in range(int(p_vector.y))]
                    else:
                        dirs += [2 for _ in range(int(-p_vector.y))]
                    direction = random.choice(dirs)

                    up, right, down, left = neumann_nbs(x, y, Map)
                    if direction == 0 and up is not None and up.v !=2:
                        cell.v = 0
                        if up.v == 1:
                            up.c = cell.c - 1
                        else:
                            up.c = cell.c
                        up.v = 2
                        if len(cell.visited_points) > 0:
                            up.visited_points.update(cell.visited_points)
                        Map_copy[encode_coords(x, y - 1)] = up 
                    elif direction == 1 and left is not None and left.v != 2: 
                        cell.v = 0
                        if left.v == 1:
                            left.c = cell.c - 1
                        else:
                            left.c = cell.c
                        left.v = 2
                        if len(cell.visited_points) > 0:
                            left.visited_points.update(cell.visited_points)
                        Map_copy[encode_coords(x - 1, y)] = left 
                    elif direction == 2 and down is not None and down.v != 2:
                        cell.v = 0
                        if down.v == 1:
                            down.c = cell.c - 1
                        else:
                            down.c = cell.c
                        down.v = 2
                        if len(cell.visited_points) > 0:
                            down.visited_points.update(cell.visited_points)
                        Map_copy[encode_coords(x, y + 1)] = down 
                    elif direction == 3 and right is not None and right.v != 2:
                        cell.v = 0
                        if right.v == 1:
                            right.c = cell.c - 1
                        else:
                            right.c = cell.c
                        right.v = 2
                        if len(cell.visited_points) > 0:
                            right.visited_points.update(cell.visited_points)
                        Map_copy[encode_coords(x + 1, y)] = right 
                else:
                    cell.v = 0

            cell_ = Map_copy[encode_coords(x, y)]
            if cell_.v != 2:
                Map_copy[encode_coords(x, y)] = cell 

    return Map_copy

def smooth_step(Map: List[Cell]):
    Map_copy = [Cell() for _ in range(width * height)]
    for x in range(width):
        for y in range(height):
            cell = copy(Map[encode_coords(x, y)])
            if cell.v == 0:
                continue
            elif cell.v == 2:
                Map_copy[encode_coords(x, y)] = cell
                continue

            nbs = moore_nbs(x, y, Map)
            alive = sum([1 for _ in nbs if _ is not None and _.v == 1])
            if alive >= 3:
                Map_copy[encode_coords(x, y)] = cell
    return Map_copy

if __name__ == '__main__':
    # local params
    scale = 6
    font_size = 16
    running = True
    Points = []
    Map = [Cell(0, 0) for _ in range(width * height)]
    for x in range(width):
        for y in range(int(height / 2), height): 
            cell = Map[encode_coords(x, y)]  
            cell.v = 1

    # initialization of pygame environment
    pygame.init()
    screen = pygame.display.set_mode([width * scale, height * scale])
    clock = pygame.time.Clock()

    while running:
        clock.tick(25)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    Map = [Cell(0, 0) for _ in range(width * height)]
                    for x in range(width):
                        for y in range(int(height / 2), height): 
                            cell = Map[encode_coords(x, y)]  
                            cell.v = 1
                    Points = []
                elif event.key == pygame.K_s:
                    pygame.image.save(screen, "img/%d.png" % time.time())
                elif event.key == pygame.K_SPACE:
                    Map = smooth_step(Map)

        left, center, right = pygame.mouse.get_pressed()
        mx, my = pygame.mouse.get_pos()
        if (left):
            cell = Map[encode_coords(int(mx / scale), int(my / scale))]
            cell.v = 2
            cell.c = 20 
        if (right):
            Points.append(Point(int(mx / scale), int(my / scale)))

        Map = step(Map, Points)

        screen.fill((0, 0, 0))
        for x in range(width):
            for y in range(height):
                cell = Map[encode_coords(x, y)]
                if cell.v == 1:
                    pygame.draw.rect(screen, color=(255, 255, 255), rect=(x * scale, y * scale, scale, scale))
                elif cell.v == 2: 
                    pygame.draw.rect(screen, color=(0, 255, 0), rect=(x * scale, y * scale, scale, scale))
        for p in Points:
            pygame.draw.circle(screen, (255, 0, 0), (p.x * scale, p.y * scale), 5)
        pygame.display.flip()
    pygame.quit()
