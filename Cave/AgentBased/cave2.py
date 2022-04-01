"""
This script makes caves using mining agent, splines and cellular automata 
"""

import random
import time
import math

from copy import copy
from typing import List

import pygame
from scipy.interpolate import CubicSpline
import numpy as np

# global params
width, height = 75, 40

class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __hash__(self):
        return hash(self.y * width + self.x)

class Wall(Point):
    pass

class Acid(Point):
    pass

class Void(Point):
    pass


def encode_coords(x: int, y: int):
    """ Encode coordinates for array access """
    return x + (width * y)


def cave_miner(sp: Point, Grid: List[Point], Points: List[Point]):
    def sort_by_distance(a: Point):
        return math.sqrt(((a.x - sp.x) ** 2) + ((a.y - sp.y) ** 2))

    if len(Points) < 3:
        raise StopIteration 

    sorted_points = [sp, ] + list(sorted(Points, key=sort_by_distance)) 
    X = list(map(lambda p: p.x, sorted_points))
    Y = list(map(lambda p: p.y, sorted_points))
   
    time = np.arange(0.0, len(X) - 1 + 0.1, 0.1)
    cx = CubicSpline([_ for _ in range(len(X))], X)
    cy = CubicSpline([_ for _ in range(len(Y))], Y) 

    queue = []
    for t in time:
        queue.append(Point(int(cx(t)), int(cy(t))))
    queue.reverse()

    pos = sp
    while len(queue) > 0:
        target = queue.pop()

        if target.x < 0 or target.x >= width or target.y < 0 or target.y >= height:
            continue

        while math.sqrt((target.x - pos.x) ** 2 + (target.y - pos.y) ** 2) > 1:
            # 0 = up, 1 = right, 2 = down, 3 = left 
            p_vector = Point(pos.x - target.x, pos.y - target.y)

            dirs = []
            if (p_vector.x > 0):
                dirs += [3 for _ in range(int(p_vector.x))]
            else:
                dirs += [1 for _ in range(int(-p_vector.x))]
            if (p_vector.y > 0):
                dirs += [0 for _ in range(int(p_vector.y))]
            else:
                dirs += [2 for _ in range(int(-p_vector.y))]

            direction = random.choice(dirs)
            if direction == 0 and pos.y > 0:
                pos = Point(pos.x, pos.y - 1)
            elif direction == 1 and pos.x < width - 1:
                pos = Point(pos.x + 1, pos.y)
            elif direction == 2 and pos.y < height - 1:
                pos = Point(pos.x, pos.y + 1)
            elif direction == 3 and pos.x > 0:
                pos = Point(pos.x - 1, pos.y)

            x_ = pos.x
            y_ = pos.y

            c_size = 1
            for x in range(x_ - c_size, x_ + c_size + 1):
                for y in range(y_ - c_size, y_ + c_size + 1):
                    if x > 0 and x < width - 1 and y > 0 and y < height - 1:
                        d = math.sqrt((x_ - x) ** 2 + (y_ - y) ** 2) 
                        if d <= c_size:
                            Grid[encode_coords(x, y)] = Void(x, y)
            yield queue, pos


def moore_nbs(x: int, y: int, Grid: List[Point]):
    """ Return values of neighbours """
    top, top_right, right, bottom_right, bottom, bottom_left, left, top_left = None, None, None, None, None, None, None, None 

    if y > 0:
        top = Grid[encode_coords(x, y - 1)]
        if x > 0:
            top_left = Grid[encode_coords(x - 1, y - 1)]
        if x < width - 1:
            top_right = Grid[encode_coords(x + 1, y - 1)]

    if y < height - 1:
        bottom = Grid[encode_coords(x, y + 1)]
        if x > 0:
            bottom_right = Grid[encode_coords(x - 1, y + 1)]
        if x < width - 1:
            bottom_left = Grid[encode_coords(x + 1, y + 1)]

    if x > 0:
        left = Grid[encode_coords(x - 1, y)]
    if x < width - 1:
        right = Grid[encode_coords(x + 1, y)]

    return top, top_right, right, bottom_right, bottom, bottom_left, left, top_left


def smooth_step(Grid: List[Point]):
    Grid_copy = copy(Grid) 
    for x in range(width):
        for y in range(height):
            cell = Grid[encode_coords(x, y)]
            nbs = moore_nbs(x, y, Grid)
            alive = sum([1 for _ in nbs if _ is not None and isinstance(_, Wall)])
            if isinstance(cell, Void) and alive > 4: 
                Grid_copy[encode_coords(x, y)] = Wall(x, y) 
            elif isinstance(cell, Wall) and alive < 4:
                Grid_copy[encode_coords(x, y)] = Void(x, y) 
    return Grid_copy

if __name__ == '__main__':
    # local params
    running = True
    scale = 8
    font_size = 24 

    # initialization of pygame environment
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode([width * scale, height * scale], 0 , 8)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("courier", font_size, True)

    # algorithm specific
    Grid = [Wall(x, y) for x in range(width) for y in range(height)]
    Points = []
    sp = Point(int(width / 2), int(height / 2))
    miner = None

    while running:
        clock.tick(120)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    Grid = [Wall(x, y) for x in range(width) for y in range(height)]
                    Points = []
                    miner = None
                elif event.key == pygame.K_s:
                    pygame.image.save(screen, "img/%d.png" % time.time())
                elif event.key == pygame.K_SPACE:
                    Grid = smooth_step(Grid)
                elif event.key == pygame.K_n:
                    if len(Points) > 2:
                        miner = cave_miner(sp, Grid, Points)
                elif event.key == pygame.K_m:
                    while True:
                        x = random.randint(5, width - 5 - 1)
                        y = random.randint(5, height - 5 - 1)
                        is_ok = 0 
                        if len(Points) == 0:
                            Points.append(Point(x, y))
                            break 
                        else:
                            for p in Points:
                                if (abs(p.x - x) > 1) and (abs(p.y - y) > 1):
                                    is_ok += 1 
                        if len(Points) == is_ok:
                            Points.append(Point(x, y))
                            break
                    if len(Points) > 2:
                        miner = cave_miner(sp, Grid, Points)
                            
        left, center, right = pygame.mouse.get_pressed()
        mx, my = pygame.mouse.get_pos()
        if (left):
            x = int(mx / scale)
            y = int(my / scale)
            Points.append(Point(x, y))


        screen.fill((0, 0, 0))
        for x in range(width):
            for y in range(height):
                cell = Grid[encode_coords(x, y)]
                if isinstance(cell, Wall):
                    pygame.draw.rect(screen, (255, 255, 255), (x * scale, y * scale, scale, scale))
                elif isinstance(cell, Acid):
                    pygame.draw.rect(screen, (0, 255, 0), (x * scale, y * scale, scale, scale))
        for p in Points:
            pygame.draw.circle(screen, (255, 0, 0), (p.x * scale, p.y * scale), 5)

        if miner is not None:
            try:
                spoints, p = next(miner)
                for point in spoints:
                    pygame.draw.circle(screen, (0, 0, 255), (point.x * scale, point.y * scale), 2)
                pygame.draw.rect(screen, (0, 255, 0), (p.x * scale, p.y * scale, scale, scale))
            except StopIteration:
                miner = None

        fps = int(clock.get_fps())
        fps_text = font.render(str(fps), True, (0, 255, 0)) 
        screen.blit(fps_text, (0, 0))
        pygame.display.flip()
    pygame.quit()
