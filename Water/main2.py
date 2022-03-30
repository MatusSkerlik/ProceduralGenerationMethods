"""
This script simulate water using Cellular Automata
"""

import random
import time
import math

from copy import copy
from typing import List, Set

import pygame

# global params
width, height = 50, 40
MaxMass = 16 
MinMass = 1

class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __hash__(self):
        return hash(self.y * width + self.x)

def encode_coords(x: int, y: int):
    """ Encode coordinates for array access """
    return x + (width * y)


def nbs(x: int, y: int, Map: List[int]):

    top, right, bottom, left = -1, -1, -1, -1

    if x > 0:
        left = Map[encode_coords(x - 1, y)]
    if y > 0:
        top = Map[encode_coords(x, y - 1)]
    if x < width - 1:
        right = Map[encode_coords(x + 1, y)]
    if y < height - 1:
        bottom = Map[encode_coords(x, y + 1)]

    return top, right, bottom, left


def step(p: Point, to_visit: Set[Point], Map: List[int], p0: Point, p1: Point):
    
    new_visit = set()

    while len(to_visit) > 0:
        point = to_visit.pop()

        # check if it is not wall of void
        if (point.x < 0):
            p0 = p
            continue
        elif point.x >= width:
            p1 = p
            continue
        elif Map[encode_coords(point.x, point.y)] != 0:
            if (point.x < p.x): # left
                p0 = Point(point.x + 1, point.y) 
                continue
            else:
                p1 = Point(point.x - 1, point.y) 
                continue

        if (point.x < p.x): # left
            _, _, bottom, left = nbs(point.x, point.y, Map)
            if bottom == 0:
                p = Point(point.x, point.y + 1)
                return p, set(), None, None 
            elif left == 0:
                new_visit.add(Point(point.x - 1, point.y))
            else:
                p0 = point
        else:
            _, right, bottom, _ = nbs(point.x, point.y, Map)
            if bottom == 0:
                p = Point(point.x, point.y + 1)
                return p, set(), None, None 
            elif right == 0:
                new_visit.add(Point(point.x + 1, point.y))
            else:
                p1 = point

    if p0 is not None and p1 is not None:
        for x in range(p0.x, p1.x + 1):
            Map[encode_coords(x, p.y)] = 2
        return None, set(), None, None

    top, right, bottom, left = nbs(p.x, p.y, Map) 

    if bottom == 0:
        p = Point(p.x, p.y + 1)
        return p, set(), None, None

    new_visit.add(Point(p.x - 1, p.y))
    new_visit.add(Point(p.x + 1, p.y))

    return p, new_visit, p0, p1 

if __name__ == '__main__':
    # local params
    running = True
    scale = 8
    font_size = 24 

    # initialization of pygame environment
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((width * scale, height * scale))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("courier", font_size, True)

    point = None
    p0 = None
    p1 = None
    to_visit: Set[Point] = set() 
    Map = [0 for _ in range(width * height)]

    while running:
        clock.tick(120)

        fps = int(clock.get_fps())
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    point = None
                    p0 = None
                    p1 = None
                    Map = [0 for _ in range(width * height)]
                    to_visit = set()
                elif event.key == pygame.K_s:
                    pygame.image.save(screen, "img/%d.png" % time.time())

        left, center, right = pygame.mouse.get_pressed()
        mx, my = pygame.mouse.get_pos()
        if left: 
            x = int(mx / scale)
            y = int(my / scale)
            if point is None:
                point = Point(x, y) 
        if right:
            x = int(mx / scale)
            y = int(my / scale)
            Map[encode_coords(x, y)] = 1

        if point is not None:
            point, to_visit, p0, p1 = step(point, to_visit, Map, p0, p1)


        screen.fill((32, 32, 32))

        # draw point
        if point is not None:
            pygame.draw.rect(screen, (0, 255, 0), (point.x * scale, point.y * scale, scale, scale))

        # draw lookup
        for p in to_visit:
            pygame.draw.rect(screen, (255, 0, 0), (p.x * scale, p.y * scale, scale, scale))

        # draw wall
        for x in range(width):
            for y in range(height):
                if Map[encode_coords(x, y)] == 1:
                    pygame.draw.rect(screen, (255, 255, 255), (x * scale, y * scale, scale, scale))
                elif Map[encode_coords(x, y)] == 2:
                    pygame.draw.rect(screen, (0, 0, 255), (x * scale, y * scale, scale, scale))
        fps_text = font.render(str(fps), True, (0, 255, 0)) 
        screen.blit(fps_text, (0, 0))
        pygame.display.flip()
    pygame.quit()
