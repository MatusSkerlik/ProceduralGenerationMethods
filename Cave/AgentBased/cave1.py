"""
This script makes chasm using agents and cellular automata
"""

import random
import time
import math

from copy import copy
from typing import List

import pygame

# global params
width, height = 75, 40

class Cell:
    def __init__(self, v = 0):
        self.v = v

class Acid(Cell):
    def __init__(self, d=0):
        super().__init__(2)
        self.d = d # durability
        self.visited_points = set()

    def __repr__(self):
        return "Acid, d = %d" % self.d

class Wall(Cell):
    def __init__(self, d=1):
        super().__init__(1)
        assert d > 0
        self.d = d 

class Void(Cell):
    def __init__(self):
        super().__init__(0)

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
    Map_copy = copy(Map) 
    for x in range(width):
        for y in range(height):
            cell = copy(Map[encode_coords(x, y)])
            if isinstance(cell, Void):
                continue
            elif isinstance(cell, Acid):
                # find closest point
                p_vector = Point(0, 0)
                min_d = int(width * height)
                for p in Points:
                    if p not in cell.visited_points:
                        if p.x == x and p.y == y:
                            cell.visited_points.add(p)
                        else:
                            p_v = Point(x - p.x, y - p.y) 
                            d = math.sqrt((p_v.x ** 2) + (p_v.y ** 2))
                            if d < 3:
                                cell.visited_points.add(p)
                            elif d < min_d:
                                min_d = d
                                p_vector = p_v

                if (len(cell.visited_points) == len(Points)) or (cell.d <= 0):
                    cell = Void()
                else: 
                    # 0 = up, 1 = right, 2 = down, 3 = left 
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

                    nbs = neumann_nbs(x, y, Map)
                    nb = nbs[direction] 
                    if not isinstance(nb, Acid):
                        if isinstance(nb, Wall):
                            if nb.d > 1:
                                nb = Wall(nb.d - 1)
                            else:
                                if cell.d > 1:
                                    nb = Acid(cell.d - 1) 
                                    nb.visited_points = cell.visited_points
                                else:
                                    nb = Void()
                                cell = Void()
                        else:
                            nb = Acid(cell.d)
                            nb.visited_points = cell.visited_points
                            cell = Void()
                        if direction == 0:
                            Map_copy[encode_coords(x, y - 1)] = nb 
                        elif direction == 1:
                            Map_copy[encode_coords(x + 1, y)] = nb 
                        elif direction == 2:
                            Map_copy[encode_coords(x, y + 1)] = nb 
                        else:
                            Map_copy[encode_coords(x - 1, y)] = nb 
                Map_copy[encode_coords(x, y)] = cell 
    return Map_copy

def smooth_step(Map: List[Cell]):
    Map_copy = copy(Map) 
    for x in range(width):
        for y in range(height):
            cell = copy(Map[encode_coords(x, y)])
            nbs = moore_nbs(x, y, Map)
            alive = sum([1 for _ in nbs if _ is not None and isinstance(_, Wall)])
            if isinstance(cell, Void) and alive > 4: 
                Map_copy[encode_coords(x, y)] = Wall() 
            elif isinstance(cell, Wall) and alive < 4:
                Map_copy[encode_coords(x, y)] = Void() 
    return Map_copy

if __name__ == '__main__':
    # local params
    running = True
    scale = 8
    font_size = 24 
    wall_durability = 1
    acid_durability = width * height 
    Points = []
    Map = [Wall(wall_durability) for _ in range(width * height)]

    # initialization of pygame environment
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode([width * scale, height * scale], 0 , 8)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("courier", font_size, True)

    while running:
        clock.tick(60)
        fps = int(clock.get_fps())
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    Map = [Wall(wall_durability) for _ in range(width * height)]
                    Points = []
                elif event.key == pygame.K_s:
                    pygame.image.save(screen, "img/%d.png" % time.time())
                elif event.key == pygame.K_SPACE:
                    Map = smooth_step(Map)
                elif event.key == pygame.K_n:
                    x_ = int(width / 2) 
                    y_ = int(height / 2)
                    for x in range(x_ - 2, x_ + 3):
                        for y in range(y_ - 2, y_ + 3):
                            Map[encode_coords(x, y)] = Acid(acid_durability)
                elif event.key == pygame.K_m:
                    count = random.randint(3, 6)
                    for _ in range(count):
                        x = random.randint(0, width - 1)
                        y = random.randint(0, height - 1)
                        Points.append(Point(x, y))
                            


        left, center, right = pygame.mouse.get_pressed()
        mx, my = pygame.mouse.get_pos()
        if (left):
            Map[encode_coords(int(mx / scale), int(my / scale))] = Acid(acid_durability)
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

        fps_text = font.render(str(fps), True, (0, 255, 0)) 
        screen.blit(fps_text, (0, 0))
        pygame.display.flip()
    pygame.quit()
