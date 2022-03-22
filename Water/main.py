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
width, height = 75, 40
MaxMass = 1.0
MaxCompress = 0.02
MinMass = 0.001

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


def get_stable_state_b(total_mass: float):
    if total_mass <= 1.0:
        return 1
    elif total_mass < (2 * MaxMass + MaxCompress):
        return ((MaxMass ** 2) + (total_mass * MaxCompress)) / (MaxMass + MaxCompress) 
    else:
        return (total_mass + MaxCompress) / 2


def step(Mass: List[float], Blocks: List[int]):
    New_Mass = copy(Mass) 
    New_Blocks = [0 for _ in range(width * height)]

    for x in range(width):
        for y in range(height):

            # -1 undefined, 0 wall, 1 air, 2 water 
            coords = encode_coords(x, y)
            cell_type = Blocks[coords]
            if cell_type != 2:
                continue

            remaining_mass = Mass[coords]
            if remaining_mass <= 0:
                continue

            top, right, bottom, left = nbs(x, y, Blocks)
            # BELOW
            if bottom > 0:
                Flow = get_stable_state_b(remaining_mass + Mass[encode_coords(x, y + 1)]) - Mass[encode_coords(x, y + 1)]
                if Flow > 0:
                    New_Mass[coords] -= Flow
                    New_Mass[encode_coords(x, y + 1)] += Flow
                    remaining_mass -= Flow

            if remaining_mass <= 0:
                continue

            if left > 0:
                Flow = (Mass[coords] - Mass[encode_coords(x - 1, y)]) / 4
                if Flow > 0:
                    New_Mass[coords] -= Flow
                    New_Mass[encode_coords(x - 1, y)] += Flow
                    remaining_mass -= Flow

            if remaining_mass <= 0:
                continue

            if right > 0:
                Flow = (Mass[coords] - Mass[encode_coords(x + 1, y)]) / 4
                if Flow > 0:
                    New_Mass[coords] -= Flow
                    New_Mass[encode_coords(x + 1, y)] += Flow
                    remaining_mass -= Flow

            if remaining_mass <= 0:
                continue
            
            if top > 0:
                Flow = remaining_mass - get_stable_state_b(remaining_mass + Mass[encode_coords(x, y - 1)])
                if Flow > 0:
                    New_Mass[coords] -= Flow
                    New_Mass[encode_coords(x, y - 1)] += Flow
                    remaining_mass -= Flow


    for x in range(width):
        for y in range(height):
            coords = encode_coords(x, y)
            if Blocks[coords] == 0:
                New_Blocks[coords] = 0
            else:
                if New_Mass[coords] > MinMass:
                    New_Blocks[coords] = 2
                else:
                    New_Blocks[coords] = 1

    return New_Mass, New_Blocks
        

if __name__ == '__main__':
    # local params
    running = True
    scale = 8
    font_size = 24 
    wall_durability = 3
    acid_durability = 15 
    Mass = [0.0 for _ in range(width * height)]
    Blocks = [-1 for _ in range(width * height)]

    # initialization of pygame environment
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((width * scale, height * scale))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("courier", font_size, True)

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
                    Mass = [0.0 for _ in range(width * height)]
                    Blocks = [-1 for _ in range(width * height)]
                elif event.key == pygame.K_s:
                    pygame.image.save(screen, "img/%d.png" % time.time())

        left, center, right = pygame.mouse.get_pressed()
        mx, my = pygame.mouse.get_pos()

        if left:
            x = int(mx / scale)
            y = int(my / scale)
            if Blocks[encode_coords(x, y)] == 1: 
                Blocks[encode_coords(x, y)] = 2
                Mass[encode_coords(x, y)] = 1.0
        if right:
            x = int(mx / scale)
            y = int(my / scale)
            Blocks[encode_coords(x, y)] = 0

        if center:
            x = int(mx / scale)
            y = int(my / scale)
            if Blocks[encode_coords(x, y)] == 0: 
                Blocks[encode_coords(x, y)] = 1

        Mass, Blocks = step(Mass, Blocks)
        
        screen.fill((32, 32, 32))
        for x in range(width):
            for y in range(height):
                cell_type = Blocks[encode_coords(x, y)]
                if cell_type == 0:
                    pygame.draw.rect(screen, color=(255, 255, 255), rect=(x * scale, y * scale, scale, scale))
                elif cell_type == 2:
                    mass = Mass[encode_coords(x, y)]
                    g = 255 - 255 * (mass / (1 + (height * MaxCompress)))
                    if g < 0:
                        g = 0
                    if g > 255:
                        g = 255
                    pygame.draw.rect(screen, color=(0, g, 255), rect=(x * scale, y * scale, scale, scale))

        fps_text = font.render(str(fps), True, (0, 255, 0)) 
        screen.blit(fps_text, (0, 0))
        pygame.display.flip()
    pygame.quit()
