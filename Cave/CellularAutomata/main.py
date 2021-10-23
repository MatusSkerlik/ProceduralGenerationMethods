import random
from typing import List

import pygame

# global params
width, height = 300, 300


def encode_coords(x, y):
    return x + (width * y)


def moore_nbs(x: int, y: int, Map: List[int]):
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

    return top, top_right, right, bottom_right, bottom, bottom_left, left, top_left


def count_alive(nbs):
    return len([1 for _ in nbs if _ == 1])


def step(survive_ratio: int, reborn_ratio: int, Map: List[int]):
    New_Map = [0 for _ in range(width * height)]
    for x in range(width):
        for y in range(height):
            alive = count_alive(moore_nbs(x, y, Map))
            if Map[encode_coords(x, y)] == 1:  # it is alive
                if alive >= survive_ratio:  # it has minimum amount of alive nbs to survive
                    New_Map[encode_coords(x, y)] = 1
            else:  # it is dead
                if alive > reborn_ratio:  # it has minimum amout of alive to give birth
                    New_Map[encode_coords(x, y)] = 1

    return New_Map


if __name__ == '__main__':

    # local params
    Map = None
    running = True
    make_step = False
    flip = True
    update_text = True
    steps = 0
    survive_ratio = 3
    reborn_ratio = 4
    birth_probability = 0.5

    # initialization of enviroment
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode([width, height + 32])
    font = pygame.font.SysFont(pygame.font.get_default_font(), 24)

    while running:
        event: pygame.event.Event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    flip = True
                    steps = 0
                elif event.key == pygame.K_r:
                    survive_ratio = 3
                    reborn_ratio = 4
                    birth_probability = 0.5
                    update_text = True
                elif event.key == pygame.K_q:
                    if survive_ratio > 1:
                        survive_ratio -= 1
                        update_text = True
                elif event.key == pygame.K_e:
                    if survive_ratio < 8:
                        survive_ratio += 1
                        update_text = True
                elif event.key == pygame.K_a:
                    if reborn_ratio > 1:
                        reborn_ratio -= 1
                        update_text = True
                elif event.key == pygame.K_d:
                    if reborn_ratio < 8:
                        reborn_ratio += 1
                        update_text = True
                elif event.key == pygame.K_w:
                    make_step = True
                    flip = True
                elif event.key == pygame.K_s:
                    pygame.image.save(screen, "img/%d-%d.png" % (survive_ratio, reborn_ratio))

        if steps == 0:
            Map = [1 if random.random() > birth_probability else 0 for _ in range(width * height)]
            steps += 1

        if make_step:
            Map = step(survive_ratio, reborn_ratio, Map)
            steps += 1
            make_step = False
            flip = True

        if flip:
            screen.fill(color=(0, 0, 0))
            for x in range(width):
                for y in range(height):
                    if Map[encode_coords(x, y)] == 1:
                        pygame.draw.rect(screen, color=(255, 255, 255), rect=(x, y, 1, 1))
            pygame.display.flip()

            flip = False
            update_text = True

        if update_text:
            if not flip:
                screen.fill(color=(0, 0, 0))
            text = font.render('sr:%d, rr:%d, bp:%.2f' % (survive_ratio, reborn_ratio, birth_probability), True,
                               (255, 255, 255))
            screen.blit(text, (4, width + 8))
            pygame.display.update((0, width, height, 32))
            update_text = False

pygame.quit()
