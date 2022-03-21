import random

import pygame
from typing import List

width, height = 400, 200

class Pixel:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

def create_ocean(base=2) -> List[Pixel]:
    pixels = []
    for x in range(width):
        y0 = base ** (-1 + (x / width))
        for y in range(height - int(y0 * height)):
            pixels.append(Pixel(x, y))
    return pixels


if __name__ == '__main__':

    # local params
    running = True
    flip = True 
    base = 2
    fps = 60

    # initialization of enviroment
    pygame.init()
    screen = pygame.display.set_mode([width, height])
    clock = pygame.time.Clock()
    Ocean = create_ocean(base)

    while running:
        clock.tick(fps)

        event: pygame.event.Event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_q:
                    if base > 2:
                        base -= 1
                        Ocean = create_ocean(base)
                        flip = True
                elif event.key == pygame.K_e:
                    base += 1
                    Ocean = create_ocean(base)
                    flip = True
                elif event.key == pygame.K_s:
                    pygame.image.save(screen, "img/fq%d.png" % fq)

        if flip:
            screen.fill((255, 218, 56))
            for p in Ocean:
                x = p.x
                y = p.y
                pygame.draw.rect(screen, (0, 0, 255), (x, y, 1 ,1))
            pygame.display.flip()
            flip = False
    pygame.quit()
