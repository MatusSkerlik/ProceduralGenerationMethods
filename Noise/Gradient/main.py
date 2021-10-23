import random

import pygame

# global params
from noise import pnoise2, snoise2

width, height = 200, 200

if __name__ == '__main__':

    # local params
    running = True
    flip = True
    fq = 1

    # initialization of enviroment
    pygame.init()
    screen = pygame.display.set_mode([width, height])

    while running:
        event: pygame.event.Event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break

                if event.key == pygame.K_SPACE:
                    flip = True

                if event.key == pygame.K_q:
                    if fq > 2:
                        fq -= 2
                        flip = True

                if event.key == pygame.K_e:
                    fq += 2
                    flip = True

                if event.key == pygame.K_s:
                    pygame.image.save(screen, "img/fq%d.png" % fq)

        if flip:

            for x in range(width):
                for y in range(height):
                    v = int(255 * (0.5 + pnoise2(x / width * fq, y / height * fq) / 1.57))
                    if v > 255:
                        v = 255
                    elif v < 0:
                        v = 0
                    pygame.draw.rect(screen, color=(v, v, v), rect=(x, y, 1, 1))
            pygame.display.flip()
            flip = False

    pygame.quit()
