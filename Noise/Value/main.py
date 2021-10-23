import math
import random
import pygame

# global params
width, height = 200, 200


def _random(x: int, y: int, seed: int):
    random.seed(17*x + 127*y)
    return random.random()


def cerp(v0, v1, t):
    v3 = (1 - math.cos(t * math.pi)) / 2
    return v0 * (1 - v3) + v1 * v3


def lerp(v0, v1, t):
    return (1 - t) * v0 + t * v1


def noise(x: int, y: int, fq: int):
    x_interval = int(width / fq)
    y_interval = int(height / fq)

    i_x = int(x / x_interval)
    i_y = int(y / y_interval)

    a = _random(i_x, i_y, 0)
    b = _random(i_x + 1, i_y, 0)
    c = _random(i_x, i_y + 1, 0)
    d = _random(i_x + 1, i_y + 1, 0)

    x_f = (x % x_interval) / x_interval
    y_f = (y % y_interval) / y_interval

    A = cerp(a, b, x_f)
    B = cerp(c, d, x_f)
    C = cerp(A, B, y_f)

    return C


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
                    v = int(255 * noise(x, y, fq))
                    pygame.draw.rect(screen, color=(v, v, v), rect=(x, y, 1, 1))
            pygame.display.flip()
            flip = False

    pygame.quit()
