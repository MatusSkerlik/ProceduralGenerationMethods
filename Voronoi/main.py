import math
import random
import time

import pygame

width = 200
height = 200


def r(a, b):
    return random.randint(a, b)


def get_paint_color(c):
    return {(r(0, width), r(0, height)): (r(0, 255), r(0, 255), r(0, 255)) for _ in range(c)}


def euclidean_distance(p0, p1):
    return math.sqrt(math.pow(p0[0] - p1[0], 2) + math.pow(p0[1] - p1[1], 2))


def manhattan_distance(p0, p1):
    return abs(p0[0] - p1[0]) + abs(p0[1] - p1[1])


def minkowski_distance(p0, p1, p):
    return (abs(p0[0] - p1[0]) ** p + abs(p0[1] - p1[1]) ** p) ** 1 / p


def run():
    global width, height

    pygame.init()

    screen = pygame.display.set_mode([width, height])
    running = True
    retry = True
    count = 5
    point_color = get_paint_color(count)
    metrics = euclidean_distance

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    retry = True
                    point_color = get_paint_color(count)
                elif event.key == pygame.K_s:
                    pygame.image.save(screen, "img/%d.png" % time.time())
                elif event.key == pygame.K_q:
                    if count > 1:
                        count -= 1
                        retry = True
                        point_color = get_paint_color(count)
                elif event.key == pygame.K_e:
                    count += 1
                    retry = True
                    point_color = get_paint_color(count)
                elif event.key == pygame.K_c:
                    if metrics is euclidean_distance:
                        metrics = manhattan_distance
                    else:
                        metrics = euclidean_distance
                    retry = True

        if retry:
            screen.fill((0, 0, 0))

            if retry:
                for x in range(width):
                    for y in range(height):
                        distances = {}
                        for i, point in enumerate(point_color.keys()):
                            d = metrics((x, y), point)
                            distances[point] = d

                        point = min(distances, key=distances.get)
                        color = point_color[point]
                        pygame.draw.rect(screen, color, (x, y, x + 1, y + 1))

                for point in point_color.keys():
                    pygame.draw.circle(screen, (0, 0, 0), point, 3)

                pygame.display.flip()
                retry = False

    pygame.quit()


if __name__ == '__main__':
    run()
