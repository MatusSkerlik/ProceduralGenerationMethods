import math
import random
from typing import Dict, Tuple, Union, List

import pygame

WIDTH = 1000
HEIGHT = 1000


def binomial_distribution(n: int, k: int, p: float) -> float:
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def distance_with_correction(points, pos):
    """ :return: List of pairs, where first item is distance from point to pos and second is correction vector """
    distance_vector_pair = []
    for point in points:
        pos_point = vec2(point.x, point.y)
        distance_vector_pair.append((pos.distance_to(pos_point), pos_point - pos))
    return distance_vector_pair


def distribute_directions(a_pair, r_pair, a_radius, r_radius):
    """ :return: Distribute unit vectors """
    result = []
    numerator = sum(map(lambda k: k[0], a_pair + r_pair))

    for d, correction_vec in a_pair:
        if 0.35 * a_radius < d < a_radius:
            normalized = correction_vec.normalize()
            for _ in range(int(numerator / d)):
                result.append(vec2(round(normalized.x), round(normalized.y)))

    for d, correction_vec in r_pair:
        if 1 < d < r_radius:
            normalized = correction_vec.normalize().rotate(180)
            for _ in range(int(numerator / d)):
                result.append(vec2(round(normalized.x), round(normalized.y)))
    return result


class vec2(pygame.math.Vector2):
    pass


class Beam:
    def __init__(self, x: int, y: int, r: int):
        self.x = x
        self.y = y
        self.radius = r

    def distance(self, x: int, y: int):
        return math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)


class Attractor(Beam):
    """ Attracts agent """
    pass


class Repulsor(Beam):
    """ Repulse agent """
    pass


class DrillAgent:
    """ Cave drill agent """

    def __init__(self, x: int, y: int, tokens: int, childs: int,
                 repulsors: Union[int, List[Repulsor]], attractors: Union[int, List[Attractor]], a_radius: int,
                 r_radius: int):
        self.x = x
        self.y = y
        self.tokens = tokens
        self.childs = childs
        self.a_radius = a_radius
        self.r_radius = r_radius

        if isinstance(repulsors, int):
            self.repulsors = [Repulsor(
                random.randint(int(WIDTH / 4), int(3 * WIDTH / 4)),  # TODO
                random.randint(int(HEIGHT / 4), int(3 * HEIGHT / 4)),  # TODO
                r_radius
            ) for _ in range(repulsors)]
        elif isinstance(repulsors, List):
            self.repulsors = repulsors
        else:
            raise ValueError("Unexpected value.")

        if isinstance(attractors, int):
            self.attractors = [Attractor(
                random.randint(int(WIDTH / 4), int(3 * WIDTH / 4)),  # TODO
                random.randint(int(HEIGHT / 4), int(3 * HEIGHT / 4)),  # TODO
                a_radius
            ) for _ in range(attractors)]
        elif isinstance(attractors, List):
            self.attractors = attractors
        else:
            raise ValueError("Unexpected value.")

    def is_active(self):
        """ Agent is active until it has tokens """
        return self.tokens >= 0

    def spawn_child(self, tokens, a_radius, r_radius):
        """ Spawn child with attractor placed at current position """
        if self.childs > 0:
            self.childs -= 1
            return DrillAgent(self.x, self.y, tokens, 0, self.repulsors, [Attractor(self.x, self.y, a_radius)],
                              a_radius, r_radius)
        else:
            raise ValueError("No more childs to spawn.")

    def dig(self, Map: Dict[Tuple[int, int], int], p_directed: float):
        """
        Dig pixel into Map
        :param Map:
        :param p_directed: probability of directed movement toward Attractor
        """
        if self.tokens >= 0:
            while True:
                nx = random.randint(-1, 1)
                ny = random.randint(-1, 1)

                if random.random() < p_directed:

                    pos = vec2(self.x, self.y)
                    directions = distribute_directions(
                        distance_with_correction(self.attractors, pos),
                        distance_with_correction(self.repulsors, pos),
                        self.a_radius,
                        self.r_radius
                    )

                    if len(directions) > 0:
                        direction = random.choice(directions)
                        nx = direction.x
                        ny = direction.y

                if (self.x + nx, self.y + ny) in Map:
                    Map[self.x + nx, self.y + ny] = 0
                    self.tokens -= 1
                    self.x += nx
                    self.y += ny
                    return self.x, self.y
                else:
                    continue
        return


def run():
    def init_diggers(count: int, tokens: int, childs: int, attractors: int, repulsors: int, a_radius: int,
                     r_radius: int):
        return [DrillAgent(int(WIDTH / 2), int(HEIGHT / 2), tokens, childs, repulsors, attractors, a_radius, r_radius)
                for _ in range(count)]

    def init_map():
        return {(x, y): 1 for x in range(WIDTH) for y in range(HEIGHT)}

    def fill_black(surface):
        surface.fill((0, 0, 0))
        pygame.display.flip()

    MODE = "VISUAL"
    # TODO params for evolution
    diggers_count, tokens, childs, child_tokens, attractors, repulsors, a_radius, r_radius, p_directed = 8, 20000, 20, 10000, 1, 5, 150, 100, 0.05

    Diggers = init_diggers(diggers_count, tokens, childs, attractors, repulsors, a_radius, r_radius)
    Map = init_map()

    pygame.init()
    screen = pygame.display.set_mode([WIDTH, HEIGHT])
    fill_black(screen)

    running = True
    lock = False
    frame = 0
    while running:
        frame += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    Map = init_map()
                    Diggers = init_diggers(diggers_count, tokens, childs, attractors, repulsors, a_radius, r_radius)
                    lock = False
                    fill_black(screen)
                elif event.key == pygame.K_q:
                    MODE = "VISUAL"
                    lock = False
                    fill_black(screen)
                elif event.key == pygame.K_e:
                    MODE = "NON_VISUAL"
                    fill_black(screen)

        for digger in Diggers:
            if digger.is_active():
                x, y = digger.dig(Map, p_directed)

                if MODE == "VISUAL":
                    pygame.draw.rect(screen, (255, 255, 255), (x, y, 1, 1))
                    pygame.display.update((x, y, 1, 1))

                if frame % 250 == 0:
                    for attractor in digger.attractors:
                        x, y = attractor.x, attractor.y
                        pygame.draw.circle(screen, (0, 255, 0), (x, y), 2)
                        pygame.display.update((x - 2, y - 2, 4, 4))
                    for repulsor in digger.repulsors:
                        x, y = repulsor.x, repulsor.y
                        pygame.draw.circle(screen, (255, 0, 0), (x, y), 2)
                        pygame.display.update((x - 2, y - 2, 4, 4))

        if childs > 0:
            diggers = []
            tokens_per_child = int(tokens / childs / 100)
            for digger in Diggers:
                if digger.childs - 1 == int((digger.tokens * childs) / tokens):
                    # binomial probability mass function for child spawn
                    p = binomial_distribution(tokens_per_child,
                                              tokens_per_child - int(digger.tokens / 100 % tokens_per_child), 0.5)
                    if random.random() < p:
                        child = digger.spawn_child(child_tokens, a_radius, r_radius)
                        diggers.append(child)
            Diggers.extend(diggers)

        if MODE != "VISUAL" and not lock:
            ready = True
            for digger in Diggers:
                if digger.is_active():
                    ready = False

            if ready:
                screen.fill((0, 0, 0))
                for x, y in Map.keys():
                    if Map[x, y] == 0:
                        pygame.draw.rect(screen, (255, 255, 255), (x, y, 1, 1))
                pygame.display.flip()
                lock = True

    pygame.quit()


if __name__ == '__main__':
    run()
