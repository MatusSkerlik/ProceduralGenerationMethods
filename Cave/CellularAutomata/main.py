import math
import random
import time
from abc import ABC, abstractmethod
from copy import copy
from functools import partial
from operator import attrgetter
from typing import List, Tuple, Any, Callable

import pygame

# global params
width, height = 50, 50

Coords = Tuple[int, int]
Rect = Tuple[int, int, int, int]


def encode_coords(x: int, y: int):
    """ Encode coordinates for array access """
    return x + (width * y)


def moore_nbs(x: int, y: int, Map: List[int]):
    """ Return values of neighbours """
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
    """ Count alive neighbours """
    return len([1 for _ in nbs if _ == 1])


def step(survive_ratio: int, reborn_ratio: int, Map: List[int]):
    """ Step function in cellular automata """

    new_map = [0 for _ in range(width * height)]
    for x in range(width):
        for y in range(height):
            alive = count_alive(moore_nbs(x, y, Map))
            if Map[encode_coords(x, y)] == 1:  # it is alive
                if alive >= survive_ratio:  # it has minimum amount of alive nbs to survive
                    new_map[encode_coords(x, y)] = 1
            else:  # it is dead
                if alive > reborn_ratio:  # it has minimum amout of alive to give birth
                    new_map[encode_coords(x, y)] = 1

    return new_map


def solve(config: List[Tuple[int, int]], birth_probability: float):
    """ Solve cellular automata """

    Solution = [1 if random.random() > birth_probability else 0 for _ in range(width * height)]
    for i in range(len(config)):
        survive_ratio, reborn_ratio = config[i]
        Solution = step(survive_ratio, reborn_ratio, Solution)
    return Solution


def flood_fill(Solution: List[int], x: int, y: int, lookup: int) -> List[Coords]:
    """ Flood fill with Von Neumann neighbourhood """
    queue = [(x, y)]
    visited = {}
    founded = {}

    while len(queue) > 0:
        coords = queue.pop()
        if coords not in visited:
            value = Solution[encode_coords(*coords)]
            visited[coords] = True
            if value == lookup:
                founded[coords] = lookup
                x0, y0 = coords
                if x0 > 0:
                    queue.append((x0 - 1, y0))
                if x0 < (width - 1):
                    queue.append((x0 + 1, y0))
                if y0 > 0:
                    queue.append((x0, y0 - 1))
                if y0 < (height - 1):
                    queue.append((x0, y0 + 1))

    return list(founded.keys())


def bounding_box(coords: List[Coords]) -> Rect:
    """ Get bounding box of coordinates """
    min_x = width
    max_x = 0
    min_y = height
    max_y = 0

    for x, y in coords:
        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y

    return min_x, min_y, max_x, max_y


class Gene(ABC):
    _value: Any

    def __repr__(self):
        return str(self._value)

    def __str__(self):
        return str(self._value)

    @classmethod
    @abstractmethod
    def initialize(cls, *args):
        raise NotImplemented

    @abstractmethod
    def mutate(self):
        """ Gene specific mutate strategy """
        raise NotImplemented

    @property
    def value(self):
        return self._value


class IntGene(Gene):
    """ Integer gene with mutation as random int between """

    @classmethod
    def initialize(cls, minimal: int, maximal: int):
        return IntGene(minimal, minimal, maximal)

    def __init__(self, value: int, minimal: int, maximal: int):
        self._min = minimal
        self._max = maximal
        self._value = value

    def mutate(self):
        if self._value < self._max:
            self._value += 1


class FloatBoundedIntervalGene(Gene):
    """ Bounded interval with mutation as random walk by diff """

    @classmethod
    def initialize(cls, diff: float):
        return FloatBoundedIntervalGene(0.5, diff)

    def __init__(self, initial: float, diff: float):
        self._diff = diff
        self._value = initial

    def mutate(self):
        if random.random() < 0.01:
            if self._diff < self._value < 1 - self._diff:
                self._value += (self._diff if random.random() > 0.5 else -self._diff)
            elif self._value < self._diff:
                self._value += self._diff
            else:
                self._value -= self._diff


class ConfigGene(Gene):
    """ Automata sequence gene with mutation  """

    @classmethod
    def initialize(cls, length: int, max_length: int):
        sequence = [(random.randint(1, 8), random.randint(1, 8)) for _ in range(length)]
        return ConfigGene(sequence, max_length)

    def __init__(self, value: List[Tuple[int, int]], max_length: int):
        self._value = value
        self._max_length = max_length

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self._value[item]
        elif isinstance(item, int):
            return self._value[item]
        else:
            raise NotImplemented

    def __len__(self):
        return len(self._value)

    def mutate(self):
        r = random.random()
        if r < 0.005 and len(self._value) < self._max_length:  # append new configuration
            self._value.append((random.randint(1, 8), random.randint(1, 8)))
        elif r < 0.01 and len(self._value) > 1:
            self._value.pop()
        elif len(self._value) > 1 and random.random() < 0.1:
            m_index = len(self._value) - 1
            p_index = random.randint(0, 1)
            ratio = self._value[m_index][p_index]
            if 1 < ratio < 8:
                ratio += 1 if random.random() > 0.5 else -1
            elif ratio == 1:
                ratio += 1
            else:
                ratio -= 1
            self._value[m_index] = (self._value[m_index][0], ratio) if p_index == 1 else (
                ratio, self._value[m_index][1])


class Chromosome:

    def __init__(self, genes: List[Gene]):
        self._genes = genes
        self._fitness = None

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self._genes[item]
        elif isinstance(item, int):
            return self._genes[item]
        else:
            raise NotImplemented

    def __len__(self):
        return len(self._genes)

    @property
    def fitness(self):
        if self._fitness is None:
            raise ValueError
        return self._fitness

    @fitness.setter
    def fitness(self, fitness: float):
        self._fitness = fitness

    @property
    def genes(self):
        return self._genes


class GeneticAlgorithm:
    """ Cellular automata GA """
    before_selection = staticmethod(lambda chromosomes: None)
    before_fitness = staticmethod(lambda chromosome: None)
    after_fitness = staticmethod(lambda chromosome: None)
    generation_start = staticmethod(lambda gen: None)
    generation_end = staticmethod(lambda gen: None)

    def __init__(self,
                 initial_genes: List[List[Gene]],
                 population_size: int,
                 fitness_function: Callable[[Any], float],
                 generations: int = 100):

        assert len(initial_genes) == population_size

        self._initial_genes = initial_genes
        self._population_size = population_size
        self._fitness_function = fitness_function
        self._generations = generations
        # initialize genetic algorithm with genes that will be mutated
        self._chromosomes = [Chromosome(genes) for genes in initial_genes]

    def initialize_chromosome(self):
        """ Will create chromosome from default genes """
        index = random.randint(0, len(self._initial_genes) - 1)
        genes = copy(self._initial_genes[index])
        for gene in genes:
            for _ in range(random.randint(0, 5)):
                gene.mutate()
        return Chromosome(genes)

    @staticmethod
    def select_parents(chromosomes: sorted, count: int):
        """ Best parents selection """
        filtered_chromosomes = list(filter(lambda ch: ch.fitness > 0, chromosomes))
        total_fitness = sum(chromosome.fitness for chromosome in filtered_chromosomes)
        if total_fitness > 0:
            # must be sorted first
            parents = []
            for _ in range(count):
                acc_p = 0
                threshold = random.random()
                for chromosome in filtered_chromosomes:
                    acc_p += chromosome.fitness / total_fitness
                    if threshold < acc_p:
                        parents.append(chromosome)
                        break
            assert len(parents) == count
            return parents
        else:
            return chromosomes[:count]

    def run(self, selection_ratio: float):

        selection_count = int(self._population_size * selection_ratio)
        new_count = self._population_size - selection_count

        for gen in range(self._generations):
            self.generation_start(gen)

            for chromosome in self._chromosomes:
                self.before_fitness(copy(chromosome))
                chromosome.fitness = self._fitness_function(*map(attrgetter('value'), chromosome[:]))
                self.after_fitness(copy(chromosome))

            # reverse = False means ascending ( min -> max )
            # reverse = True means descending ( max -> min )
            chromosomes = sorted(self._chromosomes, key=attrgetter("fitness"), reverse=True)

            self.before_selection(chromosomes)
            parents = self.select_parents(chromosomes, selection_count)
            print("Unique configurations %d" % len(set(parents)))
            for chromosome in parents:
                for gene in chromosome.genes:
                    gene.mutate()

            self._chromosomes = parents + [self.initialize_chromosome() for _ in range(new_count)]
            self.generation_end(gen)


if __name__ == '__main__':
    # local params
    r_config = 21  # how many configurations to render
    r_config_row = 7
    population_size = 250
    generations = 500
    scale = 5
    font_size = 16

    assert r_config <= population_size
    assert r_config_row <= r_config
    assert (r_config / r_config_row) % 1 == 0

    # initialization of environment
    pygame.init()
    pygame.font.init()

    font = pygame.font.SysFont(None, font_size)
    screen = pygame.display.set_mode(
        [width * scale * r_config_row, (height * scale + 64) * int(r_config / r_config_row)])


    def fitness(birth_probability: float, config: List[Tuple[int, int]]):
        """ Calculate fitness for genetic algorithm """
        caves = []
        walls = []
        visited_caves = dict()
        visited_walls = dict()

        # from genotype to phenotype
        Solution = solve(config[:], birth_probability)

        for x in range(width):
            for y in range(height):
                if (x, y) not in visited_caves and (x, y) not in visited_walls:
                    cave = flood_fill(Solution, x, y, 1)
                    wall = flood_fill(Solution, x, y, 0)
                    if len(cave) > 0:
                        coords = dict.fromkeys(cave, True)
                        visited_caves.update(coords)
                        caves.append(coords)
                    elif len(wall) > 0:
                        coords = dict.fromkeys(wall, True)
                        visited_walls.update(coords)
                        walls.append(coords)

        cave_pixels = sum(len(cave.keys()) for cave in caves)
        wall_pixels = sum(len(wall.keys()) for wall in walls)
        total_pixels = width * height

        # Entropy is maximal when cave_pixels / wall_pixels = 1
        E = 0
        if cave_pixels > 0:
            p = cave_pixels / total_pixels
            E += cave_pixels * p * -math.log(p, 2)
        if wall_pixels > 0:
            p = wall_pixels / total_pixels
            E += wall_pixels * p * -math.log(p, 2)

        if E < total_pixels * 0.5 * 0.6:
            return E
        else:
            return -(E - total_pixels * 0.5 * 0.6)


    def handle_events(_=None):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    exit()


    def render_solution(chromosomes, count: int, in_row: int):
        screen.fill(color=(0, 0, 0))
        phenotypes = chromosomes[:count]

        for i, phenotype in enumerate(phenotypes):
            x0 = (i % in_row) * width * scale
            y0 = int(i / in_row) * height * scale + int(i / in_row) * 64

            birth_probability, config = phenotype[:]
            genotype = solve(config[:], birth_probability.value)

            # render best solution
            for x in range(width):
                for y in range(height):
                    if genotype[encode_coords(x, y)] == 1:
                        pygame.draw.rect(screen, color=(255, 255, 255),
                                         rect=(x0 + x * scale, y0 + y * scale, scale, scale))

            text0 = font.render(
                "fitness: %d, steps: %s, birth_prob: %s" % (phenotype.fitness, len(config), birth_probability),
                False, (255, 255, 255))
            text1 = font.render("config: %s" % config, False, (255, 255, 255))
            screen.blit(text0, (x0 + 8, y0 + height * scale + 8))
            screen.blit(text1, (x0 + 8, y0 + height * scale + 8 + font_size))
        pygame.display.flip()
        pygame.image.save(screen, "img/%d.png" % time.time())


    initial_genes = [
        [
            FloatBoundedIntervalGene.initialize(0.01),
            ConfigGene.initialize(1, 4)
        ] for _ in range(population_size)
    ]
    GA = GeneticAlgorithm(
        initial_genes=initial_genes,
        population_size=population_size,
        generations=generations,
        fitness_function=fitness
    )
    GA.generation_start = lambda gen: print("Generation %d" % gen)
    GA.before_fitness = handle_events
    GA.before_selection = partial(render_solution, count=r_config, in_row=r_config_row)
    GA.run(0.7)

    pygame.quit()
