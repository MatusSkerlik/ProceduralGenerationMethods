import random
import time
from abc import ABC, abstractmethod
from copy import copy
from itertools import product
from operator import attrgetter
from typing import List, Tuple, Any, Callable

import pygame

# global params
width, height = 150, 150

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
        return IntGene(random.randint(minimal, maximal), minimal, maximal)

    def __init__(self, value: int, minimal: int, maximal: int):
        self._min = minimal
        self._max = maximal
        self._value = value

    def mutate(self):
        self._value = random.randint(self._min, self._max)


class FloatBoundedIntervalGene(Gene):
    """ Bounded interval with mutation as random walk by diff """

    @classmethod
    def initialize(cls, diff: float):
        return FloatBoundedIntervalGene(0.5, diff)

    def __init__(self, initial: float, diff: float):
        self._diff = diff
        self._value = initial

    def mutate(self):
        if self._diff < self._value < 1 - self._diff:
            self._value += (self._diff if random.random() > 0.5 else -self._diff)
        elif self._value < self._diff:
            self._value += self._diff
        else:
            self._value -= self._diff


class ConfigGene(Gene):
    """ Automata sequence gene with mutation  """

    @classmethod
    def initialize(cls, length: int):
        sequence = [(random.randint(1, 8), random.randint(1, 8)) for _ in range(length)]
        return ConfigGene(sequence)

    def __init__(self, value: List[Tuple[int, int]]):
        self._value = value

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self._value[item]
        elif isinstance(item, int):
            return self._value[item]
        else:
            raise NotImplemented

    def mutate(self):
        m_index = random.randint(0, len(self._value) - 1)
        p_index = random.randint(0, 1)
        ratio = self._value[m_index][p_index]
        if 1 < ratio < 8:
            ratio += 1 if random.random() > 0.5 else -1
        elif ratio == 1:
            ratio += 1
        else:
            ratio -= 1
        self._value[m_index] = (self._value[m_index][0], ratio) if p_index == 1 else (ratio, self._value[m_index][1])


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
    before_crossover = lambda self, _: None
    before_fitness = lambda self, _: None

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

    @staticmethod
    def crossover(parent_0: Chromosome, parent_1: Chromosome) -> (Chromosome, Chromosome):
        """ Generic crossover strategy """
        index = random.randrange(1, len(parent_0))
        genes_1 = parent_0[:index] + parent_1[index:]
        genes_2 = parent_1[:index] + parent_0[index:]
        return Chromosome(genes_1), Chromosome(genes_2)

    def initialize_chromosome(self):
        """ Will create chromosome from default genes """
        index = random.randint(0, len(self._initial_genes) - 1)
        genes = copy(self._initial_genes[index])
        for gene in genes:
            for _ in range(random.randint(0, 5)):
                gene.mutate()
        return Chromosome(genes)

    def select_parents(self, count: int):
        """ Best parents selection """
        self._chromosomes.sort(key=attrgetter('fitness'), reverse=True)
        return self._chromosomes[:count]

    def run(self, selection_ratio: float):

        selection_count = int(self._population_size * selection_ratio)
        new_count = self._population_size - selection_count

        for gen in range(self._generations):
            print("Generation %d" % gen)
            for chromosome in self._chromosomes:
                # TODO problem specific
                self.before_fitness(chromosome)
                chromosome.fitness = self._fitness_function(*map(attrgetter('value'), chromosome[:]))

            self.before_crossover(self._chromosomes)

            # TODO crossover strategy
            parents = self.select_parents(selection_count)
            children = []
            for chromosome_0, chromosome_1 in product(parents, parents):
                if chromosome_0 is not chromosome_1:
                    for updated_chromosome in self.crossover(chromosome_0, chromosome_1):
                        for gene in updated_chromosome.genes:
                            # TODO mutation strategy
                            if random.random() < 0.01:
                                gene.mutate()
                        children.append(updated_chromosome)
            random.shuffle(children)
            self._chromosomes = children[:selection_count] + [self.initialize_chromosome() for _ in range(new_count)]


if __name__ == '__main__':
    # local params
    population_size = 20
    generations = 500

    # initialization of environment
    pygame.init()
    pygame.font.init()

    font = pygame.font.SysFont(None, 12)
    screen = pygame.display.set_mode([width, height + 32])


    def fitness(steps: int, birth_probability: float, config: List[Tuple[int, int]]):
        """ Calculate fitness for genetic algorithm """
        caves = []
        walls = []
        visited_caves = dict()
        visited_walls = dict()

        # from genotype to phenotype
        Solution = solve(config[steps:], birth_probability)

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

        # num_of_caves = len(caves)
        # num_of_walls = len(walls)
        # num_of_cave_pixels = sum(len(cave.keys()) for cave in caves)
        # num_of_wall_pixels = sum(len(wall.keys()) for wall in walls)
        # cave_size_mean = num_of_cave_pixels / num_of_caves if num_of_caves > 0 else 0
        # wall_size_mean = num_of_wall_pixels / num_of_walls if num_of_walls > 0 else 0
        # TODO write proper fitness function
        if len(caves) > 0:
            cumulative_width = 0
            cumulative_height = 0
            for cave in caves:
                x0, y0, x1, y1 = bounding_box(cave.keys())
                w, h = x1 - x0, y1 - y0
                cumulative_width += w
                cumulative_height += h
            mean_width = cumulative_width / len(caves)
            mean_height = cumulative_height / len(caves)

            return mean_width * mean_height
        return 0


    def handle_events(_):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    exit()


    def render_solution(chromosomes):
        phenotype = max(chromosomes, key=attrgetter('fitness'))
        steps, birth_probability, config = phenotype[:]
        genotype = solve(config[steps.value:], birth_probability.value)

        # render best solution
        screen.fill(color=(0, 0, 0))
        for x in range(width):
            for y in range(height):
                if genotype[encode_coords(x, y)] == 1:
                    pygame.draw.rect(screen, color=(255, 255, 255), rect=(x, y, 1, 1))

        text0 = font.render("fitness: %d" % phenotype.fitness, False, (255, 255, 255))
        text1 = font.render("config: %s" % config[:], False, (255, 255, 255))
        screen.blit(text0, (8, height + 8))
        screen.blit(text1, (8, height + 8 + 12))
        pygame.display.flip()
        pygame.image.save(screen, "img/%d.png" % time.time())


    initial_genes = [
        [
            IntGene.initialize(1, 5),
            FloatBoundedIntervalGene.initialize(0.01),
            ConfigGene.initialize(5)
        ] for _ in range(population_size)
    ]
    GA = GeneticAlgorithm(
        initial_genes=initial_genes,
        population_size=population_size,
        generations=generations,
        fitness_function=fitness
    )
    GA.before_fitness = handle_events
    GA.before_crossover = render_solution
    GA.run(0.6)

    pygame.quit()
