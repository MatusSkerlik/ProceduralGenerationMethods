import random
import time

import matplotlib
import matplotlib.backends.backend_agg as agg
import matplotlib.pyplot as plt
import networkx as nx
import pygame

matplotlib.use("Agg")

width = 200
height = 200


class TreeVisitor:

    def __init__(self, on_enter, on_leave) -> None:
        self._on_enter = on_enter
        self._on_leave = on_leave

    def on_enter(self, *args):
        self._on_enter(*args)

    def on_leave(self, *args):
        self._on_leave(*args)


def bsp(x_mn, y_mn, x_mx, y_mx, vertical=False, r=0.2):
    if vertical:  # we are generating x
        w = x_mx - x_mn
        x0 = random.randint(int(x_mn + w * r), int(x_mx - w * r))
        return (x_mn, y_mn, x0, y_mx), (x0, y_mn, x_mx, y_mx)
    else:
        h = y_mx - y_mn
        y0 = random.randint(int(y_mn + h * r), int(y_mx - h * r))
        return (x_mn, y_mn, x_mx, y0), (x_mn, y0, x_mx, y_mx)


def tree(x0, y0, x1, y1, iterations, is_prob, visitor, r=0.2):
    if iterations > 0 and (x1 - x0) * r > 1 and (y1 - y0) * r > 1:
        root = (x0, y0, x1, y1)
        a, b = bsp(x0, y0, x1, y1, vertical=(iterations % 2) == 1, r=r)
        if is_prob:
            if random.random() < .75:
                visitor.on_enter(root, a)
                yield from tree(*a, iterations - 1, is_prob, visitor)
                visitor.on_leave(root, a)
            else:
                visitor.on_enter(root, a)
                yield a
                visitor.on_leave(root, a)
            if random.random() < .75:
                visitor.on_enter(root, b)
                yield from tree(*b, iterations - 1, is_prob, visitor)
                visitor.on_leave(root, b)
            else:
                visitor.on_enter(root, b)
                yield b
                visitor.on_leave(root, b)
        else:
            visitor.on_enter(root, a)
            yield from tree(*a, iterations - 1, is_prob, visitor)
            visitor.on_leave(root, a)

            visitor.on_enter(root, b)
            yield from tree(*b, iterations - 1, is_prob, visitor)
            visitor.on_leave(root, b)
    else:
        yield x0, y0, x1, y1


def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                     vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                     pos=pos, parent=root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


def run():
    global width, height

    fig = plt.gcf()
    fig.set_size_inches(int(width / 100), int(height / 100))

    pygame.init()
    pygame.font.init()

    screen = pygame.display.set_mode([width * 2, height])
    font_w = 16
    font = pygame.font.SysFont('Courier New', font_w, bold=True)
    alphabet = "ABCDEFGHIJKLMNOPRSTUVWXYZ"

    stochastic = False
    running = True
    retry = True
    depth = 1

    j = 0
    nodes = {(0, 0, width, height): "A0"}
    G = nx.Graph()

    print("controls")
    print("     space:  generate new")
    print("     q:      decrease tree depth")
    print("     e:      increase tree depth")
    print("     s:      save image")
    print("     tab:    toggle stochastic")

    def on_enter(a, b):
        nonlocal G, j

        if a not in nodes:
            nodes[a] = "%s%d" % (alphabet[j % 24], int(j / 24))
            j += 1
        if b not in nodes:
            nodes[b] = "%s%d" % (alphabet[j % 24], int(j / 24))
            j += 1
        G.add_edge(nodes[a], nodes[b])

    def on_leave(a, b):
        nonlocal G
        pass

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    retry = True
                elif event.key == pygame.K_s:
                    pygame.image.save(screen, "img/%d.png" % time.time())
                elif event.key == pygame.K_q:
                    if depth > 1:
                        depth -= 1
                        retry = True
                elif event.key == pygame.K_e:
                    depth += 1
                    retry = True
                elif event.key == pygame.K_TAB:
                    stochastic = not stochastic
                    retry = True

        if retry:
            screen.fill((0, 0, 0))
            fig.clf()

            G.clear()
            nodes = {}
            j = 0

            for rect in tree(0, 0, width, height, depth, stochastic, TreeVisitor(on_enter, on_leave)):
                a, b, c, d = rect
                pygame.draw.rect(screen, (255, 255, 255), rect)
                pygame.draw.rect(screen, (0, 0, 0), (a + 1, b + 1, c - 1, d - 1))

                # render cell id
                w = c - a
                h = d - b

                if w > font_w and h > font_w:
                    letter = "%s%d" % (alphabet[(j - 1) % 24], int((j - 1) / 24))
                    cell_id = font.render(letter, True, (255, 255, 255))
                    screen.blit(cell_id, (a + w / 2 - font_w / 2, b + h / 2 - font_w / 2))

            # render graph
            pos = hierarchy_pos(G, "A0")
            nx.draw(G, pos=pos, with_labels=True, font_family="Courier New", font_weight="bold", node_color="white")
            canvas = agg.FigureCanvasAgg(fig)
            canvas.draw()
            renderer = canvas.get_renderer()
            raw_data = renderer.tostring_rgb()
            graph = pygame.image.fromstring(raw_data, canvas.get_width_height(), "RGB")
            screen.blit(graph, (width, 0))

            pygame.display.flip()
            retry = False

    pygame.quit()


if __name__ == '__main__':
    run()
