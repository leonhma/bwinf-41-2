import itertools
import math
import operator
import os
import random
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import mip

ANGLE_UPPER_BOUND = 90
ANGLE_COST_FACTOR = 0
SOLVER_MAX_TIME = 60 * 3    # 3 Minuten Berechnungszeit


class ExitException(BaseException):
    pass


def distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return (
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2) ** 0.5


def dot(a, b):
    return sum(map(operator.mul, a, b))


def sub(a, b):
    return tuple(map(operator.sub, a, b))


def norm(a):
    return math.sqrt(dot(a, a))


def angle(p1: Tuple[int, int], p2: Tuple[int, int], p3: Tuple[int, int]) -> float:
    ba = sub(p1, p2)
    bc = sub(p3, p2)
    cos = max(min(dot(ba, bc) / (norm(ba) * norm(bc)), 1), -1)  # clamp to [-1, 1]
    return 180 - math.degrees(math.acos(cos))


# pylama:ignore=C901
def main(points: List[Tuple[float, float]], fname: str, solver_name: str = 'GPLK'):
    start_time = time.time()
    print()
    print('Berechne ungefähren Mittelwert der Kantenlängen...')
    avg_arc_cost = 0
    avg_arc_n = 0
    for _ in range(1000):
        i, j = random.sample(range(len(points)), 2)
        avg_arc_cost += distance(points[i], points[j])
        avg_arc_n += 1
    avg_arc_cost /= avg_arc_n

    print('\033[1A\033[2KErstelle Solver...')
    model = mip.Model(solver_name=solver_name, sense=mip.MINIMIZE)

    print('\033[1A\033[2KVorberechnung der Winkel...')
    # winkel-matrix berechnen
    a = {}
    for i, j, k in itertools.permutations(range(-1, len(points)), 3):
        if i < k and j not in (i, k):
            if -1 in (i, j, k):  # winkel beinhaltet den 'unsichbaren' Start- / Endknoten
                a[i, j, k] = 0
            else:
                a[i, j, k] = angle(points[i], points[j], points[k])

    print('\033[1A\033[2KErstelle Variablen...')
    # create 2d binary matrix [node index, next node index]
    x = {}
    for i, j in itertools.permutations(range(-1, len(points)), 2):
        if i != j:
            x[i, j] = model.add_var(name=f'x_{i}_{j}', var_type=mip.BINARY)
    # create miller-tucker-zemlin variables
    t = {i: model.add_var(name=f't_{i}', ub=len(points) - 1,
                          var_type=mip.INTEGER) for i in range(len(points))}
    # create angle upper bound variable
    angle_ub = model.add_var(name='angle_ub', ub=ANGLE_UPPER_BOUND, var_type=mip.CONTINUOUS,
                             obj=ANGLE_COST_FACTOR * avg_arc_cost * len(points))

    print(f'\033[1A\033[2KAnzahl der Variablen: \n')

    print('Erstelle Bedingungen...')
    # mtz subtour elimination
    for i, j in x:
        if -1 not in (i, j):
            # use <= and -1 on right side, linearize conditional constraint
            model += (t[i] <= t[j] - 1 + len(points) * (1 - x[i, j]))
    # every node has a next node
    for i in range(-1, len(points)):
        model += (mip.xsum(x[i, j] for i2, j in x if i2 == i) == 1)
    # every node has a previous node
    for j in range(-1, len(points)):
        model += (mip.xsum(x[i, j] for i, j2 in x if j2 == j) == 1)
    # angle <= angle_ub
    for i, j, k in a:
        if i < k and (i, j) in x and (j, k) in x:
            model += (a[i, j, k] <= angle_ub + 180 * (1 - x[i, j]) + 180 * (1 - x[j, k]))
            model += (a[i, j, k] <= angle_ub + 180 * (1 - x[k, j]) + 180 * (1 - x[j, i]))

    print(f'\033[1A\033[2K\033[1AAnzahl der Bedingungen: \n')

    print('Erstelle Ziel...')
    for i, j in x:
        if -1 not in (i, j) and i < j:
            dist = distance(points[i], points[j])
            x[i, j].obj = dist
            x[j, i].obj = dist

    print('\033[1A\033[2KFinde Lösung...')
    model.verbose = 0
    status = model.optimize(max_seconds=SOLVER_MAX_TIME)

    print('\033[1A\033[2K', end='')

    # print solution
    if status in (mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE):
        print(f'Zeit: {time.time() - start_time:.2f}s')
        print(f'Status: {status}')
        print(f'Länge: {model.objective_value:.2f}km (bound is {model.objective_bound:.2f}km)')
        print(f'Winkel: {angle_ub.x:.2f}°')

        G = nx.Graph()
        for i in range(len(points)):
            G.add_node(i, pos=points[i])

        for i, j in x:
            if -1 not in (i, j) and x[i, j].x >= 0.99:
                G.add_edge(i, j)

        for node in G.nodes:
            neighbors = list(G.neighbors(node))
            if len(neighbors) == 2:
                a, b = neighbors
                if angle(points[a], points[node], points[b]) > ANGLE_UPPER_BOUND:
                    G.nodes[node]['color'] = 'r'

        ax = plt.gca()
        ax.set_aspect('equal')
        plt.get_current_fig_manager().set_window_title(f'Lösung für {fname}')

        pos = nx.get_node_attributes(G, 'pos')
        colorsd = nx.get_node_attributes(G, 'color')
        colors = [colorsd.get(node, 'w') for node in G.nodes]

        nx.draw(G,
                pos,
                node_size=25,
                font_size=8,
                node_color=colors,
                edgecolors='k')

        plt.show()
    else:
        print('Keine mögliche Lösung gefunden.')
    print()


if __name__ == '__main__':
    try:
        while True:
            try:
                fname = f'wenigerkrumm{input("Bitte Zahl des Beispiels eingeben: ")}.txt'
                points = []
                with open(os.path.join(os.path.dirname(__file__), f'beispieldaten/{fname}')) as f:
                    points = [tuple(map(float, line.split())) for line in f.readlines()]
                main(tuple(points), fname)
            except Exception as e:
                raise e
    except ExitException as e:
        print(e)
        exit()
    except KeyboardInterrupt:
        print()
        print('Abbruch durch Benutzer.')
        exit()
