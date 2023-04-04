import operator
import time
import matplotlib.pyplot as plt
import networkx as nx
import mip

import os
import math
import itertools
import random

from typing import List, Tuple

ANGLE_UPPER_BOUND = 90
ANGLE_COST_FACTOR = 0       # 0.002
SOLVER_MAX_TIME = 60 * 3    # 3 Minuten Berechnungszeit


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
def main(points: List[Tuple[float, float]], fname: str,
         angle_upper_bound: float = ANGLE_UPPER_BOUND, angle_cost_factor: float = ANGLE_COST_FACTOR,
         solver_name: str = 'CBC') \
        -> Tuple[float, int, float, float]:  # [zeit, status, länge, winkel_ub]
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

    print('\033[1A\033[2KVorberechnung der Winkel...')
    # winkel-matrix berechnen
    a = {}
    for i, j, k in itertools.permutations(range(-1, len(points)), 3):
        if i < k and j not in (i, k):
            if -1 in (i, j, k):  # winkel beinhaltet den 'unsichbaren' Start- / Endknoten
                a[i, j, k] = 0
            else:
                a[i, j, k] = angle(points[i], points[j], points[k])

    print('\033[1A\033[2KErstelle Solver...')
    model = mip.Model(sense=mip.MINIMIZE, solver_name=solver_name)

    print('\033[1A\033[2KErstelle Variablen...')

    # Entscheidungsvariablen
    x = {}
    for i, j in itertools.permutations(range(-1, len(points)), 2):
        if i < j:
            x[i, j] = model.add_var(name=f'x_{i}_{j}',
                                    var_type=mip.BINARY)
            if -1 not in (i, j):
                x[i, j].obj = distance(points[i], points[j])

    # Winkel-UB
    angle_ub = model.add_var(name='angle_ub', lb=0, ub=angle_upper_bound,
                             obj=angle_cost_factor * len(points) * avg_arc_cost,
                             var_type=mip.INTEGER)

    # Jeder Knoten muss Grad 2 haben
    for n in range(-1, len(points)):
        model += (sum(x[i, j] for i, j in x if n in (i, j)) == 2)

    # angle <= angle_ub
    for i, j, k in a:  # => i < k
        if (i, j) in x:
            if (j, k) in x:  # => i < j < k
                model += (a[i, j, k] <= angle_ub + 180 * (1 - x[i, j]) + 180 * (1 - x[j, k]))
            else:  # => i < k < j
                model += (a[i, j, k] <= angle_ub + 180 * (1 - x[i, j]) + 180 * (1 - x[k, j]))
        elif (j, k) in x:  # => j < i < k
            model += (a[i, j, k] <= angle_ub + 180 * (1 - x[j, i]) + 180 * (1 - x[j, k]))

    # Lösung starten
    status = model.optimize(max_seconds=SOLVER_MAX_TIME)
    print(status)

    print('\033[1A\033[2K', end='')

    # print solution
    if status in (mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE):
        G = nx.Graph()
        for i in range(len(points)):
            G.add_node(i, pos=points[i])

        length = 0
        for i, j in x:
            if -1 not in (i, j) and x[i, j].x >= 0.99:
                G.add_edge(i, j)
                length += x[i, j].obj

        for node in G.nodes:
            neighbors = list(G.neighbors(node))
            if len(neighbors) == 2:
                a, b = neighbors
                if angle(points[a], points[node], points[b]) > angle_upper_bound:
                    G.nodes[node]['color'] = 'r'

        plt.clf()
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
        print()
        return (time.time() - start_time, status, length, angle_ub.x,
                model.objective_value, model.objective_bound)
    else:
        return time.time() - start_time, status, None, None, None, None


if __name__ == '__main__':
    try:
        while True:
            try:
                fname = f'wenigerkrumm{input("Bitte Zahl des Beispiels eingeben: ")}.txt'
                points = []
                with open(os.path.join(os.path.dirname(__file__), f'beispieldaten/{fname}')) as f:
                    points = [tuple(map(float, line.split())) for line in f.readlines()]
                t, s, l, a_ub, obj, lb = main(tuple(points), fname)
                print(f'Zeit: {t:.2f}s')
                print(f'Status: {mip.OptimizationStatus(s)}')
                if s in (mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE):
                    print(f'Objektivwert: {obj:.2f}')
                    print(f'Untere Schranke: {lb:.2f}')
                    print(f'Länge: {l:.2f}km')
                    print(f'Winkel-UB: {a_ub:.2f}°')
                else:
                    print('Keine Lösung gefunden.')
                plt.show()
            except Exception as e:
                raise e
    except KeyboardInterrupt:
        print()
        print('Abbruch durch Benutzer.')
        exit()
