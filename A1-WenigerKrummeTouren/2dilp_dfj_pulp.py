import operator
import matplotlib.pyplot as plt
import networkx as nx
import pulp

import os
import math
import itertools
import random

from typing import List, Tuple

ANGLE_UPPER_BOUND = 90
ANGLE_COST_FACTOR = 0       # 0.002
SOLVER_MAX_TIME = 60 * 5    # 3 Minuten Berechnungszeit


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


def main(points: List[Tuple[float, float]], fname: str):
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
    problem = pulp.LpProblem("TSP", pulp.LpMinimize)

    print('\033[1A\033[2KErstelle Variablen...')
    # Matrix von Entfernungen
    c = {}
    for i, j in itertools.permutations(range(-1, len(points)), 2):
        if i < j:
            c[i, j] = distance(points[i], points[j])

    # 3d-Matrix von Winkeln
    a = {}
    for i, j, k in itertools.permutations(range(-1, len(points)), 3):
        if i < k and j not in (i, k):
            if -1 in (i, j, k):  # winkel beinhaltet den 'unsichbaren' Start- / Endknoten
                a[i, j, k] = 0
            else:
                a[i, j, k] = angle(points[i], points[j], points[k])

    # Entscheidungsvariablen
    x = pulp.LpVariable.dicts('x', ((i, j) for i, j in
                                    itertools.permutations(range(-1, len(points)), 2)
                                    if i < j), 0, 1, pulp.LpBinary)
    # Winkel-UB
    angle_ub = pulp.LpVariable('angle_ub', 0, ANGLE_UPPER_BOUND, pulp.LpContinuous)

    # Jeder Knoten muss Grad 2 haben
    for n in range(-1, len(points)):
        problem += (pulp.lpSum(x[i, j] for i, j in x if n in (i, j)) == 2)

    # angle <= angle_ub
    for i, j, k in a:  # => i < k
        if (i, j) in x:
            if (j, k) in x:  # => i < j < k
                problem += (a[i, j, k] <= angle_ub + 180 * (1 - x[i, j]) + 180 * (1 - x[j, k]))
            else:  # => i < k < j
                problem += (a[i, j, k] <= angle_ub + 180 * (1 - x[i, j]) + 180 * (1 - x[k, j]))
        elif (j, k) in x:  # => j < i < k
            problem += (a[i, j, k] <= angle_ub + 180 * (1 - x[j, i]) + 180 * (1 - x[j, k]))

    # Ziel: Minimiere die Gesamtlänge + Winkel-Term
    problem += pulp.lpSum([c[i, j] * x[i, j] for i, j in x if -1 not in (i, j)] +
                          [angle_ub * ANGLE_COST_FACTOR * len(points) * avg_arc_cost])

    # Lösung starten
    status = problem.solve(pulp.GLPK_CMD(msg=0))
    print(pulp.LpStatus[status])

    print('\033[1A\033[2K', end='')

    # print solution
    if pulp.LpStatus[status] in ('Optimal', 'Feasible'):
        print(f'Zeit: {problem.solutionTime:.2f}s')
        print(f'Status: {pulp.LpStatus[status]}')
        # print(f'Länge: {objective.Value():.2f}km')
        print(f'Winkel-UB: {angle_ub.value():.2f}°')

        G = nx.Graph()
        for i in range(len(points)):
            G.add_node(i, pos=points[i])

        for i, j in x:
            if -1 not in (i, j) and x[i, j].value() == 1:
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
    except KeyboardInterrupt:
        print()
        print('Abbruch durch Benutzer.')
        exit()

