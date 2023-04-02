import itertools
import math
import operator
import os
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from ortools.linear_solver import pywraplp

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
def main(points: List[Tuple[float, float]], fname: str, solver_name: str = 'CP_SAT'):

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
    solver = pywraplp.Solver.CreateSolver(solver_name)
    if not solver:
        raise ExitException('Fehler beim Erstellen des Solvers. '
                            '(Ist die richtige Version von ortools installiert?)')
    solver.SetTimeLimit(SOLVER_MAX_TIME * 1000)
    solver.SetNumThreads(max(1, os.cpu_count() - 2))

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
            x[i, j] = solver.BoolVar(f'x_{i}_{j}')
    # create miller-tucker-zemlin variables
    t = {i: solver.IntVar(0, len(points) - 1, f't_{i}') for i in range(len(points))}
    # create angle upper bound variable
    angle_ub = solver.IntVar(0, ANGLE_UPPER_BOUND, 'angle_ub')

    print(f'\033[1A\033[2KAnzahl der Variablen: {solver.NumVariables()}\n')

    print('Erstelle Bedingungen...')
    # mtz subtour elimination
    for i, j in x:
        if -1 not in (i, j):
            # use <= and -1 on right side, linearize conditional constraint
            solver.Add(t[i] <= t[j] - 1 + len(points) * (1 - x[i, j]))
    # every node has a next node
    for i in range(-1, len(points)):
        solver.Add(sum(x[i, j] for i2, j in x if i2 == i) == 1)
    # every node has a previous node
    for j in range(-1, len(points)):
        solver.Add(sum(x[i, j] for i, j2 in x if j2 == j) == 1)
    # angle <= angle_ub
    for i, j, k in a:
        if i < k and (i, j) in x and (j, k) in x:
            solver.Add(a[i, j, k] <= angle_ub + 180 * (1 - x[i, j]) + 180 * (1 - x[j, k]))
            solver.Add(a[i, j, k] <= angle_ub + 180 * (1 - x[k, j]) + 180 * (1 - x[j, i]))

    print(f'\033[1A\033[2K\033[1AAnzahl der Bedingungen: {solver.NumConstraints()}\n')

    print('Erstelle Ziel...')
    objective = solver.Objective()
    for i, j in x:
        if -1 not in (i, j) and i < j:
            dist = distance(points[i], points[j])
            objective.SetCoefficient(x[i, j], dist)
            objective.SetCoefficient(x[j, i], dist)
    # add angle cost with factor
    objective.SetCoefficient(angle_ub, avg_arc_cost * len(points) * ANGLE_COST_FACTOR)
    objective.SetMinimization()

    print('\033[1A\033[2KFinde Lösung...')
    status = solver.Solve()

    print('\033[1A\033[2K', end='')

    # print solution
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        time = solver.WallTime()
        solver.VerifySolution(1e-7, True)
        status_name = 'OPTIMAL' if status == pywraplp.Solver.OPTIMAL else 'FEASIBLE'

        print(f'Zeit: {time/1000:.2f}s')
        print(f'Status: {status_name}')
        print(f'Länge: {objective.Value():.2f}km')
        print(f'Winkel-UB: {int(angle_ub.solution_value())}°')

        G = nx.Graph()
        for i in range(len(points)):
            G.add_node(i, pos=points[i])

        for i, j in x:
            if -1 not in (i, j) and x[i, j].solution_value() == 1:
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
