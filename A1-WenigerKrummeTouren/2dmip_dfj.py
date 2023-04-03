import operator
import time
import matplotlib.pyplot as plt
import networkx as nx
import mip

import os
import math
import itertools
import random

from typing import Dict, List, Set, Tuple

ANGLE_UPPER_BOUND = 90
ANGLE_COST_FACTOR = 0       # 0.002
SOLVER_MAX_TIME = 60 * 2    # 2 Minuten Berechnungszeit


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


class SubtourAndAngleConstraintGenerator(mip.ConstrsGenerator):
    def __init__(self,
                 F: List[Tuple[int, int]],
                 V: Set[int],
                 x_: Dict[Tuple[int, int], mip.Var]):
        self.F, self. V, self.x = F, V, x_

    def generate_constrs(self, model: mip.Model, depth: int = 0, npass: int = 0):
        xf, cp = model.translate(self.x), mip.CutPool()
        # check subtours
        G = nx.DiGraph()
        for (i, j) in xf:
            if xf[i, j] >= 0.99:
                G.add_edge(i, j, capacity=xf[i, j].x)
        for (u, v) in self.F:
            try:
                val, (S, _) = nx.minimum_cut(G, u, v)
                if val <= 0.99:
                    aInS = [(xf[i, j], xf[i, j].x)
                            for (i, j) in xf if (xf[i, j] and i in S and j in S)]
                    if sum(f for _, f in aInS) >= (len(S) - 1) + 1e-4:
                        cut = mip.xsum(v for v, _ in aInS) <= len(S) - 1
                        cp.add(cut)
            except Exception as e:
                print(f'Exception for i={u}, j={v}: {e}')

        for cut in cp.cuts:
            model += cut


# pylama:ignore=C901
def main(points: List[Tuple[float, float]], fname: str):
    start_time = time.time()
    print()

    V = set(range(-1, len(points)))
    E = {(i, j) for i in V for j in V if i < j}

    print('Berechne ungefähren Mittelwert der Kantenlängen...')
    avg_arc_cost = 0
    avg_arc_n = 0
    for _ in range(1000):
        i, j = random.sample(V, 2)
        avg_arc_cost += distance(points[i], points[j])
        avg_arc_n += 1
    avg_arc_cost /= avg_arc_n

    print('\033[1A\033[2KVorberechnung der Winkel...')
    # winkel-matrix berechnen
    a = {}
    for i, j, k in itertools.permutations(V, 3):
        if i < k and j not in (i, k):
            if -1 in (i, j, k):  # winkel beinhaltet den 'unsichbaren' Start- / Endknoten
                a[i, j, k] = 0
            else:
                a[i, j, k] = angle(points[i], points[j], points[k])

    print('\033[1A\033[2KErstelle Solver...')
    model = mip.Model(sense=mip.MINIMIZE, solver_name=mip.CBC)

    # temporärer Graph
    G = nx.Graph()

    print('\033[1A\033[2KErstelle Variablen...')
    # Entscheidungsvariablen
    x = {}
    for i, j in E:
        if i < j:
            dist = distance(points[i], points[j])
            # Variable hinzufügen
            x[i, j] = model.add_var(name=f'x_{i}_{j}',
                                    obj=dist,
                                    var_type=mip.BINARY)
            # Kante in temporärem Graphen hinzufügen
            G.add_edge(i, j, weight=dist)

    angle_ub = model.add_var(name='angle_ub',
                             var_type=mip.CONTINUOUS,
                             lb=0, ub=ANGLE_UPPER_BOUND,
                             obj=ANGLE_COST_FACTOR * len(V) * avg_arc_cost)

    # angle <= angle_ub
    for i, j, k in a:  # => i < k
        if (i, j) in x:
            if (j, k) in x:  # => i < j < k
                model += (a[i, j, k] <= angle_ub + 180 * (1 - x[i, j]) + 180 * (1 - x[j, k]))
            else:  # => i < k < j
                model += (a[i, j, k] <= angle_ub + 180 * (1 - x[i, j]) + 180 * (1 - x[k, j]))
        elif (j, k) in x:  # => j < i < k
            model += (a[i, j, k] <= angle_ub + 180 * (1 - x[j, i]) + 180 * (1 - x[j, k]))

    print(f'Anzahl Variablen: {len(x)}')

    # Jeder Knoten muss Grad 2 haben
    for n in V:
        model += (sum(x[i, j] for i, j in x if n in (i, j)) == 2)

    # Berechnung des am weistesten entfernten Knoten zu i für alle i in V
    F = []
    for i in V:
        _, D = nx.dijkstra_predecessor_and_distance(G, source=i)
        DS = list(D.items())
        DS.sort(key=lambda x: x[1])
        F.append((i, DS[-1][0]))

    print(F)

    # Lösung starten
    model.verbose = 0
    model.cuts_generator = SubtourAndAngleConstraintGenerator(F, V, x)
    model.lazy_constrs_generator = SubtourAndAngleConstraintGenerator(F, V, x)
    status = model.optimize(max_seconds=SOLVER_MAX_TIME)
    print(status)

    print('\033[1A\033[2K', end='')

    # print solution
    if status in (mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE):
        print(f'Zeit: {time.time() - start_time:.2f}s')
        print(f'Status: {status}')
        print(f'Länge: {model.objective_value:.2f}km (bound is {model.objective_bound:.2f}km)')

        G = nx.Graph()
        for i in range(len(points)):
            G.add_node(i, pos=points[i])

        for i, j in x:
            if -1 not in (i, j) and x[i, j].x == 1:
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

# TODO add constraint generator for angles
# TODO add constraint generator for dfj
# TODO test on solvers (gurobi)
