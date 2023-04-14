import itertools
import math
import operator
import os
import random
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from ortools.linear_solver import pywraplp

ANGLE_UPPER_BOUND = 90
ANGLE_COST_FACTOR = 0        # 0.002
SOLVER_MAX_TIME = 60 * 20    # 3 Minuten Berechnungszeit


class ExitException(BaseException):
    pass


# Skalarprodukt zweier Vektoren
def dot(a: Tuple[float, ...], b: Tuple[float, ...]):
    return sum(map(operator.mul, a, b))


# Abzug von Vektoren
def sub(a: Tuple[float, ...], b: Tuple[float, ...]):
    return tuple(map(operator.sub, a, b))


# Länge eines Vektors
def norm(a: Tuple[float, ...]):
    return math.hypot(*a)


# Winkel zwischen drei Punkten
def angle(p1: Tuple[int, int], p2: Tuple[int, int], p3: Tuple[int, int]) -> float:
    ba = sub(p1, p2)
    bc = sub(p3, p2)
    # Beschränken auf [-1, 1] um Rundungsfehler zu vermeiden
    cos = max(min(dot(ba, bc) / (norm(ba) * norm(bc)), 1), -1)
    return 180 - math.degrees(math.acos(cos))


# Distanz zwischen zwei Punkten
def distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# pylama:ignore=C901
def main(points: List[Tuple[float, float]], fname: str):

    print()
    print('Berechne ungefähren Mittelwert der Kantenlängen...')
    # Berechnung eines ungefähren Mittelwerts für die Kantenlängen,
    # für die Verwendung in der Kostengleichung
    avg_arc_cost = 0
    avg_arc_n = 0
    for _ in range(1000):
        i, j = random.choices(range(len(points)), k=2)
        avg_arc_cost += distance(points[i], points[j])
        avg_arc_n += 1
    avg_arc_cost /= avg_arc_n

    print('\033[1A\033[2KErstelle Solver...')
    solver = pywraplp.Solver.CreateSolver('CP_SAT')
    if not solver:
        raise ExitException('Fehler beim Erstellen des Solvers. '
                            '(Ist die richtige Version von ortools installiert?)')
    # Maximale Berechnungszeit in Millisekunden
    solver.SetTimeLimit(SOLVER_MAX_TIME * 1000)
    solver.SetNumThreads(max(1, os.cpu_count() - 2))    # Anzahl der Threads

    print('\033[1A\033[2KVorberechnung der Winkel...')
    # Winkel-Matrix berechnen
    a = {}
    for i, j, k in itertools.permutations(range(-1, len(points)), 3):
        if i < k and j not in (i, k):
            if -1 in (i, j, k):  # winkel beinhaltet den 'unsichbaren' Start- / Endknoten
                a[i, j, k] = 0
            else:
                a[i, j, k] = angle(points[i], points[j], points[k])

    print('\033[1A\033[2KErstelle Variablen...')
    # 2d-Binärmatrix für die Kanten
    x = {}
    for i, j in itertools.permutations(range(-1, len(points)), 2):
        if i != j:
            x[i, j] = solver.BoolVar(f'x_{i}_{j}')
    # Erstellen von Subtour-Eliminierungs-Variablen
    t = {i: solver.IntVar(0, len(points) - 1, f't_{i}')
         for i in range(len(points))}
    # Erstellen von Winkel-Upper-Bound Variable
    angle_ub = solver.IntVar(0, ANGLE_UPPER_BOUND, 'angle_ub')

    print(f'\033[1A\033[2KAnzahl der Variablen: {solver.NumVariables()}\n')

    print('Erstelle Bedingungen...')
    # Bedingungen für die Subtour Elimination
    for i, j in x:
        if -1 not in (i, j):
            # linearisierte Bedingung für die Subtour Elimination
            solver.Add(t[i] <= t[j] - 1 + len(points) * (1 - x[i, j]))
    # Jeder Knoten hat einen nächsten Knoten
    for i in range(-1, len(points)):
        solver.Add(sum(x[i, j] for i2, j in x if i2 == i) == 1)
    # Jeder Knoten hat einen vorherigen Knoten
    for j in range(-1, len(points)):
        solver.Add(sum(x[i, j] for i, j2 in x if j2 == j) == 1)
    # Jeder Winkel im Pfad muss kleiner als der Upper-Bound sein, d. h.
    # angle_ub ist >= dem größten Winkel im Pfad
    for i, j, k in a:
        if i < k and (i, j) in x and (j, k) in x:
            solver.Add(a[i, j, k] <= angle_ub + 180 *
                       (1 - x[i, j]) + 180 * (1 - x[j, k]))
            solver.Add(a[i, j, k] <= angle_ub + 180 *
                       (1 - x[k, j]) + 180 * (1 - x[j, i]))

    print(
        f'\033[1A\033[2K\033[1AAnzahl der Bedingungen: {solver.NumConstraints()}\n')

    # Erstellen der Kostenfunktion
    print('Erstelle Ziel...')
    objective = solver.Objective()
    for i, j in x:
        if -1 not in (i, j) and i < j:
            # Distanz wird als Koeffizient hinzugefügt
            dist = distance(points[i], points[j])
            objective.SetCoefficient(x[i, j], dist)
            objective.SetCoefficient(x[j, i], dist)
    # angle_ub wird als Kostenfaktor hinzugefügt
    objective.SetCoefficient(angle_ub, avg_arc_cost *
                             len(points) * ANGLE_COST_FACTOR)
    objective.SetMinimization()

    # Lösung finden
    print('\033[1A\033[2KFinde Lösung...')
    status = solver.Solve()

    print('\033[1A\033[2K', end='')

    # Lösung anzeigen
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        time = solver.WallTime()
        solver.VerifySolution(1e-7, True)
        status_name = 'OPTIMAL' if status == pywraplp.Solver.OPTIMAL else 'FEASIBLE'

        # Konstruieren des Lösungs-Graphen
        G = nx.Graph()
        for i in range(len(points)):
            G.add_node(i, pos=points[i])

        # Übersetzen der Variablen in Kanten
        end_nodes = []
        length = 0
        for i, j in x:
            if -1 not in (i, j) and x[i, j].solution_value() == 1:
                G.add_edge(i, j)
                if i not in end_nodes:
                    end_nodes.append(i)
                else:
                    end_nodes.remove(i)
                if j not in end_nodes:
                    end_nodes.append(j)
                else:
                    end_nodes.remove(j)
                length += distance(points[i], points[j])

        # Ausgabe der Lösung als Datei
        Path(os.path.join(os.path.dirname(__file__), 'output')).mkdir(
            parents=True, exist_ok=True)
        with open(os.path.join(os.path.dirname(__file__), f'output/{fname}'), 'w') as f:
            for node in nx.shortest_path(G, end_nodes[0], end_nodes[1]):
                coords = points[node]
                f.write(f'{coords[0]} {coords[1]}\n')

        # Ausgabe der Lösungswerte in der Konsole
        print(f'Zeit: {time/1000:.2f}s')
        print(f'Status: {status_name}')
        print(f'Länge: {length:.2f}km')
        print(f'Winkel-UB: {int(angle_ub.solution_value())}°')

        # Rote Farbe für Winkel > ANGLE_UPPER_BOUND
        for node in G.nodes:
            neighbors = list(G.neighbors(node))
            if len(neighbors) == 2:
                a, b = neighbors
                if angle(points[a], points[node], points[b]) > ANGLE_UPPER_BOUND:
                    G.nodes[node]['color'] = 'r'

        ax = plt.gca()
        # Damit die x- und y-Achsen gleich skaliert werden
        ax.set_aspect('equal')
        plt.get_current_fig_manager().set_window_title(f'{fname[:-4]}')

        pos = nx.get_node_attributes(G, 'pos')
        colorsd = nx.get_node_attributes(G, 'color')
        colors = [colorsd.get(node, 'w') for node in G.nodes]

        nx.draw(G,
                pos,
                node_size=10,
                font_size=8,
                node_color=colors,
                edgecolors='k')

        plt.show()      # Anzeigen des Graphen
    else:
        print('Keine mögliche Lösung gefunden.')
    print()


# Konsolen-Loop
if __name__ == '__main__':
    try:
        while True:
            try:
                fname = f'wenigerkrumm{input("Bitte Zahl des Beispiels eingeben: ")}.txt'
                points = []
                with open(os.path.join(os.path.dirname(__file__), f'beispieldaten/{fname}')) as f:
                    points = [tuple(map(float, line.split()))
                              for line in f.readlines()]
                main(tuple(points), fname)
            except Exception as e:
                print(e)
    except ExitException as e:
        print(e)
        exit()
    except KeyboardInterrupt:
        print()
        print('Abbruch durch Benutzer.')
        exit()
