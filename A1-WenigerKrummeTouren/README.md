# Weniger Krumme Touren

❔ A1 👤 64712 🧑 Leonhard Masche 📆 01.04.2023

## Inhaltsverzeichnis

1. [Lösungsidee](#lösungsidee)
2. [Umsetzung](#umsetzung)
    1. [Verbesserungen](#verbesserungen)
    2. [Qualität der Ergebnisse](#qualität-der-ergebnisse)
3. [Beispiele](#beispiele)
4. [Quellcode](#quellcode)

## Lösungsidee

Das Netz der Außenposten wird als Graph betrachtet.
Gegeben sei ein kompletter Graph $G(V, E)$, der die möglichen Verbindungen zwischen den einzelnen Knoten darstellt.
$V$ stellt Menge der Außenposten, und $E$ ist die Menge der möglichen Verbindungen dieser dar.
Nun gilt es als Lösung einen Hamiltonkreis $L(V, E_L)$ zu konstruieren, der die Bedingungen $E_L \subset E$ und $|E_L| = |V| - 1$ erfüllt.
Zusätzlich dazu müssen auch noch die Vorgaben aus der Aufgabenstellung (keine Abbiegewinkel über $90°$ und die Minimierung der Strecke) beachtet werden.

Für eine arbiträre Liste von Außenstellen und deren Koordinaten kann nicht immer eine Lösung gefunden werden. Das Liegt daran dass es sein kann, dass eine Außenstelle keine zwei Nachbaren hat, mit denen sie einen Abbiegewinkel unter $90°$ bilden kann. Hier ein Beispiel:
![Darstellung von Punkten die keinen Pfad nach Aufgabenbeschreibung zulassen](./static/illegal_turn.png)

Wie man sieht kann hier (leicht überprüfbar) kein Pfad gefunden werden, der die verlangten Anforderungen erfüllt.

Modelliert wird diese Aufgabenstellung mit einem Integer-Linear-Programming Modell, bestehend aus einer Matrix von binären Variablen die angeben, ob zwischen zwei Knoten eine Verbindung besteht.

Diese Aufgabe (die Suche nach einem optimalen Pfad) ähnelt sehr stark dem Travelling-Salesman-Problem, und teilt mit diesem auch seine Klassifizierung als NP-Schwer. Während eine Suche nach irgendeiner Lösung, die die Abiegewinkel- und Grapheigenschaften-Vorgaben erfüllt durch ILP auf ein SAT-Problem reduziert werden kann und somit NP-Komplett ist, ist die Suche nach einer optimalen Lösung NP-Schwer, da sich eine Lösung nicht in Polynom-Zeit verifizieren lässt. Ein ähnlicher Aufwand muss für den Beweis der Unauffindbarkeit einer möglichen Route vollbracht werden. Dieser Befindet sich als Umkehrung des vorher genannten SAT-Problems in der Klasse co-NP.

## Umsetzung

Wie vorher genannt wird die Aufgabenstellung als Integer-Linear-Programming Problem formuliert. $W$ sei $V\cup\set{-1}$. Hierzu wird eine 2d-Matrix an binären Variablen $x_{ij}\quad(i,j)\in W$ erstellt, die besagt, ob ein Knoten $i$ mit dem Knoten $j$ verbunden ist. Der Index $-1$ ist dafür zuständig, den Start und das Ende der Tour zu markieren und wird in der Wegkostenberechnung nicht berücksichtigt.

Um bei jedem Knoten einen Grad von $\delta(v)=2\quad v\in W$ sicherzustellen, werden zwei Bedingungen eingeführt:
$$\sum_{j\in W}x_{ij} = 1\qquad i \in W\tag 1$$
$$\sum_{i\in W}x_{ij} = 1\qquad j \in W\tag 2$$

Als weitere Bedingung müssen noch disjunkte Teilstrecken verhindert werden. Diese entstehen wenn ein Knoten mit einem Knoten verbunden ist, der schon vorher in der Tour enthalten war. Diese Bedingung wird für den Knoten $-1$ nicht durchgesetzt, da dieser sowohl am Start, als auch am Ende der Tour enthalten sein muss. Um diese Bedingung zu modellieren werden entsprechend der MTZ-Methode $t_i \quad i \in V$ weitere ganzzahlige Variablen eingeführt, welche die Position der Knotenpunkte in der Tour angeben. Zusätzlich wird diese Bedingung aufgestellt:

$$x_{ij} \implies t_i < t_j \qquad (i, j) \in V^2 \tag 3$$

Zuletzt muss noch die Winkel-Vorgabe berücksichtigt werden. Vor dem eigentlichen Vorgang des Lösens werden alle Winkel mit dem Kreuzprodukt von Vektoren vorberechnet und in einer 3d-Matrix $a$ gespeichert. So ergibt sich:
$$x_{ij} \wedge x_{jk} \implies a_{ijk} \le 90 \qquad (i, j, k) \in V^3 \tag 4$$

Als zu minimierende Funktion wird der Gesamtweg berechnet. $c_{ij}\quad (i, j)\in V^2$ sei der Abstand zwischen den Knoten $i$ und $j$.
$$\text{min}\quad\sum_{i \in V}\sum_{j \in V}c_{ij} x_{ij}\tag{5}$$

Im Quelltext sind diese Beschränkungen in linearisierter Form zu finden.
Das Programm ist in der Sprache Python umgesetzt und ab der Version `3.6` ausführbar. Zur Lösung wird die von Google entwicklete Bibliothek `ortools` neben einigen anderen Paketen verwendet, die mit `pip install -r requirements.txt` installiert werden können. Das Programm erstellt das ILP-Modell, sucht dann mit einem Zeitlimit von 3 Minuten nach einer Lösung und gibt diese aus.

### Verbesserungen

#### Jahre später

In den ersten Zeilen des Programms finden sich Konstanten, mit denen sich das Verhalten des Programms anpassen lässt. So zum Beispiel auch die maximale Berechnungszeit...

```python
ANGLE_UPPER_BOUND = 90
ANGLE_COST_FACTOR = 0       # 0.002
SOLVER_MAX_TIME = 60 * 3    # 3 Minuten Berechnungszeit
```

#### Maximaler Winkel

Anton hat ein neues Gefährt bekommen! Jetzt kann er Abbiegewinkel von `110°` meistern. In den Parametern kann auch der maximale Abbiegewinkel angepasst werden (`ANGLE_UPPER_BOUND`).

#### Abbiegewinkel-Minimierung

Einer der weiteren anpassbaren Parameter (`ANGLE_COST_FACTOR`) fügt den maximalen Abbiegewinkel als Teil der Kostenfunktion hinzu. So kann auch dieser optimiert werden. Ein guter Wert scheint `0.002` zu sein. Allerdings wird die Suche dadurch sehr viel langsamer. Hier ein Ergebnis für Beispiel 3 mit Weglänge `<TODO>` und Winkel-UB `33°`, das mit einer Maximalzeit von 20 Minuten berechnet wurde:

![Kreise](./static/3-angle-20min.png)

#### Halbierung der Anzahl der berechneten Winkel

Da der Winkel $a_{kji}$ gleich dem Winkel $a_{ijk}$ ist, wird nur letzerer berechnet, und für diesen nun Bedingungen in beide Richtungen ($x_{ij} \wedge x_{jk}$ und $x_{kj} \wedge x_{ji}$) hinzugefügt. Die Anzahl der vorberechneten Winkel wird somit halbiert.

#### Halbierung der Anzahl der berechneten Distanzen

Da die Distanz $c_{ji}$ gleich der Distanz $c_{ij}$ ist, wird nur letzere berechnet, und für diese nun Bedingungen in beide Richtungen ($x_{ij}$ und $x_{ji}$) hinzugefügt. Die Anzahl der berechneten Distanzen wird somit halbiert.

### Qualität der Ergebnisse

Das Integer-Linear-Programming Verfahren ist in der Lage, optimale Ergebnisse zu finden ('optimal' heißt hier nicht 'exklusiv optimal'). Da aber einige sehr große Instanzen bearbeitet werden, werden in drei Minuten teilweise nur sinnvolle Lösungen erreicht.

Das liegt daran, dass im ILP-Modell sowohl Variablen als auch Bedingungen in grob quadratisch wachsender Anzahl erstellt werden. Auf einem Desktop-System mit 16 logischen Kernen @4.5GHz werden alle Beispiele außer `3`, `6` und `7` optimal gelöst. Für diese Aufgaben wird aber eine zufriedenstellende mögliche Lösung gefunden.

## Beispiele


## Quellcode

*program.py*

```python
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
ANGLE_COST_FACTOR = 0       # 0.002
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
        i, j = random.choices(range(len(points)), k=2)
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
    # winkel-Matrix berechnen
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
    # Erstellen von Subtour Elimination Variablen
    t = {i: solver.IntVar(0, len(points) - 1, f't_{i}') for i in range(len(points))}
    # Erstellen von Winkel-Upper-Bound Variable
    angle_ub = solver.IntVar(0, ANGLE_UPPER_BOUND, 'angle_ub')

    print(f'\033[1A\033[2KAnzahl der Variablen: {solver.NumVariables()}\n')

    print('Erstelle Bedingungen...')
    # Bedingungen für die Subtour Elimination
    for i, j in x:
        if -1 not in (i, j):
            # linearisierte Bedingung für die Subtour Elimination
            solver.Add(t[i] <= t[j] - 1 + len(points) * (1 - x[i, j]))
    # jeder Knoten hat einen nächsten Knoten
    for i in range(-1, len(points)):
        solver.Add(sum(x[i, j] for i2, j in x if i2 == i) == 1)
    # jeder Knoten hat einen vorherigen Knoten
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
            # Distanz wird als Koeffizient hinzugefügt
            dist = distance(points[i], points[j])
            objective.SetCoefficient(x[i, j], dist)
            objective.SetCoefficient(x[j, i], dist)
    # angle_ub wird als Kostenfaktor hinzugefügt
    objective.SetCoefficient(angle_ub, avg_arc_cost * len(points) * ANGLE_COST_FACTOR)
    objective.SetMinimization()

    print('\033[1A\033[2KFinde Lösung...')
    status = solver.Solve()

    print('\033[1A\033[2K', end='')

    # Lösung anzeigen
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        time = solver.WallTime()
        solver.VerifySolution(1e-7, True)
        status_name = 'OPTIMAL' if status == pywraplp.Solver.OPTIMAL else 'FEASIBLE'

        G = nx.Graph()
        for i in range(len(points)):
            G.add_node(i, pos=points[i])

        length = 0
        for i, j in x:
            if -1 not in (i, j) and x[i, j].solution_value() == 1:
                G.add_edge(i, j)
                length += distance(points[i], points[j])

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
                print(e)
    except ExitException as e:
        print(e)
        exit()
    except KeyboardInterrupt:
        print()
        print('Abbruch durch Benutzer.')
        exit()

```
