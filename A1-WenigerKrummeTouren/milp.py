from functools import lru_cache
import itertools
from typing import List, Tuple
from ortools.linear_solver import pywraplp
import networkx as nx
import matplotlib.pyplot as plt
import operator
import math


@lru_cache(maxsize=None)    # cache the results of this function for speeed
def _distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """Return the distance between two points.
    Parameters
    ----------
    p1 : int
        The index of the first point.
    p2 : int
        The index of the second point.
    Returns
    -------
    float
        The distance between the two points.
    """
    return (
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2) ** 0.5


def dot(a, b):
    return sum(map(operator.mul, a, b))


def sub(a, b):
    return tuple(map(operator.sub, a, b))


def norm(a):
    return math.sqrt(dot(a, a))


@lru_cache(maxsize=None)
def _angle(p1: Tuple[int, int], p2: Tuple[int, int], p3: Tuple[int, int]) -> float:
    """Return the closest-to-zero angle of the flight line to the x-axis. ([-180, 180])
    Hint
    ----
        Subtracting two of these values gives the turn angle.
    Parameters
    ----------
    p1 : int
        The index of the first point.
    p2 : int
        The index of the second point.
    p3 : int
        The index of the third point.
    Returns
    -------
    float
        The angle between the two points.
    """
    ba = sub(p1, p2)
    bc = sub(p3, p2)
    return 180 - math.degrees(math.acos(dot(ba, bc) / (norm(ba) * norm(bc))))


def main(points: List[Tuple[float, float]]):

    # create the solver
    solver = pywraplp.Solver.CreateSolver('CP_SAT')
    if not solver:
        print('Could not create solver')
        return

    # create the variables
    x = {}
    for i in range(len(points)):
        for j in range(len(points)):
            if i == j:
                continue
            x[i, j] = solver.BoolVar(f'x_{i}_{j}')

    for i in range(len(points)):
        x['t', i] = solver.IntVar(0, len(points) - 1, f't_{i}')
    print(f'Created {solver.NumVariables()} variables')

    # subtour elimination
    for i, j in itertools.permutations(range(len(points)), 2):
        if i != j and (i != 0 and j != 0):
            solver.Add(x['t', j] > (x['t', i] - (len(points) - 1) * (1 - x[i, j])))

    # create the turn constraint
    for i, j, k in itertools.permutations(range(len(points)), 3):
        if i != j and j != k and i != k:
            solver.Add(_angle(points[i], points[j], points[k]) <= 90 + 90 * (1 - x[i, j]) * (1 - x[j, k]))

    # create the objective
    objective = solver.Objective()
    for i in range(len(points)):        # last row doesnt have any weight
        for j in range(len(points)):
            if i == j:
                continue
            objective.SetCoefficient(x[i, j], _distance(points[i], points[j]))
    objective.SetMinimization()

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        print('Objective value =', solver.Objective().Value())
        if (status == pywraplp.Solver.FEASIBLE):
            print('A potentially suboptimal solution.')
        print()
        print('Problem solved in %f milliseconds' % solver.wall_time())
        print('Problem solved in %d iterations' % solver.iterations())
        print('Problem solved in %d branch-and-bound nodes' % solver.nodes())

        G = nx.Graph()
        for i in range(len(points)):
            G.add_node(i, pos=points[i])
        for i in range(len(points)):
            for j in range(len(points)):
                if i != j and x[i, j].solution_value() > 0:
                    G.add_edge(i, j)

        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos, with_labels=True)
        plt.gca().set_aspect('equal')
        plt.show()
    else:
        print('The problem does not have an optimal solution.')


if __name__ == '__main__':
    fname = f'beispieldaten/wenigerkrumm{input("nummer? ")}.txt'
    points = []
    with open(fname) as f:
        points = [tuple(map(float, line.split())) for line in f.readlines()]
    main(tuple(points))
