from functools import lru_cache
from typing import Dict, Iterable, List, Tuple
import numpy as np
from ortools.linear_solver import pywraplp
import networkx as nx
import matplotlib.pyplot as plt


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
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    ba = a - b
    bc = c - b
    return 180 - np.rad2deg(np.arccos(np.dot(ba, bc) /
                                      (np.linalg.norm(ba) * np.linalg.norm(bc))))


def get_data(points: Iterable[Tuple[float, float]]) -> Dict:
    data = {}
    data['max_turn_angle'] = 90
    data['points'] = tuple(points)
    return data


def main(points: List[Tuple[float, float]]):
    data = get_data(points)

    # create the solver
    solver = pywraplp.Solver.CreateSolver('CP_SAT')
    if not solver:
        print('Could not create solver')
        return

    # create the variables
    x = {}
    for i in range(len(data['points'])):
        for j in range(len(data['points'])):
            if i == j:
                continue
            x[(i, j)] = solver.BoolVar(f'x_{i}_{j}')
    print(f'Created {solver.NumVariables()} variables')

    # create the constraints

    # one selection per position
    for i in range(len(data['points'])):
        solver.Add(solver.Sum([x[(i, j)] for j in range(len(data['points'])) if i != j]) <= 1)

    # one end index TODO: check if this is correct
    solver.Add(solver.Sum([x[(i, j)] for i in range(len(data['points'])) for j in range(len(data['points'])) if i != j]) == len(data['points']) - 1)

    # a point can only be selected (up to) once 
    for j in range(len(data['points'])):
        solver.Add(solver.Sum([x[(i, j)] for i in range(len(data['points'])) if i != j]) <= 1)

    # no two-way connections
    for i in range(len(data['points'])):
        for j in range(len(data['points'])):
            if i == j:
                continue
            solver.Add(x[(i, j)] + x[(j, i)] <= 1)

    # create the turn constraint

    # create the objective
    objective = solver.Objective()
    for i in range(len(data['points'])):        # last row doesnt have any weight
        for j in range(len(data['points'])):
            if i == j:
                continue
            objective.SetCoefficient(x[(i, j)], _distance(data['points'][i], data['points'][j]))
    objective.SetMinimization()

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print('Objective value =', solver.Objective().Value())
        print()
        print('Problem solved in %f milliseconds' % solver.wall_time())
        print('Problem solved in %d iterations' % solver.iterations())
        print('Problem solved in %d branch-and-bound nodes' % solver.nodes())

        G = nx.Graph()
        for i in range(len(data['points'])):
            G.add_node(i, pos=data['points'][i])
        for i in range(len(data['points'])):
            for j in range(len(data['points'])):
                if i != j and x[(i, j)].solution_value() > 0:
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
