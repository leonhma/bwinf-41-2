from functools import lru_cache
import itertools
import math
import operator
from typing import List, Tuple
from ortools.linear_solver import pywraplp
import networkx as nx
import matplotlib.pyplot as plt


ANGLE_UPPER_BOUND = 90
ANGLE_COST_FACTOR = 0

@lru_cache(maxsize=None)    # cache the results of this function for speeed
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
    cos = max(min(dot(ba, bc) / (norm(ba) * norm(bc)), 1), -1)  # clamp to domain of acos
    return 180 - math.degrees(math.acos(cos))


# pylama:ignore=C901
def main(points: List[Tuple[float, float]]):

    print('creating solver')
    # create the solver
    solver = pywraplp.Solver.CreateSolver('CP_SAT')
    if not solver:
        print('Could not create solver')
        return
    solver.SetTimeLimit(120000)

    print('precomputing angles')
    a = {}
    for i, j, k in itertools.permutations(range(-1, len(points)), 3):
        if i >= k
        if -1 in (i, j, k):
            a[i, j, k] = 0
        else:
            a[i, j, k] = angle(points[i], points[j], points[k])

    print('creating variables')
    # create variables
    # create 3d variable matrix where angle doesn't exceed 90°
    x = {}
    for i, j in itertools.permutations(range(-1, len(points)), 2):
        if i != j:
            x[i, j] = solver.BoolVar(f'x_{i}_{j}')
    mlz = {i: solver.IntVar(0, len(points) - 1, f'mlz_{i}') for i in range(len(points))}
    angle_ub = solver.IntVar(0, ANGLE_UPPER_BOUND, 'angle_ub')

    print(f'Number of variables: {solver.NumVariables()}')

    print('creating constraints')
    # create the constraints
    # no subtours are created
    for i, j in x:
        if -1 not in (i, j):
            # use <= and -1 on right side, linearize conditional constraint
            solver.Add(mlz[i] <= mlz[j] - 1 + len(points) * (1 - x[i, j]))
    print('done creating mtz constraints')

    # every node has a next except end index
    for i in range(-1, len(points)):
        solver.Add(sum(x[i, j] for i2, j in x if i2 == i) == 1)
    # every node has a previous except start index
    for j in range(-1, len(points)):
        solver.Add(sum(x[i, j] for i, j2 in x if j2 == j) == 1)
    # the number of selected nodes matches the number of points + 1 (start-/end marker)
    print('done creating next/prev constraints')

    # angle <= angle_ub
    for i, j, k in a:
        if i < j and (i, j) in x and (j, k) in x:
            solver.Add(a[i, j, k] <= angle_ub + 180 * (1 - x[i, j]) + 180 * (1 - x[j, k]))
    print('done creating angle constraints')

    print(f'Number of constraints: {solver.NumConstraints()}')

    print('creating objective')
    # create objective
    objective = solver.Objective()
    for i, j in x:
        if -1 not in (i, j):
            objective.SetCoefficient(x[i, j], distance(points[i], points[j]))
    # TODO: add angle_ub cost
    objective.SetMinimization()

    # solve
    status = solver.Solve()

    # print solution
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        solver.VerifySolution(1e-7, True)
        print(f'Objective value: {objective.Value()}')
        if (status == pywraplp.Solver.FEASIBLE):
            print('A potentially suboptimal solution.')

        G = nx.Graph()
        for i in range(len(points)):
            G.add_node(i, pos=points[i])
        num = 0
        for i, j in x:
            if -1 not in (i, j) and x[i, j].solution_value() == 1:
                num += 1
                G.add_edge(i, j)

        print(f'Number of selected: {num} of {len(points)}')
        for i, j in x:
            if ab := x[i, j].solution_value():
                if i == -1 and ab:
                    print('i', i, i)
                if j == -1 and ab:
                    print('j', i, j)

        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos, with_labels=True)
        plt.gca().set_aspect('equal')
        plt.show()
    else:
        print('The problem does not have a feasible solution.')


if __name__ == '__main__':
    fname = f'beispieldaten/wenigerkrumm{input("nummer? ")}.txt'
    points = []
    with open(fname) as f:
        points = [tuple(map(float, line.split())) for line in f.readlines()]
    main(tuple(points))
