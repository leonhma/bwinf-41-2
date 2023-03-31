from functools import lru_cache
import itertools
import math
import operator
from typing import List, Tuple
from ortools.linear_solver import pywraplp
import networkx as nx
import matplotlib.pyplot as plt


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
    try:
        ba = sub(p1, p2)
        bc = sub(p3, p2)
        cos = max(min(dot(ba, bc) / (norm(ba) * norm(bc)), 1), -1)  # clamp to domain of acos
        return 180 - math.degrees(math.acos(cos))
    except Exception:
        print(p1, p2, p3)
        raise ZeroDivisionError


# pylama:ignore=C901
def main(points: List[Tuple[float, float]]):

    # create the solver
    solver = pywraplp.Solver.CreateSolver('CP_SAT')
    if not solver:
        print('Could not create solver')
        return
    solver.SetTimeLimit(120000)

    # create variables
    # create 3d variable matrix where angle doesn't exceed 90°
    x = {}
    for j in range(len(points)):
        for i, k in itertools.permutations(range(-1, len(points)), 2):
            if ((i != j and j != k and k != i) and
                    ((i == -1 or k == -1) or angle(points[i], points[j], points[k]) <= 90)):
                x[i, j, k] = solver.BoolVar(f'x_{i}_{j}_{k}')
    mlz = {i: solver.IntVar(0, len(points) - 1, f'mlz_{i}') for i in range(len(points))}  # added -1 here 
    print(f'Number of variables: {solver.NumVariables()}')

    # create the constraints
    # no subtours are created
    for i, j, k in x:
        if i != -1:
            # use <= and -1 on right side, linearize conditional constraint
            solver.Add(mlz[i] <= mlz[j] - 1 + len(points) * (1 - x[i, j, k]))
        if k != -1:
            solver.Add(mlz[j] <= mlz[k] - 1 + len(points) * (1 - x[i, j, k]))

    # every start index has a node except end index
    for i in range(-1, len(points)):
        solver.Add(sum(x[i, j, k] for i2, j, k in x if i2 == i) <= 1)
    # every turn index has a node
    for j in range(len(points)):
        solver.Add(sum(x[i, j, k] for i, j2, k in x if j2 == j) == 1)
    # every end index has a node except start index
    for k in range(-1, len(points)):
        solver.Add(sum(x[i, j, k] for i, j, k2 in x if k2 == k) <= 1)
    # the number of selected nodes matches the number of points
    solver.Add(sum(x[i, j, k] for i, j, k in x) == len(points))

    # ensure i and k are only once -1
    # solver.Add(sum(x[i, j, k] for i, j, k in x if i == -1) == 1)
    # solver.Add(sum(x[i, j, k] for i, j, k in x if k == -1) == 1)

    print(f'Number of constraints: {solver.NumConstraints()}')

    # create objective
    objective = solver.Objective()
    for i, j, k in x:
        coeff = 0
        if i != -1:
            coeff += 0.5 * distance(points[i], points[j])
        if k != -1:
            coeff += 0.5 * distance(points[j], points[k])
        objective.SetCoefficient(x[i, j, k], coeff)
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
        for i, j, k in x:
            if x[i, j, k].solution_value() == 1:
                num += 1
                if i != -1:
                    G.add_edge(i, j)
                if k != -1:
                    G.add_edge(j, k)

        print(f'Number of selected: {num} of {len(points)}')
        for i, j, k in x:
            if a := x[i, j, k].solution_value():
                if i == -1 and a:
                    print('i', i, i, j)
                if k == -1 and a:
                    print('k', i, j, k)

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
