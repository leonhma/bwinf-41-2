import functools
import itertools
import math
import operator
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from ortools.sat.python import cp_model


@functools.lru_cache(maxsize=None)    # cache the results of this function for speeed
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


def main(points: List[Tuple[float, float]]):
    model = cp_model.CpModel()

    # create variables
    # create 3d variable matrix where angle doesn't exceed 90°
    x = {}
    for j in range(len(points)):
        for i, k in itertools.permutations(range(-1, len(points)), 2):
            if ((i != j and j != k and k != i) and
                    ((i == -1 or k == -1) or angle(points[i], points[j], points[k]) <= 90)):
                x[i, j, k] = model.NewBoolVar(f'x_{i}_{j}_{k}')
    mlz = {i: model.NewIntVar(0, len(points) - 1, f'mlz_{i}') for i in range(len(points))}
    print(f'Number of variables: {len(x)}/{len(mlz)}')

    # create the constraints
    # no subtours are created
    for i, j, k in x:
        if i != -1:
            model.Add(mlz[i] + 1 == mlz[j]).OnlyEnforceIf(x[i, j, k])
        if k != -1:
            model.Add(mlz[j] + 1 == mlz[k]).OnlyEnforceIf(x[i, j, k])

    # the number of selected nodes matches the number of points
    # model.Add(cp_model.LinearExpr.Sum(tuple(x[i, j, k] for i, j, k in x)) == len(points))

    # ensure i and k are only once -1
    # every i is <= 1
    for i in range(-1, len(points)):
        model.AddAtMostOne(tuple(x[i, j, k] for i2, j, k in x if i2 == i))
    # every j is 1
    for j in range(len(points)):
        model.AddExactlyOne(tuple(x[i, j, k] for i, j2, k in x if j2 == j))
    # every k is <= 1
    for k in range(-1, len(points)):
        model.AddAtMostOne(tuple(x[i, j, k] for i, j, k2 in x if k2 == k))

    def coeff(i, j, k):
        coeff = 0
        if i != -1:
            coeff += 0.5 * distance(points[i], points[j])
        if k != -1:
            coeff += 0.5 * distance(points[j], points[k])
        return coeff

    model.Minimize(
        cp_model.LinearExpr.WeightedSum(
            tuple(x[i, j, k] for i, j, k in x),
            tuple(coeff(i, j, k) for i, j, k in x)
        ))
    
    solver = cp_model.CpSolver()
    status = solver.Solve(model)


# print solution
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        if (status == cp_model.FEASIBLE):
            print('A potentially suboptimal solution.')

        G = nx.Graph()
        for i in range(len(points)):
            G.add_node(i, pos=points[i])
        num = 0
        for i, j, k in x:
            if solver.Value(x[i, j, k]) == 1:
                num += 1
                if i != -1:
                    G.add_edge(i, j)
                if k != -1:
                    G.add_edge(j, k)

        print(f'Number of selected: {num} of {len(points)}')
        for i, j, k in x:
            if a := solver.Value(x[i, j, k]):
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
