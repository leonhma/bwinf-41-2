from functools import lru_cache
import itertools
import math
import operator
from typing import List, Tuple
from more_itertools import sliding_window
from ortools.sat.python import cp_model
import networkx as nx
import matplotlib.pyplot as plt


@lru_cache(maxsize=None)
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


@lru_cache(maxsize=None)
def angle(p1: Tuple[int, int], p2: Tuple[int, int], p3: Tuple[int, int]) -> float:
    ba = sub(p1, p2)
    bc = sub(p3, p2)
    cos = max(min(dot(ba, bc) / (norm(ba) * norm(bc)), 1), -1)  # clamp to domain of acos
    return 180 - math.degrees(math.acos(cos))


def main(points: List[Tuple[float, float]]):
    model = cp_model.CpModel()

    # create the variables
    nums = [model.NewIntVarFromDomain(cp_model.Domain.FromIntervals([[-1, i - 1], [i + 1, len(points) - 1]]), f'num_{i}') for i in range(len(points))]

    # TODO remove self assign

    # create the constraints
    model.AddAllDifferent(*nums)

    # create the turn constraint
    print('creating angle contraints')
    for i, j, k in sliding_window(map(lambda x: x % len(points), range(len(points) + 2)), 3):
        model \
            .Add(angle(points[i], points[j], points[k]) <= 90) \
            .OnlyEnforceIf(nums[i] == j) \
            .OnlyEnforceIf(nums[j] == k)

    model.Minimize(
        cp_model.LinearExpr.WeightedSum(
            tuple(distance(points[i], points[j]) for i, j in itertools.permutations(range(len(points), 2))),
            tuple(distance(i, j) if -1 not in (i, j) else 0 for i, j in itertools.permutations(range(len(points), 2)))
        ))

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print('Objective value =', solver.Objective().Value())
        if (status == cp_model.FEASIBLE):
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

if __name__ == '__main__':
    fname = f'beispieldaten/wenigerkrumm{input("nummer? ")}.txt'
    points = []
    with open(fname) as f:
        points = [tuple(map(float, line.split())) for line in f.readlines()]
    main(tuple(points))
