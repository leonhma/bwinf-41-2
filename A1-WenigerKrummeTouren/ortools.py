from functools import lru_cache
from typing import Tuple
import numpy as np
from ortools.sat.python import cp_model
import networkx as nx
import matplotlib.pyplot as plt

from utils import sliding_window

def main(points: Tuple[Tuple[float, float]]):
    model = cp_model.CpModel()
    vars = {}
    for node in range(len(points)):
        for to in range(len(points)):
            if node == to:
                continue
            vars[(node, to)] = model.NewBoolVar(f'{node}->{to}')

    for node in range(len(points)):
        model.AddExactlyOne(vars[(node, to)] for to in range(len(points)) if node != to)

    for to in range(len(points)):
        model.AddExactlyOne(vars[(node, to)] for node in range(len(points)) if node != to)

    @lru_cache(maxsize=None)    # cache the results of this function for speeed
    def _distance(p1: int, p2: int) -> float:
        a1, a2 = points[p1]
        b1, b2 = points[p2]
        return (
            (b1 - a1) ** 2 +
            (b2 - a2) ** 2) ** 0.5

    @lru_cache(maxsize=None)
    def _is_illegal_turn(p1: int, p2: int, p3: int) -> float:
        a = np.array(points[p1])
        b = np.array(points[p2])
        c = np.array(points[p3])

        ba = a - b
        bc = c - b

        return np.rad2deg(np.arccos(np.dot(ba, bc) /
                                    np.linalg.norm(ba) / np.linalg.norm(bc))) < 90

    # add constraints for angles less than 90 degrees
    for a in range(len(points)):
        for b in range(len(points)):
            if a == b:
                continue
            for c in range(len(points)):
                if a == c or b == c:
                    continue
                model.Add(vars[(a, b)] * vars[(b, c)] * _is_illegal_turn(a, b, c) <= 0)

    solver = cp_model.CpSolver()

    class PartialSolutionPrinter(cp_model.CpSolverSolutionCallback):
        """Print intermediate solutions."""

        def __init__(self, vars, points, solution_limit=5):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self._vars = vars
            self._points = points
            self._solution_count = 0
            self._solution_limit = solution_limit

        def on_solution_callback(self):
            self._solution_count += 1
            print('Solution %i' % self._solution_count)

            sequence = []
            current = 0
            for _ in range(len(self._points)):
                for to in range(len(self._points)):
                    if current == to:
                        continue
                    if self.Value(self._vars[(current, to)]):
                        sequence.append(to)
                        current = to
                        break

            print(f'length: {sum(_distance(a, b) for a, b, in sliding_window(sequence, 2))}')
            G = nx.Graph()
            G.add_edges_from(sliding_window(sequence, 2))
            nx.draw(G, self._points, with_labels=True)
            plt.show()

            if self._solution_count >= self._solution_limit:
                print('Stop search after %i solutions' % self._solution_limit)
                self.StopSearch()

        def solution_count(self):
            return self._solution_count

    # Display the first five solutions.
    solution_limit = 5
    solution_printer = PartialSolutionPrinter(vars, points, solution_limit)

    solver.Solve(model, solution_printer)

    # Statistics.
    print('\nStatistics')
    print('  - conflicts      : %i' % solver.NumConflicts())
    print('  - branches       : %i' % solver.NumBranches())
    print('  - wall time      : %f s' % solver.WallTime())
    print('  - solutions found: %i' % solution_printer.solution_count())


if __name__ == '__main__':
    fname = f'beispieldaten/wenigerkrumm{input("nummer? ")}.txt'
    points = []
    with open(fname) as f:
        points = [tuple(map(float, line.split())) for line in f.readlines()]
    main(tuple(points))
