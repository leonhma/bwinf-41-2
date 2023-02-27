from functools import lru_cache
from itertools import chain
from math import atan2, copysign, degrees
from os import path
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx


class Counter:
    """A custom counter class."""
    n: int
    counter: List[int]
    last_increment_idx: int
    needs_increment: bool

    def __init__(self, n: int):
        self.n = n
        self.counter = [0] * n
        self.last_increment_idx = 0
        self.needs_increment = False

    def increment(self, idx: int = None):
        """Increment the counter.

        Parameters
        ----------
        idx : int, optional
            The index to increment, by default `-1`.
        """
        if idx is None:
            idx = self.n - 1
        else:
            self.counter = self.counter[:idx + 1] + [0] * (self.n - idx - 1)

        self.counter[idx] += 1
        self.last_increment_idx = idx

        for i in range(idx, -1, -1):
            if self.counter[i] >= (self.n - i):
                self.counter[i] = 0
                if i > 0:
                    self.counter[i - 1] += 1
                    self.last_increment_idx = i - 1
                else:
                    raise StopIteration
            else:
                break

    def carry(self):
        self.increment(self.last_increment_idx)
        self.counter = self.counter[:self.last_increment_idx + 1] \
            + [0] * (self.n - self.last_increment_idx - 1)
        self.needs_increment = False

    def __next__(self):
        if self.needs_increment:
            self.increment()
        self.needs_increment = True
        return tuple(self.counter)


class WKT:
    outposts: Tuple[Tuple[float, float]]

    def __init__(
            self,
            fp: str):
        """Initialize the WKT class.

        Parameters
        ----------
        fp : str
            The path to the file containing the outposts.
        """

        with open(path.join(
            path.dirname(path.abspath(__file__)),
            fp
        ), 'r') as f:
            self.outposts = tuple(tuple(map(float, line.split()))
                                  for line in f.readlines())  # load outposts from file

    def show(self, points: Tuple[int, ...], other:  Tuple[int, ...] = ()):
        # TODO highlight sharp angles
        plt.clf()
        G = nx.Graph()
        G.add_edges_from([(points[i], points[i+1]) for i in range(len(points)-1)])
        G.add_nodes_from(other)
        pos = {i: self.outposts[i] for i in range(len(self.outposts))}
        nx.draw(G, pos, with_labels=True, node_color='r', node_size=500, edge_color='b')
        plt.show(block=False)
        plt.pause(0.1)

    @lru_cache(maxsize=None)    # cache the results of this function for speeed
    def _distance(self, p1: int, p2: int) -> float:
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
            (self.outposts[p1][0] - self.outposts[p2][0]) ** 2 +
            (self.outposts[p1][1] - self.outposts[p2][1]) ** 2) ** 0.5

    @lru_cache(maxsize=None)
    def _angle(self, p1: int, p2: int) -> float:
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

        Returns
        -------
        float
            The angle between the two points.
        """
        return degrees(
            atan2(
                self.outposts[p2][1] - self.outposts[p1][1],
                self.outposts[p2][0] - self.outposts[p1][0]))

    @lru_cache(maxsize=None)
    def get_next(self,
                 sequence: Tuple[int, ...],
                 available: List[int]
                 ) -> Tuple[Tuple[int, float], ...]:
        return tuple(
            sorted(
                filter(lambda x: -90 <= ((x[3] + 180) % 360 - 180) <= 90, (
                    (i,
                     add_pos,
                     self._distance(sequence[idx], i) if sequence else float('inf'),
                     self._angle(sequence[idx + copysign(1, idx)]) - self._angle(sequence[idx], i)
                     if len(sequence > 1) else 0)
                    for i in available
                    for idx, add_pos in ((0, 0), (-1, len(sequence))))
                ), key=lambda x: x[2]))


    def solve(self) -> Tuple[Tuple[float, float]]:
        """Return the optimized sequence.

        Returns
        -------
        Tuple[Tuple[float, float]]
            The sequence of outposts to visit.
        """
        c = Counter(len(self.outposts))

        sequence = []

        while True:
            try:
                idx = next(c)
            except StopIteration:
                break

            sequence = []
            available = list(range(len(self.outposts)))
            weight = 0

            for i in idx:
                nexts = self.get_next(tuple(sequence), tuple(available))
                if not nexts:
                    break
                next_, add_pos, distance, _ = nexts[i]
                sequence.insert(add_pos, next_)
                available.remove(next_)
                weight += distance

            self.show(sequence, available)


        


wkt = WKT('beispieldaten/wenigerkrumm2.txt')
solution = wkt.solve()

plt.pause(100)
