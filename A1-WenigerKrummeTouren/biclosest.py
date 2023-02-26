from functools import lru_cache, reduce
from itertools import chain
from math import atan2, degrees
from os import path
from random import choice, choices
import time
from typing import Tuple
import networkx as nx


import matplotlib.pyplot as plt

class WKT:
    outposts: Tuple[Tuple[float, float]]
    max_n_of_its_wo_improv: int
    explore_its: int

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

    def solve(self) -> Tuple[Tuple[float, float]]:
        """Return the optimized sequence.

        Returns
        -------
        Tuple[Tuple[float, float]]
            The sequence of outposts to visit.
        """
        to_add = [i for i in range(len(self.outposts)) if i != 0]
        sequence = [0]

        while to_add:
            self.show(sequence, to_add)
            next_ = min(chain((i, self._distance(sequence[idx], i), idx) for i in to_add for idx in (0, -1)), key=lambda x: x[1])
            print(next_)
            sequence.insert(next_[2], next_[0])
            to_add.remove(next_[0])
            
        return tuple(self.outposts[i] for i in sequence)


wkt = WKT('beispieldaten/wenigerkrumm2.txt')
solution = wkt.solve()

plt.pause(100)
