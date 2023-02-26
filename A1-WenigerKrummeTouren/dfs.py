from functools import lru_cache, reduce
from math import atan2, degrees, factorial
from os import path
from random import choice, choices
from typing import List, Tuple


import matplotlib.pyplot as plt
from tqdm import tqdm


def show_plot(points: Tuple[Tuple[float, float]], block: bool = True):
    """Show a plot of the given points.

    Parameters
    ----------
    points : Tuple[Tuple[float, float]]
        The points to plot.
    """
    plt.clf()
    plt.plot(tuple(x[0] for x in points), tuple(x[1] for x in points), f'{"b" if block else "y"}o-')

    def reduce_fn(acc, x):
        coords, last, last_angle = acc
        if last is not None:
            angle = degrees(atan2(x[1] - last[1], x[0] - last[0]))
        else:
            return coords, x, None
        if last_angle is not None:
            current_angle = (last_angle - angle + 180) % 360 - 180
            if abs(current_angle) > 90:
                coords.append(last)
        return coords, x, angle
    to_highlight = reduce(reduce_fn, points, ([], None, None))[0]
    plt.plot(tuple(x[0] for x in to_highlight), tuple(x[1] for x in to_highlight), 'ro')
    plt.pause(0.001)
    plt.show(block=block)


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

    def _fitness(self, genome: Tuple[int]) -> float:
        """Return the fitness of a genome. (lower is better)

        Parameters
        ----------
        genome : Tuple[int]
            The genome to evaluate.

        Returns
        -------
        float
            The fitness of the genome.
        """
        lastAngle = None
        weight = 0

        for i in range(1, len(genome)):
            # angle calculation
            angle = self._angle(genome[i - 1], genome[i])
            if lastAngle is not None:
                currentAngle = (lastAngle - angle + 180) % 360 - 180
                if abs(currentAngle) > 90:
                    return 0
            lastAngle = angle

            # weight calculation
            weight += self._distance(genome[i - 1], genome[i])

        return 1 / weight
    
    
    def solve(self) -> Tuple[Tuple[float, float]]:
        """Return the optimized sequence.

        Returns
        -------
        Tuple[Tuple[float, float]]
            The sequence of outposts to visit.
        """
        with tqdm(total=factorial(len(self.outposts))) as pbar:
            def dfs(stack: List):
                if len(stack) == len(self.outposts):
                    pbar.update()
                    return stack
                return max((dfs(stack + [i]) for i in range(len(self.outposts)) if i not in stack),
                           key=self._fitness)

            return tuple(self.outposts[i] for i in dfs([]))


wkt = WKT('beispieldaten/wenigerkrumm4.txt')
solution = wkt.solve()

show_plot(solution)
