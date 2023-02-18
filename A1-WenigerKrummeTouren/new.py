from functools import lru_cache, reduce
from math import atan2, degrees
from os import path
from random import choice, choices
from typing import Tuple


import matplotlib.pyplot as plt


ILLEGAL_TURN_WEIGHT = 1000

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
    max_n_of_its_wo_improv: int
    explore_its: int

    def __init__(
            self,
            fp: str, max_n_of_its_wo_improv: int = 1000,
            explore_its: int = 30):
        """Initialize the WKT class.

        Parameters
        ----------
        fp : str
            The path to the file containing the outposts.
        max_n_of_its_wo_improv : int, optional
            The maximum number of iterations without improvement, by default 1000
        explore_its : int, optional
            The number of iterations to further explore infeasible solutions for, by default 30
        """
        self.max_n_of_its_wo_improv = max_n_of_its_wo_improv
        self.explore_its = explore_its

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

    def _mutability(self, genome: Tuple[int, ...]) -> Tuple[float, ...]:
        """Return the mutation probability of each point in the genome.

        Parameters
        ----------
        genome : Tuple[int, ...]
            The genome to evaluate.

        Returns
        -------
        Tuple[float, ...]
            The mutation probability of each point in the genome. (normalized to a sum of 1)
        """
        lastAngle = None
        mutability = [1] * len(genome)

        for i in range(1, len(genome)):
            # angle calculation
            angle = self._angle(genome[i - 1], genome[i])
            if lastAngle is not None:
                currentAngle = (lastAngle - angle + 180) % 360 - 180
                if abs(currentAngle) > 90:
                    mutability[i - 1] = ILLEGAL_TURN_WEIGHT
            lastAngle = angle

            # weight calculation
            currentWeight = self._distance(genome[i - 1], genome[i])
            mutability[i] += currentWeight * 0.5
            mutability[i - 1] += currentWeight * 0.5

        # double the weights of the first and last point
        # since they dont have a previous or next point
        mutability[0] = mutability[0] * 2
        mutability[-1] = mutability[-1] * 2

        return tuple(map(lambda x: x / (sum(mutability)), mutability))

    def _cost(self, genome: Tuple[int]) -> float:
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
                    weight += ILLEGAL_TURN_WEIGHT
            lastAngle = angle

            # weight calculation
            weight += self._distance(genome[i - 1], genome[i])

        return weight

    def _mutate(self, genome: Tuple[int, ...]) -> Tuple[int, ...]:
        """Mutate a genome.

        Parameters
        ----------
        genome : Tuple[int, ...]
            The genome to mutate.

        Returns
        -------
        Tuple[int, ...]
            The mutated genome.
        """
        [s1] = choices(tuple(i for i in range(len(genome))),
                       self._mutability(genome), k=1)
        s2 = choice(tuple(i for i in range(len(genome))))
        new_genome = list(genome)
        new_genome[s1], new_genome[s2] = new_genome[s2], new_genome[s1]
        return tuple(new_genome)

    def _cooldown(self, temp: float) -> float:
        return temp * self.cooldown_multiplier

    def solve(self) -> Tuple[Tuple[float, float]]:
        """Return the optimized sequence.

        Returns
        -------
        Tuple[Tuple[float, float]]
            The sequence of outposts to visit.
        """
        n_of_its_wo_improv = 0
        best = tuple(i for i in range(len(self.outposts)))
        best_cost = self._cost(best)
        while n_of_its_wo_improv < self.max_n_of_its_wo_improv:
            current = best
            for _ in range(self.explore_its):
                current = self._mutate(current)
                current_cost = self._cost(current)
                if current_cost < best_cost:
                    best = current
                    best_cost = current_cost
                    print(f'found better solution with cost {best_cost}')
                    n_of_its_wo_improv = 0
                    show_plot(tuple(self.outposts[i] for i in best), False)
                    break
            n_of_its_wo_improv += 1

        return tuple(self.outposts[i] for i in best)


wkt = WKT('beispieldaten/wenigerkrumm1.txt', 1000, 60)
solution = wkt.solve()

show_plot(solution)
