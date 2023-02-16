from functools import lru_cache
from math import atan2, degrees
from os import path
from random import choices
from typing import Tuple

from utils import BestList

import matplotlib.pyplot as plt


ILLEGAL_TURN_WEIGHT = 999999


class WKT:
    outposts: Tuple[Tuple[float, float]]
    max_n_of_its_wo_improv: int
    top_n: int

    def __init__(
            self, fp: str, max_n_of_its_wo_improv: int = 1000, top_n: int = 10):
        """Initialize the WKT class.

        Parameters
        ----------
        fp : str
            The path to the file containing the outposts.
        max_n_of_its_wo_improv : int, optional
            The maximum number of iterations without improvement, by default 10000.
        top_n : int, optional
            The number of top genomes to keep track of, by default 10.
        """
        self.max_n_of_its_wo_improv = max_n_of_its_wo_improv
        self.top_n = top_n
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
        """Return the CCW angle of the flight line to the x-axis. ([-180, 180])

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
                abs(self.outposts[p2][0] - self.outposts[p1][0])))

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
                currentAngle = abs(angle - lastAngle)
                if currentAngle > 90:
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

    def _fitness(self, genome: Tuple[int]) -> float:
        """Return the fitness of a genome.

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
                currentAngle = abs(angle - lastAngle)
                if currentAngle > 90:
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
        [s2] = choices(tuple(i for i in range(len(genome))), tuple(
            map(lambda x: 1 / x, self._mutability(genome))), k=1)
        new_genome = list(genome)
        new_genome[s1], new_genome[s2] = new_genome[s2], new_genome[s1]
        return tuple(new_genome)

    def solve(self) -> Tuple[Tuple[float, float]]:
        """Return the optimized sequence.

        Returns
        -------
        Tuple[Tuple[float, float]]
            The sequence of outposts to visit.
        """
        n_of_its_wo_improvement = 0
        best = BestList(self.top_n, self._fitness)
        best.add(tuple(i for i in range(len(self.outposts))))

        while n_of_its_wo_improvement < self.max_n_of_its_wo_improv:
            n_of_its_wo_improvement += 1
            for genome in best:
                new_genome = self._mutate(genome)
                if self._fitness(new_genome) < self._fitness(genome):
                    best.add(new_genome)
                    n_of_its_wo_improvement = 0
                    print(
                        f'found better genome. new fitness {self._fitness(new_genome)}')

        return tuple(self.outposts[i] for i in best[0])


wkt = WKT('beispieldaten/wenigerkrumm2.txt')
solution = wkt.solve()

plt.plot(tuple(x[0] for x in solution), tuple(x[1] for x in solution), 'o-')
plt.show()
