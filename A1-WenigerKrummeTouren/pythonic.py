from math import atan2, degrees
import random
from typing import Iterable, List, Tuple

from utils import TabuList, BestList, index_it


class WKT:
    """"Weniger Krumme Touren"

    Create an optimized path visiting all outposts without turning over 90° using heuristics.
    """
    outposts: List[Tuple[float, float]]
    tabu_tenure: int
    max_no_improvement: int

    distance_cache: List[List[float]]

    def __init__(
            self, outposts: List[Tuple[float, float]],
            tabu_tenure: int = 20, max_no_improvement: int = 1000):
        """
        Initialize the WKT class.

        Parameters
        ----------
        outposts : List[Tuple[float, float]]
            A list of (x, y) coordinates of the outposts.
        tabu_tenure : int, optional
            The number of generations to ignore previously visited genomes for, by default 20
        max_no_improvement : int, optional
            The maximum number of iterations without improvement, by default 1000
        """
        self.outposts = outposts
        self.tabu_tenure = tabu_tenure
        self.max_no_improvement = max_no_improvement

        self.distance_cache = [
            [self._distance(outposts[from_],
                            self.outposts[to])
             for to in range(len(self.outposts))]
            for from_ in range(len(self.outposts))]  # [from_][to]

    def _angle(self,
               p1: Tuple[float, float],
               p2: Tuple[float, float]) -> float:
        """Return the CCW angle of the flight line to the x-axis. ([-180, 180])

            Hint
            ----
                Subtracting two of these values gives the turn angle.
            """
        return degrees(atan2(p2[1] - p1[1], abs(p2[0] - p1[0])))

    def _distance(self,
                  p1: Tuple[float, float],
                  p2: Tuple[float, float]) -> float:
        """Return the distance between two points."""
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def _fitness(self, genome: Tuple[int]) -> float:
        """Return the fitness of a genome."""
        return sum(self.distance_cache[genome[i]][genome[i + 1]]
                   for i in range(len(genome) - 1))

    def _genome_to_sequence(self, genome: Tuple[int]) -> Tuple[Tuple[int, int]]:
        """Return the sequence of outposts to visit."""
        return tuple(self.outposts[i] for i in genome)
    
    def _mutate(self, genome: Tuple[int, ...]) -> Tuple[int]:
        """Mutate a genome."""
        # swap two outposts
        i, j = (random.randrange(0, len(genome)) for _ in range(2))
        genome_ = list(genome)
        genome_[i], genome_[j] = genome_[j], genome_[i]
        for i in map(
                lambda x: x % len(genome) - 1, range(-1, len(genome) - 1)):
            if genome_[i] == i:
                genome_[i], genome_[i + 1] = genome_[i + 1], genome_[i]
        return tuple(genome_)

    def solve(self) -> Tuple[Tuple[int, int], float]:
        """Solve the WKT problem."""
        current = list(range(1, len(self.outposts) + 1))
        current[-1] = 0
        current = tuple(current)

        no_improvement = 0
        best = BestList(
            n=self.tabu_tenure + 100,
            sorting_key=self._fitness)
        best.add(current)
        best_fitness = None
        tabu = TabuList(default_tenure=self.tabu_tenure)

        # breed new generations
        while no_improvement < self.max_no_improvement:
            no_improvement += 1
            tabu.tick()

            # mutate the current genome
            new = self._mutate(
                index_it(
                    filter(
                        lambda x: not tabu.get(x),
                        best),
                    0))

            tabu.add(new)
            best.add(new)
            new_fitness = self._fitness(new)
            if not best_fitness or best_fitness > new_fitness:
                no_improvement = 0
                best_fitness = new_fitness

        # get the sequence of outposts
        print('done')
        return self._genome_to_sequence(best[0])
    


