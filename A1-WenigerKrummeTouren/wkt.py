from math import atan2, degrees
import random
from typing import List, Tuple

from tabu import TabuList

# distance units: km
# angle units: degrees


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
            for from_ in range(len(outposts))]  # [from_][to]

    def _angle(self,
               p1: Tuple[float, float],
               p2: Tuple[float, float]) -> float:
        """Return the CCW angle of the flight line to the x-axis.

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

    def _genome_to_sequence(self, genome: Tuple[int, ...]) -> float:
        """Get the sequence of visited outposts from a genome."""

        current_outpost = genome[0]  # arbitrary starting point
        current_angle, longest_edge_length, longest_edge_idx, illegal_turn_idx = [
            None] * 4

        for i in range(len(self.outposts) + 1):
            angle = self._angle(
                self.outposts[current_outpost],
                self.outposts[genome[current_outpost]])
            if current_angle is not None:
                turn = current_angle - angle
                if abs(turn) > 90:
                    if illegal_turn_idx is None:
                        illegal_turn_idx = i
                    else:
                        print(f'too many illegal turns {abs(turn)} at {i=}')
                        if (abs(turn) > 180):
                            print(
                                f'{turn=} {angle=} {current_angle=} {genome=} {self.outposts[current_outpost]=} {self.outposts[genome[current_outpost]]=}')
                            exit()
                        return None
            current_angle = angle

            length = self.distance_cache[current_outpost][
                genome[current_outpost]]

            if longest_edge_length is not None:
                if length > longest_edge_length:
                    longest_edge_length = length
                    # longest edge from current_outpost to genome[current_outpost]
                    longest_edge_idx = i
            else:
                longest_edge_length = 0

            current_outpost = genome[current_outpost]

        # get sequence of indices
        current = (illegal_turn_idx or longest_edge_idx) % len(self.outposts)
        # skip longest edge if no illegal turns
        sequence: List[Tuple[float, float]] = [
            self.outposts[current]] if illegal_turn_idx else []
        for i in range(len(self.outposts)):
            current = genome[current]
            sequence.append(self.outposts[current])

        # remove longest edge next to illegal_turn_idx
        if illegal_turn_idx:
            if (self._distance(sequence[0], sequence[1])
                    > self._distance(sequence[-2], sequence[-1])):
                sequence.pop(0)
            else:
                sequence.pop(-1)

        return sequence

    def _fitness(self, genome: Tuple[int, ...]) -> float:
        """Return the fitness of a genome. Smaller is better."""
        sequence = self._genome_to_sequence(genome)
        if sequence is None:
            return float('inf')
        total_distance = 0
        for i in range(len(sequence) - 1):
            total_distance += self.distance_cache[i][i + 1]
        return total_distance

    def _mutate(self, genome: Tuple[int, ...]) -> List[int]:
        """Mutate a genome."""
        # swap two outposts
        i, j = (random.randrange(0, len(genome)) for _ in range(2))
        genome_ = list(genome)
        genome_[i], genome_[j] = genome_[j], genome_[i]
        # TODO prevent i at genome[i]
        for i in map(lambda x: x % len(genome) - 1, range(-1, len(genome) - 1)):
            if genome_[i] == i:
                print(f'swapping at {i=} of {len(genome)=}')
                genome_[i], genome_[i + 1] = genome_[i + 1], genome_[i]
        return tuple(genome_)

    def solve(self) -> List[Tuple[float, float]]:
        """Solve the WKT problem.

        Returns
        -------
        List[Tuple[float, float]]
            A list of (x, y) coordinates of the outposts in the order they should be visited.
        """
        # create a random genome
        current = list(range(len(self.outposts)))
        random.shuffle(current)
        current = tuple(current)

        no_improvement = 0
        best = current
        tabu = TabuList(default_tenure=self.tabu_tenure)

        # breed new generations
        while no_improvement < self.max_no_improvement:
            no_improvement += 1
            tabu.tick()

            while True:
                # mutate the current genome
                mutated = self._mutate(current)
                if tabu.get(mutated) > 0:
                    continue
                tabu.add(current)
                current = mutated
                break

            if self._fitness(current) > self._fitness(best):
                print('new best')
                best = current
                no_improvement = 0

        # get the sequence of outposts
        sequence = self._genome_to_sequence(best)

        return sequence
