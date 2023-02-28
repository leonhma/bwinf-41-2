from copy import deepcopy
from functools import lru_cache, reduce
from itertools import chain
from math import atan2, degrees, exp
from os import path
from random import choice, choices, random, shuffle
from typing import Tuple


import matplotlib.pyplot as plt


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
    start_temperature: float
    restart_after_n_its_wo_improv: int
    max_restart_n: int

    def __init__(
            self,
            fp: str,
            start_temperature: float = 10000,
            restart_after_n_its_wo_improv: int = 1000,
            max_restart_n: int = 10):
        """Initialize the WKT class.

        Parameters
        ----------
        fp : str
            The path to the file containing the outposts.
        start_temperature : float, optional
            The initial temperature of the simulated annealing algorithm, by default 10000
        restart_after_n_its_wo_improv : int, optional
            The number of iterations without improvement after which the algorithm restarts,
            by default 1000
        max_restart_n : int, optional
            The maximum number of restarts, by default 10
        """
        self.start_temperature = start_temperature
        self.restart_after_n_its_wo_improv = restart_after_n_its_wo_improv
        self.max_restart_n = max_restart_n

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

    def _cost(self, sequence: Tuple[int]) -> float:
        """Return the fitness of a sequence. (lower is better)

        Parameters
        ----------
        sequence : Tuple[int]
            The sequence to evaluate.

        Returns
        -------
        float
            The fitness of the sequence.
        """
        lastAngle = None
        weight = 0

        for i in range(1, len(sequence)):
            # angle calculation
            angle = self._angle(sequence[i - 1], sequence[i])
            if lastAngle is not None:
                currentAngle = (lastAngle - angle + 180) % 360 - 180
                if abs(currentAngle) > 90:
                    weight += 1000000
            lastAngle = angle

            # weight calculation
            weight += self._distance(sequence[i - 1], sequence[i])

        return weight

    def _mutability(self, sequence: Tuple[int, ...]) -> Tuple[float, ...]:
        """Return the mutation probability of each point in the sequence.

        Parameters
        ----------
        sequence : Tuple[int, ...]
            The sequence to evaluate.

        Returns
        -------
        Tuple[float, ...]
            The mutation probability of each point in the sequence. (normalized to a sum of 1)
        """
        lastAngle = None
        mutability = [1] * len(sequence)
        illegal_turn_indices = []

        for i in range(1, len(sequence)):
            # angle calculation
            angle = self._angle(sequence[i - 1], sequence[i])
            if lastAngle is not None:
                currentAngle = (lastAngle - angle + 180) % 360 - 180
                if abs(currentAngle) > 90:
                    illegal_turn_indices.append(i - 1)
            lastAngle = angle

            # weight calculation
            currentWeight = self._distance(sequence[i - 1], sequence[i])
            mutability[i] += currentWeight * 0.5
            mutability[i - 1] += currentWeight * 0.5

        # replace all illegal turn weights with the calculated value
        illegal_turn_weight = sum(
            filter(lambda x: x, mutability)
        )
        for i in illegal_turn_indices:
            mutability[i] = illegal_turn_weight

        # double the weights of the first and last point
        # since they dont have a previous or next point
        mutability[0] = mutability[0] * 2
        mutability[-1] = mutability[-1] * 2

        return tuple(map(lambda x: x / (sum(mutability)), mutability))

    # TODO: test swapping vs removing and inserting vs random segment flips
    def _mutate(self, sequence: Tuple[int, ...]) -> Tuple[int, ...]:
        """Mutate a sequence by removing and inserting a city.

        Parameters
        ----------
        sequence : Tuple[int, ...]
            The sequence to mutate.

        Returns
        -------
        Tuple[int, ...]
            The mutated sequence.
        """
        [s1] = choices(tuple(i for i in range(len(sequence))),
                       self._mutability(sequence), k=1)
        s2 = choice(tuple(i for i in range(len(sequence) + 1)))
        new_sequence = list(sequence)
        new_sequence.insert(s2, new_sequence.pop(s1))
        return tuple(new_sequence)

    def solve(self) -> Tuple[Tuple[float, float]]:
        """Return the optimized sequence.

        Returns
        -------
        Tuple[Tuple[float, float]]
            The sequence of outposts to visit.
        """

        # create initial sequence using greedy algorithm
        # to_add = [i for i in range(len(self.outposts)) if i != 0]
        # sequence = [0]

        # while to_add:
        #     next_ = min(
        #         chain(
        #             (i, self._distance(sequence[idx], i), add_pos)
        #             for i in to_add
        #             for idx, add_pos in ((0, 0), (-1, len(sequence)))
        #         ), key=lambda x: x[1])
        #     sequence.insert(next_[2], next_[0])
        #     to_add.remove(next_[0])

        sequence = list(range(len(self.outposts)))
        shuffle(sequence)

        # optimize using simulated annealing
        global_best = tuple(sequence)
        global_best_cost = self._cost(global_best)

        n_restart = 0
        restarting = True
        while restarting and n_restart < self.max_restart_n:
            restarting = False

            temp = self.start_temperature
            current = deepcopy(sequence)
            current_cost = self._cost(current)
            n_of_its_wo_improvement = 0
            while temp > 0:
                new_sequence = self._mutate(sequence)
                new_cost = self._cost(new_sequence)
                prob = 1 if new_cost < current_cost else exp((current_cost - new_cost) / temp)
                if random() < prob:
                    current = new_sequence
                    current_cost = new_cost
                    if current_cost < global_best_cost:
                        global_best = current
                        global_best_cost = current_cost
                n_of_its_wo_improvement += 1
                if n_of_its_wo_improvement > self.restart_after_n_its_wo_improv:
                    print('restarting')
                    n_restart += 1
                    restarting = True
                    break
                temp = self._cooldown(temp)

        return tuple(self.outposts[i] for i in global_best)


wkt = WKT('beispieldaten/wenigerkrumm4.txt', 10000, 10000, 10)
solution = wkt.solve()

show_plot(solution)
