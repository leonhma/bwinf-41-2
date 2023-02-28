from copy import deepcopy
from functools import lru_cache, reduce
from itertools import chain
from math import atan2, degrees, exp
from os import path
from random import choice, choices, random, shuffle
from typing import Tuple


import matplotlib.pyplot as plt


class WKT:
    outposts: Tuple[Tuple[float, float]]
    start_temperature: float
    end_temperature: float
    adjustment_factor: float
    delta_temp_stabilized_threshold: float
    # TODO maybe multiple runs / restart?

    def __init__(
            self,
            fp: str,
            start_temperature: float = 10,
            end_temperature: float = 100,
            adjustment_factor: float = 0.95,
            delta_temp_stabilized_threshold: float = 1):
        """Initialize the WKT class.

        Parameters
        ----------
        fp : str
            The path to the file containing the outposts.
        start_temperature : float, optional
            The initial temperature of the simulated annealing algorithm, by default 10000
        end_temperature : float, optional
            The final temperature of the simulated annealing algorithm, by default 100
        adjustment_factor : float, optional
            The factor by which the temperature is adjusted each iteration, by default 0.95
        delta_temp_stabilized_threshold : float, optional
            The threshold for the temperature change to be considered stable, by default 1
        """
        self.start_temperature = start_temperature
        self.end_temperature = end_temperature
        self.adjustment_factor = adjustment_factor
        self.delta_temp_stabilized_threshold = delta_temp_stabilized_threshold

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
                    weight += 999
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

        temps = []
        costs = []
        n = 0

        # optimize using simulated annealing
        temp = self.end_temperature + 10
        delta_temp = float('inf')
        delta_cost_total = 0
        delta_entropy_total = 0
        current_cost = self._cost(sequence)

        while temp > self.end_temperature or delta_temp > self.delta_temp_stabilized_threshold:
            # -----
            temps.append(temp)
            costs.append(current_cost)
            n += 1
            # -----

            new_sequence = self._mutate(sequence)
            new_cost = self._cost(new_sequence)
            delta_cost = new_cost - current_cost
            prob = 1 if delta_cost < 0 else exp(-delta_cost / temp)

            if random() < prob:
                sequence = new_sequence
                current_cost = new_cost
                delta_cost_total += delta_cost
            if delta_cost > 0:
                delta_entropy_total -= delta_cost / temp
            if delta_cost_total >= 0 or delta_entropy_total == 0:
                delta_temp = temp - self.start_temperature
                temp = self.start_temperature
            else:
                new_temp = self.adjustment_factor * (delta_cost_total / delta_entropy_total)
                delta_temp = temp - new_temp
                temp = new_temp

        points = tuple(self.outposts[i] for i in sequence)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.plot(tuple(x[0] for x in points), tuple(x[1] for x in points), 'bo-')

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
        ax1.plot(tuple(x[0] for x in to_highlight), tuple(x[1] for x in to_highlight), 'ro')

        ax3.plot(temps, range(n))
        ax4.plot(costs, range(n))

        plt.show(block=True)
        return tuple(self.outposts[i] for i in sequence)


wkt = WKT('beispieldaten/wenigerkrumm2.txt')
solution = wkt.solve()
