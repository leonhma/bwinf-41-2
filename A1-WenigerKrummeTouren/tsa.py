from functools import lru_cache, reduce
from itertools import chain
from os import path
from random import choices, random
from time import time
from typing import Tuple
import numpy as np


import matplotlib.pyplot as plt


class WKT:
    outposts: Tuple[Tuple[float, float]]
    start_temperature: float
    end_temperature: float
    adjustment_factor: float
    delta_temp_stabilized_threshold: float
    mutate_mode: int
    weight_mode: int
    n_runs: int
    max_runtime: float

    def __init__(
            self,
            fp: str,
            start_temperature: float = 10,
            end_temperature: float = 100,
            adjustment_factor: float = 0.95,
            delta_temp_stabilized_threshold: float = 1,
            mutate_mode: int = 0,
            weight_mode: int = 0,
            n_runs: int = 10,
            max_runtime: float = 30):
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
        mutate_mode : int
            The mode of mutation to use, by default 0.
            0: Remove city and insert it at a random position
            1: Swap two cities
            2: Flip a random segment of the sequence
            3: Move a random segment of the sequence to a random position
        weight_mode : int
            The mode of weight calculation to use, by default 0.
            0: Uniform weight
            1: Weight by distance and illegal turns
        n_runs : int
            The number of runs to perform, by default 10
        max_runtime : float
            The maximum runtime of the algorithm in seconds, by default 30

        Note
        ----
            weight_mode is ignored if mutate_mode is 2 or 3.

        """
        self.start_temperature = start_temperature
        self.end_temperature = end_temperature
        self.adjustment_factor = adjustment_factor
        self.delta_temp_stabilized_threshold = delta_temp_stabilized_threshold
        self.mutate_mode = mutate_mode
        self.weight_mode = weight_mode
        self.n_runs = n_runs
        self.max_runtime = max_runtime

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
    def _angle(self, p1: int, p2: int, p3: int) -> float:
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
        p3 : int
            The index of the third point.

        Returns
        -------
        float
            The angle between the two points.
        """
        a = np.array(self.outposts[p1])
        b = np.array(self.outposts[p2])
        c = np.array(self.outposts[p3])

        ba = a - b
        bc = c - b

        return 180 - np.rad2deg(np.arccos(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))))

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
        weight = 0

        for i in range(1, len(sequence)):
            # angle calculation
            if i >= 2:
                if abs(self._angle(sequence[i - 2], sequence[i - 1], sequence[i])) > 90:
                    weight += 1000

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
        mutability = [1] * len(sequence)
        illegal_turn_indices = []

        for i in range(1, len(sequence)):
            # angle calculation
            if i >= 2:
                if abs(self._angle(sequence[i - 2], sequence[i - 1], sequence[i])) > 90:
                    illegal_turn_indices.append(i - 1)

            # weight calculation
            currentWeight = self._distance(sequence[i - 1], sequence[i])
            mutability[i] += currentWeight * 0.5
            mutability[i - 1] += currentWeight * 0.5

        # replace all illegal turn weights with the calculated value
        if illegal_turn_indices:
            illegal_turn_weight = sum(
                filter(lambda x: x, mutability)
            ) / len(illegal_turn_indices)

            for i in illegal_turn_indices:
                mutability[i] = illegal_turn_weight

        # double the weights of the first and last point
        # since they dont have a previous or next point
        mutability[0] = mutability[0] * 2
        mutability[-1] = mutability[-1] * 2

        return tuple(map(lambda x: x / (sum(mutability)), mutability))

    # TODO: test swapping vs removing and inserting vs random segment flips
    def _mutate(self, sequence: Tuple[int, ...],
                mutate_mode: int,
                weight_mode: int) -> Tuple[int, ...]:
        """Mutate a sequence by removing and inserting a city.

        Parameters
        ----------
        sequence : Tuple[int, ...]
            The sequence to mutate.
        mutate_mode : int
            The mode of mutation to use, by default 0.
            0: Remove city and insert it at a random position
            1: Swap two cities
            2: Flip a random segment of the sequence
            3: Move a random segment of the sequence to a random position
        weight_mode : int
            The mode of weight calculation to use, by default 0.
            0: Uniform weight
            1: Weight by distance and illegal turns

        Note
        ----
            The weight mode is ignored if mutate_mode is 2 or 3.

        Returns
        -------
        Tuple[int, ...]
            The mutated sequence.
        """
        probs = self._mutability(sequence) if weight_mode else tuple(1 for _ in sequence)
        match mutate_mode:
            case 0:
                return self._mutate_remove_insert(sequence, probs)
            case 1:
                return self._mutate_swap(sequence, probs)
            case 2:
                return self._mutate_flip(sequence)
            case 3:
                return self._mutate_move(sequence)

    def _mutate_remove_insert(self, sequence: Tuple[int, ...],
                              probs: Tuple[float, ...]) -> Tuple[int, ...]:
        s1, s2 = choices(tuple(i for i in range(len(sequence))),
                         probs, k=2)
        new_sequence = list(sequence)
        new_sequence.insert(s2, new_sequence.pop(s1))
        return tuple(new_sequence)

    def _mutate_swap(self, sequence: Tuple[int, ...], probs: Tuple[float, ...]) -> Tuple[int, ...]:
        s1, s2 = choices(tuple(i for i in range(len(sequence))), probs, k=2)
        new_sequence = list(sequence)
        new_sequence[s1], new_sequence[s2] = new_sequence[s2], new_sequence[s1]
        return tuple(new_sequence)

    def _mutate_flip(self, sequence: Tuple[int, ...]) -> Tuple[int, ...]:
        s1, s2 = choices(tuple(i for i in range(len(sequence))), k=2)
        new_sequence = list(sequence)
        new_sequence[s1:s2] = new_sequence[s2:s1:-1]
        return tuple(new_sequence)

    def _mutate_move(self, sequence: Tuple[int, ...]) -> Tuple[int, ...]:
        s1, s2, s3 = sorted(choices(tuple(i for i in range(len(sequence))), k=3))
        new_sequence = list(sequence)
        new_sequence = new_sequence[:s1] + new_sequence[s2:s3] + \
            new_sequence[s1:s2] + new_sequence[s3:]
        return tuple(new_sequence)

    def solve(self) -> Tuple[Tuple[float, float]]:
        """Return the optimized sequence.

        Returns
        -------
        Tuple[Tuple[float, float]]
            The sequence of outposts to visit.
        """

        # create initial sequence using greedy algorithm
        to_add = [i for i in range(len(self.outposts)) if i != 0]
        initial_sequence = [0]

        while to_add:
            next_ = min(
                chain(
                    (i, self._distance(initial_sequence[idx], i), add_pos)
                    for i in to_add
                    for idx, add_pos in ((0, 0), (-1, len(initial_sequence)))
                ), key=lambda x: x[1])
            initial_sequence.insert(next_[2], next_[0])
            to_add.remove(next_[0])

        best_sequence = None
        best_cost = float('inf')
        all_temps = []
        all_best_temp_idx = None
        all_costs = []
        all_best_cost_idx = None
        best_n = None
        start_time = time()

        for run_idx in range(self.n_runs):
            if time() - start_time > self.max_runtime:
                break

            sequence = tuple(initial_sequence)

            temps = []
            costs = []
            n = 0

            # optimize using simulated annealing
            temp = self.start_temperature
            delta_temp = float('inf')
            delta_cost_total = 0
            delta_entropy_total = 0
            current_cost = self._cost(sequence)

            while (temp > self.end_temperature or
                   delta_temp > self.delta_temp_stabilized_threshold) \
                    and time() - start_time < self.max_runtime:
                # -----
                temps.append(temp)
                costs.append(current_cost)
                n += 1
                # -----

                new_sequence = self._mutate(sequence, self.mutate_mode, self.weight_mode)
                new_cost = self._cost(new_sequence)
                delta_cost = new_cost - current_cost
                prob = 1 if delta_cost < 0 else np.exp(-delta_cost / temp)

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

            all_temps.append(temps)
            all_costs.append(costs)

            if current_cost < best_cost:
                best_cost = current_cost
                best_sequence = sequence
                all_best_temp_idx = run_idx
                all_best_cost_idx = run_idx
                best_n = n

        points = tuple(self.outposts[i] for i in best_sequence)
        fig, axs = plt.subplot_mosaic([["graph", "graph"], ["graph", "graph"], ["temp", "cost"]],
                                      constrained_layout=True)
        fig.suptitle(f"Thermodynamic Simulated Annealing (n={best_n})")
        axs['graph'].set_title("Graph")
        axs['graph'].plot(tuple(x[0] for x in points), tuple(x[1] for x in points), 'bo-')

        def reduce_fn(acc, x):
            coords, last, llast = acc
            if last and llast:
                if abs(self._angle(llast, last, x)) > 90:
                    coords.append(last)

            return coords, x, last
        to_highlight = [self.outposts[i] for i in reduce(reduce_fn, best_sequence, ([], None, None))[0]]
        axs['graph'].plot(tuple(x[0] for x in to_highlight), tuple(x[1] for x in to_highlight),
                          'ro')

        axs['temp'].set_title("Temperature")
        for temps in all_temps:
            axs['temp'].plot(temps, '--', color='0.7', zorder=0.5)
        axs['temp'].plot(all_temps[all_best_temp_idx], '-', color='blue', zorder=1)
        axs['cost'].set_title("Cost")
        for costs in all_costs:
            axs['cost'].plot(costs, '--', color='0.7', zorder=0.5)
        axs['cost'].plot(all_costs[all_best_cost_idx], '-', color='blue', zorder=1)

        plt.show(block=True)
        return tuple(self.outposts[i] for i in best_sequence)


wkt = WKT('beispieldaten/wenigerkrumm4.txt',
          start_temperature=100,
          end_temperature=1,
          adjustment_factor=0.95,
          delta_temp_stabilized_threshold=0.5,
          mutate_mode=1,
          weight_mode=0,
          n_runs=5,
          max_runtime=30)
solution = wkt.solve()
