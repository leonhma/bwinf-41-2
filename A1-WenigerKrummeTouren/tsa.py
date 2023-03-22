from functools import lru_cache, reduce
from itertools import chain
from os import path
from random import choices, random
from time import time
from typing import Tuple
import numpy as np


import matplotlib.pyplot as plt


class WKT:
    max_runtime: float
    outposts: Tuple[Tuple[float, float]]

    def __init__(
            self,
            fp: str,
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

        return 180 - np.rad2deg(np.arccos(np.dot(ba, bc) /
                                          (np.linalg.norm(ba) * np.linalg.norm(bc))))

    def to_outposts(self, sequence: Tuple[int, ...]) -> Tuple[Tuple[int, int]]:
        print(f'converted sequence {sequence} to outposts')
        return tuple(self.outposts[i] for i in sequence)

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
                if self._angle(sequence[i - 2], sequence[i - 1], sequence[i]) > 90:
                    weight += 700 * len(sequence)

            # weight calculation
            weight += self._distance(sequence[i - 1], sequence[i])

        return weight

    def _contains_illegal_turn(self, sequence: Tuple[int, ...]) -> bool:
        for i in range(2, len(sequence)):
            if self._angle(sequence[i - 2], sequence[i - 1], sequence[i]) > 90:
                return True
        return False

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
                if self._angle(sequence[i - 2], sequence[i - 1], sequence[i]) > 90:
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
        if mutate_mode == 0:
            return self._mutate_remove_insert(sequence, probs)
        elif mutate_mode == 1:
            return self._mutate_swap(sequence, probs)
        elif mutate_mode == 2:
            return self._mutate_flip(sequence)
        elif mutate_mode == 3:
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

    def create_initial_biclosest(self) -> Tuple[int, ...]:
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

        return tuple(initial_sequence)

    def optimize(self,
                 start_temperature: float,
                 end_temperature: float,
                 adjustment_factor: float,
                 delta_temp_stabilized_threshold: float,
                 mutate_mode: int,
                 weight_mode: int,
                 sequence: Tuple[int, ...] = ()) -> Tuple[Tuple[float, float]]:
        """Return the optimized sequence.

        Returns
        -------
        Tuple[Tuple[float, float]]
            The sequence of outposts to visit.
        """
        start_time = time()

        # optimize using simulated annealing
        temp = start_temperature
        delta_temp = float('inf')
        delta_cost_total = 0
        delta_entropy_total = 0
        current_cost = self._cost(sequence)

        best_sequence = sequence
        best_cost = current_cost

        while (temp > end_temperature or
               delta_temp > delta_temp_stabilized_threshold) \
                and time() - start_time < self.max_runtime:
            new_sequence = self._mutate(sequence, mutate_mode, weight_mode)
            new_cost = self._cost(new_sequence)
            delta_cost = new_cost - current_cost
            prob = 1 if delta_cost < 0 else np.exp(-delta_cost / temp)
            if random() < prob:
                sequence = new_sequence
                current_cost = new_cost
                delta_cost_total += delta_cost
            if new_cost < best_cost and not self._contains_illegal_turn(new_sequence):
                best_cost = new_cost
                best_sequence = sequence
            if delta_cost > 0:
                delta_entropy_total -= delta_cost / temp
            if delta_cost_total >= 0 or delta_entropy_total == 0:
                delta_temp = temp - start_temperature
                temp = start_temperature
            else:
                new_temp = adjustment_factor * (delta_cost_total / delta_entropy_total)
                delta_temp = temp - new_temp
                temp = new_temp
        if current_cost < best_cost:
            best_cost = current_cost
            best_sequence = sequence

        return best_sequence


def run(example: int):
    wkt = WKT(f'beispieldaten/wenigerkrumm{example}.txt',
              max_runtime=30)

    initial_sequence = wkt.create_initial_biclosest()
    print(f'Initial sequence cost: {wkt._cost(initial_sequence)}')

    solution = None
    start_time = time()

    while time() - start_time < 30:
        if solution is not None:
            break
        flip_optimized = wkt.optimize(100, 1, 1, 0.1, 2, 1, initial_sequence)
        print(f'Flip optimized sequence cost: {wkt._cost(flip_optimized)}')
        if not wkt._contains_illegal_turn(flip_optimized):
            solution = flip_optimized

        # solution = wkt.optimize(1, 1, 0.95, 1, 1, 0, initial_sequence)
        # print(f'Swap optimized sequence cost: {wkt._cost(solution)}')
    if not solution:
        print('No solution found')
        return
    return wkt.to_outposts(solution)

# points = tuple(self.outposts[i] for i in best_sequence)
# fig, axs = plt.subplot_mosaic([["graph", "graph"], ["graph", "graph"], ["temp", "cost"]],
#                               constrained_layout=True)
# fig.suptitle(f"Thermodynamic Simulated Annealing (n={best_n})")
# axs['graph'].set_title("Graph")
# axs['graph'].plot(tuple(x[0] for x in points), tuple(x[1] for x in points), 'bo-')
# def reduce_fn(acc, x):
#     coords, last, llast = acc
#     if last and llast:
#         if abs(self._angle(llast, last, x)) > 90:
#             coords.append(last)
#     return coords, x, last
# to_highlight = [self.outposts[i] for i in reduce(reduce_fn, best_sequence, ([], None, None))[0]]
# axs['graph'].plot(tuple(x[0] for x in to_highlight), tuple(x[1] for x in to_highlight),
#                   'ro')
# axs['temp'].set_title("Temperature")
# for temps in all_temps:
#     axs['temp'].plot(temps, '--', color='0.7', zorder=0.5)
# axs['temp'].plot(all_temps[all_best_temp_idx], '-', color='blue', zorder=1)
# axs['cost'].set_title("Cost")
# for costs in all_costs:
#     axs['cost'].plot(costs, '--', color='0.7', zorder=0.5)
# axs['cost'].plot(all_costs[all_best_cost_idx], '-', color='blue', zorder=1)
# plt.show(block=True)


while True:
    try:
        solution = run(input('Nummer des Beispiels: > '))
        if solution is not None:
            print(len(solution))
    except KeyboardInterrupt:
        print('Stopped')
        break
