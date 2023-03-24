from functools import lru_cache
from os import path
from typing import Callable, List, Tuple
from random import random, shuffle, choices

import numpy as np

from utils import sliding_window, ilist

import networkx as nx
import matplotlib.pyplot as plt


class WKT:
    outposts: Tuple[Tuple[float, float]]
    constraint_factor: float
    n_shots: int
    initial_temperature: float
    cooldown_fn: Callable[[float, float, int], float]
    end_temperature: float

    def __init__(
            self,
            fp: str,
            constraint_factor: float = 1,
            n_shots: int = 10,
            initial_temperature: float = 100,
            cooldown_fn: Callable[[float, float, int], float] = lambda t, t0, i: t0 / i,
            end_temperature: float = 0.1):

        self.constraint_factor = constraint_factor
        self.n_shots = n_shots
        self.initial_temperature = initial_temperature
        self.cooldown_fn = cooldown_fn
        self.end_temperature = end_temperature

        with open(path.join(
            path.dirname(path.abspath(__file__)), 'beispieldaten',
            fp
        ), 'r') as f:
            self.outposts = tuple(tuple(map(float, line.split()))
                                  for line in f.readlines())  # load outposts from file

    @lru_cache(maxsize=None)    # cache the results of this function for speeed
    def _distance(self, p1: int, p2: int) -> float:
        a1, a2 = self.outposts[p1]
        b1, b2 = self.outposts[p2]
        return (
            (b1 - a1) ** 2 +
            (b2 - a2) ** 2) ** 0.5

    @lru_cache(maxsize=None)
    def _angle(self, p1: int, p2: int, p3: int) -> float:
        a = np.array(self.outposts[p1])
        b = np.array(self.outposts[p2])
        c = np.array(self.outposts[p3])

        ba = a - b
        bc = c - b

        return 180 - np.rad2deg(np.arccos(np.dot(ba, bc) /
                                          np.linalg.norm(ba) / np.linalg.norm(bc)))

    def _get_shortest_path_and_cost(self, sequence: Tuple[int]) -> Tuple[Tuple[int, ...], float, bool]:  # sequence length is_legal
        longest_edge: List = []  # index, length
        illegal_turns: List[int] = []  # indices of illegal turns

        length = 0

        for a, b, c in sliding_window(map(lambda x: x % (len(sequence)), range(-1, len(sequence) + 1)), n=3):
            # calculate length of current edge
            dist = self._distance(b, c)
            if (not longest_edge) or dist > longest_edge[1]:
                longest_edge = [b, dist]
            length += dist

            # check if turn is illegal
            if self._angle(a, b, c) > 90:
                illegal_turns.append(b)

        # three or more illegal turns, or two non-adjacent illegal turns
        if len(illegal_turns) > 2 or (len(illegal_turns) == 2 and abs(illegal_turns[0] - illegal_turns[1]) != 1):
            print(f'{len(illegal_turns)} illegal turns')
            return (sequence, length * (1 + self.constraint_factor * len(illegal_turns)), False)

        # only two adjacent illegal turns
        if len(illegal_turns) == 2:
            print('two adjacent illegal turns')
            return (sequence[illegal_turns[1]:] + sequence[:illegal_turns[0]+1],
                    (length - self._distance(illegal_turns[0], illegal_turns[1])) * (1 + self.constraint_factor),
                    True)

        # one illegal turn
        if len(illegal_turns) == 1:
            print('one illegal turn')
            if (self._distance(illegal_turns[0], (illegal_turns[0] - 1) % len(sequence))
                    > self._distance(illegal_turns[0], (illegal_turns[0] + 1) % len(sequence))):
                return (sequence[illegal_turns[0]:] + sequence[:illegal_turns[0]],
                        (length - self._distance(illegal_turns[0], (illegal_turns[0] - 1) % len(sequence))) * (1 + self.constraint_factor),
                        True)
            else:
                return (sequence[((illegal_turns[0] + 1) % len(sequence)):] + sequence[:((illegal_turns[0] + 1) % len(sequence))],
                        (length - self._distance(illegal_turns[0], (illegal_turns[0] + 1) % len(sequence))) * (1 + self.constraint_factor),
                        True)

        # no illegal turns
        print('no illegal turns')
        return (sequence[((longest_edge[0] + 1) % len(sequence)):] + sequence[:((longest_edge[0] + 1) % len(sequence))],
                length - longest_edge[1],
                True)

    def _mutate(self, sequence: Tuple[int, ...]) -> Tuple[int, ...]:
        mutability = ilist(dft=0)  # mutation probability of each outpost
        illegal_turns: List[int] = []  # indices of illegal turns

        for a, b, c in sliding_window(map(lambda x: x % (len(sequence)), range(-1, len(sequence) + 1)), n=3):
            # calculate length of current edge
            mutability[b] = self._distance(b, c)

            # check if turn is illegal
            if self._angle(a, b, c) > 90:
                illegal_turns.append(b)

        # multiply mutability by constraint factor for illegal turns
        for i in illegal_turns:
            mutability[i] *= 1 + self.constraint_factor

        [s1, s2] = sorted(choices(range(len(sequence)), weights=mutability, k=2))
        print(f'chose {s1} and {s2} to flip')
        return tuple(sequence[:s1 + 1] + sequence[s2:s1:-1] + sequence[s2 + 1:])

    def _to_outposts(self, sequence: Tuple[int, ...]) -> Tuple[Tuple[int, int]]:
        return tuple(self.outposts[i] for i in sequence)

    def solve(self) -> Tuple[Tuple[Tuple[int, int], ...], float]:
        sequences: List[Tuple[Tuple[int, ...], float, bool]] = []
        temperature = self.initial_temperature
        iteration = 1

        for _ in range(self.n_shots):
            sequence = list(range(len(self.outposts)))
            shuffle(sequence)
            sequence = tuple(sequence)
            _, cost, is_legal = self._get_shortest_path_and_cost(sequence)
            sequences.append((sequence, cost, is_legal))

        while temperature > self.end_temperature:
            for i in range(len(sequences)):
                sequence, cost, is_legal = sequences[i]
                print('hash', hash(sequence))
                new_sequence = self._mutate(sequence)
                _, new_cost, new_is_legal = self._get_shortest_path_and_cost(sequence)
                print('cost', new_cost)
                probability = np.exp(-(cost - new_cost) / temperature)
                print(f'{probability=}, {temperature=}, {cost=}, {new_cost=}')
                if new_cost < cost or random() < probability:
                    print('better sequence')
                    sequences[i] = (new_sequence, new_cost, new_is_legal)

            temperature = self.cooldown_fn(temperature, self.initial_temperature, iteration)
            iteration += 1

        shortest_path, shortest_cost, _ = self._get_shortest_path_and_cost(min(filter(lambda x: x[2], sequences), key=lambda x: x[1])[0])
        return self._to_outposts(shortest_path), shortest_cost


while True:
    path_ = input('Enter path to file: ')
    wkt = WKT(f'wenigerkrumm{path_}.txt', n_shots=1, initial_temperature=10000, cooldown_fn=lambda t, t0, i: t * 0.99)
    solution, cost = wkt.solve()
    print(' -> '.join(map(lambda x: f'({x[0]}|{x[1]})', solution)))
    print(f'Cost: {cost}')
    print()
    G = nx.Graph()
    G.add_edges_from(sliding_window(range(len(solution)), n=2))
    nx.draw(G, pos=solution, with_labels=True)
    plt.gca().set_aspect('equal')
    plt.show()
