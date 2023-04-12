# fully connected graph - obviously illegal connections are pruned
# IDEAS
# - for every node calculate the largest number of nodes that can be connected from it in one path,
#   check if that number is >= the number of nodes remaining

# NP-Complex probably

import collections
import os
from pathlib import Path
import time
from typing import List, Set, Tuple


class ExitException(BaseException):
    pass


class Counter:
    state: List[int]

    def __init__(self, size: int):
        self.state = [0] * size

    def __iter__(self):
        ret = self.state.copy()
        self.state[-1] += 1
        return ret

    def carry(self, i: int):
        self.state[i:] = [0] * (len(self.state) - i)
        self.state[i - 1] += 1


# pylama:ignore=C901
def main(stack: List[Tuple[int, int]]):
    start_time = time.time()
    # create lookup
    lookup = collections.defaultdict(set)
    for i, (a, b) in enumerate(stack):
        lookup[a].add(i)
        lookup[b].add(i)

    def get_neighbors(i: int) -> Set[int]:
        a, b = stack[i]
        ret = (lookup[a] | lookup[b]) - {i}
        return ret

    # min_start = min(map(lambda x: x[0], stack))
    # start_nodes_i = list(filter(lambda i: stack[i][0] == min_start, range(len(stack))))
    # TODO go with the start node that improves the quickest
    start_nodes_i = list(sorted(range(len(stack)), key=lambda i: stack[i][0] * stack[i][1]))
    seen = set()
    # node, size, to_check
    path = []
    avg_n_neigh = 0
    avg_n_neigh_cnt = 0
    while True:
        if not path:
            if start_nodes_i:
                start = start_nodes_i.pop(0)
                path = [[start,
                        tuple(stack[start] + (1,)),
                        None]]
                continue
            raise Exception('no solution found')
        current, size, to_check = path[-1]
        # add current node to seen
        seen.add(current)
        # check if path is complete and return it
        if len(seen) == len(stack):
            return list(map(lambda x: stack[x[0]], path)), \
                size, time.time() - start_time, avg_n_neigh / avg_n_neigh_cnt
        # if next_neighbors is none, generate them (calculate new size and check for fit and seen)
        if to_check is None:
            to_check = set()
            for i in get_neighbors(current):
                if i in seen:
                    continue
                ab = set(stack[i])
                if ab == set(size[:2]):
                    to_check.add((i, (size[0], size[1], size[2] + 1)))
                elif ab == set(size[1:]):
                    to_check.add((i, (size[1], size[2], size[0] + 1)))
                elif ab == set(size[::2]):
                    to_check.add((i, (size[2], size[0], size[1] + 1)))
        # if neighbors is empty or none, remove current node from path and seen
        # happens if a deadend is encountered
        if not to_check:
            path.pop()
            seen.remove(current)
            continue
        else:
            avg_n_neigh += len(to_check)
            avg_n_neigh_cnt += 1
        # set in path
        path[-1][2] = to_check
        # if neighbors is not empty or just generated, choose one and add it to path,
        # store neigh in to_check
        next_ = to_check.pop()
        path.append([next_[0], next_[1], None])
        #
        # add new size to next node
        # neighbors is none in next node


if __name__ == '__main__':
    try:
        while True:
            try:
                fname = f'kaese{input("Bitte Zahl des Beispiels eingeben: ")}.txt'
                print()
                with open(os.path.join(os.path.dirname(__file__), f'beispieldaten/{fname}')) as f:
                    stack = [tuple(sorted(map(int, f.readline().split())))
                             for _ in range(int(f.readline()))]
                path, size, time_, cmplx_base = main(stack)

                # Ausgabe der Lösung als Datei
                Path(os.path.join(os.path.dirname(__file__), 'output')).mkdir(parents=True, exist_ok=True)
                with open(os.path.join(os.path.dirname(__file__), f'output/{fname}'), 'w') as f:
                    for slice in path:
                        f.write(f'{slice[0]} {slice[1]}\n')

                print(f'Zeit: {time_:.2f}s')
                print(f'Komplexitäts-Basis: {cmplx_base:.2f}')
                print('Reihenfolge: ', end='')
                if len(path) < 100:
                    print(' -> '.join(map(lambda x: f'{x[0]}x{x[1]}', path)))
                else:
                    print(f'Zu groß für die Konsole ({len(path)} Stück). Siehe output/{fname}.')
                print(f'Größe: {size[0]}x{size[1]}x{size[2]}')
            except Exception as e:
                print(f'Fehler: {e}')
            finally:
                print()
    except ExitException as e:
        print(e)
        exit()
    except KeyboardInterrupt:
        print()
        print('Abbruch durch Benutzer.')
        exit()
