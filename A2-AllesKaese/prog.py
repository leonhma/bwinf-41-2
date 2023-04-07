import os
from collections import defaultdict
from typing import Dict, List, Set, Tuple


class ExitException(BaseException):
    pass


def main(kaesestack: List[Tuple[int, int]], fname: str):

    print('Indexiere Käsestapel...')
    lookup: Dict[int, Set[int]] = defaultdict(set)
    for i, (a, b) in enumerate(kaesestack):
        lookup[a].add(i)
        lookup[b].add(i)
    print('\033[1A\033[2K', end='')


    def get_neighbours(i: int) -> Set[int]:
        a, b = kaesestack[i]
        ret = (lookup[a] | lookup[b]) - {i}
        return ret

    def dfs(current: int, current_size: Tuple[int, int, int], visited: Set[int]):       # TODO recursion limit :(
        visited.add(current)
        if len(visited) == len(kaesestack):     # TODO last gets added no matter what
            return [current], current_size
        current_size = tuple(sorted(current_size))
        for j in get_neighbours(current):
            if j not in visited:
                slice = kaesestack[j]
                if slice == current_size[:2]:
                    current_size = (current_size[0],
                                    current_size[1],
                                    current_size[2] + 1)
                    res = dfs(j, current_size, visited)
                    if res:
                        return [current] + res[0], res[1]
                elif slice == current_size[1:]:
                    current_size = (current_size[0] + 1,
                                    current_size[1],
                                    current_size[2])
                    res = dfs(j, current_size, visited)
                    if res:
                        return [current] + res[0], res[1]
                elif slice == current_size[::2]:
                    current_size = (current_size[0],
                                    current_size[1] + 1,
                                    current_size[2])
                    res = dfs(j, current_size, visited)
                    if res:
                        return [current] + res[0], res[1]
        visited.remove(current)

    start = kaesestack[0]
    s = [start[0], start[1], 1]
    path, size = dfs(0, s, set())

    print('Reihenfolge: ', end='')
    print(' -> '.join(map(lambda x: f'{kaesestack[x][0]}x{kaesestack[x][1]}', path)))
    print(f'Größe: {size[0]}x{size[1]}x{size[2]}')


if __name__ == '__main__':
    try:
        while True:
            try:
                fname = f'kaese{input("Bitte Zahl des Beispiels eingeben: ")}.txt'
                print()
                with open(os.path.join(os.path.dirname(__file__), f'beispieldaten/{fname}')) as f:
                    kaesestack = [tuple(sorted(map(int, f.readline().split())))
                                  for _ in range(int(f.readline()))]
                main(kaesestack, fname)
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
