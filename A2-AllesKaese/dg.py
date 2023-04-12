import os
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import networkx as nx
import matplotlib.pyplot as plt


class ExitException(BaseException):
    pass


# pylama:ignore=C901
def main(kaesestack: List[Tuple[int, int]], fname: str):

    print('Rotiere Käsestapel...')
    kaesestack = list(map(lambda x: tuple(sorted(x)), kaesestack))
    print('\033[1A\033[2K', end='')

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

    # Graph konstruieren
    print('Erstelle Graph...')
    print('> Erstelle Knoten...')
    G = nx.Graph()
    for i, (a, b) in enumerate(kaesestack):
        G.add_node(i, name=f'{a}x{b}')
    print('\033[1A\033[2K', end='')

    print('> Erstelle Kanten...')
    for i in range(len(kaesestack)):
        for j in get_neighbours(i):
            if i > j:
                G.add_edge(i, j)
    print('\033[1A\033[2K', end='')
    print('\033[1A\033[2K', end='')

    nx.draw(G, with_labels=True)
    plt.show()

    # print('Suche Pfad...')
    # min_dim = min(map(lambda x: x[0], kaesestack))
    # first_nodes = list(filter(lambda x: x[0] == min_dim, G.nodes))
    # print(f'Anzahl möglicher Startknoten: {len(first_nodes)}: {first_nodes}')

    # for start in first_nodes:
    #     print(f'> Suche Pfad von {start}...')
    #     path = nx.shortest_path(G, start, None, 'count')
    #     print(f'Pfad gefunden: {path}')
    #     print(f'Größe: {path[-1][0]}x{path[-1][1]}x{len(path)}')

if __name__ == '__main__':
    try:
        while True:
            try:
                fname = f'kaese{input("Bitte Zahl des Beispiels eingeben: ")}.txt'
                print()
                with open(os.path.join(os.path.dirname(__file__), f'beispieldaten/{fname}')) as f:
                    kaesestack = [tuple(sorted(map(int, f.readline().split())))
                                  for _ in range(int(f.readline()))]
                path, size = main(kaesestack, fname)
                print('Reihenfolge: ', end='')
                print(' -> '.join(map(lambda x: f'{kaesestack[x][0]}x{kaesestack[x][1]}', path)))
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
