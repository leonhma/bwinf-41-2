import collections
import itertools
import time
from typing import List, Set, Tuple


# pylama:ignore=C901
def vanilla(stack: List[Tuple[int, int]]) -> Tuple:
    """Lösen des Problems mit vollständigem Käsestack und einem Startkäse."""
    start_time = time.time()

    # Hashmap für schnellen Zugriff auf potentielle Nachbarn
    lookup = collections.defaultdict(set)
    for i, (a, b) in enumerate(stack):
        lookup[a].add(i)
        lookup[b].add(i)

    # Funktion zum Zugriff auf potentielle Nachbarn
    def get_neighbors(i: int) -> Set[int]:
        a, b = stack[i]
        ret = (lookup[a] | lookup[b]) - {i}
        return ret

    # Liste der Start-Scheiben (sortiert nach Größe)
    start_nodes_i = list(sorted(range(len(stack)), key=lambda i: stack[i][0] * stack[i][1]))
    # Besuchte Knoten
    seen = set()
    # Besuchter Pfad mit Größe und zu prüfenden Nachbarn (für backtracking)
    path = []
    while True:
        # Wenn kein Pfad existiert, wähle einen Startknoten
        if not path:
            if start_nodes_i:
                start = start_nodes_i.pop(0)
                path = [[start,
                        tuple(stack[start] + (1,)),
                        None]]
                continue
            # Wenn keine Startknoten mehr existieren, ist keine Lösung möglich
            raise Exception('no solution found')
        # Aktueller Knoten, Größe und zu prüfende Nachbarn
        current, size, to_check = path[-1]
        # Füge den aktuellen Knoten zu den besuchten Knoten hinzu
        seen.add(current)
        # Wenn der Pfad vollständig ist, wurde eine Lösung gefunden
        if len(seen) == len(stack):
            return list(map(lambda x: stack[x[0]], path)), \
                size, time.time() - start_time
        # Wenn noch keine Nachbarn geprüft wurden, generiere sie
        if to_check is None:
            to_check = set()        # Set für mögliche Nachbarn
            seen_sizes = set()      # Set für bereits besuchte Größen
            for i in get_neighbors(current):
                if i in seen:    # Nachbarn, die bereits besucht wurden, überspringen
                    continue
                ab = set(stack[i])
                new_size = None
                # Prüfe, ob die Nachbarn die gleiche Größe (in 2 Dimensionen) haben
                if ab == set(size[:2]):
                    new_size = tuple(sorted((size[0], size[1], size[2] + 1)))
                elif ab == set(size[1:]):
                    new_size = tuple(sorted((size[1], size[2], size[0] + 1)))
                elif ab == set(size[::2]):
                    new_size = tuple(sorted((size[2], size[0], size[1] + 1)))
                if new_size is not None and new_size not in seen_sizes:
                    seen_sizes.add(new_size)
                    # Füge den Nachbarn zu den zu prüfenden Nachbarn hinzu
                    to_check.add((i, new_size))
        # Wenn es keine Nachbarn gibt, entferne den aktuellen Knoten aus dem Pfad (backtracking)
        if not to_check:
            path.pop()
            seen.remove(current)
            continue
        # Aktualisiere die zu prüfenden Nachbarn im Pfad
        path[-1][2] = to_check
        # Der nächste Knoten ist der erste Nachbar, der noch nicht besucht wurde
        next_ = to_check.pop()
        path.append([next_[0], next_[1], None])


def itplus1(it: List | Tuple) -> List:
    """Iterator, der das nächste Element vorab liest."""
    return itertools.chain(it, (it[i] + 1 for i in range(len(it))))


def covers(it1, it2) -> bool:
    """Prüft, ob das erste Element das zweite komplett enthält."""
    it1_ = list(it1)
    for i in itplus1(it2):
        if i in it1_:
            it1_.remove(i)
    return not it1_


# pylama:ignore=C901
def search(stack: List[Tuple[int, int]]) -> Tuple:      # step through solutions to the problem (includes missing cheese). Return a solution before backtracking starts or when all nodes have been used up
    """Lösen des Problems mit fehlerhaftem Käsestack und mehreren Käseblöcken."""

    # Hashmap für schnellen Zugriff auf potentielle Nachbarn
    lookup = collections.defaultdict(set)
    for i, (a, b) in enumerate(stack):
        lookup[a].add(i)
        lookup[b].add(i)

    # Funktion zum Zugriff auf potentielle Nachbarn
    def get_neighbors(i: int) -> Set[int]:
        a, b = stack[i]
        ret = (lookup[a] | lookup[b] | lookup[a + 1] | lookup[b + 1]) - {i}
        return ret

    # Liste der Start-Scheiben (sortiert nach Größe)
    start_nodes_i = list(sorted(range(len(stack)), key=lambda i: stack[i][0] * stack[i][1]))
    # Besuchte Knoten
    seen = set()
    # Besuchter Pfad mit Größe und zu prüfenden Nachbarn (für backtracking)
    path = []
    while True:
        has_yielded = False
        # Wenn kein Pfad existiert, wähle einen Startknoten
        if not path:
            if start_nodes_i:
                start = start_nodes_i.pop(0)
                path = [[start,
                        tuple(stack[start] + (1,)),
                        None]]
                continue
            # Wenn keine Startknoten mehr existieren, ist keine Lösung möglich
            return
        # Aktueller Knoten, Größe und zu prüfende Nachbarn
        current, size, to_check = path[-1]
        print(f'current path is: {path}')
        # Füge den aktuellen Knoten zu den besuchten Knoten hinzu
        seen.add(current)
        # Wenn der Pfad vollständig ist, wurde eine Lösung gefunden
        if len(seen) == len(stack):
            has_yielded = True
            yield list(map(lambda x: x[0], path)), size
        else:
            # Wenn noch keine Nachbarn geprüft wurden, generiere sie
            if to_check is None:
                to_check = set()        # Set für mögliche Nachbarn
                seen_sizes = set()      # Set für bereits besuchte Größen
                for i in get_neighbors(current):
                    if i in seen:    # Nachbarn, die bereits besucht wurden, überspringen
                        continue
                    new_sizes = set()
                    # Prüfe, ob die Nachbarn die gleiche Größe (in 2 Dimensionen) haben
                    ab = list(stack[i])
                    if covers(ab, size[:2]):
                        new_sizes.add(tuple(sorted(ab + [size[2] + 1])))
                    if covers(ab, size[1:]):
                        new_sizes.add(tuple(sorted(ab + [size[0] + 1])))
                    if covers(ab, size[::2]):
                        new_sizes.add(tuple(sorted(ab + [size[1] + 1])))
                    for new_size in new_sizes - seen_sizes:
                        seen_sizes.add(new_size)
                        # Füge den Nachbarn zu den zu prüfenden Nachbarn hinzu
                        to_check.add((i, new_size))
        # Aktualisiere die zu prüfenden Nachbarn im Pfad
        path[-1][2] = to_check
        # Wenn es keine Nachbarn gibt -> backtracking
        if not to_check:
            print(f'backtracking: {path}, size: {size}')
            if not has_yielded:
                yield list(map(lambda x: x[0], path)), size
            while path and not path[-1][2]:
                i = path.pop()[0]
                seen.remove(i)
            continue

        # Der nächste Knoten ist der erste Nachbar, der noch nicht besucht wurde
        next_ = to_check.pop()
        path.append([next_[0], next_[1], None])


def make_stacks(n: int):
    # run through all solutions in complex and check if make_stacks(n-1) of the rest is possible
    pass
