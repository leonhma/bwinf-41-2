import collections
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
    start_nodes_i = list(
        sorted(range(len(stack)), key=lambda i: stack[i][0] * stack[i][1])
    )
    # Besuchte Knoten
    seen = set()
    # Besuchter Pfad mit Größe und zu prüfenden Nachbarn (für backtracking)
    path = []
    while True:
        # Wenn kein Pfad existiert, wähle einen Startknoten
        if not path:
            if start_nodes_i:
                start = start_nodes_i.pop(0)
                path = [[start, tuple(stack[start] + (1,)), None]]
                continue
            # Wenn keine Startknoten mehr existieren, ist keine Lösung möglich
            raise Exception("no solution found")
        # Aktueller Knoten, Größe und zu prüfende Nachbarn
        current, size, to_check = path[-1]
        # Füge den aktuellen Knoten zu den besuchten Knoten hinzu
        seen.add(current)
        # Wenn der Pfad vollständig ist, wurde eine Lösung gefunden
        if len(seen) == len(stack):
            return (
                list(map(lambda x: stack[x[0]], path)),
                size,
                time.time() - start_time,
            )
        # Wenn noch keine Nachbarn geprüft wurden, generiere sie
        if to_check is None:
            to_check = set()  # Set für mögliche Nachbarn
            seen_sizes = set()  # Set für bereits besuchte Größen
            for i in get_neighbors(current):
                if i in seen:  # Nachbarn, die bereits besucht wurden, überspringen
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


def covers(it1, it2) -> Tuple | bool:
    it1_ = tuple(sorted(it1))
    masks = ((0, 0), (0, 1), (1, 0))
    for mask in masks:
        if tuple(sorted(map(lambda x: x[0] + x[1], zip(it2, mask)))) == it1_:
            print('returning', mask)
            return mask


def create_virtual(size: List, partial_mask: Tuple, ignored_dim: int) -> Tuple or None:
    """Erstellt eine virtuelle Scheibe aus einer Scheibe und einer Maske."""
    mask = []
    removed = 0
    for i in range(3):
        if i == ignored_dim:
            mask.append(0)
            removed = 1
        else:
            mask.append(partial_mask[i - removed])

    if not any(mask):
        return

    print('adding virtual', size, mask)
    return tuple(sorted(map(lambda x: x[0] if not x[1] else 1, zip(size, mask))))


# pylama:ignore=C901
# step through solutions to the problem (includes missing cheese).
# Return a solution before backtracking starts
def search(stack: List[Tuple[int, int]]):
    """Lösen des Problems mit fehlerhaftem Käsestack und mehreren Käseblöcken."""

    # Dictionary der Lösungen und ihrer Größe
    solutions = {}

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

    # Liste der Start-Scheiben (sortiert nach Größe, dedupliziert nach Größe)
    start_nodes_i = []
    seen_start_sizes = set()
    for node in list(
        sorted(range(len(stack)), key=lambda i: stack[i][0] * stack[i][1])
    ):
        size = tuple(sorted(stack[node]))
        if size not in seen_start_sizes:
            start_nodes_i.append(node)
            seen_start_sizes.add(size)

    # Besuchte Knoten
    seen = set()
    # Besuchter Pfad mit Größe und zu prüfenden Nachbarn (für backtracking)
    path = []

    while True:
        # Wenn kein Pfad existiert, wähle einen Startknoten
        if not path:
            if start_nodes_i:
                start = start_nodes_i.pop(0)
                path = [[0, start, tuple(stack[start] + (1,)), None]]
                continue
            # Wenn keine Startknoten mehr existieren, sind alle Lösungen gefunden
            return solutions.items()
        # Aktueller Knoten, Größe und zu prüfende Nachbarn, und eingefügte 'aufgegessene' Scheiben
        _, current, size, to_check = path[-1]
        print("current path is", path)
        # Füge den aktuellen Knoten zu den besuchten Knoten hinzu
        seen.add(current)
        # Wenn noch keine Nachbarn geprüft wurden, generiere sie
        if to_check is None:
            to_check = set()  # Set für mögliche Nachbarn
            seen_sizes = set()  # Set für bereits besuchte Größen
            for i in get_neighbors(current):
                if i in seen:  # Nachbarn, die bereits besucht wurden, überspringen
                    continue
                new_sizes = set()  # new_size, virtual
                # Prüfe, ob die Nachbarn die gleiche Größe (in 2 Dimensionen) haben
                ab = list(stack[i])
                if s := covers(ab, size[:2]):
                    new_sizes.add(
                        (tuple(sorted(ab + [size[2] + 1])), create_virtual(size, s, 2))
                    )
                if s := covers(ab, size[1:]):
                    new_sizes.add(
                        (tuple(sorted(ab + [size[0] + 1])), create_virtual(size, s, 0))
                    )
                if s := covers(ab, size[::2]):
                    new_sizes.add(
                        (tuple(sorted(ab + [size[1] + 1])), create_virtual(size, s, 1))
                    )
                for new_size, virtual in new_sizes:
                    if new_size not in seen_sizes:
                        seen_sizes.add(new_size)
                        # Füge den Nachbarn zu den zu prüfenden Nachbarn hinzu
                        to_check.add((i, new_size, None, virtual))
        # Aktualisiere die zu prüfenden Nachbarn im Pfad
        path[-1][3] = to_check
        # Wenn es keine Nachbarn gibt -> backtracking
        if not to_check:
            path_ = tuple(map(lambda x: x[1], filter(lambda x: not x[0], path)))
            print(f"returning {path_}")
            solutions[path_] = min(
                (size, solutions.get(path_, size)), key=lambda x: sum(x)
            )
            while path and not path[-1][3]:
                i = path.pop()[1]
                seen.remove(i)
                # remove virtual
                if path and path[-1][0] == 1:
                    path.pop()
            continue

        # Der nächste Knoten ist der erste Nachbar, der noch nicht besucht wurde
        next_ = to_check.pop()
        next_node, next_size, next_to_check, virtual = next_
        if virtual:
            path.append([1, virtual])  # 1, virtual size
        path.append([0, next_node, next_size, next_to_check])


def make_stacks(n: int):
    # run through all solutions in complex and check if make_stacks(n-1) of the rest is possible
    pass
