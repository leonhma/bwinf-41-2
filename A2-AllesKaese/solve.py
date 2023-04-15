import collections
import itertools
from typing import List, Set, Tuple


# pylama:ignore=C901
def vanilla(stack: List[Tuple[int, int]]) -> Tuple:
    """Lösen des Problems mit vollständigem Käsestack und einem Startkäse."""

    # Hashmap für schnellen Zugriff auf potentielle Nachbarn
    lookup = collections.defaultdict(set)
    try:
        for i, (a, b) in enumerate(stack):
            lookup[a].add(i)
            lookup[b].add(i)
    except ValueError:
        raise Exception(
            "ValueError: Die Eingabe-Datei ist wahrscheinlich nicht korrekt formatiert."
        )

    # Funktion zum Zugriff auf potentielle Nachbarn
    def get_neighbors(i: int) -> Set[int]:
        a, b = stack[i]
        ret = (lookup[a] | lookup[b]) - {i}
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
    avg_n_neigh = 0
    avg_n_neigh_n = 0
    while True:
        # Wenn kein Pfad existiert, wähle einen Startknoten
        if not path:
            if start_nodes_i:
                start = start_nodes_i.pop(0)
                path = [[start, tuple(stack[start] + (1,)), None]]
                continue  # Diese Iteration überspringen
            else:
                # Es kann keine Lösung gefunden werden
                return
        # Aktueller Knoten, Größe und zu prüfende Nachbarn
        current, size, to_check = path[-1]
        # Füge den aktuellen Knoten zu den besuchten Knoten hinzu
        seen.add(current)
        # Wenn der Pfad vollständig ist, wurde eine Lösung gefunden
        if len(seen) == len(stack):
            return ((list(map(lambda x: x[0], path)), size, 0),)
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
                # Wenn zwei mögliche Nachbarn zur gleichen Größe führen, überspringe den Nachbarn
                if new_size is not None and new_size not in seen_sizes:
                    seen_sizes.add(new_size)
                    # Füge den Nachbarn zu den zu prüfenden Nachbarn hinzu
                    to_check.add((i, new_size))
        # Wenn es keine Nachbarn gibt, entferne den aktuellen Knoten aus dem Pfad (backtracking)
        if not to_check:
            path.pop()
            seen.remove(current)
            continue
        else:
            avg_n_neigh += len(to_check)
            avg_n_neigh_n += 1
        # Aktualisiere die zu prüfenden Nachbarn im Pfad
        # (nötig wenn die Nachbarn gerade generiert wurden)
        path[-1][2] = to_check
        # Der nächste Knoten ist der erste Nachbar, der noch nicht besucht wurde
        next_ = to_check.pop()
        path.append([next_[0], next_[1], None])


def covers(it1, it2) -> Tuple | None:
    """Prüft, ob eine Scheibe auf eine andere Scheibe passt.
    Auch Unterschiede von 1 in einer Dimension sind erlaubt.
    Gibt eine Maske von hinzugefügten Scheiben zurück."""
    it1_ = tuple(sorted(it1))
    masks = ((0, 0), (0, 1), (1, 0))
    for mask in masks:
        if tuple(sorted(map(lambda x: x[0] + x[1], zip(it2, mask)))) == it1_:
            return mask


def create_virtual(ab: List, mask: Tuple, ignoredsize: int) -> Tuple or None:
    """Erstellt eine aufgegessene (virtuelle) Scheibe aus einer Scheibe und einer Maske."""
    if not any(mask):  # Nicht nötig eine virtuelle Scheibe zu erstellen
        return

    # Länge in die Dimension, die nicht passend sein muss x Länge der passenden Dimension
    return (ignoredsize, ab[mask.index(0)])


def remove(slice: Tuple[int, int], size: List[int]):
    """Verändere `size` indem eine Scheibe `slice` entfernt wird."""
    ab = set(slice)
    if ab == set(size[:2]):
        size[2] -= 1
    elif ab == set(size[1:]):
        size[0] -= 1
    elif ab == set(size[::2]):
        size[1] -= 1
    else:
        raise Exception("what")


# pylama:ignore=C901
def fuzzy(stack: List[Tuple[int, int]]):
    """Lösen des Problems mit fehlerhaftem Käsestack und mehreren Käseblöcken.
    Gibt eine Liste von möglichen Lösungen zurück."""

    # Dictionary der Lösungen, ihrer Größe und der Anzahl der hinzugefügten/aufgegessenen Scheiben
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
    # Anzahl der hinzugefügten/'aufgegessenen' Scheiben
    n_virtual = 0

    while True:
        # Wenn kein Pfad existiert, wähle einen Startknoten
        if not path:
            if start_nodes_i:
                start = start_nodes_i.pop(0)
                path = [[0, start, tuple(stack[start] + (1,)), None]]
                continue  # Nächste Iteration
            # Wenn keine Startknoten mehr existieren, sind alle Lösungen gefunden
            # und können zurückgegeben werden
            for path, size, n_virtual, n_nodes in solutions.values():
                yield (path, size, n_virtual, n_nodes)
            return  # Ende der Funktion
        # Aktueller Knoten, Größe und zu prüfende Nachbarn, und eingefügte 'aufgegessene' Scheiben
        _, current, size, to_check = path[-1]
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
                        (
                            tuple(sorted(ab + [size[2] + 1])),
                            create_virtual(ab, s, size[2]),
                        )
                    )
                if s := covers(ab, size[1:]):
                    new_sizes.add(
                        (
                            tuple(sorted(ab + [size[0] + 1])),
                            create_virtual(ab, s, size[0]),
                        )
                    )
                if s := covers(ab, size[::2]):
                    new_sizes.add(
                        (
                            tuple(sorted(ab + [size[1] + 1])),
                            create_virtual(ab, s, size[1]),
                        )
                    )
                # Dedupliziere Nachbarn mit gleicher Größe
                for new_size, virtual in new_sizes:
                    if new_size not in seen_sizes:
                        seen_sizes.add(new_size)
                        # Füge den Nachbarn zu den zu prüfenden Nachbarn hinzu
                        if virtual and sorted(virtual) == sorted(stack[i]):
                            virtual = None
                        to_check.add((i, new_size, None, virtual))
        # Aktualisiere die zu prüfenden Nachbarn im Pfad
        path[-1][3] = to_check
        # Wenn es keine Nachbarn gibt -> backtracking
        if not to_check:
            # Lösung speichern
            filteredpath_ = tuple(map(lambda x: x[1], filter(lambda x: not x[0], path)))
            path_ = tuple(map(lambda x: x[1], path))
            solutions[filteredpath_] = min(
                (
                    (path_, size, n_virtual, len(seen)),
                    solutions.get(path_, (path_, size, n_virtual, len(seen))),
                ),
                key=lambda x: sum(x[1]) * x[2],
            )
            # Alle besuchten Knoten ohne alternativen Nachbarn entfernen
            while path and not path[-1][3]:
                i = path.pop()[1]
                seen.remove(i)
                # remove virtual
                if path and path[-1][0] == 1:
                    path.pop()
                    n_virtual -= 1
            continue  # Nächste Iteration

        # Der nächste Knoten ist der erste Nachbar, der noch nicht besucht wurde
        next_ = to_check.pop()
        next_node, next_size, next_to_check, virtual = next_
        if virtual:  # Wenn eine virtuelle Scheibe hinzugefügt werden muss
            path.append([1, virtual])  # 1, virtual size
            n_virtual += 1
        path.append([0, next_node, next_size, next_to_check])


def make_stacks(stack: List[Tuple[int, int]], n: int):
    """Erstelle `n` Blöcke aus `stack`"""
    # Für jede Kombination aus `n` Pfaden
    for c in itertools.combinations(fuzzy(stack), n):
        # Wird die Anzahl der Überflüssigen Scheiben berechnet
        overflow = sum(x[3] for x in c) - len(stack)
        # Wenn die Pfade insgesamt zu wenige Schieben haben, überspringen
        if overflow < 0:
            continue
        # Für jede Kombination die Pfade zu kürzen
        for x in itertools.product(range(overflow + 1), repeat=n):
            if (
                sum(x) != overflow
            ):  # sodass die richtige Anzahl an Scheiben entfernt wird
                continue
            try:
                # werden alle Pfade gekürzt
                paths = []
                for i, p in enumerate(c):
                    path, size, n_virtual = list(p[0]), list(p[1]), p[2]
                    for _ in range(x[i]):
                        i = path.pop()
                        remove(stack[i], size)
                        if path and isinstance(path[-1], tuple):
                            s = path.pop()
                            remove(s, size)
                            n_virtual -= 1

                    paths.append((tuple(path), tuple(size), n_virtual))
            # Wenn ein Pfad zu kurz ist, um gekürzt zu werden,
            # wird die nächste Kombination ausprobiert
            except IndexError:
                continue
            # Jetzt wird das Ergebnis überprüft,
            if any(len(p) == 0 for p, _, _ in paths):
                continue
            seen = set()
            for p, _, _ in paths:
                for i in p:
                    if isinstance(i, tuple):
                        continue
                    seen.add(i)
            # und wenn vollständig, zurückgegeben
            if len(seen) == len(stack):
                yield tuple(paths)
