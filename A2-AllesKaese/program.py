import os
import pathlib
import time

from solve import vanilla, make_stacks


class ExitException(BaseException):
    pass


# pylama:ignore=C901
if __name__ == "__main__":
    try:
        while True:
            try:
                num = int(input("Bitte Zahl des Beispiels eingeben: "))
                fname = f"kaese{num}.txt"
                start_time = time.time()
                print()
                print(f"Lade {fname}...")
                with open(
                    os.path.join(os.path.dirname(__file__), f"beispieldaten/{fname}")
                ) as f:
                    stack = [
                        tuple(map(int, f.readline().split()))
                        for _ in range(int(f.readline()))
                    ]
                print("\033[1A\033[2K", end="")
                res = None
                easy = True
                try:
                    print("Löse Problem...")
                    res = vanilla(stack)
                finally:
                    print("\033[1A\033[2K", end="")

                if res is None:
                    try:
                        print(
                            "Keine Lösung für einfaches Problem gefunden. "
                            "Verwende erweitertes Problem..."
                        )
                        for i in range(1, len(stack) + 1):
                            results = list(make_stacks(stack, i))
                            if len(results) == 0:
                                continue
                            results.sort(key=lambda x: sum(y[2] for y in x))
                            res = results[0]
                            break
                        easy = False
                    finally:
                        print("\033[1A\033[2K", end="")

                if res is None or len(res) == 0:
                    raise Exception("Keine Lösung gefunden.")

                if num in tuple(range(1, 8)):  # Einfaches Problem
                    path, size, n_virtual = res[0]

                    # Ausgabe der Lösung als Datei
                    pathlib.Path(
                        os.path.join(os.path.dirname(__file__), "output")
                    ).mkdir(parents=True, exist_ok=True)
                    with open(
                        os.path.join(os.path.dirname(__file__), f"output/{fname}"), "w"
                    ) as f:
                        for slice in path:
                            f.write(f"{stack[slice][0]} {stack[slice][1]}\n")

                    print(f"Zeit: {time.time()-start_time:.2f}s")
                    print("Reihenfolge: ", end="")
                    if len(path) < 100:
                        print(
                            " -> ".join(
                                map(lambda x: f"{stack[x][0]}x{stack[x][1]}", path)
                            )
                        )
                    else:
                        print(
                            f"Zu groß für die Konsole ({len(path)} Stück). Siehe output/{fname}."
                        )
                    print(f"Größe: {size[0]}x{size[1]}x{size[2]}")
                else:
                    print(f"Zeit: {time.time()-start_time:.2f}s")
                    print(f"Käseblöcke: {len(res)}")
                    for i, (path, size, n_virtual) in enumerate(res):
                        print("------------------------")
                        print(f"Käseblock {i+1}:")
                        print("Reihenfolge: ", end="")
                        started = False
                        for slice in path:
                            if started:
                                print(" -> ", end="")
                            else:
                                started = True
                            if isinstance(slice, tuple):
                                print(
                                    f"{slice[0]}x{slice[1]} (wurde aufgegessen)", end=""
                                )
                            else:
                                print(f"{stack[slice][0]}x{stack[slice][1]}", end="")
                        print()
                        print(f"Größe: {size[0]}x{size[1]}x{size[2]}")
                        print(f"Anzahl aufgegessener Scheiben: {n_virtual}")
            except Exception as e:
                print(f"Fehler: {e}")
            finally:
                print()
    except ExitException as e:
        print(e)
        exit()
    except KeyboardInterrupt:
        print()
        print("Abbruch durch Benutzer.")
        exit()
