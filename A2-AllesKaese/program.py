
import os
import pathlib

from solve import vanilla


class ExitException(BaseException):
    pass


# pylama:ignore=C901
if __name__ == '__main__':
    try:
        while True:
            try:
                fname = f'kaese{input("Bitte Zahl des Beispiels eingeben: ")}.txt'
                print()
                print(f'Lade {fname}...')
                with open(os.path.join(os.path.dirname(__file__), f'beispieldaten/{fname}')) as f:
                    stack = [tuple(map(int, f.readline().split()))
                             for _ in range(int(f.readline()))]
                print('\033[1A\033[2K', end='')
                res = None
                try:
                    print('Löse Problem...')
                    res = vanilla(stack)
                except Exception as e:
                    print(f'Fehler: {e}')
                finally:
                    print('\033[1A\033[2K', end='')
                if res is None:
                    raise ExitException('Keine Lösung gefunden.')

                path, size, time_ = res

                # Ausgabe der Lösung als Datei
                pathlib.Path(os.path.join(os.path.dirname(__file__), 'output')).mkdir(parents=True,
                                                                                      exist_ok=True)
                with open(os.path.join(os.path.dirname(__file__), f'output/{fname}'), 'w') as f:
                    for slice in path:
                        f.write(f'{slice[0]} {slice[1]}\n')

                print(f'Zeit: {time_:.2f}s')
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
