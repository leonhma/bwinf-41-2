
import os
from typing import List, Tuple


class ExitException(BaseException):
    pass


def main(kaesestack: List[Tuple[int, int]], fname: str):
    kaesestack.sort(key=lambda x: (min(x), max(x)))
    print(kaesestack)
    print()
    current = kaesestack.pop(0)
    size = [min(current), max(current), 1]
    print(f'start size is {size}')
    while kaesestack:
        current = kaesestack.pop(0)
        print(f'current is {current}')
        if current == (min(size[:2]), max(size[:2])):
            size[2] += 1
            print(f'added {current} to new {size}')
        elif current == (min(size[1:]), max(size[1:])):
            size[0] += 1
            print(f'added {current} to new {size}')
        elif current == (min(size[::2]), max(size[::2])):
            size[1] += 1
            print(f'added {current} to new {size}')
        else:
            raise Exception('Käsestack ist nicht stapelbar.')
    print(f'Käse ist {size[0]}x{size[1]}x{size[2]} groß.')


if __name__ == '__main__':
    try:
        while True:
            try:
                fname = f'kaese{input("Bitte Zahl des Beispiels eingeben: ")}.txt'
                points = []
                with open(os.path.join(os.path.dirname(__file__), f'beispieldaten/{fname}')) as f:
                    kaesestack = [tuple(sorted(map(int, f.readline().split()))) for _ in range(int(f.readline()))]
                main(kaesestack, fname)
            except Exception as e:
                print(e)
    except ExitException as e:
        print(e)
        exit()
    except KeyboardInterrupt:
        print()
        print('Abbruch durch Benutzer.')
        exit()
