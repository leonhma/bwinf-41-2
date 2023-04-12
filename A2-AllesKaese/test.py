from functools import reduce
import operator
import random
from alive_progress import alive_it

from prog import main

for _ in alive_it(range(1000)):
    sizei = [random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)]
    size = sizei.copy()
    slices = []

    while reduce(operator.mul, size) > 0:
        dim = random.randint(0, 2)
        slice = size.copy()
        slice.pop(dim)
        slice.sort()
        slices.append(tuple(slice))
        size[dim] -= 1

    try:
        pathr, sizer = main(slices, '')

        if sorted(sizei) != sorted(sizer):
            print('Size mismatch')
            print(sizei, sizer)
            print(slices)
            print(pathr)
            break
    except Exception:
        pass
