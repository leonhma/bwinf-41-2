from bisect import insort
import itertools
from pathlib import Path

from typing import Dict, Hashable


class TabuList:
    tabu: Dict[Hashable, int]
    offset: int
    default_tenure: int
    cleanup_freq: int

    def __init__(self, default_tenure: int, cleanup_freq: int = 20):
        self.tabu = {}
        self.offset = 0
        self.default_tenure = default_tenure
        self.cleanup_freq = cleanup_freq

    def _cleanup(self):
        to_delete = [k for k, v in self.tabu.items() if v + self.offset <= 0]
        for k in to_delete:
            del self.tabu[k]

    def add(self, item: Hashable, tenure: int = None):
        self.tabu[item] = (tenure or self.default_tenure) - self.offset

    def get(self, item: Hashable) -> int:
        if item in self.tabu:
            val = self.tabu[item] + self.offset
            return max(val, 0)
        return 0

    def tick(self):
        self.offset -= 1
        if self.offset % self.cleanup_freq == 0:
            self._cleanup()


class BestList:
    def __init__(self, n, sorting_key=lambda x: x):
        self.n = n
        self.data = []
        self.sorting_key = sorting_key

    def add(self, item):
        self.data.append(item)
        insort(self.data, item, key=self.sorting_key)
        self.data = self.data[:self.n]

    def __getitem__(self, i):
        return self.data[i]


def index_it(iterable, i: int):
    while i > 0:
        i -= 1
        next(iterable)
    return next(iterable)


def sliding_window(iterable, n=2):
    iterables = itertools.tee(iterable, n)

    for iterable, num_skipped in zip(iterables, itertools.count()):
        for _ in range(num_skipped):
            next(iterable, None)

    return zip(*iterables)


class ilist(list):
    def __init__(self, r=None, dft=None):
        if r is None:
            r = []
        list.__init__(self, r)
        self.dft = dft

    def _ensure_length(self, n):
        maxindex = n
        if isinstance(maxindex, slice):
            maxindex = maxindex.indices(len(self))[1]
        while len(self) <= maxindex:
            self.append(self.dft)

    def __getitem__(self, n):
        self._ensure_length(n)
        return super(ilist, self).__getitem__(n)

    def __setitem__(self, n, val):
        self._ensure_length(n)
        return super(ilist, self).__setitem__(n, val)


def r_path(path):
    return Path(__file__).parent / path
