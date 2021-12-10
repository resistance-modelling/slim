from __future__ import annotations

import math
from queue import Empty

from numba import njit, types
import numpy as np
from typing import Dict, MutableMapping, Union, List, Iterator
from abc import ABC

from numba.experimental import jitclass

Allele = str
Alleles = str

@njit
def largest_remainder(nums: np.ndarray) -> np.ndarray:
    # a vectorised implementation of largest remainder
    assert np.all(nums >= 0) or np.all(nums <= 0)

    nums = nums.astype(np.float32)
    approx = np.trunc(nums)

    while True:
        diff = nums - approx
        diff_sum = int(np.rint(np.sum(diff)))

        if diff_sum == 0:
            break

        positions = np.argsort(diff)
        tweaks = min(len(nums), abs(diff_sum))
        if diff_sum > 0:
            approx[positions[::-1][:tweaks]] += 1.0
        else:
            approx[positions[:tweaks]] -= 1.0

    #assert np.rint(np.sum(approx)) == np.rint(np.sum(nums)), f"{approx} is not summing to {nums}, as the first sums to {np.sum(approx)} and the second to {np.sum(nums)}"
    return approx

from heapq import heapify, heappush, heappop
class UnsafePriorityQueue():
    def __init__(self):
       self.queue = []

    def qsize(self):
        return len(self.queue)

    def peek(self):
        if self.qsize() == 0:
            raise Empty
        return self.queue[0]

    def put(self, item):
        heappush(self.queue, item)

    def get(self):
        return heappop(self.queue)

    def heapify(self):
        heapify(self.queue)
"""
@jitclass([('_store', types.DictType(*(types.unicode_type, types.float64)))])
class GenoDistrib():
    def __init__(self, store):
        print("In GenoDistrib:", store)
        self._store = store
        print(store)
        print(self._store)
        print(store)

    def normalise_to(self, population: int) -> GenoDistrib:
        keys = self.keys()
        values = list(self.values())
        np_values = np.array(values)
        # np.isclose is slow and not needed: all values are granted to be real and positive.
        if np.sum(np_values) < 1 or population == 0:
            return GenoDistrib({k: 0 for k in keys})

        np_values = np_values * population / np.sum(np_values)

        rounded_values = largest_remainder(np_values)

        store = {}
        keys = list(keys)
        for i in range(len(keys)):
            store[keys[i]] = rounded_values[i]
        print(store)
        return GenoDistrib(store)

    def keys(self):
        return list(self._store.keys())

    def values(self):
        return list(self._store.values())

    def set(self, other: int):
        result = self.normalise_to(other)
        self._store.clear()
        self._store.update(result)

    @property
    def gross(self) -> int:
        return int(sum(self.values()))

    def copy(self: GenoDistrib) -> GenoDistrib:
        return GenoDistrib(self._store.copy())


def __add__(self, other: Union[GenoDistrib, int]) -> GenoDistrib:
    '''Overload add operation'''
    if isinstance(other, GenoDistrib):
        res = GenoDistrib()
        res._store.update(self)
        for k, v in other.items():
            res[k] = self[k] + v
        return res
    else:
        return self.normalise_to(self.gross + other)

def __sub__(self, other: Union[GenoDistrib, int]) -> GenoDistrib:
    '''Overload sub operation'''
    if isinstance(other, GenoDistrib):
        res = GenoDistrib()
        res._store.update(self)
        for k, v in other.items():
            res[k] = self[k] - v
        return res
    else:
        return self.normalise_to(self.gross - other)

def __mul__(self, other: Union[float, GenoDistrib]) -> GenoDistrib:
    '''Multiply a distribution.'''
    if isinstance(other, float) or isinstance(other, int):
        keys = self.keys()
        values = [v * other for v in self.values()]
        if isinstance(other, float) and math.trunc(other) == other:
            values = map(int, largest_remainder(np.array(values)).tolist())
    else:
        # pairwise product
        # Sometimes it may be helpful to not round
        keys = set(list(self.keys()) + list(other.keys()))
        values = [self[k] * other[k] for k in keys]
    return GenoDistrib(dict(zip(keys, values)))

def __eq__(self, other: Union[GenoDistrib, Dict[Alleles, int]]) -> bool:
    # Make equality checks a bit more flexible
    if isinstance(other, dict):
        other = GenoDistrib(other)
    return self._store == other._store

def __le__(self, other: Union[GenoDistrib, float]) -> bool:
    '''A <= B if B has all the keys of A and A[k] <= B[k] for every k
    Note that __gt__ is not implemented as its behaviour is not "greater than" but rather
    "there may be an unbounded frequency".
    '''
    if isinstance(other, float):
        return all(v <= other for v in self.values())

    merged_keys = set(list(self.keys()) + list(other.keys()))
    return all(self[k] <= other[k] for k in merged_keys)

def is_positive(self) -> bool:
    return all(v >= 0 for v in self.values())

def __getitem__(self, key) -> float:
    return self._store.setdefault(key, 0)

def __setitem__(self, key, value):
    self._store[key] = value

def __delitem__(self, key):
    del self._store[key]

def __iter__(self) -> Iterator[Alleles]:
    return iter(self._store)

def __len__(self) -> int:
    return len(self._store)

def __repr__(self) -> str:
    return repr(self._store)

def __str__(self) -> str:
    return "GenoDistrib(" + str(self._store) + ")"

def to_json_dict(self) -> Dict[str, float]:
    return {"".join(k): v for k, v in self.items()}

def truncate_negatives(self):
    for k in self:
        self[k] = max(self[k], 0)

# from numba.typed import Dict as NumbaDict

@njit
def test():
    a = GenoDistrib({"A": 1.0})
    print(a._store["A"])
    assert a._store["A"] == 1
    b = a.normalise_to(10)
    print(b)
    # assert list(b.values()) == [10.0]
    #assert len(a.keys()) == 1
test()
"""
