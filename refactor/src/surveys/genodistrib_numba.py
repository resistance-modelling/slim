from __future__ import annotations

from types import GeneratorType

from numba import jit, njit, types, float64, float64
from numba.experimental import jitclass
from numba.typed import Dict as NumbaDict
import numpy as np
from typing import Dict, Optional, Union, Iterable, Tuple, List


@njit(float64[:](float64[:]))
def largest_remainder(nums: np.ndarray) -> np.ndarray:
    # a vectorised implementation of largest remainder
    assert np.all(nums >= 0) or np.all(nums <= 0)

    nums = nums.astype(np.float64)
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

    #assert np.rint(sum(approx)) == np.rint(np.sum(
    #    nums)), f"{approx} is not summing to {nums}, as the first sums to {np.sum(approx)} and the second to {np.sum(nums)}"
    return approx


Alleles = str
_JitStoreType = (types.unicode_type, types.float64)

@jitclass([("_store", types.DictType(types.unicode_type, types.float64))])
class JitGenoDistrib:
    def __init__(self, store: Dict[Alleles, float]):
        self._store = NumbaDict.empty(*_JitStoreType)
        for k, v in store.items():
            self._store[k] = v

    @staticmethod
    def from_zip(keys: List[Alleles], values: List[float]) -> JitGenoDistrib:
        assert len(keys) == len(values)
        store = NumbaDict.empty(*_JitStoreType)
        for i in range(len(keys)):
            store[keys[i]] = values[i]
        return JitGenoDistrib(store)

    def normalise_to(self, population: int) -> JitGenoDistrib:
        keys = self.keys()
        values = list(self.values())
        np_values = np.array(values)
        # np.isclose is slow and not needed: all values are granted to be real and positive.
        if np.sum(np_values) < 1 or population == 0:
            return JitGenoDistrib({k: 0 for k in keys})

        np_values = np_values * population / np.sum(np_values)

        rounded_values = largest_remainder(np_values)

        return self.from_zip(keys, rounded_values)

    def keys(self):
        return list(self._store.keys())

    def values(self):
        return list(self._store.values())

    def __str__(self):
        return "GenoDistrib(" + str(self._store) + ")"


'''
class GenoDistrib(JitGenoDistrib):
    def __init__(self, params: Optional[Union[
        GenoDistrib,
        Dict[Alleles, float],
        Iterable[Tuple[Alleles, float]]
    ]] = None):
        # asdict() will call this constructor with a stream of tuples (Alleles, v)
        # to prevent the default behaviour (Counter counts all the instances of the elements) we have to check
        # for generators

        store = NumbaDict()
        if params:
            if isinstance(params, GeneratorType):
                for k, v in params:
                    store[k] = float(v)
        super().__init__(store)
        """
        elif isinstance(params, dict):
            self._store = cast(Dict[Alleles, float], params)
        elif isinstance(params, GenoDistrib):
            self._store.update(params._store)
        """
'''


@jit
def test1():
    a = JitGenoDistrib({"A": 1.0, "Aa": 2.0})
    print(str(a))
    print(a.normalise_to(10))

test1()