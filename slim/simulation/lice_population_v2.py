"""
This module provides optimised implementations of genotype distributions
and manipulation tools.

Hic sunt leones.
"""
from __future__ import annotations

__all__ = [
    "LifeStage",
    "GeneType",
    "GenoDistrib",
    "GenoFrequencyType",
    "GrossLiceDistrib",
    "largest_remainder",
    "LICE_STAGES_BIO_LABELS",
    "PATHOGENIC_STAGES",
]

import datetime as dt
from enum import IntEnum
from heapq import heappush, heappop
import math
import timeit
from typing import Dict

import numpy as np
from numba import njit, float64, int64
from numba.experimental import jitclass

# from numba.typed import Dict as NumbaDict

from slim.simulation.lice_population import GenoDistrib as OldGenoDistrib

# Needed as numba's decorator breaks the type checker
from typing_extensions import Self

# ---------- TYPE DECLARATIONS --------------
# With Numba Enums are (finally!) efficient

#: A sea lice life stage
class LifeStage(IntEnum):
    L1 = 0
    L2 = 1
    L3 = 2
    L4 = 3
    L5f = 4
    L5m = 5


#: The type of a gene
GeneType = int
#: The type of the frequency of a particular allele
GenoDistribFrequency = float
#: The numba type of the frequency of a particular allele
GenoFrequencyType = float64[:]
#: The type of a serialised (JSON-style) representation of a :class:`GenoDistrib`
GenoDistribSerialisable = Dict[str, float]
#: The type of a mapping between each recessive base and its probability
#  E.g. "a" -> 0.2 means the probability that the first of the two heterozygous
#  bases is an "a" is :math:`0.2`; the probability to have a fully homozygous recessive
#  gene is thus :math:`0.2^2 = 0.04`, for the heterozygous dominant it is :math:`2 \cdot 0.2 \cdot (1-0.2) = 0.32`
#  and the dominant it is :math:`(1-0.2)^2 = 0.64`.
#  Also note that all the bases are assumed to be uncorrelated lest combinatorial explosion occurs.
GenoRates = Dict[str, float]
#: The type of a dictionary
GrossLiceDistrib = Dict[LifeStage, int]
# --------------------------------------------


@njit(float64[:](float64[:]), fastmath=True)
def largest_remainder(nums):
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

    return approx


# If this code looks nasty, please remember the following:
# 1. Numba's jitclass is very experimental
# 2. Forget about OOP, static methods, variable arguments, Union types (isinstance is broken) etc.
# 3. Castings must always be explicit!
# 4. Until https://github.com/numba/numba/pull/5877 gets merged, there is no overload support for jitclass (oddly
#    enough, except for __getitem__ and __setitem__). Thus, the result is going to be very Java-esque. We could also
#    install that PR but that'd be a pain for the CI...

GenoDistribV2_spec = [
    ("num_alleles", int64),
    ("_store", float64[:]),
    ("_gross", float64),
]


@njit
def bitstring_to_str(seq: int, k: int) -> str:
    """
    :param seq: a bit sequence of :math:`2k` bits
    :param k: the aforementioned :math:`k`

    :returns: a human-readable string representation of such genes
    """
    rhs_mask = 0b1
    lhs_mask = 0b1 << k
    gene_name = ord("a")
    res = ""

    for i in range(k):
        rhs = seq & rhs_mask
        lhs = (seq & lhs_mask) >> k
        if rhs & lhs:
            res += chr(gene_name).upper()
        elif not (rhs | lhs):
            res += chr(gene_name)
        else:
            res += chr(gene_name).upper() + chr(gene_name)

        rhs_mask <<= 1
        lhs_mask <<= 1
        gene_name += 1

    return res


@jitclass(GenoDistribV2_spec)
class GenoDistrib:
    """
    An enriched Dictionary-style container of a statistical population of sea lice arranged by gene.

    Internal implementation: we assume a simple scheme where a genotype is a pair of two bitsets
    (or in this case two concatenated ones). When a bit at a position i is 0 the allele is considered recessive,
    whereas when it is 1 it is considered dominant. The combination of the ith and (2^k+i)th bit will give us
    the desired genotype.

    Technically all the operations here have O(2^(2k)) time and space complexity, which is fine for small k.
    """

    def __init__(self, num_alleles: int):
        """
        :param num_alleles: the number of alleles to generate. Please keep this value below 5
        """
        self.num_alleles = num_alleles
        length = 1 << (2 * num_alleles)
        self._store = np.zeros(length, dtype=np.float64)
        self._gross = 0

    def __sum__(self, o) -> Self:
        assert o.num_alleles == self.num_alleles
        new = GenoDistrib(self.num_alleles)
        for i in range(self.num_alleles * 2):
            new._store[i] = self._store[i] + o._store[i]
        return new

    def __getitem__(self, idx):
        """
        Get the number of dominant, recessive and partially dominant genes for the ith bit.

        This method has :math:`\mathcal{\Theta}(2^{2k})` complexity with :math:`k` being the
        number of loci.
        """
        dominant = 0
        recessive = 0
        partial_dominant = 0
        # TODO: inspect whether AVX instructions are generated here (they likely should...)
        for i in range(1 << (self.num_alleles * 2)):
            rhs = i & (1 << idx)
            lhs = (i & (1 << (self.num_alleles + idx))) >> (self.num_alleles)

            if not (rhs | lhs):  # NOR = homozygous recessive
                recessive += self._store[i]

            elif rhs ^ lhs:  # XOR = heterozygous dominant
                print(i, rhs, lhs, self._store[i])
                partial_dominant += self._store[i]

            else:  # == AND = homozygous dominant
                dominant += self._store[i]

        return recessive, partial_dominant, dominant

    def __setitem__(self, key, value):
        self._gross = self.gross + (value - self._store[key])
        self._store[key] = value

    def _normalise_to(self, population: int):
        store = self._store / np.sum(self._store)
        return largest_remainder(store * population)

    def normalise_to(self, population: int) -> Self:
        """
        Transform the distribution so that the overall sum changes to the given number
        but the ratios are preserved.
        Note: this operation yields undefined behaviour if the genotype distribution is 0

        :param population: the new population
        :returns: the new distribution
        """

        new_geno = GenoDistrib(self.num_alleles)
        if population == 0:  # small optimisation shortcut
            return new_geno
        truncated_store = self._normalise_to(population)
        new_geno._store = truncated_store
        new_geno._gross = population
        return new_geno

    def set(self, population: int) -> Self:
        """This is the inplace version of :meth:`normalise_to`.

        :param population: the new population
        """
        self._store = self._normalise_to(population)
        self._gross = float(population)

    @property
    def gross(self):
        return self._gross

    def copy(self):
        """Create a copy"""
        new_geno = GenoDistrib(self.num_alleles)
        new_geno._store = self._store.copy()
        new_geno._gross = self.gross
        return new_geno

    def add(self, other: Self) -> Self:
        """Overload add operation for GenoDistrib + GenoDistrib operations"""
        res = GenoDistrib(self.num_alleles)
        res._store = self._store + other._store
        res._gross = self.gross + other.gross
        return res

    def add_to_scalar(self, other: int):
        """Add to scalar."""
        return self.normalise_to(self.gross + other)

    def sub(self, other: Self) -> Self:
        """Overload sub operation"""
        res = GenoDistrib(self.num_alleles)
        res._store = self._store + other._store
        res._gross = self.gross - other.gross
        return res

    # def __mul__(self, other: Union[float, GenoDistrib]) -> GenoDistrib:
    def mul(self, other: Self) -> Self:
        """Multiply a distribution by another distribution.


        In the second case it is simply a pairwise product.

        :param other: a similar distribution
        :returns: the new genotype distribution
        """
        res = GenoDistrib(self.num_alleles)
        res._store = self._store * other._store
        res._gross = self.gross * other.gross
        return res

    def mul_by_scalar(self, other: float) -> Self:
        """
        Multiply a distribution by a scalar.

        In the first case, the multiplication by a scalar will multiply each bin
        by the scalar. If the new sum is expected to be an integer a rounding step
        is applied.

        :param other: a scalar
        :returns: the new genotype distribution
        """
        truncate = math.trunc(other) == other

        res = GenoDistrib(self.num_alleles)
        res._store = self._store * other
        if truncate:
            res._store = largest_remainder(res._store)
        res._gross = np.sum(res._store)
        return res

    def equals(self, other: Self):
        """Overloaded eq operator"""
        # this is mainly useful for simple testing
        return self._store == other._store

    def is_positive(self) -> bool:
        """:returns True if all the bins are positive."""
        return all(v >= 0 for v in self._store)

    def values(self):
        """
        Avoid calling this.
        """
        return self._store

    def to_json_dict(self) -> GenoDistribSerialisable:
        # starting from the LSB, the genes are just called "a", "b" etc.
        to_return = {}

        for idx, freq in enumerate(self._store):
            to_return[bitstring_to_str(idx, self.num_alleles)] = freq

        return to_return

    def _truncate_negatives(self):
        for idx in range(1 << 2 * self.num_alleles):
            clamped = max(self._store[idx], 0.0)
            delta = self._store[idx] - clamped
            self._store[idx] = clamped
            self._gross += delta


GenoLifeStageDistrib = Dict[LifeStage, GenoDistrib]

_lice_stages_list = list(LifeStage)
LICE_STAGES_BIO_LABELS = dict(
    zip(_lice_stages_list, ["R", "CO", "CH", "PA", "AF", "AM"])
)
lice_stages_bio_long_names = dict(
    zip(
        _lice_stages_list,
        [
            "Recruitment",
            "Copepopid",
            "Chalimus",
            "Preadult",
            "Adult Female",
            "Adult Male",
        ],
    )
)

PATHOGENIC_STAGES = _lice_stages_list[3:]

# -------------- PRIORITY QUEUE ------------------


@dataclass(order=True)
class DamAvailabilityBatch:
    availability_date: dt.datetime  # expected return time
    geno_distrib: GenoDistrib = field(compare=False)

    @property
    def event_time(self):
        return self.availability_date


@jitclass()
class _PriorityQueue:
    """A mutex-free reimplementation of a priority queue via heapification."""

    def __init__(self):
        self.queue = []

    def qsize(self):
        return len(self.queue)

    def put(self, item):
        heappush(self.queue, item)

    def get(self):
        return heappop(self.queue)


# ------------- END PRIORITY QUEUE ---------------

# ------------- LICE POPULATION ------------------

LicePopulation_spec = [
    ("num_alleles", int64),
    ("_store", float64[:]),
    ("_gross", float64),
]


@jitclass
class LicePopulation:
    """
    A cage-specific population of lice.

    This class provides two "views" of the population: as gross numbers and fine-grained
    to the genotype. Both of these are common use cases and require special attention
    due to the hidden computational cost.

    Furthermore, it takes into account busy female lice from being considered for mating
    purposes.
    """

    def __init__(self, geno_data: GenoLifeStageDistrib, generic_ratios: GenoRates):
        self._cache: Dict[LifeStage, int] = {}
        self._busy_dams = _PriorityQueue()
        self._busy_dams_cache = GenoDistrib()
        self.genetic_ratios = generic_ratios
        for stage, distrib in geno_data.items():
            self._cache[stage] = distrib.gross

    def __setitem__(self, key, value):
        pass


x = LicePopulation()
print(x.lice_stages)

"""
test = GenoDistrib(2)
test[0b0000] = 50
test[0b0001] = 50
test[0b0100] = 100
test[0b0101] = 100
test[0b1000] = 50
test[0b1001] = 50
test[0b1111] = 25
print(test[0])
print(test[1])
print(test.to_json_dict())
test2 = test.normalise_to(1000)
print(test2.gross)
print(test2._store)
print(test.normalise_to(0))
print(test.add(test2)._store)
print(test.sub(test2)._store)
print(test.mul(test2)._store)
print(test.mul_by_scalar(1 / 3)._store)

test._truncate_negatives()

print(timeit.timeit("test.normalise_to(1000)", globals={"test": test}, number=10000))

test3 = OldGenoDistrib({("a",): 50, ("A",): 100, ("A", "a"): 150})
print(timeit.timeit("test.normalise_to(1000)", globals={"test": test3}, number=10000))

test4 = test.copy()
test4[0b0000] = 20
print("Testing copy")
print(test._store)
print(test4._store)

print(timeit.timeit("test._truncate_negatives()", globals={"test": test}, number=10000))
print(
    timeit.timeit("test._truncate_negatives()", globals={"test": test3}, number=10000)
)

"""
