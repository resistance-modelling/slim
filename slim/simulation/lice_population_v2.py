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


@njit
def geno_to_idx(key):
    geno_idx = ord(key[0].lower()) - ord("a")
    dominant = key[0].isupper()
    recessive = key[-1].islower()

    if dominant:
        if recessive:
            allele_variant = 2
        else:
            allele_variant = 1
    else:
        allele_variant = 0

    return geno_idx, allele_variant


# If this code looks nasty, please remember the following:
# 1. Numba's jitclass is very experimental
# 2. Forget about OOP, static methods, variable arguments, Union types (isinstance is broken) etc.
# 3. Castings must always be explicit!
# 4. Until https://github.com/numba/numba/pull/5877 gets merged, there is no overload support for jitclass (oddly
#    enough, except for __getitem__ and __setitem__). Thus, the result is going to be very Java-esque. We could also
#    install that PR but that'd be a pain for the CI...

# the store is an k x 3 array representing populations.
# To make the maths easier: the order is always a, A, Aa
GenoDistribV2_spec = [
    ("num_alleles", int64),
    ("_store", float64[:, :]),
    ("_gross", float64),
    ("_default_probs", float64[:]),
]


@njit
def geno_to_str(gene_idx, allele_idx):
    gene_str = chr(ord("a") + gene_idx)
    if allele_idx == 0:
        return gene_str
    if allele_idx == 1:
        return gene_str.upper()
    else:
        return gene_str.upper() + gene_str


@jitclass(GenoDistribV2_spec)
class GenoDistrib:
    """
    An enriched Dictionary-style container of a statistical population of sea lice arranged by gene.

    The interpretation is the following:
    a genotype is made of distinct genes, each of them appearing in at most 3 different modes:
    recessive (e.g. a), homozygous dominant (e.g. A), heterozygous dominant (e.g. Aa).
    In theory all combinations can occur (e.g. ABbcDEe -> dom. A, het. dom. B, rec. c,
    dom. D, het. dom. e) but we chose not to consider joint probabilities here to save.

    The value of this distribution at a specific allele is the number of lice that possess
    that trait. Note that a magnitude change in one allele will result in a change in others.

    Assuming all genes are i.i.d. we allocate O(k) space.
    """

    def __init__(self, num_alleles: int, default_probs: np.ndarray):
        """
        :param num_alleles: the number of alleles to generate. Please keep this value below 5
        :param default_probs: if the number of lice for a bucket is 0, use this array as a "normalised" probability
        """
        self.num_alleles = num_alleles
        self._store = np.zeros((num_alleles, 3), dtype=np.float64)
        self._default_probs = default_probs
        self._gross = 0.0

    def __sum__(self, o) -> Self:
        assert o.num_alleles == self.num_alleles
        new = GenoDistrib(self.num_alleles)
        for i in range(self.num_alleles):
            new._store[i] = self._store[i] + o._store[i]
        return new

    def __getitem__(self, key: str) -> float64:
        """
        Get the population of the required allele
        See the class documentation for instruction on how to interpret the key.

        :param key: the key
        :returns: the number of lice with the given allele.
        """

        # extract the gene
        geno_id, allele_variant = geno_to_idx(key)

        return self._store[geno_id][allele_variant]

    def __setitem__(self, key, value):
        geno_id, allele_variant = geno_to_idx(key)
        self._gross = self.gross + (value - self._store[geno_id][allele_variant])
        self._store[geno_id][allele_variant] = value
        for i in range(self.num_alleles):
            if i != geno_id:
                self._normalise_single_gene(i, self.gross)

    def _normalise_single_gene(self, gene_idx: int, population: float):
        gene_store = self._store[gene_idx]
        current_gene_gross = np.sum(gene_store)
        if current_gene_gross == 0.0:
            normalised_store = self._default_probs * population
        else:
            normalised_store = (gene_store / current_gene_gross) * population
        self._store[gene_idx] = largest_remainder(normalised_store)

    def _normalise_to(self, population: float):
        new_store = np.empty_like(self._store)
        for idx in range(self.num_alleles):
            line = self._store[idx]
            store_normalised = largest_remainder(line / np.sum(line) * population)
            new_store[idx] = store_normalised

        return new_store

    def normalise_to(self, population: float) -> Self:
        """
        Transform the distribution so that the overall sum changes to the given number
        but the ratios are preserved.
        Note: this operation yields undefined behaviour if the genotype distribution is 0

        :param population: the new population
        :returns: the new distribution
        """

        new_geno = GenoDistrib(self.num_alleles, self._default_probs)
        if population == 0.0:  # small optimisation shortcut
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
        new_geno = GenoDistrib(self.num_alleles, self._default_probs)
        new_geno._store = self._store.copy()
        new_geno._gross = self.gross
        return new_geno

    def add(self, other: Self) -> Self:
        """Overload add operation for GenoDistrib + GenoDistrib operations"""
        res = GenoDistrib(self.num_alleles, self._default_probs)
        res._store = self._store + other._store
        res._gross = self.gross + other.gross
        return res

    def add_to_scalar(self, other: float):
        """Add to scalar."""
        return self.normalise_to(self.gross + other)

    def sub(self, other: Self) -> Self:
        """Overload sub operation"""
        res = GenoDistrib(self.num_alleles, self._default_probs)
        res._store = self._store + other._store
        res._gross = self.gross - other.gross
        return res

    def mul(self, other: Self) -> Self:
        """Multiply a distribution by another distribution.


        In the second case it is simply a pairwise product.

        :param other: a similar distribution
        :returns: the new genotype distribution
        """
        res = GenoDistrib(self.num_alleles, self._default_probs)
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

        res = GenoDistrib(self.num_alleles, self._default_probs)
        res._store = self._store * other
        if truncate:
            res._store = self._normalise_to(res._gross * other)
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
        Get a copy of the store.
        """
        return self._store

    def to_json_dict(self) -> GenoDistribSerialisable:
        """
        Get a JSON representation of the different genotypes.
        """
        to_return = {}

        for gene in range(self.num_alleles):
            for allele in range(3):
                to_return[(geno_to_str(gene, allele))] = self._store[gene][allele]

        return to_return

    def _truncate_negatives(self):
        for idx in range(self.num_alleles):
            for allele in range(3):
                clamped = max(self._store[idx][allele], 0.0)
                delta = self._store[idx][allele] - clamped
                self._store[idx] = clamped
                self._gross += delta

        # Ensure consistency across all stages
        for idx in range(self.num_alleles):
            self._normalise_single_gene(idx, self._gross)


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

'''
# -------------- PRIORITY QUEUE ------------------

# Note: as we cannot define the __lt__ we cannot use dataclasses inside
# our priority queue. The obvious solution, to use pairs, does not work as
# __lt__ for tuples requires GenoDistrib to have one as well.


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
'''

if __name__ == "__main__":
    test = GenoDistrib(2, np.array([0.2, 0.8, 0.2]))
    test["a"] = 50
    test["A"] = 50
    test["Aa"] = 100
    print(test["A"])
    print(test["b"])
    test["b"] = 100
    print(test.to_json_dict())
    test["B"] = 50
    test["Bb"] = 50
    print(test["A"])
    print(test["b"])

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

    test3 = OldGenoDistrib({("a",): 50, ("A",): 100, ("A", "a"): 150})

    print("Testing access of a single genotype")
    print(
        "new: {}".format(
            timeit.timeit("test['A']", globals={"test": test}, number=10000)
        )
    )
    print(
        "old: {}".format(
            timeit.timeit("test[('A',)]", globals={"test": test3}, number=10000)
        )
    )

    print("Testing normalise_to")
    print(
        "new: {}".format(
            timeit.timeit(
                "test.normalise_to(1000)", globals={"test": test}, number=10000
            )
        )
    )

    print(
        "old: {}".format(
            timeit.timeit(
                "test.normalise_to(1000)", globals={"test": test3}, number=10000
            )
        )
    )

    test4 = test.copy()
    test4["A"] = 20
    print("Testing copy")
    print(test._store)
    print(test4._store)

    print("Testing copy speed")
    print(
        "new (deep): {}".format(
            timeit.timeit("test.copy()", globals={"test": test}, number=10000)
        )
    )
    print(
        "old (shallow): {}".format(
            timeit.timeit("test.copy()", globals={"test": test3}, number=10000)
        )
    )

    print(
        "old (deep): {}".format(
            timeit.timeit("test.__deepcopy__()", globals={"test": test3}, number=10000)
        )
    )

    print("Testing truncating negatives")
    print(
        "new: {}".format(
            timeit.timeit(
                "test._truncate_negatives()", globals={"test": test}, number=10000
            )
        )
    )
    print(
        "old: {}".format(
            timeit.timeit(
                "test._truncate_negatives()", globals={"test": test3}, number=10000
            )
        )
    )
