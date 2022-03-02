"""
Hic sunt leones.
"""
import math
import timeit
from typing import Any, Union, Dict

import numpy as np
from numba import njit, float64, int64, vectorize
from numba.core.extending import overload_method
from numba.experimental import jitclass
from numba.extending import overload

from slim.simulation.lice_population import GenoDistrib

from typing_extensions import Self

AlleleType = int
Alleles = int


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
def bitstring_to_str(seq: int, k: int):
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
class GenoDistribV2:
    """
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
        new = GenoDistribV2(self.num_alleles)
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

        new_geno = GenoDistribV2(self.num_alleles)
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
        new_geno = GenoDistribV2(self.num_alleles)
        new_geno._store = self._store.copy()
        new_geno._gross = self.gross
        return new_geno

    def add(self, other: Self) -> Self:
        """Overload add operation for GenoDistrib + GenoDistrib operations"""
        res = GenoDistribV2(self.num_alleles)
        res._store = self._store + other._store
        res._gross = self.gross + other.gross
        return res

    def add_to_scalar(self, other: int):
        """Add to scalar."""
        return self.normalise_to(self.gross + other)

    def sub(self, other: Self) -> Self:
        """Overload sub operation"""
        res = GenoDistribV2(self.num_alleles)
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
        res = GenoDistribV2(self.num_alleles)
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

        res = GenoDistribV2(self.num_alleles)
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

    def to_json_dict(self) -> Dict[str, float]:
        # starting from the LSB, the genes are just called "a", "b" etc.
        to_return = {}

        for idx, freq in enumerate(self._store):
            to_return[bitstring_to_str(idx, self.num_alleles)] = freq

        return to_return

    def _truncate_negatives(self):
        for k in self._store:
            self[k] = max(self._store[k], 0.0)


test = GenoDistribV2(2)
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
"""
test2 = test.normalise_to(1000)
print(test2.gross)
print(test2._store)
print(test.normalise_to(0))
print(test.add(test2)._store)
print(test.sub(test2)._store)
print(test.mul(test2)._store)
print(test.mul_by_scalar(1 / 3)._store)


print(timeit.timeit("test.normalise_to(1000)", globals={"test": test}, number=10000))

test3 = GenoDistrib({("a",): 50, ("A",): 100, ("A", "a"): 150})
print(timeit.timeit("test.normalise_to(1000)", globals={"test": test3}, number=10000))

test4 = test.copy()
test4[0b0000] = 20
print("Testing copy")
print(test._store)
print(test4._store)

"""
