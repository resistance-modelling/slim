"""
This module provides two important classes:

* :class:`GenoDistrib`
* :class:`LicePopulation`

The first is a numba-optimised Dictionary-style container for
gene frequency. Because of that, the API has been geared towards efficiency over
"pythonicity".

For convenience, use the following snippet to create
empty distribution with a default per-allele distribution

>>> cfg = Config(...)
>>> empty_geno = empty_geno_from_cfg(cfg)
>>> nonempty_geno = empty_geno.normalise_to(100)

The second class is a wrapper to manage multiple :class:`GenoDistrib` arranged by
life stage.
"""
from __future__ import annotations

__all__ = [
    "largest_remainder",
    "Allele",
    "GenoDistribDict",
    "GenoDistribSerialisable",
    "GenoDistrib",
    "GenoLifeStageDistrib",
    "GenoTreatmentDistrib",
    "GenoTreatmentValue",
    "GrossLiceDistrib",
    "LifeStage",
    "LicePopulation",
    "from_ratios",
    "from_ratios_rng",
    "from_dict",
    "genorates_to_dict",
]

from functools import lru_cache
from typing import NamedTuple, List, TYPE_CHECKING, Union, Optional

from slim import logger


# from enum import IntEnum
import math
from typing import Dict

import numpy as np
from numba import njit as _njit, float64, int64
from numba.core.types import unicode_type, Array
from numba.experimental import jitclass

from numba.typed.typeddict import Dict as NumbaDict


if TYPE_CHECKING:
    from slim.simulation.config import Config

# Cache by default
def njit(*args, **kwargs):
    return _njit(*args, **kwargs, cache=True)


# ---------- TYPE DECLARATIONS --------------

"""
# With Numba Enums are (finally!) efficient

#: A sea lice life stage
class LifeStage(IntEnum):
    L1 = 0
    L2 = 1
    L3 = 2
    L4 = 3
    L5f = 4
    L5m = 5
"""

#: The type of a gene
Gene = str
#: The type of an allele key
Allele = str
#: The type of a lifestage
LifeStage = str
#: The type of the frequency of a particular allele
GenoDistribFrequency = float
#: The numba type of the frequency of a particular allele
GenoFrequencyType = float64[:]
#: The type of a serialised (JSON-style) representation of a :class:`GenoDistrib`
GenoDistribSerialisable = Dict[Allele, float]
#: The type of a dictionary
GrossLiceDistrib = Dict[LifeStage, int]
#: These are used when GenoDistrib is not available
GenoDistribDict = Dict[Allele, float]
GenoRates = np.ndarray


class GenoTreatmentValue(NamedTuple):
    mortality_rate: float
    susceptible_stages: List[LifeStage]


GenoTreatmentDistrib = Dict[Allele, GenoTreatmentValue]

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


@njit(float64[:, :](float64[:, :]))
def largest_remainder_matrix(nums: np.ndarray):
    res = np.empty_like(nums)
    for idx in range(nums.shape[0]):
        res[idx] = largest_remainder(nums[idx])
    return res


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


@njit
def geno_to_str(gene_idx, allele_idx):
    gene_str = chr(ord("a") + gene_idx)
    if allele_idx == 0:
        return gene_str
    if allele_idx == 1:
        return gene_str.upper()
    else:
        return gene_str.upper() + gene_str


def geno_config_to_matrix(geno_dict: GenoDistribDict) -> np.ndarray:
    """
    Convert a dictionary of genes into a matrix.

    We recommend using :func:`from_dict` as it will call this helper as well.
    """
    num_genes = len(geno_dict) // 3
    translation_dict = _alleles_to_geno_dict(num_genes)
    matrix = np.zeros((num_genes, 3), dtype=np.float64)
    for k, v in translation_dict.items():
        row = v // 3
        col = v % 3
        matrix[row, col] = geno_dict[k]
    return matrix


@lru_cache(maxsize=None)
def geno_to_alleles(gene: Union[Gene, int]) -> List[Allele]:
    """
    Get the alleles that stem from a single gene.
    Note that here "allele" is conflated to represent pairs of such alleles at the same
    gene locus.
    :param gene: the gene
    :returns: the alleles
    """
    if isinstance(gene, int):
        gene = geno_to_str(gene, 0)

    dominant = gene[0].upper()
    recessive = gene[0].lower()
    return [recessive, dominant, dominant + recessive]


@lru_cache(maxsize=None)
def _alleles_to_geno_dict(num_genes):
    geno_dict = {}
    for i in range(num_genes):
        keys = geno_to_alleles(i)
        for j, k in enumerate(keys):
            geno_dict[k] = i * 3 + j
    return geno_dict


@njit
def genorates_to_dict(store: GenoRates) -> GenoDistribSerialisable:
    """
    Get a JSON representation of the geno rates.

    :param store: an array of genotype rates as plain matrix
    :returns: a JSON representation
    """
    to_return = dict()
    num_alleles = store.shape[0]

    for gene in range(num_alleles):
        for allele in range(3):
            to_return[(geno_to_str(gene, allele))] = store[gene][allele]

    return to_return


# -------------- GenoDistrib class ---------------
# If this code looks nasty, please remember the following:
# 1. Numba's jitclass is very experimental
# 2. Forget about OOP, static methods (they do exist but are inaccessible inside a no-object
#    pipeline), variable arguments, Union types (isinstance is broken) etc.
# 3. Castings must always be explicit!
# 4. Until https://github.com/numba/numba/pull/5877 gets merged, there is no overload support for jitclass (oddly
#    enough, except for __getitem__ and __setitem__). Thus, the result is going to be very Java-esque. We could also
#    install that PR but that'd be a pain for the CI...

# the store is an k x 3 array representing populations.
# To make the maths easier: the order is always a, A, Aa
GenoDistribV2_spec = [
    ("_store", float64[:, :]),
    ("_gross", float64),
    ("_default_probs", float64[:, :]),
]


@jitclass(GenoDistribV2_spec)
class GenoDistrib:
    """
    An enriched Dictionary-style container of a statistical population of sea lice arranged by gene.

    The interpretation is the following:
    a genotype is made of distinct genes, each of them appearing in at most 3 different modes:
    recessive (e.g. a), homozygous dominant (e.g. A), heterozygous dominant (e.g. Aa).

    The value of this distribution at a specific allele is the number of lice that possess
    that trait. Note that a magnitude change in one allele will result in a change in others.

    Assuming all genes are i.i.d. we allocate O(k) space.
    """

    def __init__(self, default_probs: np.ndarray, _no_init: bool = False):
        """
        Creates an empty `GenoDistrib`. An "empty" genotype distribution is a distribution
        with zero lice but preallocated data structures, therefore both k and the
        default ratios are required in advance.


        :param num_alleles: the number of alleles to generate. Please keep this value below 5
        :param default_probs: if the number of lice for a bucket is 0, use this array as a "normalised" probability
        :param _no_init: (ADVANCED) if provided do not initialise the store matrix with 0.
        """
        if _no_init:
            self._store = np.empty_like(default_probs, dtype=np.float64)
        else:
            self._store = np.zeros_like(default_probs, dtype=np.float64)
        self._default_probs = default_probs
        self._gross = 0.0

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
        for i in range(self.num_genes):
            if i != geno_id:
                self._store[i] = self._normalise_single_gene(i, self.gross)

    def setcounts(self, counts: np.ndarray):
        self._store = counts
        self._gross = np.sum(self._store[0])

    def _normalise_single_gene(self, gene_idx: int, population: float) -> np.ndarray:
        gene_store = self._store[gene_idx]
        current_gene_gross = np.sum(gene_store)
        if current_gene_gross == 0.0:
            normalised_store = largest_remainder(
                self._default_probs[gene_idx] * population
            )
        else:
            normalised_store = largest_remainder(
                (gene_store / current_gene_gross) * population
            )

        return normalised_store

    def _normalise_to(self, population: float):
        new_store = np.empty_like(self._store)
        for idx in range(self.num_genes):
            normalised = self._normalise_single_gene(idx, population)
            new_store[idx] = normalised

        return new_store

    def normalise_to(self, population: float):
        """
        Transform the distribution so that the overall sum changes to the given number
        but the ratios are preserved.
        Note: this operation yields undefined behaviour if the genotype distribution is 0

        :param population: the new population
        :returns: the new distribution
        """

        new_geno = GenoDistrib(self._default_probs, False)
        if population == 0.0:  # small optimisation shortcut
            return new_geno
        truncated_store = self._normalise_to(population)
        new_geno._store = truncated_store
        new_geno._gross = population
        return new_geno

    def set(self, population: float):
        """This is the inplace version of :meth:`normalise_to`.

        :param population: the new population
        """
        self._store = self._normalise_to(population)
        self._gross = float(population)

    @property
    def gross(self):
        return self._gross

    @property
    def num_genes(self):
        return len(self._store)

    def copy(self):
        """Create a copy"""
        new_geno = GenoDistrib(self._default_probs, False)
        new_geno._store = self._store.copy()
        new_geno._gross = self.gross
        return new_geno

    def add(self, other):
        """Add operation between GenoDistrib and GenoDistrib"""
        res = GenoDistrib(self._default_probs, False)
        res._store = self._store + other._store
        res._gross = self.gross + other.gross
        return res

    def iadd(self, other):
        """Inplace add operation between GenoDistrib and GenoDistrib"""
        self._store += other._store
        self._gross += other.gross

    def iadd_scalar(self, other: float):
        self.set(self.gross + other)

    def __iadd__(self, other):
        """Overloaded inplace add operator.
        Note: available only when the JIT is disabled.
        """

        if isinstance(other, float):
            self.iadd_scalar(other)
        else:
            self.iadd(other)

    def add_to_scalar(self, other: float):
        """Add to scalar."""
        return self.normalise_to(self.gross + other)

    def __add__(self, other: Union[GenoDistrib, float]):
        """Overloaded add operator.

        Note: available only when the JIT is disabled.
        """

        if isinstance(other, float):
            return self.add_to_scalar(other)
        return self.add(other)

    def sub(self, other: GenoDistrib) -> GenoDistrib:
        """Overload sub operation"""
        res = GenoDistrib(self._default_probs, False)
        res._store = self._store - other._store
        res._gross = self.gross - other.gross
        return res

    def __sub__(self, other):
        """
        Overloaded sub operator

        Note: available only when the JIT is disabled.
        """
        return self.sub(other)

    def mul(self, other: GenoDistrib) -> GenoDistrib:
        """Multiply a distribution by another distribution.

        :param other: a similar distribution
        :returns: the new genotype distribution
        """
        res = GenoDistrib(self._default_probs, False)
        res._store = self._store * other._store
        res._gross = self.gross * other.gross
        return res

    def mul_by_scalar(self, other: float) -> GenoDistrib:
        """
        Multiply a distribution by a scalar.

        The multiplication by a scalar will multiply each bin by the scalar.
        If the new sum is expected to be an integer a rounding step
        is applied.

        :param other: a scalar
        :returns: the new genotype distribution
        """
        truncate = math.trunc(other) == other

        res = GenoDistrib(self._default_probs, False)
        res._store = self._store * other
        res._gross = self._gross * other
        if truncate:
            res._store = self._normalise_to(res._gross)
        return res

    def __mul__(self, other: Union[GenoDistrib, float]):
        """Overloaded mul operator.

        Note: available only when the JIT is disabled.
        """

        if isinstance(other, float):
            return self.mul_by_scalar(other)
        return self.mul(other)

    def equals(self, other: GenoDistrib):
        """Overloaded eq operator"""
        return np.all(self._store == other._store)

    def equals_dict(self, other: GenoDistribDict):
        """Overloaded eq operator for dicts.
        Note: this is mainly meant for testing. Be sure to test without the JIT on,
        otherwise you may have to manually allocate a numba dict.
        """
        return self.to_json_dict() == other

    def __eq__(self, other: Union[GenoDistrib, GenoDistribDict]):
        """Overloaded eq operator.

        Note: this function is currently inaccessible as long as JIT is enabled.
        Use this function for testing only.
        """
        if isinstance(other, dict):
            return self.equals_dict(other)
        return self.equals(other)

    def is_positive(self) -> bool:
        """:returns: `True` if all the bins are positive."""
        return bool(np.all(self._store >= 0))

    def keys(self):
        """
        :returns the list of keys
        """
        return [geno_to_str(i, j) for j in range(3) for i in range(self.num_genes)]

    def values(self):
        """
        :returns: a reference of the store
        """
        return self._store

    def to_json_dict(self) -> GenoDistribSerialisable:
        """
        Get a JSON representation of the different genotypes.
        """
        return genorates_to_dict(self._store)

    def __repr__(self):
        """Note: only works when no JIT-ing occurs. Mainly intended for testing"""
        return "GenoDistrib(" + repr(self.to_json_dict()) + ")"

    @staticmethod
    def batch_sum(batches: List[GenoDistrib]) -> GenoDistrib:
        """
        Calculate the sum of multiple :class:`GenoDistrib` instances.
        This function is a bit faster than manually invoking a folding operation
        due to reduced overhead.

        The caller must ensure the list of batches is non-empty and homogeneous.
        The default rates used for the generation are the same of the first element.

        :param batches: either a list of distribution or a list of dictionaries (useful for pandas)
        :returns: a :class:`GenoDistrib`
        """

        x = np.zeros_like(batches[0]._store)
        gross = 0.0
        for batch in batches:
            x += batch._store
            gross += batch.gross

        default_distrib = batches[0]._default_probs

        res = GenoDistrib(default_distrib, False)
        res._store = x
        res._gross = gross

        return res

    def _truncate_negatives(self):
        for idx in range(self.num_genes):
            for allele in range(3):
                clamped = max(self._store[idx][allele], 0.0)
                delta = self._store[idx][allele] - clamped
                self._store[idx] = clamped
                self._gross += delta

        # Ensure consistency across all stages
        self._store = self._normalise_to(self._gross)

    def mutate(self):
        pass

    def mutate(self, mutations: int):
        """
        Mutate the genotype distribution in-place.

        There are only three directions: R->ID, ID->D, ID->R, D->ID. Note that
        R->D or D->R are impossible via a single mutation, and the probability
        of a double mutation is low enough to be ignored. Self-mutations are ignored.

        The function uses a best-effort algorithm to derive mutations.

        :param mutations: the number of mutations to perform
        """

        # Thus, our objective is to sample in such a way that:
        # Na' = Na + x - y
        # NA' = NA + z - w
        # NAa' = NAa + (y + w) - (x + z)
        # Na', NA', NAa' >= 0
        # 0 <= x, y, z
        # x + y + z <= n

        # The algorithm here is a probabilistic algorithm that attempts mutation in the general case,
        # or removes selectively from one bin if the others are empty (e.g. when starting a simulation)

        p = np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4], dtype=np.float64)
        for gene in range(self.num_genes):
            gene_array = self._store[gene]
            for i in range(10):
                # Note: numba does not support random states
                swaps = np.random.multinomial(mutations, p)
                # Does destructuring work in numba?
                x = swaps[0]
                y = swaps[1]
                z = swaps[2]
                w = swaps[3]

                new_array = gene_array + np.array(
                    [x - y, z - w, y + w - x - z], dtype=np.float64
                )
                if np.all(new_array >= 0.0):
                    self._store[gene] = new_array


GenoLifeStageDistrib = Dict[LifeStage, GenoDistrib]


def _dict_to_numba(x: GenoDistribDict):
    p_numba = NumbaDict.empty(unicode_type, float64)
    for k, v in x.items():
        p_numba[k] = v

    return p_numba


def from_ratios_rng(n: int, pvals: GenoRates, rng: np.random.Generator) -> GenoDistrib:
    """
    Create a :class:`GenoDistrib` with a given number of lice and a probability distribution
    Note that this function uses a *statistical* approach to building such distribution.
    If you prefer a faster but deterministic alternative consider :func:`from_ratios` .

    :param n: the number of lice
    :param pvals: the probability of each genotype
    :param rng: the random number generator to use for sampling.
    :returns: the new GenoDistrib
    """

    values = np.stack([rng.multinomial(n, p) for p in pvals]).astype(np.float64)

    res = GenoDistrib(pvals, False)
    res._store = values
    res._gross = n
    return res


# @njit
def _from_ratios(p, n):
    config_matrix = p
    sums = np.sum(config_matrix, axis=1)[:, np.newaxis]
    probas = config_matrix / sums
    res = GenoDistrib(probas, False)
    if n > -1:
        return res.normalise_to(n)
    return res


def from_ratios(p: Union[GenoDistribDict, GenoRates], n: int = -1) -> GenoDistrib:
    """
    Create a :class:`GenoDistrib` with a given number of lice and fixed ratios.
    Note that this function uses a *deterministic* approach based on the
    largest remainder method. If you prefer a noisier approach consider
    :func:`from_ratios_rng` .

    :param p: the probability of each genotype
    :param n: if not -1, set the distribution so that the gross value is equal to n
    :returns: the new GenoDistrib
    """
    if isinstance(p, dict):
        p = geno_config_to_matrix(p)
    return _from_ratios(p, n)


def from_dict(p: GenoDistribDict) -> GenoDistrib:
    """
    Generate a GenoDistrib from a dictionary.

    The default rates will be assumed to be the same as the current population arranged by gene.

    Useful for testing and quick prototyping.
    """
    keys = ["a", "A", "Aa"]
    population = int(sum(p[k] for k in keys))
    return from_ratios(p, population)


def from_counts(counts: np.ndarray, cfg: Config):
    default_probs = geno_config_to_matrix(cfg.initial_genetic_ratios)
    geno = GenoDistrib(default_probs)
    geno.setcounts(counts)
    return geno


def empty_geno_from_cfg(cfg: Config):
    """
    Generate an empty GenoDistrib with the correct default rates and number alleles.

    :param cfg: the current configuration

    :returns: an empty GenoDistrib
    """

    return from_ratios(cfg.initial_genetic_ratios)


# See https://stackoverflow.com/a/7760938
class LicePopulation:
    """
    Wrapper to keep the global population and genotype information updated.

    This class provides both a gross-based and a geno-based view of a lice population.
    It also manages busy dam distributions when mating is involved.

    The stages population can be accessed via either dictionary access (gross) or via
    the ``geno_by_lifestage` attribute (detailed).
    Additionally we provide two "virtual" stages ("L5f_busy" and "L5f_free") as read-only
    convenience accessors in place of the verbose `busy_dams` and `available_dams` attributes.
    """

    lice_stages = ["L1", "L2", "L3", "L4", "L5f", "L5m"]
    lice_stages_bio_labels = dict(zip(lice_stages, ["R", "CO", "CH", "PA", "AF", "AM"]))
    lice_stages_bio_long_names = dict(
        zip(
            lice_stages,
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

    susceptible_stages = lice_stages[2:]
    pathogenic_stages = lice_stages[3:]

    def __init__(
        self,
        geno_data: GenoLifeStageDistrib,
        generic_ratios: GenoDistribDict,
        busy_dam_waiting_time: float,
    ):
        """
        :param geno_data: the default genetic ratios
        :param generic_ratios: a config to use
        :param busy_dam_waiting_time: the average pregnancy period for dams (in days)
        """
        self.geno_by_lifestage = geno_data.copy()
        self.genetic_ratios = generic_ratios
        self._busy_dam_arrival_rate = 0.0
        self._busy_dam_waiting_time = busy_dam_waiting_time

    def __setitem__(self, stage: LifeStage, value: int):
        # If one attempts to make gross modifications to the population these will be repartitioned according to the
        # current genotype information.
        old_value = self[stage]
        if old_value == 0:
            if value > 0:
                logger.warning(
                    f"Trying to initialise population {stage} with null genotype distribution. Using default genotype information."
                )
            self.geno_by_lifestage[stage] = from_ratios(
                self.genetic_ratios
            ).normalise_to(value)
        else:
            if value < 0:
                logger.warning(
                    f"Trying to assign population stage {stage} a negative value. It will be truncated to zero, but this probably means an invariant was broken."
                )
            elif value == old_value:
                return
            value = max(value, 0)
            self.geno_by_lifestage[stage] = self.geno_by_lifestage[stage].normalise_to(
                value
            )

            # When L5f lice die the arrival rate should drop
            # TODO: should we replace the arrival rate with the inverted load?
            if stage == "L5f":
                self._busy_dam_arrival_rate = min(
                    [
                        self["L5f"] / self._busy_dam_waiting_time - 1,
                        self._busy_dam_arrival_rate,
                    ]
                )

    def __getitem__(self, stage: LifeStage) -> int:
        if stage == "L5f_busy":
            geno = self.busy_dams
        elif stage == "L5f_free":
            geno = self.available_dams
        else:
            geno = self.geno_by_lifestage[stage]
        return int(geno.gross)

    def __len__(self) -> int:
        return len(self.geno_by_lifestage)

    def __repr__(self) -> str:
        return "LicePopulation" + repr(self.to_json_dict())

    def __str__(self) -> str:
        return f"LicePopulation(" + str(self.as_dict()) + ")"

    def as_dict(self) -> Dict[LifeStage, int]:
        to_return = {
            stage: distrib.gross for stage, distrib in self.geno_by_lifestage.items()
        }

        return to_return

    def as_full_dict(self):
        return {
            stage: distrib.to_json_dict()
            for stage, distrib in self.geno_by_lifestage.items()
        }

    def clear_busy_dams(self):
        self._busy_dam_arrival_rate = 0.0

    @property
    def available_dams(self) -> GenoDistrib:
        return self.geno_by_lifestage["L5f"].sub(self.busy_dams)

    @property
    def busy_dams(self) -> GenoDistrib:
        # Little's law
        return self.geno_by_lifestage["L5f"].normalise_to(
            round(self._busy_dam_waiting_time * self._busy_dam_arrival_rate)
        )

    def add_busy_dams_batch(self, num_dams: int):
        # To "emulate" a queue we perform a very simple weighted mean over time
        assert num_dams <= self["L5f"], "Busy dam overflow detected"
        self._busy_dam_arrival_rate = (
            1 - 1 / self._busy_dam_waiting_time
        ) * self._busy_dam_arrival_rate + (1 / self._busy_dam_waiting_time) * num_dams

    def remove_negatives(self):
        for distrib in self.geno_by_lifestage.values():
            distrib._truncate_negatives()

    @staticmethod
    def get_empty_geno_distrib(cfg: Config) -> GenoLifeStageDistrib:
        return {stage: empty_geno_from_cfg(cfg) for stage in LicePopulation.lice_stages}

    def to_json_dict(self) -> GenoDistribSerialisable:
        return self.as_dict()

    def __eq__(self, other):
        """Note: only works when not JIT-ing"""
        return self.to_json_dict() == other

    def values(self):
        return [int(geno.gross) for geno in self.geno_by_lifestage.values()]
