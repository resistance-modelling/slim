from __future__ import annotations

__all__ = [
    "largest_remainder",
    "Allele",
    "Alleles",
    "GenoDistribDict",
    "GenoDistribSerialisable",
    "GenoDistrib",
    "GenoLifeStageDistrib",
    "GenoTreatmentValue",
    "GrossLiceDistrib",
    "LicePopulation"
]

import math
from abc import ABC
import datetime as dt
from heapq import heapify
from queue import PriorityQueue
from types import GeneratorType
from typing import Dict, MutableMapping, Tuple, Union, NamedTuple, Optional, \
    Iterable, TYPE_CHECKING, List, cast, Iterator

import numpy as np

from slim import logger
from slim.types.QueueTypes import pop_from_queue

if TYPE_CHECKING:  # pragma: no cover
    from slim.types.QueueTypes import DamAvailabilityBatch

################ Basic type aliases #####################
LifeStage = str
Allele = str
Alleles = Tuple[Allele, ...]
# These are used when GenoDistrib is not available
GenoDistribSerialisable = Dict[str, float]
GenoDistribDict = Dict[Alleles, float]


def largest_remainder(nums: np.ndarray) -> np.ndarray:
    """
    An implementation of the Largest Remainder method.
    The aim of this function is to round an array so that the integer sum
    is preserved.

    :param nums: the number to truncate
    :returns: an array of numbers such that the integer sum is preserved
    """

    # a vectorised implementation of largest remainder
    assert np.all(nums >= 0) or np.all(nums <= 0)

    nums = nums.astype(np.float32)
    approx = np.trunc(nums, dtype=np.float32)

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

    assert np.rint(np.sum(approx)) == np.rint(np.sum(nums)), f"{approx} is not summing to {nums}, as the first sums to {np.sum(approx)} and the second to {np.sum(nums)}"
    return approx


class GenoDistrib(MutableMapping[Alleles, float], ABC):
    """
    A GenoDistrib is a distribution of genotypes.

    Each GenoDistrib provides custom operator overloading operations and suitable
    rescaling operations.
    Internally, this is built to mimic the Counter type but implements smarter move semantics
    when possible.
    """

    alleles: List[Alleles] = [('a',), ('A', 'a'), ('A',)]
    allele_labels = ["".join(allele) for allele in alleles]

    def __init__(self, params: Optional[Union[
        GenoDistrib,
        Dict[Alleles, float],
        Iterable[Tuple[Alleles, float]]
    ]] = None):
        """A :class:`GenoDistrib` can be built in the following ways:

        * Empty

          >>> a = GenoDistrib()

        * From a dictionary. The keys must be the same in alleles

          >>> b = GenoDistrib({('A',): 10})
        * From a generator

          >>> c = GenoDistrib(zip(GenoDistrib.alleles, [1, 2, 3]))
        * From another :class:`GenoDistrib`. This will create a distinct copy of keys and values.

          >>> d = GenoDistrib(c)
          >>> assert c != d

        Note that the class does not enforce any type as this job is left to the type checker of choice
        for efficiency reasons.
        """
        # asdict() will call this constructor with a stream of tuples (Alleles, v)
        # to prevent the default behaviour (Counter counts all the instances of the elements) we have to check
        # for generators

        self._store: Dict[Alleles, float] = {}

        if params:
            if isinstance(params, GeneratorType):
                self._store = cast(Dict[Alleles, float], dict(params))
            elif isinstance(params, dict):
                self._store = cast(Dict[Alleles, float], params)
            elif isinstance(params, GenoDistrib):
                self._store.update(params._store)

    @staticmethod
    def from_ratios(n: int, p: Union[np.ndarray, List[float]], rng: np.random.Generator) -> GenoDistrib:
        """
        Create a :class:`GenoDistrib` with a given number of lice and a probability distribution

        Note that this function uses a *statistical* approach to building such distribution.
        If you prefer a faster but deterministic alternative you could do this:

        >>> GenoDistrib(dict(zip(GenoDistrib.alleles), p)).normalise_to(n)

        :param n: the number of lice
        :param p: the probability of each genotype
        :param rng: the random number generator to use for sampling.

        :returns: the new GenoDistrib
        """

        assert np.isclose(np.sum(p), 1.0), "p must be a probability distribution"

        keys = GenoDistrib.alleles
        values = rng.multinomial(n, p).tolist()
        return GenoDistrib(dict(zip(keys, values)))

    def normalise_to(self, population: int) -> GenoDistrib:
        """
        Transform the distribution so that the overall sum changes to the given number
        but the ratios are preserved.

        :param population: the new population
        :returns: the new distribution
        """

        assert np.rint(population) == population, f"{population} is not an integer"
        keys = self.keys()
        values = list(self.values())
        np_values = np.array(values)
        # np.isclose is slow and not needed: all values are granted to be real and positive.
        if np.sum(np_values) < 1 or population == 0:
            return GenoDistrib({k: 0 for k in keys})

        np_values = np_values * population / np.sum(np_values)

        rounded_values = largest_remainder(np_values).tolist()
        return GenoDistrib(dict(zip(keys, map(int, rounded_values))))

    def set(self, other: int):
        """This is the inplace version of :meth:`normalise_to`.

        :param other: the new population
        """
        result = self.normalise_to(other)
        self._store.clear()
        self._store.update(result)

    @property
    def gross(self) -> int:
        """Return the gross number of lice, i.e. the distribution sum."""
        return int(sum(self.values()))

    def copy(self) -> GenoDistrib:
        """Create a copy"""
        return GenoDistrib(self._store.copy())

    def __add__(self, other: Union[GenoDistrib, int]) -> GenoDistrib:
        """Overload add operation"""
        if isinstance(other, GenoDistrib):
            res = GenoDistrib()
            res._store.update(self)
            for k, v in other.items():
                res[k] = self[k] + v
            return res
        else:
            return self.normalise_to(self.gross + other)

    def __sub__(self, other: Union[GenoDistrib, int]) -> GenoDistrib:
        """Overload sub operation"""

        if isinstance(other, GenoDistrib):
            res = GenoDistrib()
            res._store.update(self)
            for k, v in other.items():
                res[k] = self[k] - v
            return res
        else:
            return self.normalise_to(self.gross - other)

    def __mul__(self, other: Union[float, GenoDistrib]) -> GenoDistrib:
        """Multiply a distribution by either a scalar or another distribution.

        In the first case, the multiplication by a scalar will multiply each bin
        by the scalar. If the new sum is expected to be an integer a rounding step
        is applied.

        In the second case it is simply a pairwise product.

        :param other: either a float or a similar distribution
        :returns: the new genotype distribution
        """
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
        """Overloaded eq operator"""
        # Make equality checks a bit more flexible
        if isinstance(other, dict):
            other = GenoDistrib(other)
        return self._store == other._store

    def __le__(self, other: Union[GenoDistrib, float]) -> bool:
        """A <= B if B has all the keys of A and A[k] <= B[k] for every k
        Note that __gt__ is not implemented as its behaviour is not "greater than" but rather
        "there may be an unbounded frequency"."""
        if isinstance(other, float):
            return all(v <= other for v in self.values())

        merged_keys = set(list(self.keys()) + list(other.keys()))
        return all(self[k] <= other[k] for k in merged_keys)

    def is_positive(self) -> bool:
        """:returns True if all the bins are positive."""
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

    def _truncate_negatives(self):
        for k in self:
            self[k] = max(self[k], 0)

    @staticmethod
    def batch_sum(batches: Union[List[GenoDistrib], List[Dict[str, float]]],
                  as_pure_dict=False) -> Union[GenoDistrib, GenoDistribSerialisable]:
        """
        Calculate the sum of multiple :class:`GenoDistrib` instances.
        This function is a bit faster than manually invoking a folding operation
        due to reduced overhead.

        :param batches: either a list of distribution or a list of dictionaries (useful for pandas)
        :param as_pure_dict: if True return the final distribution as a dictionary rather than a :class:`GenoDistrib`
        :returns: either a dictionary or a :class:`GenoDistrib`
        """

        # this function is ugly but it's a hotspot.
        # TODO: automatically generate these from the class attrib
        alleles_to_idx = {('a',): 0, ('a'): 0,
                          ('A', 'a'): 1,
                          ('Aa'): 1,
                          ('A',): 2,
                          ('A'): 2}
        res_as_vector = [0, 0, 0]
        for batch in batches:
            for allele, value in batch.items():
                res_as_vector[alleles_to_idx[allele]] += value

        if as_pure_dict:
            return dict(zip(["a", "Aa", "A"], res_as_vector))

        return GenoDistrib(dict(zip(GenoDistrib.alleles, res_as_vector)))


GenoLifeStageDistrib = Dict[LifeStage, GenoDistrib]
GrossLiceDistrib = Dict[LifeStage, int]


class GenoTreatmentValue(NamedTuple):
    mortality_rate: float
    num_susc: int


GenoTreatmentDistrib = Dict[Alleles, GenoTreatmentValue]


# See https://stackoverflow.com/a/7760938
class LicePopulation(MutableMapping[LifeStage, int]):
    """
    Wrapper to keep the global population and genotype information updated
    This is definitely a convoluted way to do this, but I wanted to memoise as much as possible.
    """

    lice_stages = ["L1", "L2", "L3", "L4", "L5f", "L5m"]
    lice_stages_bio_labels = dict(zip(lice_stages, ["R", "CO", "CH", "PA", "AF", "AM"]))
    lice_stages_bio_long_names = dict(zip(lice_stages, [
        "Recruitment", "Copepopid", "Chalimus", "Preadult", "Adult Female", "Adult Male"]))

    susceptible_stages = lice_stages[2:]
    pathogenic_stages = lice_stages[3:]

    def __init__(self, geno_data: GenoLifeStageDistrib, generic_ratios: GenoDistribDict):
        """
        :param geno_data: a Genotype distribution
        :param generic_ratios: a config to use
        """
        self._cache: Dict[LifeStage, int] = {}
        self.geno_by_lifestage = GenotypePopulation(self, geno_data)
        self.genetic_ratios = generic_ratios
        self._busy_dams: PriorityQueue[DamAvailabilityBatch] = PriorityQueue()
        self._busy_dams_cache = GenoDistrib()
        for stage, distrib in geno_data.items():
            self._cache[stage] = distrib.gross

    def __setitem__(self, stage: LifeStage, value: int):
        # If one attempts to make gross modifications to the population these will be repartitioned according to the
        # current genotype information.
        old_value = self._cache[stage]
        if old_value == 0:
            if value > 0:
                logger.warning(
                    f"Trying to initialise population {stage} with null genotype distribution. Using default genotype information.")
            self.geno_by_lifestage[stage] = GenoDistrib(self.genetic_ratios).normalise_to(value)
        else:
            if value < 0:
                logger.warning(
                    f"Trying to assign population stage {stage} a negative value. It will be truncated to zero, but this probably means an invariant was broken.")
            elif value == old_value:
                return
            value = max(value, 0)
            self.geno_by_lifestage[stage] = self.geno_by_lifestage[stage].normalise_to(value)
        if stage == "L5f":
            self.__rescale_busy_dams(round(self.busy_dams.gross * value / old_value) if old_value > 0 else 0)

    def __getitem__(self, stage: LifeStage) -> int:
        return self._cache[stage]

    def __delitem__(self, stage: LifeStage):
        raise NotImplementedError()

    def __iter__(self) -> Iterator[LifeStage]:
        return iter(self._cache)

    def __len__(self) -> int:
        return len(self._cache)

    def __repr__(self) -> str:
        return repr(self._cache)

    def __str__(self) -> str:
        return f"LicePopulation({str(self._cache)})"

    def as_dict(self) -> Dict[LifeStage, int]:
        return self._cache

    def raw_update_value(self, stage: LifeStage, value: int):
        self._cache[stage] = value

    def clear_busy_dams(self):
        self._busy_dams = PriorityQueue()

    @property
    def available_dams(self) -> GenoDistrib:
        return self.geno_by_lifestage["L5f"] - self.busy_dams

    @property
    def busy_dams(self) -> GenoDistrib:
        return self._busy_dams_cache

    def free_dams(self, cur_time: dt.datetime) -> GenoDistrib:
        """
        Return the number of available dams

        :param cur_time: the current time
        :returns: the genotype population of dams that return available today
        """
        delta_avail_dams = GenoDistrib()

        def cts(event):
            nonlocal delta_avail_dams
            delta_avail_dams += event.geno_distrib
            self._busy_dams_cache = self._busy_dams_cache - event.geno_distrib

        logger.debug(f"\t\t\t\tIn free dams: self._busy_dams has {self._busy_dams.qsize()} events")
        pop_from_queue(self._busy_dams, cur_time, cts)

        return delta_avail_dams

    def add_busy_dams_batch(self, delta_dams_batch: DamAvailabilityBatch):
        dams_distrib = self.geno_by_lifestage["L5f"]
        dams_to_add = delta_dams_batch.geno_distrib

        # clip the distribution, just in case
        for k in dams_to_add:
            dams_to_add[k] = max(dams_to_add[k], 0)

        if not(self.busy_dams + dams_to_add <= dams_distrib):
            delta_dams_batch.geno_distrib = dams_distrib - self.busy_dams
        self._busy_dams.put(delta_dams_batch)
        self._busy_dams_cache = self._busy_dams_cache + delta_dams_batch.geno_distrib

    def remove_negatives(self):
        for stage, distrib in self.geno_by_lifestage.items():
            keys = distrib.keys()
            values = distrib.values()
            distrib_purged = GenoDistrib(dict(zip(keys, [max(v, 0) for v in values])))
            self.geno_by_lifestage[stage] = distrib_purged

    def _flush_busy_dams_cache(self):
        self._busy_dams_cache = cast(GenoDistrib, GenoDistrib.batch_sum([x.geno_distrib for x in self._busy_dams.queue]))

    def __clear_empty_dams(self):
        self._busy_dams.queue = [x for x in self._busy_dams.queue if x.geno_distrib.gross > 0]
        # re-ensure heaping condition is respected.
        # TODO: no synchronisation is done here. Determine if this causes threading issues
        heapify(self._busy_dams.queue)
        self._flush_busy_dams_cache()

    def __rescale_busy_dams(self, num: int):
        # rescale the number of busy dams to the given number.
        if num <= 0:
            self.clear_busy_dams()

        elif self._busy_dams.qsize() == 0 or self.busy_dams.gross == 0:
            raise ValueError("Busy dam queue is empty")

        else:
            self.__clear_empty_dams()
            partitions = np.array([event.geno_distrib.gross for event in self._busy_dams.queue])
            total_busy = np.sum(partitions)
            proportions = largest_remainder(num * partitions / total_busy)

            # TODO: rounding this with iteround isn't easy so we need to make sure dams are clipped
            for event, new_population in zip(self._busy_dams.queue, proportions):
                event.geno_distrib.set(new_population)

            self._clip_dams_to_stage()

        assert self.busy_dams <= self.geno_by_lifestage["L5f"]

    def _clip_dams_to_stage(self):
        # make sure that self.busy_dams <= self.geno_by_lifestage["L5f"]
        busy_sum = self.busy_dams
        offset = busy_sum - self.geno_by_lifestage["L5f"]

        #logger.debug(f"\t\t\t In _clip_dams_to_stage: L5f is {self.geno_by_lifestage['L5f']}")

        for allele in offset:
            off = offset[allele]
            if off <= 0:
                continue
            for event in self._busy_dams.queue:
                to_reduce = min(event.geno_distrib[allele], off)
                offset[allele] -= to_reduce
                off -= to_reduce
                event.geno_distrib[allele] -= to_reduce

        self._flush_busy_dams_cache()
        assert self.busy_dams <= self.geno_by_lifestage["L5f"]

    @staticmethod
    def get_empty_geno_distrib() -> GenoLifeStageDistrib:
        return {stage: GenoDistrib() for stage in LicePopulation.lice_stages}

    def to_json_dict(self) -> GenoDistribSerialisable:
        return self._cache


class GenotypePopulation(MutableMapping[LifeStage, GenoDistrib]):
    def __init__(
        self,
        gross_lice_population: LicePopulation,
        geno_data: GenoLifeStageDistrib
    ):
        """
        A GenotypePopulation is a mapping between the life stage and the actual
        population rather than a gross projection.

        :param gross_lice_population: the parent LicePopulation to update
        :param geno_data: the initial geno population
        """
        self._store:  Dict[LifeStage, GenoDistrib] = {}

        self._lice_population = gross_lice_population
        for k, v in geno_data.items():
            self._store[k] = v.copy()

    def __setitem__(self, stage: LifeStage, value: GenoDistrib):
        # update the value and the gross population accordingly
        value._truncate_negatives()
        old_value = self[stage]
        old_value_sum = self._lice_population[stage]
        delta = value - old_value
        self._store[stage] = value#.copy()
        value_sum = value.gross
        self._lice_population.raw_update_value(stage, value_sum)

        # if L5f increases all the new lice are by default free
        # if it decreases we subtract the delta from each dam event
        if stage == "L5f" and value_sum < old_value_sum:
            # calculate the deltas and
            busy_dams_denom = self._lice_population.busy_dams.gross
            if busy_dams_denom == 0:
                self._lice_population.clear_busy_dams()
            else:
                for event in self._lice_population._busy_dams.queue:
                    geno_distrib = event.geno_distrib
                    delta_geno = delta * (geno_distrib.gross/ busy_dams_denom)
                    geno_distrib += delta_geno

        self._lice_population._clip_dams_to_stage()

    def __getitem__(self, stage: LifeStage) -> GenoDistrib:
        return self._store[stage]

    def __delitem__(self, stage: LifeStage):
        raise NotImplementedError()

    def __iter__(self) -> Iterator[LifeStage]:
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return repr(self._store)

    def __str__(self) -> str:
        return f"GenericGenoDistrib({str(self._store)})"

    def to_json_dict(self) -> Dict[LifeStage, GenoDistribSerialisable] :
        return {k: v.to_json_dict() for k, v in self.items()}

    def as_dict(self) -> Dict[LifeStage, GenoDistrib]:
        return self._store