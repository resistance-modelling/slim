from __future__ import annotations

from heapq import heapify
from queue import PriorityQueue
from types import GeneratorType
from typing import Dict, MutableMapping, Tuple, Union, NamedTuple, TypeVar, Generic, Counter as CounterType, Optional, \
    Iterable, TYPE_CHECKING, List

import iteround
import numpy as np

from src import logger
from src.QueueTypes import pop_from_queue

if TYPE_CHECKING:  # pragma: no cover
    from src.QueueTypes import DamAvailabilityBatch

################ Basic type aliases #####################
LifeStage = str
Allele = str
Alleles = Union[Tuple[Allele, ...]]

GenoKey = TypeVar('GenoKey', Alleles, float)

@profile
def largest_remainder(nums: np.ndarray):
    # a vectorised implementation of largest remainder
    approx = np.trunc(nums)
    diff = nums - approx
    diff_sum = np.sum(diff)
    positions = np.argsort(diff)
    tweaks = int(min(len(nums), abs(diff_sum)))
    if diff_sum < 0:
        approx[positions[:tweaks]] -= 1
    else:
        approx[positions[::-1][:tweaks]] += 1

    return approx

class GenericGenoDistrib(CounterType[GenoKey], Generic[GenoKey]):
    """
    A GenoDistrib is a distribution of genotypes.

    Each GenoDistrib provides custom operator overloading operations and suitable
    rescaling operations.
    Internally, this is built on top of a Counter. As a consequence, this always
    represents an actual count and not a PMF. Float values are (currently) not allowed.
    """

    def __init__(self, params: Optional[Union[
        GenericGenoDistrib[GenoKey],
        Dict[GenoKey, float],
        Iterable[Tuple[GenoKey, float]]
    ]] = None):
        # asdict() will call this constructor with a stream of tuples (Alleles, v)
        # to prevent the default behaviour (Counter counts all the instances of the elements) we have to check
        # for generators
        if params:
            if isinstance(params, GeneratorType):
                super().__init__(dict(params))
            else:
                super().__init__(params)
        else:
            super().__init__()

    @profile
    def normalise_to(self, population: int) -> GenericGenoDistrib[GenoKey]:
        keys = self.keys()
        values = list(self.values())
        np_values = np.array(values)
        # np.isclose is slow and not needed: all values are granted to be real and positive.
        if np.sum(np_values) < 1 or population == 0:
            return GenericGenoDistrib({k: 0 for k in keys})

        np_values = np_values * population / np.sum(np_values)

        # TODO: rewrite saferound to be vectorised
        rounded_values = largest_remainder(np_values).tolist()
        return GenericGenoDistrib(dict(zip(keys, map(int, rounded_values))))

    def set(self, other: int):
        result = self.normalise_to(other)
        self.clear()
        self.update(result)

    @property
    def gross(self) -> int:
        return sum(self.values())

    def copy(self: GenericGenoDistrib[GenoKey]) -> GenericGenoDistrib[GenoKey]:
        return GenericGenoDistrib(super().copy())

    def __add__(self, other: Union[GenericGenoDistrib[GenoKey], int]) -> GenericGenoDistrib[GenoKey]:
        """Overload add operation"""
        res = self.copy()

        if isinstance(other, GenericGenoDistrib):
            res = GenericGenoDistrib()
            res.update(self)
            for k, v in other.items():
                res[k] = self[k] + v
            return res
        else:
            return self.normalise_to(self.gross + other)

    def __sub__(self, other: Union[GenericGenoDistrib[GenoKey], int]) -> GenericGenoDistrib[GenoKey]:
        """Overload sub operation"""

        if isinstance(other, GenericGenoDistrib):
            res = GenericGenoDistrib()
            res.update(self)
            for k, v in other.items():
                res[k] = self[k] - v
            return res
        else:
            return self.normalise_to(self.gross - other)


        """
        if isinstance(other, GenericGenoDistrib):
            res = GenericGenoDistrib()
            for k, v in other.items():
                res[k] = self[k] - v
            return res

        return self.normalise_to(self.gross - other)

        """

    def __mul__(self, other: Union[float, GenericGenoDistrib[GenoKey]]) -> GenericGenoDistrib[GenoKey]:
        """Multiply a distribution."""
        if isinstance(other, float) or isinstance(other, int):
            keys = self.keys()
            values = [v * other for v in self.values()]
            if isinstance(other, float):
                values = largest_remainder(np.array(values)).tolist()
        else:
            # pairwise product
            # Sometimes it may be helpful to not round
            keys = set(list(self.keys()) + list(other.keys()))
            values = [self[k] * other[k] for k in keys]
        return GenericGenoDistrib(dict(zip(keys, values)))

    def __eq__(self, other: Union[GenericGenoDistrib[GenoKey], Dict[GenoKey, int]]) -> bool:
        # Make equality checks a bit more flexible
        if isinstance(other, dict):
            other = GenericGenoDistrib(other)
        return super().__eq__(other)

    def __le__(self, other: Union[GenericGenoDistrib[GenoKey], float]):
        """A <= B if B has all the keys of A and A[k] <= B[k] for every k
        Note that __gt__ is not implemented as its behaviour is not "greater than" but rather
        "there may be an unbounded frequency"."""
        if isinstance(other, float):
            return all(v <= other for v in self.values())

        merged_keys = set(list(self.keys()) + list(other.keys()))
        return all(self[k] <= other[k] for k in merged_keys)

    def to_json_dict(self):
        return {"".join(k): v for k, v in self.items()}

    def truncate_negatives(self):
        for k in self:
            self[k] = max(self[k], 0)

    @staticmethod
    @profile
    def batch_sum(batches: List[GenoDistrib]) -> GenoDistrib:
        # TODO: it's quite official quantitative is not the way to go
        # this function is ugly but it's a hotspot.
        alleles = [('a',), ('A', 'a'), ('A',)]
        res = GenericGenoDistrib()
        for allele in alleles:
            for batch in batches:
                res[allele] += batch[allele]

        return res


GenoDistrib = GenericGenoDistrib[Alleles]
QuantitativeGenoDistrib = GenericGenoDistrib[float]

GenoLifeStageDistrib = Dict[LifeStage, GenericGenoDistrib]
GrossLiceDistrib = Dict[LifeStage, int]


class GenoTreatmentValue(NamedTuple):
    mortality_rate: float
    num_susc: int


GenoTreatmentDistrib = Dict[GenoKey, GenoTreatmentValue]


# See https://stackoverflow.com/a/7760938
class LicePopulation(MutableMapping[LifeStage, int]):
    """
    Wrapper to keep the global population and genotype information updated
    This is definitely a convoluted way to do this, but I wanted to memoise as much as possible.
    """

    lice_stages = ["L1", "L2", "L3", "L4", "L5f", "L5m"]

    def __init__(self, geno_data: GenoLifeStageDistrib, generic_ratios):
        """
        :param geno_data a Genotype distribution
        :param generic_ratios a config to use
        """
        self._cache: Dict[LifeStage, int] = dict()
        self.geno_by_lifestage = GenotypePopulation(self, geno_data)
        self.genetic_ratios = generic_ratios
        self._busy_dams: PriorityQueue[DamAvailabilityBatch] = PriorityQueue()
        for stage, distrib in geno_data.items():
            self._cache[stage] = distrib.gross

    @profile
    def __setitem__(self, stage: LifeStage, value: int):
        # If one attempts to make gross modifications to the population these will be repartitioned according to the
        # current genotype information.
        old_value = self._cache[stage]
        if old_value == 0:
            if value > 0:
                logger.warning(
                    f"Trying to initialise population {stage} with null genotype distribution. Using default genotype information.")
            self.geno_by_lifestage[stage] = GenericGenoDistrib(self.genetic_ratios).normalise_to(value)
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

    def __getitem__(self, stage: LifeStage):
        return self._cache[stage]

    def __delitem__(self, stage: LifeStage):
        raise NotImplementedError()

    def __iter__(self):
        return iter(self._cache)

    def __len__(self):
        return len(self._cache)

    def __repr__(self):
        return repr(self._cache)

    def __str__(self):
        return f"LicePopulation({str(self._cache)})"

    def as_dict(self):
        return self._cache

    def raw_update_value(self, stage: LifeStage, value: int):
        self._cache[stage] = value

    def clear_busy_dams(self):
        self._busy_dams = PriorityQueue()

    @property
    def available_dams(self):
        return self.geno_by_lifestage["L5f"] - self.busy_dams

    @property
    def busy_dams(self):
        return GenericGenoDistrib.batch_sum([x.geno_distrib for x in self._busy_dams.queue])

    def free_dams(self, cur_time) -> GenericGenoDistrib:
        """
        Return the number of available dams

        :param cur_time the current time
        :returns the genotype population of dams that return available today
        """
        delta_avail_dams = GenericGenoDistrib()

        def cts(event):
            nonlocal delta_avail_dams
            delta_avail_dams += event.geno_distrib

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

    def remove_negatives(self):
        for stage, distrib in self.geno_by_lifestage.items():
            keys = distrib.keys()
            values = distrib.values()
            distrib_purged = GenoDistrib(dict(zip(keys, [max(v, 0) for v in values])))
            self.geno_by_lifestage[stage] = distrib_purged

    def __clear_empty_dams(self):
        self._busy_dams.queue = [x for x in self._busy_dams.queue if x.geno_distrib.gross > 0]
        # re-ensure heaping condition is respected.
        # TODO: no synchronisation is done here. Determine if this causes threading issues
        heapify(self._busy_dams.queue)

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
            ratios = partitions / total_busy

            # TODO: rounding this with iteround isn't easy so we need to make sure dams are clipped
            for event, ratio in zip(self._busy_dams.queue, ratios):
                event.geno_distrib.set(num * ratio)

            self._clip_dams_to_stage()

        assert self.busy_dams <= self.geno_by_lifestage["L5f"]

    @profile
    def _clip_dams_to_stage(self):
        # make sure that self.busy_dams <= self.geno_by_lifestage["L5f"]
        offset = self.busy_dams - self.geno_by_lifestage["L5f"]

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

        assert self.busy_dams <= self.geno_by_lifestage["L5f"]

    @staticmethod
    def get_empty_geno_distrib() -> GenoLifeStageDistrib:
        return {stage: GenoDistrib() for stage in LicePopulation.lice_stages}

    def to_json_dict(self):
        return self._cache


class GenotypePopulation(MutableMapping[LifeStage, GenericGenoDistrib]):
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
        self._store:  Dict[LifeStage, GenericGenoDistrib] = {}

        self._lice_population = gross_lice_population
        for k, v in geno_data.items():
            self._store[k] = v.copy()

    @profile
    def __setitem__(self, stage: LifeStage, value: GenoDistrib):
        # update the value and the gross population accordingly
        value.truncate_negatives()
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

    def __getitem__(self, stage: LifeStage):
        return self._store[stage]

    def __delitem__(self, stage: LifeStage):
        raise NotImplementedError()

    def __iter__(self):
        return iter(self._store)

    def __len__(self):
        return len(self._store)

    def __repr__(self):
        return repr(self._store)

    def __str__(self):
        return f"GenericGenoDistrib({str(self._store)})"

    def to_json_dict(self):
        return {k: v.to_json_dict() for k, v in self.items()}

    def as_dict(self):
        return self._store