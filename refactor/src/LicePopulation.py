from __future__ import annotations

import copy
from queue import PriorityQueue
from typing import Dict, MutableMapping, Tuple, Union, NamedTuple, TypeVar, Generic, Counter as CounterType, Optional, \
    TYPE_CHECKING

import iteround
import numpy as np

from src.Config import Config

if TYPE_CHECKING:  # pragma: no cover
    from src.QueueTypes import DamAvailabilityBatch

################ Basic type aliases #####################
LifeStage = str
Allele = str
Alleles = Union[Tuple[Allele, ...]]

GenoKey = TypeVar('GenoKey', Alleles, float)


class GenericGenoDistrib(CounterType[GenoKey], Generic[GenoKey]):
    """
    A GenoDistrib is a distribution of genotypes.

    Each GenoDistrib provides custom operator overloading operations and suitable
    rescaling operations.
    Internally, this is built on top of a Counter. As a consequence, this always
    represents an actual count and not a PMF. Float values are (currently) not allowed.
    """

    def normalise_to(self, population: int) -> GenericGenoDistrib[GenoKey]:
        keys = self.keys()
        values = list(self.values())
        np_values = np.array(values)
        if np.isclose(np.sum(np_values), 0.0) or population == 0:
            return GenericGenoDistrib({k: 0 for k in keys})

        np_values = np_values * population / np.sum(np_values)

        rounded_values = iteround.saferound(np_values.tolist(), 0)
        return GenericGenoDistrib(dict(zip(keys, map(int, rounded_values))))

    def set(self, other: int):
        result = self.normalise_to(other)
        self.clear()
        self.update(result)

    def gross(self) -> int:
        return sum(self.values())

    def copy(self: GenericGenoDistrib[GenoKey]) -> GenericGenoDistrib[GenoKey]:
        return GenericGenoDistrib(super().copy())

    def __add__(self, other: GenericGenoDistrib[GenoKey]) -> GenericGenoDistrib[GenoKey]:
        """Overload add operation"""
        res = self.copy()
        for k, v in other.items():
            res[k] += v
        return res

    def __sub__(self, other: GenericGenoDistrib[GenoKey]) -> GenericGenoDistrib[GenoKey]:
        """Overload sub operation"""
        res = self.copy()
        for k, v in other.items():
            res[k] -= v
        return res

    def __mul__(self, other: Union[float, GenericGenoDistrib[GenoKey]]) -> GenericGenoDistrib[GenoKey]:
        """Multiply a distribution."""
        if isinstance(other, float) or isinstance(other, int):
            keys = self.keys()
            values = [v * other for v in self.values()]
            if isinstance(other, float):
                values = iteround.saferound(values, 0)
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

    def __round__(self, n: Optional[int] = None) -> GenericGenoDistrib[GenoKey]:
        """Perform rounding to get an actual population distribution."""
        if n is None:
            n = 0

        values = iteround.saferound(list(self.values()), n)
        return GenericGenoDistrib(dict(zip(self.keys(), values)))


GenoDistrib = GenericGenoDistrib[Alleles]
QuantitativeGenoDistrib = GenericGenoDistrib[float]

GenoLifeStageDistrib = Dict[LifeStage, GenericGenoDistrib]
GrossLiceDistrib = Dict[LifeStage, int]


class GenoTreatmentValue(NamedTuple):
    mortality_rate: float
    num_susc: int


GenoTreatmentDistrib = Dict[GenoKey, GenoTreatmentValue]


# See https://stackoverflow.com/a/7760938
class LicePopulation(dict, MutableMapping[LifeStage, int]):
    """
    Wrapper to keep the global population and genotype information updated
    This is definitely a convoluted way to do this, but I wanted to memoise as much as possible.
    """

    lice_stages = ["L1", "L2", "L3", "L4", "L5f", "L5m"]

    def __init__(self, initial_population: GrossLiceDistrib, geno_data: GenoLifeStageDistrib, cfg: Config):
        super().__init__()
        self.geno_by_lifestage = GenotypePopulation(self, geno_data)
        self.genetic_ratios = cfg.genetic_ratios
        self.logger = cfg.logger
        self._busy_dams = PriorityQueue()  # type: PriorityQueue[DamAvailabilityBatch]
        for k, v in initial_population.items():
            super().__setitem__(k, v)

    def __setitem__(self, stage: LifeStage, value: int):
        # If one attempts to make gross modifications to the population these will be repartitioned according to the
        # current genotype information.
        old_value = self[stage]
        if old_value == 0:
            if value > 0:
                self.logger.warning(
                    f"Trying to initialise population {stage} with null genotype distribution. Using default genotype information.")
            self.geno_by_lifestage[stage] = GenericGenoDistrib(self.genetic_ratios).normalise_to(value)
        else:
            if value < 0:
                self.logger.warning(
                    f"Trying to assign population stage {stage} a negative value. It will be truncated to zero, but this probably means an invariant was broken.")
            elif value == old_value:
                return
            value = max(value, 0)
            self.geno_by_lifestage[stage] = self.geno_by_lifestage[stage].normalise_to(value)
        if stage == "L5f":
            self.rescale_busy_dams(round(self.busy_dams.gross() * value / old_value) if old_value > 0 else 0)

    def raw_update_value(self, stage: LifeStage, value: int):
        super().__setitem__(stage, value)

    def clear_busy_dams(self):
        self._busy_dams = PriorityQueue()

    @property
    def available_dams(self):
        return self.geno_by_lifestage["L5f"] - self.busy_dams

    @property
    def busy_dams(self):
        return sum(map(lambda x: x.geno_distrib, self._busy_dams.queue), GenericGenoDistrib())

    def free_dams(self, cur_time) -> GenericGenoDistrib:
        """
        Return the number of available dams

        :param cur_time the current time
        :returns the genotype population of dams that return available today
        """
        delta_avail_dams = GenericGenoDistrib()

        while not self._busy_dams.empty() and self._busy_dams.queue[0].availability_date <= cur_time:
            event = self._busy_dams.get()
            delta_avail_dams += event.geno_distrib

        return delta_avail_dams

    def add_busy_dams_batch(self, delta_dams_batch: DamAvailabilityBatch):
        for k, v in delta_dams_batch.geno_distrib.items():
            assert v <= self.geno_by_lifestage['L5f'][k]
        self._busy_dams.put(delta_dams_batch)

    def remove_negatives(self):
        for stage, distrib in self.geno_by_lifestage.items():
            keys = distrib.keys()
            values = distrib.values()
            distrib_purged = GenoDistrib(dict(zip(keys, [max(v, 0) for v in values])))
            self.geno_by_lifestage[stage] = distrib_purged

    def rescale_busy_dams(self, num: int):
        # rescale the number of busy dams to the given number.
        if num <= 0:
            self.clear_busy_dams()

        elif self._busy_dams.qsize() == 0 or self.busy_dams.gross() == 0:
            raise ValueError("Busy dam queue is empty")

        else:
            partitions = np.array([sum(event.geno_distrib.values()) for event in self._busy_dams.queue])
            total_busy = np.sum(partitions)
            ratios = total_busy / partitions

            # TODO: rounding this with iteround isn't easy...
            for event, ratio in zip(self._busy_dams.queue, ratios):
                event.geno_distrib.set(num * ratio)

    @staticmethod
    def get_empty_geno_distrib() -> GenoLifeStageDistrib:
        return {stage: GenoDistrib() for stage in LicePopulation.lice_stages}


class GenotypePopulation(Dict[LifeStage, GenericGenoDistrib]):
    def __init__(self, gross_lice_population: LicePopulation, geno_data: GenoLifeStageDistrib):
        super().__init__()
        self._lice_population = gross_lice_population
        for k, v in geno_data.items():
            super().__setitem__(k, v.copy())

    def __setitem__(self, stage: LifeStage, value: GenoDistrib):
        # update the value and the gross population accordingly
        old_value = self[stage].copy()
        old_value_sum = self._lice_population[stage]
        old_value_sum_2 = old_value.gross()
        assert old_value_sum == old_value_sum_2
        super().__setitem__(stage, value.copy())
        value_sum = value.gross()
        self._lice_population.raw_update_value(stage, value_sum)

        # if L5f increases all the new lice are by default free
        # if it decreases we subtract the delta from each dam event
        if stage == "L5f" and value_sum < old_value_sum:
            # calculate the deltas and
            delta = value - old_value
            busy_dams_denom = self._lice_population.busy_dams.gross()
            if busy_dams_denom == 0:
                self._lice_population.clear_busy_dams()
            else:
                for event in self._lice_population._busy_dams.queue:
                    geno_distrib = event.geno_distrib
                    delta_geno = delta * (geno_distrib.gross() / busy_dams_denom)
                    geno_distrib += delta_geno

    def raw_update_value(self, stage: LifeStage, value: GenoDistrib):
        super().__setitem__(stage, value)
