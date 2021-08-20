from __future__ import annotations

from collections import Counter
import copy
from typing import Dict, MutableMapping, Tuple, Union, NamedTuple, TypeVar, Generic, cast, Counter as CounterType

import iteround
import numpy as np

from src.Config import Config

# Basic type aliases
LifeStage = str
Allele = str
Alleles = Union[Tuple[Allele, ...]]

GenoKey = TypeVar('GenoKey', Alleles, float)


class GenericGenoDistrib(CounterType[GenoKey], Generic[GenoKey]):
    """
    A GenoDistrib is a distribution of genotypes.

    Each GenoDistrib provides custom operator overloading operations and suitable
    rescaling operations.
    Internally, this is built on top of a Counter.
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

    def __eq__(self, other: Union[GenericGenoDistrib[GenoKey], Dict[GenoKey, int]]) -> bool:
        # Make equality checks a bit more flexible
        if isinstance(other, dict):
            other = GenericGenoDistrib(other)
        return super().__eq__(other)


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
    def __init__(self, initial_population: GrossLiceDistrib, geno_data: GenoLifeStageDistrib, cfg: Config):
        super().__init__()
        self.geno_by_lifestage = GenotypePopulation(self, geno_data)
        self._available_dams = copy.deepcopy(self.geno_by_lifestage["L5f"])  # type: GenoDistrib
        self.genetic_ratios = cfg.genetic_ratios
        self.logger = cfg.logger
        for k, v in initial_population.items():
            super().__setitem__(k, v)

    def __setitem__(self, stage: LifeStage, value: int):
        # If one attempts to make gross modifications to the population these will be repartitioned according to the
        # current genotype information.
        if sum(self.geno_by_lifestage[stage].values()) == 0:
            if value > 0:
                self.logger.warning(f"Trying to initialise population {stage} with null genotype distribution. Using default genotype information.")
            self.geno_by_lifestage[stage] = GenericGenoDistrib(self.genetic_ratios).normalise_to(value)
        else:
            if value < 0:
                self.logger.warning(f"Trying to assign population stage {stage} a negative value. It will be truncated to zero, but this probably means an invariant was broken.")
            value = max(value, 0)
            self.geno_by_lifestage[stage].set(value)
        if stage == "L5f":
            self._available_dams.set(value)
        super().__setitem__(stage, value)

    def raw_update_value(self, stage: LifeStage, value: int):
        super().__setitem__(stage, value)

    @property
    def available_dams(self):
        return self._available_dams

    @available_dams.setter
    def available_dams(self, new_value: GenoDistrib):
        for geno in new_value:
            assert self.geno_by_lifestage["L5f"][geno] >= new_value[geno], \
                f"current population geno {geno}:{self.geno_by_lifestage['L5f'][geno]} is smaller than new value geno {new_value[geno]}"

        self._available_dams = new_value

    def get_empty_geno_stage_distrib(self) -> GenoDistrib:
        # A little factory method to get empty genos
        genos = self.geno_by_lifestage["L1"].keys()
        return GenoDistrib({geno: 0 for geno in genos})

    def get_empty_geno_distrib(self) -> GenoLifeStageDistrib:
        return {stage: self.get_empty_geno_stage_distrib() for stage in self.keys()}


class GenotypePopulation(Dict[Alleles, GenericGenoDistrib]):
    def __init__(self, gross_lice_population: LicePopulation, geno_data: GenoLifeStageDistrib):
        super().__init__()
        self._lice_population = gross_lice_population
        for k, v in geno_data.items():
            super().__setitem__(k, v)

    def __setitem__(self, stage: LifeStage, value: GenoDistrib):
        # update the value and the gross population accordingly
        super().__setitem__(stage, value)
        self._lice_population.raw_update_value(stage, sum(value.values()))

    def raw_update_value(self, stage: LifeStage, value: GenoDistrib):
        super().__setitem__(stage, value)

