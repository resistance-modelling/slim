import copy
from typing import Dict, MutableMapping, Tuple, Union

import numpy as np

from src.Config import Config

LifeStage = str
Allele = str
Alleles = Tuple[Allele, ...]
GenoDistrib = Dict[Alleles, Union[int, float]]
GenoLifeStageDistrib = Dict[LifeStage, GenoDistrib]
GrossLiceDistrib = Dict[LifeStage, int]


# See https://stackoverflow.com/a/7760938
class LicePopulation(dict, MutableMapping[LifeStage, int]):
    """
    Wrapper to keep the global population and genotype information updated
    This is definitely a convoluted way to do this, but I wanted to memoise as much as possible.
    """
    def __init__(self, initial_population: GrossLiceDistrib, geno_data: GenoLifeStageDistrib, cfg: Config):
        super().__init__()
        self.geno_by_lifestage = GenotypePopulation(self, geno_data)
        self._available_dams = copy.deepcopy(self.geno_by_lifestage["L5f"])
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
            self.geno_by_lifestage.raw_update_value(stage, self.multiply_distrib(self.genetic_ratios, value))
        else:
            self.geno_by_lifestage.raw_update_value(stage, self.multiply_distrib(self.geno_by_lifestage[stage], value))
        if stage == "L5f":
            self._available_dams = self.multiply_distrib(self._available_dams, value)
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

    @staticmethod
    def multiply_distrib(distrib: dict, population: int):
        keys = distrib.keys()
        values = list(distrib.values())
        np_values = np.array(values)
        if np.isclose(np.sum(np_values), 0.0) or population == 0:
            return dict(zip(keys, map(int, np.zeros_like(np_values))))
        np_values = np_values * population / np.sum(np_values)
        np_values = np_values.round().astype('int32')
        # correct casting errors
        np_values[-1] += population - np.sum(np_values)

        return dict(zip(keys, map(int, np_values)))

    @staticmethod
    def update_distrib_discrete_add(distrib_delta, distrib):
        """
        I've assumed that both distrib and delta are dictionaries
        and are *not* normalised (that is, they are effectively counts)
        I've also assumed that we never want a count below zero
        Code is naive - interpret as algorithm spec
        combine these two functions with a negation of a dict?
        """

        for geno in distrib_delta:
            if geno not in distrib:
                distrib[geno] = 0
            distrib[geno] += distrib_delta[geno]

    @staticmethod
    def update_distrib_discrete_subtract(distrib_delta, distrib):
        for geno in distrib:
            if geno in distrib_delta:
                distrib[geno] -= distrib_delta[geno]
            if distrib[geno] < 0:
                distrib[geno] = 0


class GenotypePopulation(dict, MutableMapping[LifeStage, GenoDistrib]):
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
