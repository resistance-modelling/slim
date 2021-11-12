from __future__ import annotations

from abc import abstractmethod, ABC
from decimal import Decimal
from enum import Enum
from typing import Dict, cast

import numpy as np

# A few extra general types
from src.LicePopulation import LicePopulation, GenoDistrib, GenoTreatmentValue, Alleles

Money = Decimal


class Treatment(Enum):
    """
    A stub for treatment types
    TODO: add other treatments here
    """
    emb = 0
    thermolicer = 1


class GeneticMechanism(Enum):
    """
    Genetic mechanism to be used when generating egg genotypes
    """
    discrete = 1
    maternal = 2


class HeterozygousResistance(Enum):
    """
    Resistance in a monogenic, heterozygous setting.
    """
    dominant = 1
    incompletely_dominant = 2
    recessive = 3


TreatmentResistance = Dict[HeterozygousResistance, float]


class TreatmentParams(ABC):
    """
    Abstract class for all the treatments
    """
    name = ""

    def __init__(self, payload):
        self.pheno_resistance = self.parse_pheno_resistance(payload["pheno_resistance"])
        self.price_per_kg = Money(payload["price_per_kg"])
        self.quadratic_fish_mortality_coeffs = np.array(payload["quadratic_fish_mortality_coeffs"])

        self.effect_delay: int = payload["effect_delay"]
        self.durability_temp_ratio: float = payload["durability_temp_ratio"]
        self.application_period: int = payload["application_period"]

    @staticmethod
    def parse_pheno_resistance(pheno_resistance_dict: dict) -> TreatmentResistance:
        return {HeterozygousResistance[key]: val for key, val in pheno_resistance_dict.items()}

    def get_mortality_pp_increase(self, temperature: float, fish_mass: float) -> float:
        """Get the mortality percentage point difference increase.

        :param temperature: the temperature in Celsius
        :param fish_mass: the fish mass (in grams)
        :returns Mortality percentage point difference increase
        """
        # TODO: is this the right way to solve this?
        fish_mass_indicator = 1 if fish_mass > 2000 else 0

        input = np.array([1, temperature, fish_mass_indicator, temperature ** 2, temperature * fish_mass_indicator,
                          fish_mass_indicator ** 2])
        return max(float(self.quadratic_fish_mortality_coeffs.dot(input)), 0)

    @abstractmethod
    def delay(self, average_temperature: float):  # pragma: no cover
        pass

    @staticmethod
    def get_allele_heterozygous_trait(alleles: Alleles):
        """
        Get the allele heterozygous type
        """
        # should we move this?
        if 'A' in alleles:
            if 'a' in alleles:
                trait = HeterozygousResistance.incompletely_dominant
            else:
                trait = HeterozygousResistance.dominant
        else:
            trait = HeterozygousResistance.recessive
        return trait


class EMB(TreatmentParams):
    name = "EMB"

    def delay(self, average_temperature: float):
        return self.durability_temp_ratio / average_temperature

    def get_lice_treatment_mortality_rate(self, lice_population: LicePopulation, _temperature=None):
        susceptible_populations = [lice_population.geno_by_lifestage[stage] for stage in
                                   LicePopulation.susceptible_stages]
        num_susc_per_geno = GenoDistrib.batch_sum(susceptible_populations)

        geno_treatment_distrib = {geno: GenoTreatmentValue(0.0, 0) for geno in num_susc_per_geno}

        for geno, num_susc in num_susc_per_geno.items():
            trait = self.get_allele_heterozygous_trait(geno)
            susceptibility_factor = 1.0 - self.pheno_resistance[trait]
            geno_treatment_distrib[geno] = GenoTreatmentValue(susceptibility_factor, cast(int, num_susc))

        return geno_treatment_distrib


class Thermolicer(TreatmentParams):
    name = "Thermolicer"

    def delay(self, _):
        return 0

    def get_lice_treatment_mortality_rate(self, lice_population: LicePopulation, temperature: float):
        pass
