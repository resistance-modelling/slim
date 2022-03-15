from __future__ import annotations

import functools
from abc import abstractmethod, ABC
from enum import Enum
from typing import Dict, cast, List

import numpy as np

# A few extra general types
from slim.simulation.lice_population import (
    LicePopulation,
    GenoDistrib,
    GenoTreatmentValue,
    Allele,
    GenoTreatmentDistrib,
    Gene,
    geno_to_alleles,
)


class Treatment(Enum):
    """
    A stub for treatment types
    TODO: add other treatments here
    """

    EMB = 0
    THERMOLICER = 1


TREATMENT_NO = len(Treatment)


class GeneticMechanism(Enum):
    """
    Genetic mechanism to be used when generating egg genotypes
    """

    DISCRETE = 1
    MATERNAL = 2


class HeterozygousResistance(Enum):
    """
    Resistance in a monogenic, heterozygous setting.
    """

    DOMINANT = 1
    INCOMPLETELY_DOMINANT = 2
    RECESSIVE = 3


TreatmentResistance = Dict[HeterozygousResistance, float]


class TreatmentParams(ABC):
    """
    Abstract class for all the treatments
    """

    name = ""
    susceptible_stages: List[str] = []

    def __init__(self, payload):
        self.quadratic_fish_mortality_coeffs = np.array(
            payload["quadratic_fish_mortality_coeffs"]
        )
        self.effect_delay: int = payload["effect_delay"]
        self.application_period: int = payload["application_period"]
        self.gene: Gene = payload["gene"]

    @staticmethod
    def parse_pheno_resistance(pheno_resistance_dict: dict) -> TreatmentResistance:
        return {
            HeterozygousResistance[key.upper()]: val
            for key, val in pheno_resistance_dict.items()
        }

    def __get_mortality_pp_increase(
        self, temperature: float, fish_mass: float
    ) -> float:
        """Get the mortality percentage point difference increase.

        :param temperature: the temperature in Celsius
        :param fish_mass: the fish mass (in grams)
        :returns: Mortality percentage point difference increase
        """
        # TODO: is this the right way to solve this?
        fish_mass_indicator = 1 if fish_mass > 2000 else 0

        input = np.array(
            [
                1,
                temperature,
                fish_mass_indicator,
                temperature**2,
                temperature * fish_mass_indicator,
                fish_mass_indicator**2,
            ]
        )
        return max(float(self.quadratic_fish_mortality_coeffs.dot(input)), 0)

    @abstractmethod
    def delay(self, average_temperature: float):  # pragma: no cover
        """
        Delay before treatment should have a noticeable effect
        """

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def get_allele_heterozygous_trait(gene: Gene, alleles: Allele):
        """
        Get the allele heterozygous type
        """
        # should we move this?
        if gene.upper() in alleles:
            if gene.lower() in alleles:
                trait = HeterozygousResistance.INCOMPLETELY_DOMINANT
            else:
                trait = HeterozygousResistance.DOMINANT
        else:
            trait = HeterozygousResistance.RECESSIVE
        return trait

    @abstractmethod
    def get_lice_treatment_mortality_rate(
        self, temperature: float
    ) -> GenoTreatmentDistrib:
        """
        Calculate the mortality rates of this treatment

        :param temperature: the water temperature

        :returns: the mortality rates, arranged by geno.
        """

    def get_fish_mortality_occurrences(
        self,
        temperature: float,
        fish_mass: float,
        num_fish: float,
        efficacy_window: float,
        mortality_events: int,
    ):
        """Get the number of fish that die due to treatment

        :param temperature: the temperature of the cage
        :param num_fish: the number of fish
        :param fish_mass: the average fish mass (in grams)
        :param efficacy_window: the length of the efficacy window
        :param mortality_events: the number of fish mortality events to subtract from
        """
        predicted_pp_increase = self.__get_mortality_pp_increase(temperature, fish_mass)

        mortality_events_pp = 100 * mortality_events / num_fish
        predicted_deaths = (
            (predicted_pp_increase + mortality_events_pp) * num_fish / 100
        ) - mortality_events
        predicted_deaths /= efficacy_window

        return predicted_deaths


class ChemicalTreatment(TreatmentParams):
    """Trait for all chemical treatments"""

    def __init__(self, payload):
        super().__init__(payload)
        self.pheno_resistance = self.parse_pheno_resistance(payload["pheno_resistance"])
        self.price_per_kg: float = payload["price_per_kg"]
        self.dosage_per_fish_kg: float = payload["dosage_per_fish_kg"]

        self.durability_temp_ratio: float = payload["durability_temp_ratio"]


class ThermalTreatment(TreatmentParams):
    """Trait for all thermal-based treatments"""

    def __init__(self, payload):
        super().__init__(payload)
        self.price_per_application = payload["price_per_application"]
        # NOTE: these are currently unused
        # self.exposure_temperature: float = payload["exposure_temperature"]
        # self.exposure_length: float = payload["efficacy"]


class EMB(ChemicalTreatment):
    """Emamectin Benzoate"""

    name = "EMB"
    susceptible_stages = ["L3", "L4", "L5m", "L5f"]

    def delay(self, average_temperature: float):
        return self.durability_temp_ratio / average_temperature

    def get_lice_treatment_mortality_rate(self, _temperature=None):
        geno_treatment_distrib = {}

        for geno in geno_to_alleles(self.gene):
            # TODO: we could optimise this
            trait = self.get_allele_heterozygous_trait(self.gene, geno)
            susceptibility_factor = 1.0 - self.pheno_resistance[trait]
            geno_treatment_distrib[geno] = GenoTreatmentValue(
                susceptibility_factor, self.susceptible_stages
            )

        return geno_treatment_distrib


class Thermolicer(ThermalTreatment):
    name = "Thermolicer"
    susceptible_stages = ["L3", "L4", "L5m", "L5f"]

    @functools.lru_cache
    def delay(self, _):
        return 1  # effects noticeable the next day

    def get_lice_treatment_mortality_rate(
        self, temperature: float
    ) -> GenoTreatmentDistrib:
        if temperature >= 12:
            efficacy = 0.8
        else:
            efficacy = 0.99

        geno_treatment_distrib = {
            geno: GenoTreatmentValue(efficacy, self.susceptible_stages)
            # TODO
            # for geno in GenoDistrib.alleles_from_gene(self.gene)
            for geno in GenoDistrib.alleles
        }
        return geno_treatment_distrib
