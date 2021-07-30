from enum import Enum
from typing import Dict


class Treatment(Enum):
    """
    A stub for treatment types
    TODO: add other treatments here
    """
    emb = 1


class GeneticMechanism(Enum):
    """
    Genetic mechanism to be used when generating egg genotypes
    """
    discrete = 1
    maternal = 2
    quantitative = 3


class HeterozygousResistance(Enum):
    """
    Resistance in a monogenic, heterozygous setting.
    """
    dominant = 1
    incompletely_dominant = 2
    recessive = 3


TreatmentResistance = Dict[HeterozygousResistance, float]


class TreatmentParams:

    name = ""

    def __init__(self, data):
        self.data = data
        self.pheno_resistance = self.parse_pheno_resistance(data["pheno_resistance"]["value"])

    def __getattr__(self, name):
        if name in self.data:
            return self.data[name]["value"]
        else:
            raise AttributeError("{} not found in {} parameters".format(name, self.name))

    def parse_pheno_resistance(self, pheno_resistance_dict: dict) -> TreatmentResistance:
        return {HeterozygousResistance[key]: val for key, val in pheno_resistance_dict.items()}


class EMB(TreatmentParams):
    name = "EMB"
