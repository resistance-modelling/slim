from abc import abstractmethod, ABC
from decimal import Decimal
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


class TreatmentParams(ABC):

    name = ""

    def __init__(self, data):
        self.data = data
        self.pheno_resistance = self.parse_pheno_resistance(data["pheno_resistance"]["value"])
        self.price_per_kg = Decimal(data["price_per_kg"]["value"])

    def __getattr__(self, name):
        if name in self.data:
            return self.data[name]["value"]
        else: # pragma: no cover
            raise AttributeError("{} not found in {} parameters".format(name, self.name))

    @staticmethod
    def parse_pheno_resistance(pheno_resistance_dict: dict) -> TreatmentResistance:
        return {HeterozygousResistance[key]: val for key, val in pheno_resistance_dict.items()}

    @abstractmethod
    def delay(self, average_temperature: float):  # pragma: no cover
        pass


class EMB(TreatmentParams):
    name = "EMB"

    def delay(self, average_temperature: float):
        return self.durability_temp_ratio / average_temperature

