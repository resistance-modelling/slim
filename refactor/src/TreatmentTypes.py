from enum import Enum
from typing import Dict
from mypy_extensions import TypedDict


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


TreatmentResistance = Dict[Treatment, Dict[HeterozygousResistance, float]]
InfectionDelay = Dict[Treatment, TypedDict("infection_data", {"time": int, "prob": float})]
