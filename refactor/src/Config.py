import logging
from dataclasses import dataclass
import datetime as dt
import json
import os
from pathlib import Path
from typing import Tuple, Dict, Optional, TYPE_CHECKING

import jsonschema
import numpy as np

from src.TreatmentTypes import Treatment, TreatmentParams, GeneticMechanism, EMB, Money

if TYPE_CHECKING:
    from src.LicePopulation import LifeStage


def to_dt(string_date) -> dt.datetime:
    """Convert from string date to datetime date

    :param string_date: Date as string timestamp
    :type string_date: str
    :return: Date as Datetime
    :rtype: [type]
    """
    dt_format = "%Y-%m-%d %H:%M:%S"
    return dt.datetime.strptime(string_date, dt_format)


def override(data, override_options: dict):
    for k, v in override_options.items():
        if k in data and v is not None:
            data[k] = v


class RuntimeConfig:
    """Simulation parameters and constants"""

    def __init__(self, hyperparam_file, _override_options):
        with open(hyperparam_file) as f:
            data = json.load(f)

        override(data, _override_options)
        hyperparam_dir = Path(hyperparam_file).parent
        with (hyperparam_dir / "config.schema.json").open() as f:
            schema = json.load(f)

        jsonschema.validate(data, schema)

        # Evolution constants
        self.stage_age_evolutions = data["stage_age_evolutions"]  # type: Dict[LifeStage, float]
        self.delta_p = data["delta_p"]  # type: Dict[LifeStage, float]
        self.delta_s = data["delta_s"]  # type: Dict[LifeStage, float]
        self.delta_m10 = data["delta_m10"]  # type: Dict[LifeStage, float]
        self.smolt_mass_params = SmoltParams(**data["smolt_mass_params"])

        # Infection constants
        self.infection_main_delta = data["infection_main_delta"]  # type: float
        self.infection_weight_delta = data["infection_weight_delta"]  # type: float
        self.delta_expectation_weight_log = data["delta_expectation_weight_log"]  # type: float

        # Treatment constants
        self.emb = EMB(data["treatments"][0])

        # Fish mortality constants
        self.fish_mortality_center = data["fish_mortality_center"]  # type: float
        self.fish_mortality_k = data["fish_mortality_k"]  # type: float
        self.male_detachment_rate = data["male_detachment_rate"]  # type: float

        # Background lice mortality constants
        self.background_lice_mortality_rates = data["background_lice_mortality_rates"]  # type: Dict[LifeStage, float]

        # Reproduction and recruitment constants
        self.reproduction_eggs_first_extruded: int = data["reproduction_eggs_first_extruded"]
        self.reproduction_age_dependence: float = data["reproduction_age_dependence"]
        self.dam_unavailability: int = data["dam_unavailability"]
        self.genetic_mechanism = GeneticMechanism[data["genetic_mechanism"]]
        self.geno_mutation_rate: float = data["geno_mutation_rate"]

        # TODO: take into account processing of non-discrete keys
        self.genetic_ratios = {tuple(sorted(key.split(","))): val for key, val in data["genetic_ratios"].items()}

        # Farm data
        self.farm_data = data["farm_data"]

        # Other reward/payoff constants
        self.gain_per_kg = Money(data["gain_per_kg"])

        # Other constraints
        self.aggregation_rate_threshold = data["aggregation_rate_threshold"]  # type: float

        # load in the seed if provided
        # otherwise don't use a seed
        self.seed = data.get("seed", 0)

        self.rng = np.random.default_rng(seed=self.seed)


class Config(RuntimeConfig):
    """One-stop class to hold constants, farm setup and other settings."""

    def __init__(
        self,
        config_file: str,
        simulation_dir: str,
        override_params: Optional[dict]= None,
        save_rate: Optional[int] = None
    ):
        """Read the configuration from files

        :param config_file: Path to the environment JSON file
        :type config_file: string
        :param simulation_dir: path to the simulator parameters JSON file
        :param override_params: options that override the config
        :param save_rate if True
        """

        if override_params is None:
            override_params = dict()
        super().__init__(config_file, override_params)

        # read and set the params
        with open(os.path.join(simulation_dir, "params.json")) as f:
            data = json.load(f)

        override(data, override_params)

        with open(os.path.join(simulation_dir, "../params.schema.json")) as f:
            schema = json.load(f)

        jsonschema.validate(data, schema)

        # time and dates
        self.start_date = to_dt(data["start_date"])
        self.end_date = to_dt(data["end_date"])

        # general parameters
        self.ext_pressure = data["ext_pressure"]

        self.monthly_cost = Money(data["monthly_cost"])
        self.name = data["name"]  # type: str

        # farms
        self.farms = [FarmConfig(farm_data)
                      for farm_data in data["farms"]]
        self.nfarms = len(self.farms)

        self.interfarm_times = np.loadtxt(os.path.join(simulation_dir, "interfarm_time.csv"), delimiter=",")
        self.interfarm_probs = np.loadtxt(os.path.join(simulation_dir, "interfarm_prob.csv"), delimiter=",")

        # driver-specific settings
        self.save_rate = save_rate

    def get_treatment(self, treatment_type: Treatment) -> TreatmentParams:
        return [self.emb][treatment_type.value]


class FarmConfig:
    """Config for individual farm"""

    def __init__(self, data: dict):
        """Create farm configuration

        :param data: Dictionary with farm data
        """

        # set params
        self.num_fish = data["num_fish"]  # type: int
        self.n_cages = data["ncages"]  # type: int
        self.farm_location = data["location"]  # type: Tuple[int, int]
        self.farm_start = to_dt(data["start_date"])
        self.cages_start = [to_dt(date)
                            for date in data["cages_start_dates"]]
        self.max_num_treatments = data["max_num_treatments"]  # type: int
        self.sampling_spacing = data["sampling_spacing"]  # type: int

        # TODO: a farm may employ different chemicals
        self.treatment_type = Treatment[data["treatment_type"]]

        # Farm-specific capital
        self.start_capital = Money(data["start_capital"])

        # Defection probability
        self.defection_proba: float = data["defection_proba"]

        # fixed treatment schedules
        self.treatment_starts = [to_dt(date) for date in data["treatment_dates"]]


@dataclass
class SmoltParams:
    max_mass: float
    skewness: float
    x_shift: float
