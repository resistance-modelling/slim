import logging
from dataclasses import dataclass
import datetime as dt
import json
import os
from typing import Tuple, Dict, Optional, TYPE_CHECKING


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
        if k in data:
            data[k]["value"] = v


class RuntimeConfig:
    """Simulation parameters and constants"""

    def __init__(self, hyperparam_file, _override_options):
        with open(hyperparam_file) as f:
            data = json.load(f)

        override(data, _override_options)

        # Evolution constants
        self.stage_age_evolutions = data["stage_age_evolutions"]["value"]  # type: Dict[LifeStage, float]
        self.delta_p = data["delta_p"]["value"]  # type: Dict[LifeStage, float]
        self.delta_s = data["delta_s"]["value"]  # type: Dict[LifeStage, float]
        self.delta_m10 = data["delta_m10"]["value"]  # type: Dict[LifeStage, float]
        self.smolt_mass_params = SmoltParams(**data["smolt_mass_params"]["value"])

        # Infection constants
        self.infection_main_delta = data["infection_main_delta"]["value"]  # type: float
        self.infection_weight_delta = data["infection_weight_delta"]["value"]  # type: float
        self.delta_expectation_weight_log = data["delta_expectation_weight_log"]["value"]  # type: float

        # Treatment constants
        self.emb = EMB(data["treatments"]["value"]["emb"]["value"])

        # Fish mortality constants
        self.fish_mortality_center = data["fish_mortality_center"]["value"]  # type: float
        self.fish_mortality_k = data["fish_mortality_k"]["value"]  # type: float
        self.male_detachment_rate = data["male_detachment_rate"]["value"]  # type: float

        # Background lice mortality constants
        self.background_lice_mortality_rates = data["background_lice_mortality_rates"]["value"]  # type: Dict[LifeStage, float]

        # Reproduction and recruitment constants
        self.reproduction_eggs_first_extruded = data["reproduction_eggs_first_extruded"]["value"]
        self.reproduction_age_dependence = data["reproduction_age_dependence"]["value"]
        self.dam_unavailability = data["dam_unavailability"]["value"]
        self.genetic_mechanism = GeneticMechanism[data["genetic_mechanism"]["value"]]
        self.geno_mutation_rate = data["geno_mutation_rate"]["value"]

        # TODO: take into account processing of non-discrete keys
        self.genetic_ratios = {tuple(sorted(key.split(","))): val for key, val in data["genetic_ratios"]["value"].items()}

        # Farm data
        self.farm_data = data["farm_data"]["value"]

        # Other reward/payoff constants
        self.gain_per_kg = Money(data["gain_per_kg"]["value"])

        # Other constraints
        self.aggregation_rate_threshold = data["aggregation_rate_threshold"]["value"]  # type: float

        # load in the seed if provided
        # otherwise don't use a seed
        seed_dict = data.get("seed", 0)
        self.seed = seed_dict["value"] if seed_dict else None

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

        # time and dates
        self.start_date = to_dt(data["start_date"]["value"])
        self.end_date = to_dt(data["end_date"]["value"])

        # general parameters
        self.ext_pressure = data["ext_pressure"]["value"]

        self.monthly_cost = Money(data["monthly_cost"]["value"])
        self.name = data["name"]["value"]  # type: str
        self.defection_proba = data["defection_proba"]["value"]  # type: float

        # farms
        self.farms = [FarmConfig(farm_data["value"])
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
        self.num_fish = data["num_fish"]["value"]  # type: int
        self.n_cages = data["ncages"]["value"]  # type: int
        self.farm_location = data["location"]["value"]  # type: Tuple[int, int]
        self.farm_start = to_dt(data["start_date"]["value"])
        self.cages_start = [to_dt(date)
                            for date in data["cages_start_dates"]["value"]]
        self.max_num_treatments = data["max_num_treatments"]["value"]  # type: int
        self.sampling_spacing = data["sampling_spacing"]["value"]  # type: int

        # TODO: a farm may employ different chemicals
        self.treatment_type = Treatment[data["treatment_type"]["value"]]

        # Farm-specific capital
        self.start_capital = Money(data["start_capital"]["value"])

        # fixed treatment schedules
        self.treatment_starts = [to_dt(date) for date in data["treatment_dates"]["value"]]


@dataclass
class SmoltParams:
    max_mass: float
    skewness: float
    x_shift: float
