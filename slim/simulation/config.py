__all__ = ["to_dt", "Config"]

import argparse
import datetime as dt
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, TYPE_CHECKING, Union

import jsonschema
import numpy as np

from slim.types.treatments import (
    Treatment,
    TreatmentParams,
    GeneticMechanism,
    EMB,
    Thermolicer,
)

if TYPE_CHECKING:
    from slim.simulation.lice_population import LifeStage, GenoDistribDict


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
        self.stage_age_evolutions: Dict[LifeStage, float] = data["stage_age_evolutions"]
        self.delta_p: Dict[LifeStage, float] = data["delta_p"]
        self.delta_s: Dict[LifeStage, float] = data["delta_s"]
        self.delta_m10: Dict[LifeStage, float] = data["delta_m10"]
        self.lice_development_rates: Dict[LifeStage, float] = data[
            "lice_development_rates"
        ]
        self.smolt_mass_params = SmoltParams(**data["smolt_mass_params"])

        # Infection constants
        self.infection_main_delta: float = data["infection_main_delta"]
        self.infection_weight_delta: float = data["infection_weight_delta"]
        self.delta_expectation_weight_log: float = data["delta_expectation_weight_log"]

        # Treatment constants
        self.emb = EMB(data["treatments"][0])
        self.thermolicer = Thermolicer(data["treatments"][1])

        # Fish mortality constants
        self.fish_mortality_center: float = data["fish_mortality_center"]
        self.fish_mortality_k: float = data["fish_mortality_k"]
        self.male_detachment_rate: float = data["male_detachment_rate"]

        # Background lice mortality constants
        self.background_lice_mortality_rates: Dict[LifeStage, float] = data[
            "background_lice_mortality_rates"
        ]

        # Reproduction and recruitment constants
        self.reproduction_eggs_first_extruded: int = data[
            "reproduction_eggs_first_extruded"
        ]
        self.reproduction_age_dependence: float = data["reproduction_age_dependence"]
        self.dam_unavailability: int = data["dam_unavailability"]
        self.genetic_mechanism = GeneticMechanism[data["genetic_mechanism"].upper()]
        self.geno_mutation_rate: float = data["geno_mutation_rate"]

        # TODO: take into account processing of non-discrete keys
        self.reservoir_offspring_integration_ratio: float = data[
            "reservoir_offspring_integration_ratio"
        ]
        self.reservoir_offspring_average: int = data["reservoir_offspring_average"]

        # load in the seed if provided
        # otherwise don't use a seed
        self.seed = data.get("seed", 0)

        self.rng = np.random.default_rng(seed=self.seed)

    def multivariate_hypergeometric(
        self, bins: np.ndarray, balls: Union[int, np.integer]
    ):
        """A wrapper onto numpy's hypergeometric sampler.
        Because numpy cannot handle large bins, we approximate the distribution to a
        multinomial distribution.

        :param bins: the number of bins
        :param balls: the number of balls to sample from the bins

        :returns the sampled distribution
        """
        s = np.sum(bins)
        if s < 100:
            method = "count"
        elif s < 1e6:
            method = "marginals"
        else:
            return self.rng.multinomial(balls, bins / s)
        return self.rng.multivariate_hypergeometric(bins, balls, method=method)


class Config(RuntimeConfig):
    """One-stop class to hold constants, farm setup and other settings."""

    def __init__(self, config_file: str, simulation_dir: str, **kwargs):
        """Read the configuration from files

        :param config_file: Path to the environment JSON file
        :type config_file: string
        :param simulation_dir: path to the simulator parameters JSON file
        :param override_params: options that override the config
        :param save_rate: if not null it determines how often (in terms of days) the simulator saves the state.
        """

        # driver-specific settings
        self.save_rate = kwargs.pop("save_rate", 1)
        self.checkpoint_rate = kwargs.pop("checkpoint_rate", 0)
        self.farms_per_process = kwargs.pop("farms_per_process", -1)
        if self.farms_per_process is None:
            self.farms_per_process = -1

        super().__init__(config_file, kwargs)

        self.experiment_id = simulation_dir

        # read and set the params
        with open(os.path.join(simulation_dir, "params.json")) as f:
            data = json.load(f)

        override(data, kwargs)

        with open(os.path.join(simulation_dir, "../params.schema.json")) as f:
            schema = json.load(f)

        jsonschema.validate(data, schema)

        self.name: str = data["name"]

        # time and dates
        self.start_date = to_dt(data["start_date"])
        self.end_date = to_dt(data["end_date"])

        # Experiment-specific genetic ratios
        self.min_ext_pressure = data["ext_pressure"]
        self.initial_genetic_ratios: GenoDistribDict = data["genetic_ratios"]
        self.genetic_learning_rate: float = data["genetic_learning_rate"]
        self.monthly_cost: float = data["monthly_cost"]
        self.gain_per_kg: float = data["gain_per_kg"]
        self.infection_discount: float = data["infection_discount"]

        # Other constraints
        self.aggregation_rate_threshold: float = data["aggregation_rate_threshold"]
        self.aggregation_rate_enforcement_limit: float = data[
            "aggregation_rate_enforcement_limit"
        ]
        self.aggregation_rate_enforcement_weeks: int = data[
            "aggregation_rate_enforcement_weeks"
        ]

        # Policy
        self.treatment_strategy: str = data["treatment_strategy"]

        # farms
        self.farms = [FarmConfig(farm_data) for farm_data in data["farms"]]
        self.nfarms = len(self.farms)

        self.interfarm_times = np.loadtxt(
            os.path.join(simulation_dir, "interfarm_time.csv"), delimiter=","
        )
        self.interfarm_probs = np.loadtxt(
            os.path.join(simulation_dir, "interfarm_prob.csv"), delimiter=","
        )
        self.loch_temperatures = np.loadtxt(
            os.path.join(simulation_dir, "temperatures.csv"), delimiter=","
        )

    def get_treatment(self, treatment_type: Treatment) -> TreatmentParams:
        return [self.emb, self.thermolicer][treatment_type.value]

    @staticmethod
    def generate_argparse_from_config(
        cfg_schema_path: str, simulation_schema_path: str
    ):  # pragma: no cover
        parser = argparse.ArgumentParser(description="Sea lice simulation")

        # TODO: we are parsing the config twice.
        with open(cfg_schema_path) as fp:
            cfg_dict: dict = json.load(fp)

        with open(simulation_schema_path) as fp:
            simulation_dict: dict = json.load(fp)

        def add_to_group(group_name, data):
            group = parser.add_argument_group(group_name)
            schema_types_to_python = {"string": str, "number": float, "integer": int}

            for k, v in data.items():
                choices = None
                nargs = None
                type_ = None
                if type(v) != dict:
                    continue
                if "type" not in v:
                    if "enum" in v:
                        choices = v["enum"]
                        type_ = "string"
                    else:
                        # skip property
                        continue
                else:
                    type_ = v["type"]

                if type_ == "array":
                    nargs = v.get("minLength", "*")
                    if "items" in v:
                        type_ = v["items"]["type"]  # this breaks with object arrays

                if type_ == "object":
                    continue  # TODO: deal with them later, e.g. prop_a.prop_b for dicts?
                description = v["description"]
                value_type = schema_types_to_python.get(type_, type_)

                group.add_argument(
                    f"--{k.replace('_', '-')}",
                    type=value_type,
                    help=description,
                    choices=choices,
                    nargs=nargs,
                )

        add_to_group("Organisation parameters", simulation_dict["properties"])
        add_to_group("Runtime parameters", cfg_dict["properties"])

        return parser


class FarmConfig:
    """Config for individual farm"""

    def __init__(self, data: dict):
        """Create farm configuration

        :param data: Dictionary with farm data
        """

        # set params
        self.name: str = data["name"]
        self.num_fish: int = data["num_fish"]
        self.n_cages: int = data["ncages"]
        self.farm_location: Tuple[int, int] = data["location"]
        self.farm_start = to_dt(data["start_date"])
        self.cages_start = [to_dt(date) for date in data["cages_start_dates"]]
        self.max_num_treatments: int = data["max_num_treatments"]
        self.sampling_spacing: int = data["sampling_spacing"]

        self.treatment_types = [
            Treatment[treatment.upper()] for treatment in data["treatment_types"]
        ]

        # Defection probability
        self.defection_proba: float = data["defection_proba"]

        # fixed treatment schedules
        self.treatment_dates = [
            (to_dt(date), Treatment[treatment.upper()])
            for (date, treatment) in data["treatment_dates"]
        ]


@dataclass
class SmoltParams:
    max_mass: float
    skewness: float
    x_shift: float
