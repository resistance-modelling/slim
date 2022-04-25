from pathlib import Path

import numpy as np
import pytest
import datetime


# ignore profiling
import builtins

import ray

builtins.__dict__["profile"] = lambda x: x

# Disable numba's jitting for now
import os

os.environ["NUMBA_DISABLE_JIT"] = "2"

from slim.simulation.config import Config
from slim.simulation.organisation import Organisation
from slim.simulation.simulator import get_env, SimulatorPZEnv
from slim.types.queue import EggBatch
from slim.simulation.lice_population import (
    LicePopulation,
    GenoDistrib,
    empty_geno_from_cfg,
    geno_config_to_matrix,
    from_ratios_rng,
    from_dict,
)


@pytest.fixture(autouse=True, scope="session")
def run_destroy():
    # async actors cannot use local mode
    ray.init(local_mode=False, num_cpus=1, num_gpus=0)
    yield
    ray.shutdown()


@pytest.fixture
def initial_lice_population():
    return {"L1": 150, "L2": 0, "L3": 30, "L4": 30, "L5f": 10, "L5m": 10}


@pytest.fixture
def farm_config():
    np.random.seed(0)

    # Make PyCharm (which believes your working directory must be in tests/) happy

    path = Path.cwd() / "config_data"
    if not path.exists():
        path = Path.cwd().parent / "config_data"

    cfg = Config(str(path / "config.json"), str(path / "test_Fyne"))

    return cfg


@pytest.fixture
def no_prescheduled_config(farm_config):
    farm_config.farms[0].treatment_dates = []
    return farm_config


@pytest.fixture
def no_prescheduled_organisation(no_prescheduled_config, initial_lice_population):
    return Organisation(no_prescheduled_config, initial_lice_population)


@pytest.fixture
def no_prescheduled_farm(no_prescheduled_organisation):
    return no_prescheduled_organisation.farms[0]


@pytest.fixture
def no_prescheduled_cage(no_prescheduled_farm):
    return no_prescheduled_farm.cages[0]


@pytest.fixture
def organisation(farm_config, initial_lice_population):
    return Organisation(farm_config, initial_lice_population)


@pytest.fixture
def first_farm(organisation):
    return organisation.farms[0]


@pytest.fixture
def second_farm(organisation):
    return organisation.farms[1]


@pytest.fixture
def first_cage(first_farm):
    return first_farm.cages[0]


@pytest.fixture
def first_cage_population(first_cage):
    return first_cage.lice_population


@pytest.fixture
def cur_day(first_cage):
    return first_cage.date + datetime.timedelta(days=1)


@pytest.fixture
def null_offspring_distrib(farm_config):
    return empty_geno_from_cfg(farm_config)


@pytest.fixture
def sample_offspring_distrib(null_offspring_distrib):
    # just an alias
    return from_dict(
        {
            "A": 100,
            "a": 200,
            "Aa": 300,
        }
    )


@pytest.fixture
def sim_env(farm_config):
    return get_env(farm_config)


@pytest.fixture
def sim_env_unwrapped(farm_config):
    return SimulatorPZEnv(farm_config)


@pytest.fixture
def null_hatched_arrivals(null_offspring_distrib, first_farm):
    return {first_farm.start_date: null_offspring_distrib}


@pytest.fixture
def null_egg_batch(null_offspring_distrib, first_farm):
    return EggBatch(first_farm.start_date, null_offspring_distrib)


@pytest.fixture
def null_dams_batch():
    return 0


@pytest.fixture
def sample_eggs_by_hatch_date(sample_offspring_distrib, first_farm):
    return {first_farm.start_date: null_offspring_distrib}


@pytest.fixture
def empty_distrib(farm_config):
    return empty_geno_from_cfg(farm_config)


@pytest.fixture
def empty_distrib_v2(farm_config):
    farm_config.initial_genetic_ratios = {
        "A": 0.25,
        "a": 0.25,
        "Aa": 0.5,
        "B": 0.1,
        "b": 0.8,
        "Bb": 0.1,
    }
    return empty_geno_from_cfg(farm_config)


@pytest.fixture
def null_treatment_mortality(farm_config):
    return LicePopulation.get_empty_geno_distrib(farm_config)


@pytest.fixture
def sample_treatment_mortality(
    farm_config,
    first_cage,
    first_cage_population,
    null_offspring_distrib,
    empty_distrib,
):
    mortality = first_cage_population.get_empty_geno_distrib(farm_config)

    # create a custom rng to avoid breaking other tests
    rng = np.random.default_rng(0)

    probs = np.array([[0.01, 0.9, 0.09]])

    # create n stages
    target_mortality = {"L1": 0, "L2": 0, "L3": 10, "L4": 10, "L5m": 5, "L5f": 5}

    for stage, target in target_mortality.items():
        mortality[stage] = from_ratios_rng(
            min(target, first_cage_population[stage]), probs, rng
        )

    return mortality


@pytest.fixture
def planctonic_only_population(farm_config, first_cage):
    lice_pop = {"L1": 100, "L2": 200, "L3": 0, "L4": 0, "L5f": 0, "L5m": 0}
    geno = {stage: empty_geno_from_cfg(farm_config) for stage in lice_pop.keys()}
    geno["L1"][("a",)] = 100
    geno["L2"][("a",)] = 200
    return LicePopulation(geno, first_cage.cfg.initial_genetic_ratios, 0.0)


@pytest.fixture
def initial_external_ratios(farm_config):
    return geno_config_to_matrix(farm_config.initial_genetic_ratios)


@pytest.fixture
def initial_external_inflow(farm_config):
    return farm_config.min_ext_pressure
