import numpy as np
import pytest
import datetime


# ignore profiling
import builtins
builtins.__dict__['profile'] = lambda x: x

from src.Config import Config
from src.Simulator import Organisation
from src.QueueTypes import EggBatch, DamAvailabilityBatch
from src.LicePopulation import LicePopulation, GenoDistrib



@pytest.fixture
def initial_lice_population():
    return {"L1": 150, "L2": 0, "L3": 30, "L4": 30, "L5f": 10, "L5m": 10}


@pytest.fixture
def farm_config():
    np.random.seed(0)
    cfg = Config("config_data/config.json", "config_data/Fyne")
    return cfg


@pytest.fixture
def no_prescheduled_config(farm_config):
    farm_config.farms[0].treatment_starts = []
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
def null_offspring_distrib():
    return GenoDistrib({
        ('A',): 0,
        ('a',): 0,
        ('A', 'a'): 0,
    })


@pytest.fixture
def sample_offspring_distrib():
    return GenoDistrib({
        ('A',): 100,
        ('a',): 200,
        ('A', 'a'): 300,
    })

@pytest.fixture
def null_hatched_arrivals(null_offspring_distrib, first_farm):
    return {first_farm.start_date: null_offspring_distrib}


@pytest.fixture
def null_egg_batch(null_offspring_distrib, first_farm):
    return EggBatch(first_farm.start_date, null_offspring_distrib)


@pytest.fixture
def null_dams_batch(null_offspring_distrib, first_farm):
    return DamAvailabilityBatch(first_farm.start_date, null_offspring_distrib)


@pytest.fixture
def sample_eggs_by_hatch_date(sample_offspring_distrib, first_farm):
    return {first_farm.start_date: null_offspring_distrib}


@pytest.fixture
def null_treatment_mortality():
    return LicePopulation.get_empty_geno_distrib()


@pytest.fixture
def sample_treatment_mortality(first_cage, first_cage_population, null_offspring_distrib):
    mortality = first_cage_population.get_empty_geno_distrib()

    # create a custom rng to avoid breaking other tests
    rng = np.random.default_rng(0)

    probs = [0.01, 0.9, 0.09]

    # create n stages
    target_mortality = {"L1": 0, "L2": 0, "L3": 10, "L4": 10, "L5m": 5, "L5f": 5}

    for stage, target in target_mortality.items():
        bins = rng.multinomial(min(target, first_cage_population[stage]), probs).tolist()
        alleles = null_offspring_distrib.keys()
        mortality[stage] = GenoDistrib(dict(zip(alleles, bins)))

    return mortality


@pytest.fixture
def planctonic_only_population(first_cage):
    lice_pop = {"L1": 100, "L2": 200, "L3": 0, "L4": 0, "L5f": 0, "L5m": 0}
    geno = {stage: GenoDistrib({("a",): 0, ("A", "a"): 0, ("A",): 0}) for stage in lice_pop.keys()}
    geno["L1"][("a",)] = 100
    geno["L2"][("a",)] = 200
    return LicePopulation(geno, first_cage.cfg.genetic_ratios)
