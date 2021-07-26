import logging
import numpy as np
import pytest
import datetime

from src.Config import Config
from src.Farm import Farm
from src.QueueBatches import EggBatch, DamAvailabilityBatch
from src.LicePopulation import LicePopulation


@pytest.fixture
def initial_lice_population():
    return {"L1": 150, "L2": 0, "L3": 30, "L4": 30, "L5f": 10, "L5m": 10}


@pytest.fixture
def farm_config():
    np.random.seed(0)
    cfg = Config("config_data/config.json", "config_data/Fyne", logging.getLogger('dummy'))
    return cfg


@pytest.fixture
def farm(farm_config, initial_lice_population):
    return Farm(0, farm_config, initial_lice_population)


@pytest.fixture
def farm_two(farm_config, initial_lice_population):
    # use config of farm 1 but change the name
    farm = Farm(0, farm_config, initial_lice_population)
    farm.name = 1
    return farm


@pytest.fixture
def first_cage(farm):
    return farm.cages[0]


@pytest.fixture
def cur_day(first_cage):
    return first_cage.date + datetime.timedelta(days=1)


@pytest.fixture
def null_offspring_distrib():
    return {
        ('A',): 0,
        ('a',): 0,
        ('A', 'a'): 0,
    }


@pytest.fixture
def sample_offspring_distrib():
    return {
        ('A',): 100,
        ('a',): 200,
        ('A', 'a'): 300,
    }


@pytest.fixture
def null_hatched_arrivals(null_offspring_distrib, farm):
    return {farm.start_date: null_offspring_distrib}


@pytest.fixture
def null_egg_batch(null_offspring_distrib, farm):
    return EggBatch(farm.start_date, null_offspring_distrib)


@pytest.fixture
def null_dams_batch(null_offspring_distrib, farm):
    return DamAvailabilityBatch(farm.start_date, null_offspring_distrib)


@pytest.fixture
def sample_eggs_by_hatch_date(sample_offspring_distrib, farm):
    return {farm.start_date: null_offspring_distrib}


@pytest.fixture
def planctonic_only_population(first_cage):
    lice_pop = {"L1": 100, "L2": 200, "L3": 0, "L4": 0, "L5f": 0, "L5m": 0}
    geno = {stage: {("a",): 0, ("A", "a"): 0, ("A",): 0} for stage in lice_pop.keys()}
    geno["L1"][("a",)] = 100
    geno["L2"][("a",)] = 200
    return LicePopulation(lice_pop, geno, first_cage.cfg)
