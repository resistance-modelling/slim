import logging
import numpy as np
import pytest
import datetime

from src.Config import Config, to_dt
from src.Farm import Farm
from src.Cage import EggBatch, DamAvailabilityBatch


@pytest.fixture
def farm_config():
    np.random.seed(0)
    cfg = Config("config_data/config.json", "config_data/Fyne", logging.getLogger('dummy'))
    return cfg


@pytest.fixture
def farm(farm_config):
    return Farm(0, farm_config)

@pytest.fixture
def farm_two(farm_config):
    # use config of farm 1 but change the name
    farm = Farm(0, farm_config)
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
