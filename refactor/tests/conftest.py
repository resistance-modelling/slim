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
    cfg = Config("config_data/test.json", "config_data/params.json", logging.getLogger('dummy'))
    return cfg


@pytest.fixture
def farm(farm_config):
    return Farm(0, farm_config)


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
def null_egg_batch(null_offspring_distrib, farm):
    return EggBatch(farm.start_date, null_offspring_distrib)


@pytest.fixture
def null_dams_batch(null_offspring_distrib, farm):
    return DamAvailabilityBatch(farm.start_date, null_offspring_distrib)
