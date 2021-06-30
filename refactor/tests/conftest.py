import logging
import numpy as np
import pytest

from src.Config import Config, to_dt
from src.Farm import Farm
from src.Reservoir import Reservoir

@pytest.fixture
def farm_config():
    np.random.seed(0)
    cfg = Config("config_data/test.json", "config_data/params.json", logging.getLogger('dummy'))
    return cfg

@pytest.fixture
def farm(farm_config):
    return Farm(0, farm_config)


@pytest.fixture
def reservoir(farm_config):
    total = 1000
    farm_config.reservoir_num_lice = total
    return Reservoir(farm_config)


@pytest.fixture
def first_cage(farm):
    return farm.cages[0]