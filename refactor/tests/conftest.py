import logging
import numpy as np
import pytest

from src.Config import Config, to_dt
from src.Farm import Farm

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