import logging
import numpy as np
import pytest

from src.Config import Config, to_dt
from src.Farm import Farm

@pytest.fixture
def farm():
    np.random.seed(0)
    cfg = Config("config_data/test.json", logging.getLogger('dummy'))
    return Farm(0, cfg)

@pytest.fixture
def first_cage(farm):
    return farm.cages[0]
