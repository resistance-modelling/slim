import numpy as np
import pytest

from src.Config import Config
from src.Farm import Farm

@pytest.fixture
def farm():
    np.random.seed(0)
    cfg = Config("config_data/test.json", None)
    return Farm(0, cfg)

class TestFarm:
    def test_farm_loads_params(self, farm):
        assert farm.name == 0
        assert len(farm.cages) == 6
        assert farm.loc_x == 190300
        assert farm.loc_y == 665300
