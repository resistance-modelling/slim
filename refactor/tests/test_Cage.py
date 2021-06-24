import numpy as np
import pytest

from src.Config import Config, to_dt
from src.Farm import Farm
from .fixtures import farm

@pytest.fixture
def first_cage(farm):
    return farm.cages[0]

class TestCage:
    def test_cage_loads_params(self, first_cage):
        assert first_cage.id == 0
        assert first_cage.start_date == to_dt("2017-10-01 00:00:00")
        assert first_cage.num_fish == 4000
        assert first_cage.num_infected_fish == 0
        # this will likely break
        assert first_cage.lice_population == {
            'L1': 150,
            'L2': 0,
            'L3': 30,
            'L4': 30,
            'L5f': 10,
            'L5m': 0
        }

    def test_cage_lice_background_mortality_one_day(self, first_cage):
        # NOTE: this currently relies on Stien's approach.
        # Changing this approach will break things
        dead_lice_dist = first_cage.update_background_lice_mortality(first_cage.lice_population, 1)
        dead_lice_dist_np = np.array(list(dead_lice_dist.values()))
        expected_dead_lice = np.array([26, 0, 0, 2, 0, 0])
        assert np.alltrue(dead_lice_dist_np >= 0.0)
        assert np.alltrue(np.isclose(dead_lice_dist_np, expected_dead_lice))