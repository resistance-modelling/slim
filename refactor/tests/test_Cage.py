import datetime as dt

import numpy as np
import pytest

from src.Config import Config, to_dt
from src.Farm import Farm
from .fixtures import farm, first_cage

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

    def test_cage_update_lice_treatment_mortality_no_effect(self, farm, first_cage):
        treatment_dates = farm.treatment_dates
        assert(treatment_dates == sorted(treatment_dates))

        # before a 14-day activation period there should be no effect
        # TODO:
        for i in range(-14, 14):
            cur_day = treatment_dates[0] - dt.timedelta(days=i)
            mortality_updates = first_cage.update_lice_treatment_mortality(cur_day)
            assert all(rate == 0 for rate in mortality_updates.values())

    def test_cage_update_lice_treatment_mortality(self, farm, first_cage):
        # TODO: this does not take into account water temperature!
        treatment_dates = farm.treatment_dates

        # first useful day
        cur_day = treatment_dates[0] + dt.timedelta(days=5)
        mortality_updates = first_cage.update_lice_treatment_mortality(cur_day)

        # FIXME: This test is broken for now. No lice die for some reason.
        #assert mortality_updates.values() == {
        #    "L1": 0,
        #    "L2": 0,
        #    "L3": 2,
        #    "L4": 4,
        #    "L5m": 0,
        #    "L5f": 1
        #}