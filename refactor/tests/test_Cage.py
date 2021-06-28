import datetime as dt
import itertools

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
        for i in range(-14, first_cage.cfg.delay_EMB):
            cur_day = treatment_dates[0] + dt.timedelta(days=i)
            mortality_updates = first_cage.update_lice_treatment_mortality(cur_day)
            assert all(rate == 0 for rate in mortality_updates.values())

    def test_cage_update_lice_treatment_mortality(self, farm, first_cage):
        # TODO: this does not take into account water temperature!
        treatment_dates = farm.treatment_dates

        # first useful day
        cur_day = treatment_dates[0] + dt.timedelta(days=5)
        mortality_updates = first_cage.update_lice_treatment_mortality(cur_day)

        assert mortality_updates == {
            "L1": 0,
            "L2": 0,
            "L3": 0,
            "L4": 3,
            "L5m": 0,
            "L5f": 0
        }

    def test_get_stage_ages_respects_constraints(self, first_cage):
        test_num_lice = 1000
        development_days = 15
        min_development_stage = 4
        mean_development_stage = 8

        for i in range(100):
            stage_ages = first_cage.get_stage_ages(
                test_num_lice,
                min=min_development_stage,
                mean=mean_development_stage,
                development_days=development_days
            )

            assert len(stage_ages) == test_num_lice

            assert np.min(stage_ages) >= min_development_stage
            assert np.max(stage_ages) < development_days

            # This test luckily doesn't fail
            assert abs(np.mean(stage_ages) - mean_development_stage) < 1

    def test_get_stage_ages_edge_cases(self, first_cage):
        test_num_lice = 1000
        development_days = 15
        min_development_stages = list(range(10))
        mean_development_stages = list(range(10))

        for min, mean in itertools.product(min_development_stages, mean_development_stages):
            if mean <= min or min == 0:
                with pytest.raises(AssertionError):
                    first_cage.get_stage_ages(
                    test_num_lice,
                    min=min,
                    mean=mean,
                    development_days=development_days
                )
            else:
                first_cage.get_stage_ages(
                    test_num_lice,
                    min=min,
                    mean=mean,
                    development_days=development_days
                )

    def test_update_lice_lifestage(self, first_cage):
        new_l2, new_l4, new_females, new_males = first_cage.update_lice_lifestage(1)

        assert new_l2 == 0
        assert new_l4 == 30 and first_cage.lice_population["L3"] > 0
        assert new_males == 2 and first_cage.lice_population["L4"] > 0
        assert new_females == 2

    def test_update_fish_growth(self, first_cage):
        # TODO: infestation has not been implemented, this does not take into account the actual lice population!
        natural_death, lice_death = first_cage.update_fish_growth(1, 1)

        assert natural_death > 0
        assert lice_death >= 0

    def test_update_fish_growth_no_lice(self, first_cage):
        # See above
        first_cage.num_infected_fish = 0
        for k in first_cage.lice_population:
            first_cage.lice_population = 0

        _, lice_death = first_cage.update_fish_growth(1, 1)
        # assert lice_death == 0