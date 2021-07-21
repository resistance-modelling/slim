import numpy as np
import pytest
from src.Config import to_dt
import datetime as dt


class TestFarm:
    def test_farm_loads_params(self, farm):
        assert farm.name == 0
        assert len(farm.cages) == 6
        assert farm.loc_x == 190300
        assert farm.loc_y == 665300

    def test_year_temperatures(self, farm):
        tarbert = farm.cfg.farm_data["tarbert"]["temperatures"]
        ardrishaig = farm.cfg.farm_data["ardrishaig"]["temperatures"]
        temps = np.stack([tarbert, ardrishaig])
        min_temps = np.round(np.min(temps, axis=0), 1)
        max_temps = np.round(np.max(temps, axis=0), 1)
        assert all(min_temps <= farm.year_temperatures)
        assert all(farm.year_temperatures <= max_temps)

    def test_farm_update(self, farm):
        # TODO: test integration across **all** cages. Requires further refactoring.
        pass

    def test_get_cage_pressures(self, farm):

        farm.cfg.ext_pressure = 100
        farm.cages = [0] * 10

        pressures = farm.get_cage_pressures()

        assert len(pressures) == len(farm.cages)
        assert sum(pressures) == farm.cfg.ext_pressure

        for pressure in pressures:
            assert pressure >= 0

    def test_get_cage_pressures_negative_pressure(self, farm):

        farm.cfg.ext_pressure = -100
        farm.cages = [0] * 10

        with pytest.raises(Exception):
            farm.get_cage_pressures()

    def test_get_cage_pressures_zero_pressure(self, farm):
        farm.cfg.ext_pressure = 0
        farm.cages = [0] * 10

        pressures = farm.get_cage_pressures()

        assert len(pressures) == len(farm.cages)
        assert sum(pressures) == farm.cfg.ext_pressure

    def test_get_cage_pressures_no_cages(self, farm):

        farm.cfg.ext_pressure = 100
        farm.cages = []

        with pytest.raises(Exception):
            farm.get_cage_pressures()

    @pytest.mark.parametrize(
        "eggs_by_hatch_date,nbins", [
            (
                {to_dt("2017-02-01 00:00:00"): {
                                                ('A',): 100,
                                                ('a',): 200,
                                                ('A', 'a'): 300,
                                               },
                 to_dt("2017-02-10 00:00:00"): {
                                                ('A',): 100,
                                                ('a',): 200,
                                                ('A', 'a'): 300,
                                               }},
                10
            ),
            (
                {},
                10
            ),
        ])
    def test_get_egg_allocation(self, farm, eggs_by_hatch_date, nbins):

        allocation = farm.get_egg_allocation(nbins, eggs_by_hatch_date)

        allocation_list = [n for bin_dict in allocation for hatch_dict in bin_dict.values() for n in hatch_dict.values()]
        sum_eggs_by_hatch_date = sum([n for hatch_dict in eggs_by_hatch_date.values() for n in hatch_dict.values()])

        assert sum(allocation_list) == sum_eggs_by_hatch_date
        assert len(allocation) == 10

        for n in allocation_list:
            assert n >= 0

        allocation_keys_list = [list(bin_dict.keys()) for bin_dict in allocation]
        hatch_keys = list(eggs_by_hatch_date.keys())
        for allocation_keys in allocation_keys_list:
            assert allocation_keys == hatch_keys

    @pytest.mark.parametrize("nbins", [(0), (-10)])
    def test_get_egg_allocation_nonpositive_bins(self, farm, nbins):
        with pytest.raises(Exception):
            farm.get_egg_allocation(nbins, {})

    def test_disperse_offspring(self, farm, farm_two):
        farms = [farm, farm_two]
        eggs_by_hatch_date = {to_dt("2017-01-05 00:00:00"): {
                                                ('A',): 100,
                                                ('a',): 100,
                                                ('A', 'a'): 100,
                                               }}
        cur_date = to_dt("2017-01-01 00:00:00")

        new_rng = np.random.default_rng(seed=2021)
        farms[0].cfg.rng = new_rng
        farms[0].cages = farms[0].cages[:2]
        farms[1].cages = farms[1].cages[:2]

        farms[0].disperse_offspring(eggs_by_hatch_date, farms, cur_date)

        for farm in farms:
            for cage in farm.cages:
                assert cage.arrival_events.qsize() == 1

    def test_get_cage_arrivals_stats(self, farm, cur_day):

        next_day = cur_day + dt.timedelta(1)

        cage_1 = {
            cur_day: {
                        ('A',): 10,
                        ('a',): 10,
                        ('A', 'a'): 10,
                     },
            next_day: {
                        ('A',): 10,
                        ('a',): 10,
                        ('A', 'a'): 20,
                     }}

        cage_2 = {
            cur_day: {
                        ('A',): 5,
                        ('a',): 5,
                        ('A', 'a'): 5,
                     },
            next_day: {
                        ('A',): 5,
                        ('a',): 5,
                        ('A', 'a'): 10,
                     }}

        arrivals = [cage_1, cage_2]

        total, by_cage = farm.get_cage_arrivals_stats(arrivals)

        assert total == 105
        assert by_cage == [70, 35]

    def test_farm_update_before_start(self, farm):
        cur_date = farm.start_date - dt.timedelta(1)
        offspring = farm.update(cur_date, 1)

        assert offspring == {}
