import datetime as dt
import json

import numpy as np
import pytest
from src.Cage import Cage
from src.Config import to_dt


class TestFarm:
    def test_farm_loads_params(self, farm):
        assert farm.name == 0
        assert len(farm.cages) == 6
        assert farm.loc_x == 190300
        assert farm.loc_y == 665300

    def test_farm_str(self, farm):
        farm_str = str(farm)
        assert isinstance(farm_str, str)
        assert len(farm_str) > 0
        assert "id: 0" in farm_str

    def test_farm_repr(self, farm):
        farm_repr = repr(farm)
        loaded_farm_data = json.loads(farm_repr)
        assert isinstance(loaded_farm_data, dict)

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
    def test_get_cage_allocation(self, farm, eggs_by_hatch_date, nbins):

        allocation = farm.get_cage_allocation(nbins, eggs_by_hatch_date)

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
    def test_get_cage_allocation_nonpositive_bins(self, farm, nbins):
        with pytest.raises(Exception):
            farm.get_cage_allocation(nbins, {})

    def test_get_farm_allocation(self, farm, farm_two, sample_offspring_distrib):
        farm.cfg.interfarm_probs[farm.name][farm_two.name] = 0.1

        total_eggs_by_date = {farm.start_date: sample_offspring_distrib}
        farm_eggs_by_date = farm.get_farm_allocation(farm_two, total_eggs_by_date)

        assert len(farm_eggs_by_date) == len(total_eggs_by_date)
        assert farm_eggs_by_date[farm.start_date].keys() == total_eggs_by_date[farm.start_date].keys()
        for geno in total_eggs_by_date[farm.start_date]:
            assert farm_eggs_by_date[farm.start_date][geno] <= total_eggs_by_date[farm.start_date][geno]

    def test_get_farm_allocation_empty(self, farm, farm_two):
        farm_eggs_by_date = farm.get_farm_allocation(farm_two, {})
        assert farm_eggs_by_date == {}

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
        offspring, cost = farm.update(cur_date)

        assert offspring == {}
        assert cost > 0 # fallowing

    def test_update(self, farm, sample_offspring_distrib):

        # ensure number of matings
        initial_lice_pop = {stage: 100 for stage in Cage.lice_stages}

        # ensure number of different hatching dates through a number of cages
        farm.cfg.farms[farm.name].cages_start = [farm.start_date for i in range(10)]
        farm.cages = [Cage(i, farm.cfg, farm, initial_lice_pop) for i in range(10)]

        eggs_by_hatch_date, cost = farm.update(farm.start_date)

        for hatch_date in eggs_by_hatch_date:
            assert hatch_date > farm.start_date

            for geno in eggs_by_hatch_date[hatch_date]:
                assert eggs_by_hatch_date[hatch_date][geno] > 0

        # TODO: need to check that the second farm has a positive infrastructure cost
        assert cost == 0

    def test_eq(self, farm, farm_two):
        assert farm == farm
        assert farm != farm_two
        assert farm != 0
        assert farm != "dummy"
