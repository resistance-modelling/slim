import datetime as dt
import json

import numpy as np
import pytest

from src.LicePopulation import LicePopulation
from src.Cage import Cage
from src.Config import to_dt
from src.TreatmentTypes import Treatment
from src.QueueTypes import SampleRequestCommand


class TestFarm:
    def test_farm_loads_params(self, first_farm):
        assert first_farm.name == 0
        assert len(first_farm.cages) == 6
        assert first_farm.loc_x == 190300
        assert first_farm.loc_y == 665300
        # accounts for pre-sceduled treaments, but only when applicable
        assert first_farm.available_treatments == 7

    def test_farm_str(self, first_farm):
        farm_str = str(first_farm)
        assert isinstance(farm_str, str)
        assert len(farm_str) > 0
        assert "id: 0" in farm_str

    def test_farm_repr(self, first_farm):
        farm_repr = repr(first_farm)
        loaded_farm_data = json.loads(farm_repr)
        assert isinstance(loaded_farm_data, dict)

    def test_year_temperatures(self, first_farm):
        tarbert = first_farm.cfg.loch_temperatures[1][1:]
        ardrishaig = first_farm.cfg.loch_temperatures[0][1:]
        temps = np.stack([tarbert, ardrishaig])
        min_temps = np.round(np.min(temps, axis=0), 1)
        max_temps = np.round(np.max(temps, axis=0), 1)
        assert all(min_temps <= first_farm.year_temperatures)
        assert all(first_farm.year_temperatures <= max_temps)

    def test_farm_update(self, first_farm):
        # TODO: test integration across **all** cages. Requires further refactoring.
        pass

    def test_get_cage_pressures(self, first_farm, initial_external_inflow):

        first_farm.cfg.min_ext_pressure = 100
        first_farm.cages = [0] * 10

        pressures = first_farm.get_cage_pressures(100)

        assert len(pressures) == len(first_farm.cages)
        assert sum(pressures) == first_farm.cfg.min_ext_pressure * 10

        for pressure in pressures:
            assert pressure >= 0

    def test_get_cage_pressures_negative_pressure(self, first_farm):

        first_farm.cfg.min_ext_pressure = -100
        first_farm.cages = [0] * 10

        with pytest.raises(Exception):
            first_farm.get_cage_pressures()

    def test_get_cage_pressures_zero_pressure(self, first_farm):
        first_farm.cfg.min_ext_pressure = 0
        first_farm.cages = [0] * 10

        pressures = first_farm.get_cage_pressures(0)

        assert len(pressures) == len(first_farm.cages)
        assert sum(pressures) == first_farm.cfg.min_ext_pressure

    def test_get_cage_pressures_no_cages(self, first_farm):

        first_farm.cfg.min_ext_pressure = 100
        first_farm.cages = []

        with pytest.raises(Exception):
            first_farm.get_cage_pressures()

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
    def test_get_cage_allocation(self, first_farm, eggs_by_hatch_date, nbins):

        allocation = first_farm.get_cage_allocation(nbins, eggs_by_hatch_date)

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

    @pytest.mark.parametrize("nbins", [0, (-10)])
    def test_get_cage_allocation_nonpositive_bins(self, first_farm, nbins):
        with pytest.raises(Exception):
            first_farm.get_cage_allocation(nbins, {})

    def test_get_farm_allocation(self, first_farm, second_farm, sample_offspring_distrib):
        first_farm.cfg.interfarm_probs[first_farm.name][second_farm.name] = 0.1

        total_eggs_by_date = {first_farm.start_date: sample_offspring_distrib}
        farm_eggs_by_date = first_farm.get_farm_allocation(second_farm, total_eggs_by_date)

        assert len(farm_eggs_by_date) == len(total_eggs_by_date)
        assert farm_eggs_by_date[first_farm.start_date].keys() == total_eggs_by_date[first_farm.start_date].keys()
        for geno in total_eggs_by_date[first_farm.start_date]:
            assert farm_eggs_by_date[first_farm.start_date][geno] <= total_eggs_by_date[first_farm.start_date][geno]

    def test_get_farm_allocation_empty(self, first_farm, second_farm):
        farm_eggs_by_date = first_farm.get_farm_allocation(second_farm, {})
        assert farm_eggs_by_date == {}

    def test_disperse_offspring(self, first_farm, second_farm):
        farms = [first_farm, second_farm]
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

    def test_get_cage_arrivals_stats(self, first_farm, cur_day):

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

        total, by_cage, _ = first_farm.get_cage_arrivals_stats(arrivals)

        assert total == 105
        assert by_cage == [70, 35]

    def test_farm_update_before_start(self, first_farm, initial_external_inflow, initial_external_ratios):
        cur_date = first_farm.start_date - dt.timedelta(1)
        offspring, cost = first_farm.update_farm(cur_date, initial_external_inflow, initial_external_ratios)

        assert offspring == {}
        assert cost > 0  # fallowing

    # Currently fixtures are not automatically loaded in the parametrisation, so they need to be manually invoked
    # See https://github.com/pytest-dev/pytest/issues/349
    @pytest.mark.parametrize("test_farm, expected_cost", [('first_farm', 0), ('second_farm', 0)])
    def test_update(
        self,
        sample_offspring_distrib,
        initial_external_ratios,
        initial_external_inflow,
        test_farm,
        expected_cost,
        request
    ):

        # ensure number of matings
        initial_lice_pop = {stage: 100 for stage in LicePopulation.lice_stages}

        test_farm = request.getfixturevalue(test_farm)

        # ensure number of different hatching dates through a number of cages
        test_farm.cfg.farms[test_farm.name].cages_start = [test_farm.start_date for i in range(10)]
        test_farm.cages = [Cage(i, test_farm.cfg, test_farm, initial_lice_pop) for i in range(10)]

        eggs_by_hatch_date, cost = test_farm.update_farm(
            test_farm.start_date, initial_external_inflow, initial_external_ratios)

        for hatch_date in eggs_by_hatch_date:
            assert hatch_date > test_farm.start_date

            for geno in eggs_by_hatch_date[hatch_date]:
                assert eggs_by_hatch_date[hatch_date][geno] > 0

        # TODO: need to check that the second farm has a positive infrastructure cost
        assert cost == expected_cost

    def test_eq(self, first_farm, second_farm):
        assert first_farm == first_farm
        assert first_farm != second_farm
        assert first_farm != 0
        assert first_farm != "dummy"

    def test_treatment_limit(self, first_farm, first_cage):
        treatment_step_size = dt.timedelta(days=50)
        cur_day = first_farm.farm_cfg.treatment_starts[-1] + treatment_step_size

        for i in range(7):
            assert first_farm.add_treatment(Treatment.EMB, cur_day)
            assert first_cage.treatment_events.qsize() == 3 + i + 1  # the first treatment cannot be applied
            assert first_farm.available_treatments == 7 - i - 1
            cur_day += treatment_step_size

        assert not first_farm.add_treatment(Treatment.EMB, cur_day)
        assert first_farm.available_treatments == 0

    def test_prescheduled_sampling_events(self, first_farm, cur_day):
        assert first_farm.farm_to_org.qsize() == 0
        first_farm.report_sample(cur_day)
        assert first_farm.farm_to_org.qsize() == 1
        assert first_farm.farm_to_org.queue[0].detected_rate <= 0.1

        first_farm.farm_to_org.get()

        # One test today, one test in 14 days, another test in 28 days
        # The first has already been consumed
        first_farm.report_sample(cur_day + dt.timedelta(days=21))
        assert first_farm.farm_to_org.qsize() == 1
        first_farm.farm_to_org.get()

        # The second too
        first_farm.report_sample(cur_day + dt.timedelta(days=28))
        assert first_farm.farm_to_org.qsize() == 1

    def test_handle_reporting_event(self, first_farm, cur_day):
        first_farm._Farm__sampling_events.queue = []
        assert first_farm.farm_to_org.qsize() == 0
        first_farm.command_queue.put(SampleRequestCommand(cur_day))
        first_farm.handle_events(cur_day)
        assert first_farm.farm_to_org.qsize() == 1

    def test_ask_for_treatment_no_defection(self, no_prescheduled_farm, cur_day, initial_external_ratios):
        first_cage = no_prescheduled_farm.cages[0]
        assert first_cage.treatment_events.qsize() == 0
        first_available_day = cur_day + dt.timedelta(days=30)
        no_prescheduled_farm.ask_for_treatment(first_available_day, False)
        assert first_cage.treatment_events.qsize() == 1

        # Asking again will not work, but it requires some internal cage update
        first_cage.update_farm(first_available_day, 0, initial_external_ratios)
        no_prescheduled_farm.ask_for_treatment(first_available_day, False)
        assert first_cage.treatment_events.qsize() == 0

    def test_ask_for_treatment(self, no_prescheduled_farm, no_prescheduled_cage, cur_day):
        assert no_prescheduled_cage.treatment_events.qsize() == 0

        for i in range(12):
            first_available_day = cur_day + dt.timedelta(days=30*i)
            no_prescheduled_farm.ask_for_treatment(first_available_day)
        assert no_prescheduled_cage.treatment_events.qsize() == 9
