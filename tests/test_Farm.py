import datetime as dt
import json

import numpy as np
import pytest

from slim.simulation.lice_population import LicePopulation, from_dict, geno_to_alleles
from slim.simulation.cage import Cage
from slim.simulation.config import to_dt
from slim.types.treatments import Treatment

# from slim.types.queue import SampleCommand


class TestFarm:
    def test_farm_loads_params(self, first_farm):
        assert first_farm.id_ == 0
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
        "eggs_by_hatch_date",
        [
            {
                to_dt("2017-02-01 00:00:00"): from_dict(
                    {
                        "A": 100,
                        "a": 200,
                        "Aa": 300,
                    }
                ),
                to_dt("2017-02-10 00:00:00"): from_dict(
                    {
                        "A": 100,
                        "a": 200,
                        "Aa": 300,
                    }
                ),
            },
            {},
        ],
    )
    def test_get_cage_allocation(self, first_farm, eggs_by_hatch_date):

        allocation = first_farm.get_cage_allocation(6, eggs_by_hatch_date)

        # Originally, this would extract the allocation for each gene
        allocation_list = [
            np.sum(n)
            for bin_dict in allocation
            for hatch_dict in bin_dict.values()
            for n in hatch_dict.values()
        ]
        if eggs_by_hatch_date == {}:
            sum_eggs_by_hatch_date = 0
        else:
            sum_eggs_by_hatch_date = sum(
                sum(
                    [
                        n
                        for hatch_dict in eggs_by_hatch_date.values()
                        for n in hatch_dict.values()
                    ]
                )
            )

        assert sum(allocation_list) == sum_eggs_by_hatch_date
        assert len(allocation) == len(first_farm.cages)

        for n in allocation_list:
            assert n >= 0

        allocation_keys_list = [list(bin_dict.keys()) for bin_dict in allocation]
        hatch_keys = list(eggs_by_hatch_date.keys())
        for allocation_keys in allocation_keys_list:
            assert allocation_keys == hatch_keys

    def test_get_farm_allocation(
        self, first_farm, second_farm, sample_offspring_distrib
    ):
        first_farm.cfg.interfarm_probs[first_farm.id_][second_farm.id_] = 0.1

        total_eggs_by_date = {first_farm.start_date: sample_offspring_distrib}
        farm_eggs_by_date = first_farm.get_farm_allocation(
            second_farm.id_, total_eggs_by_date
        )

        assert len(farm_eggs_by_date) == len(total_eggs_by_date)
        assert (
            farm_eggs_by_date[first_farm.start_date].keys()
            == total_eggs_by_date[first_farm.start_date].keys()
        )
        for geno in geno_to_alleles("a"):
            assert (
                farm_eggs_by_date[first_farm.start_date][geno]
                <= total_eggs_by_date[first_farm.start_date][geno]
            )

    def test_get_farm_allocation_empty(self, first_farm, second_farm):
        farm_eggs_by_date = first_farm.get_farm_allocation(second_farm, {})
        assert farm_eggs_by_date == {}

    def test_disperse_offspring(self, first_farm, second_farm):
        farms = [first_farm, second_farm]
        eggs_by_hatch_date = {
            to_dt("2017-01-05 00:00:00"): from_dict(
                {
                    "A": 100,
                    "a": 100,
                    "Aa": 100,
                }
            )
        }
        cur_date = to_dt("2017-01-01 00:00:00")

        new_rng = np.random.default_rng(seed=2021)
        farms[0].cfg.rng = new_rng
        farms[0].cages = farms[0].cages[:2]
        farms[1].cages = farms[1].cages[:2]

        arrivals_farm0 = farms[0].disperse_offspring(eggs_by_hatch_date, cur_date)
        arrivals_farm1 = farms[1].disperse_offspring(eggs_by_hatch_date, cur_date)
        farms[0].update_arrivals(arrivals_farm0[0])
        farms[0].update_arrivals(arrivals_farm1[0])

        # Only for the calling cage
        for cage in first_farm.cages:
            assert cage.arrival_events.qsize() >= 1

    def test_get_cage_arrivals_stats(self, first_farm, cur_day):

        next_day = cur_day + dt.timedelta(1)

        cage_1 = {
            cur_day: from_dict(
                {
                    "A": 10,
                    "a": 10,
                    "Aa": 10,
                }
            ),
            next_day: from_dict(
                {
                    "A": 10,
                    "a": 10,
                    "Aa": 20,
                }
            ),
        }

        cage_2 = {
            cur_day: from_dict(
                {
                    "A": 5,
                    "a": 5,
                    "Aa": 5,
                }
            ),
            next_day: from_dict(
                {
                    "A": 5,
                    "a": 5,
                    "Aa": 10,
                }
            ),
        }

        arrivals = [cage_1, cage_2]

        total, by_cage, _ = first_farm.get_cage_arrivals_stats(arrivals)

        assert total == 105
        assert by_cage == [70, 35]

    def test_farm_update_before_start(
        self, first_farm, initial_external_inflow, initial_external_ratios
    ):
        cur_date = first_farm.start_date - dt.timedelta(1)
        offspring, cost = first_farm.update(
            cur_date, initial_external_inflow, initial_external_ratios
        )

        assert offspring == {}
        assert cost > 0  # fallowing

    # Currently fixtures are not automatically loaded in the parametrisation, so they need to be manually invoked
    # See https://github.com/pytest-dev/pytest/issues/349
    @pytest.mark.parametrize(
        "test_farm, expected_cost", [("first_farm", 0), ("second_farm", 0)]
    )
    def test_update(
        self,
        sample_offspring_distrib,
        initial_external_ratios,
        initial_external_inflow,
        test_farm,
        expected_cost,
        request,
    ):

        # ensure number of matings
        initial_lice_pop = {stage: 100 for stage in LicePopulation.lice_stages}

        test_farm = request.getfixturevalue(test_farm)

        # ensure number of different hatching dates through a number of cages
        test_farm.cfg.farms[test_farm.id_].cages_start = [
            test_farm.start_date for i in range(10)
        ]
        test_farm.cages = [
            Cage(i, test_farm.cfg, test_farm, initial_lice_pop) for i in range(10)
        ]

        eggs_by_hatch_date, cost = test_farm.update(
            test_farm.start_date, initial_external_inflow, initial_external_ratios
        )

        for hatch_date in eggs_by_hatch_date:
            assert hatch_date > test_farm.start_date

            for geno in geno_to_alleles("a"):
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
        cur_day = first_farm.farm_cfg.treatment_dates[-1][0] + treatment_step_size

        for i in range(7):
            assert first_farm.add_treatment(Treatment.EMB, cur_day)
            assert (
                first_cage.treatment_events.qsize() == 3 + i + 1
            )  # the first treatment cannot be applied
            assert first_farm.available_treatments == 7 - i - 1
            cur_day += treatment_step_size

        assert not first_farm.add_treatment(Treatment.EMB, cur_day)
        assert first_farm.available_treatments == 0

    def test_prescheduled_sampling_events(self, first_farm, cur_day):
        first_farm._report_sample(cur_day)
        assert first_farm._get_aggregation_rate() <= 0.1

        # One test today, one test in 14 days, another test in 28 days
        # The first has already been consumed
        first_farm._report_sample(cur_day + dt.timedelta(days=21))
        assert first_farm._get_aggregation_rate() > 0.0

        # The second too
        first_farm._report_sample(cur_day + dt.timedelta(days=28))
        assert first_farm._get_aggregation_rate() > 0.0

    """
    def test_ask_for_treatment_no_defection(
        self, no_prescheduled_farm, cur_day, initial_external_ratios
    ):
        first_cage = no_prescheduled_farm.cages[0]
        assert first_cage.treatment_events.qsize() == 0
        first_available_day = cur_day + dt.timedelta(days=30)
        no_prescheduled_farm.ask_for_treatment(first_available_day, False)
        assert first_cage.treatment_events.qsize() == 1

        # Asking again will not work, but it requires some internal cage update
        first_cage.update(first_available_day, 0, initial_external_ratios)
        no_prescheduled_farm.ask_for_treatment(first_available_day, False)
        assert first_cage.treatment_events.qsize() == 0

    def test_ask_for_treatment(
        self, no_prescheduled_farm, no_prescheduled_cage, cur_day
    ):
        assert no_prescheduled_cage.treatment_events.qsize() == 0

        for i in range(12):
            first_available_day = cur_day + dt.timedelta(days=30 * i)
            no_prescheduled_farm.ask_for_treatment()
        assert no_prescheduled_cage.treatment_events.qsize() == 9

    """
