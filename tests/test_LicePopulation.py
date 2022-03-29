import datetime as dt

import numpy as np
import pytest

from slim.simulation.lice_population import (
    largest_remainder,
    GenoDistrib,
    empty_geno_from_cfg,
    LicePopulation,
    from_dict,
)


class TestGenoDistrib:
    def test_largest_remainder(self):
        x = np.array([1, 2, 3], dtype=np.float64)
        assert np.all(largest_remainder(x) == x)

        x = np.array([0, 0, 0], dtype=np.float64)
        assert np.all(largest_remainder(x) == x)

        x = np.array([1.4, 1.6], dtype=np.float64)
        assert np.all(largest_remainder(x) == [1, 2])

        x = np.array([0.5, 0.5, 1], dtype=np.float64)
        res = largest_remainder(x)
        assert np.all(res == [0.0, 1.0, 1.0]) or np.all(res == [1.0, 0.0, 1.0])

        x = np.array([200.5, 200.5, 1], dtype=np.float64)
        assert np.all(largest_remainder(x) == [200.0, 201.0, 1.0])

        x = np.array([2.45, 2.45, 4.10], dtype=np.float64)
        assert np.all(largest_remainder(x) == [2.0, 3.0, 4.0])

        x = np.array([0.91428571, 1.02857143, 2.05714286])
        assert np.all(largest_remainder(x) == [1.0, 1.0, 2.0])

        # this test case is interesting because 1.4 and 2.1 cannot be exactly represented
        x = np.array([1.4, 2.1, 3.5])
        assert np.all(largest_remainder(x) == [1.0, 2.0, 4.0])

        # Similarly for the infamous 0.3
        x = np.array([0.3, 0.3, 0.4])
        assert np.all(largest_remainder(x) == [0, 0, 1])

        x = np.array([0.9, 0.9, 0.2])
        assert np.all(largest_remainder(x) == [1, 1, 0])

        x = np.array([1.8, 2.4, 1.2, 0.6])
        assert np.all(largest_remainder(x) == [2, 2, 1, 1])

        x = np.array([2 / 3, 1 / 3])
        assert np.all(largest_remainder(x) == [1, 0])

        x = np.array([-5, -8, -13], dtype=np.float64)
        assert np.all(largest_remainder(x) == x)

    def test_GenoDistrib_init(self, farm_config):
        x = empty_geno_from_cfg(farm_config)
        assert x.num_genes == 1
        assert np.all(x._default_probs == [0.25, 0.25, 0.5])
        assert x.gross == 0
        assert x == {"A": 0, "a": 0, "Aa": 0}

    def test_GenoDistrib_ops(self, empty_distrib, empty_distrib_v2):
        empty_distrib["a"] = 50.0
        empty_distrib["A"] = 50.0
        empty_distrib["Aa"] = 100.0

        assert empty_distrib == {"a": 50, "A": 50, "Aa": 100}

        distr_v2 = empty_distrib_v2.copy()

        distr_v2["a"] = 50.0
        distr_v2["A"] = 50.0
        distr_v2["Aa"] = 100.0

        assert distr_v2 == {
            "a": 50,
            "A": 50,
            "Aa": 100,
            "b": 160,
            "B": 20,
            "Bb": 20,
        }

        assert distr_v2 != empty_distrib_v2
        assert empty_distrib_v2.gross == 0

        normalised_1000 = distr_v2.normalise_to(1000)

        assert normalised_1000 == {
            "a": 250,
            "A": 250,
            "Aa": 500,
            "b": 800,
            "B": 100,
            "Bb": 100,
        }

        normalised_1200 = distr_v2 + normalised_1000
        assert normalised_1200.gross == 1200
        assert normalised_1200 == {
            "a": 300,
            "A": 300,
            "Aa": 600,
            "b": 960,
            "B": 120,
            "Bb": 120,
        }

        normalised_800 = normalised_1000 - distr_v2
        assert normalised_800.gross == 800
        assert normalised_800 == {
            "a": 200,
            "A": 200,
            "Aa": 400,
            "b": 640,
            "B": 80,
            "Bb": 80,
        }

        assert distr_v2.normalise_to(0) == empty_distrib_v2

        assert distr_v2 * (1 / 5) == {
            "a": 10,
            "A": 10,
            "Aa": 20,
            "b": 32,
            "B": 4,
            "Bb": 4,
        }

    def test_from_dict(self):
        d = {"A": 1, "Aa": 2, "a": 3}
        assert from_dict(d) == d


class TestLicePopulation:
    def test_init(self, empty_distrib):
        geno_rates = {
            stage: empty_distrib.copy() for stage in LicePopulation.lice_stages
        }
        genetic_ratios = {"a": 0.5, "A": 0.2, "Aa": 0.3}
        lice_pop = LicePopulation(geno_rates, genetic_ratios, 3.0)
        assert lice_pop.geno_by_lifestage == geno_rates
        assert lice_pop.genetic_ratios == genetic_ratios

    def test_getitem(self, first_cage_population, initial_lice_population):
        assert first_cage_population["L1"] == 150
        assert first_cage_population["L4"] == 30
        assert first_cage_population == initial_lice_population

    def test_setitem(self, first_cage_population):
        assert first_cage_population["L5f"] == 10
        assert first_cage_population["L5m"] == 10
        first_cage_population["L5f"] = 100
        first_cage_population["L5m"] = 100
        assert first_cage_population["L5f"] == 100
        assert first_cage_population["L5m"] == 100

        assert first_cage_population.geno_by_lifestage["L5f"].is_positive()
        assert first_cage_population.geno_by_lifestage["L5m"].is_positive()

    def test_no_busy_dams(self, first_cage_population):
        # No busy dams
        assert first_cage_population.busy_dams.gross == 0
        assert (
            first_cage_population.available_dams
            == first_cage_population.geno_by_lifestage["L5f"]
        )

    def test_busy_dams_overflow_should_raise(self, first_cage_population):
        with pytest.raises(AssertionError):
            first_cage_population.add_busy_dams_batch(20)

    def test_busy_dams(self, first_cage_population):
        first_cage_population.add_busy_dams_batch(3)
        assert first_cage_population.available_dams.is_positive()
        assert first_cage_population.available_dams.gross < 10
        first_cage_population.add_busy_dams_batch(3)
        assert first_cage_population.available_dams.gross < 6
        first_cage_population.add_busy_dams_batch(3)
        assert first_cage_population.available_dams.gross < 5
        # Note: there's no concept of "freeing dams": the arrival rate should approach 3
        assert 2.0 < first_cage_population._busy_dam_arrival_rate < 3.0

        first_cage_population.clear_busy_dams()

        for i in range(10):
            to_add = min(5, first_cage_population.available_dams.gross)
            first_cage_population.add_busy_dams_batch(to_add)
            assert first_cage_population.available_dams.is_positive()

        assert 2 <= first_cage_population.available_dams.gross <= 4
