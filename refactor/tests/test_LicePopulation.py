import datetime as dt

import numpy as np

from src.QueueTypes import DamAvailabilityBatch
from src.LicePopulation import largest_remainder

class TestLicePopulation:
    def test_largest_remainder(self):
        x = np.array([1, 2, 3])
        assert np.all(largest_remainder(x) == [1, 2, 3])

        x = np.array([0.5, 0.5, 1])
        assert np.all(largest_remainder(x) == [0.0, 1.0, 1.0])

        x = np.array([200.5, 200.5, 1])
        assert np.all(largest_remainder(x) == [200.0, 201.0, 1.0])

        x = np.array([2.45, 2.45, 4.10])
        assert np.all(largest_remainder(x) == [2.0, 3.0, 4.0])

        x = np.array([0.91428571, 1.02857143, 2.05714286])
        assert np.all(largest_remainder(x) == [1., 1., 2.])


    def test_avail_dams_freed_early(self, first_cage, first_cage_population, cur_day):
        dams, _ = first_cage.do_mating_events()

        first_cage_population.add_busy_dams_batch(DamAvailabilityBatch(cur_day + dt.timedelta(days=1), dams))
        assert all(x == 0 for x in first_cage_population.free_dams(cur_day).values())

    def test_avail_dams_freed_same_day_once(self, first_cage, first_cage_population, cur_day):
        first_cage_population["L5m"] = 1000
        first_cage_population["L5f"] = 1000
        dams, _ = first_cage.do_mating_events()
        target_dams = {('A',): 188,
                       ('a',): 269,
                       ('A', 'a'): 469}

        first_cage_population.add_busy_dams_batch(DamAvailabilityBatch(cur_day + dt.timedelta(days=1), dams))
        assert first_cage_population.free_dams(cur_day + dt.timedelta(days=1)) == target_dams

    def test_avail_dams_freed_same_day_thrice(self, first_cage, first_cage_population, cur_day):
        first_cage_population["L5m"] = 1000
        first_cage_population["L5f"] = 1000
        dams, _ = first_cage.do_mating_events()
        # After 3 days all the dams must have been freed.
        target_dams = first_cage_population.available_dams.copy()

        for i in range(3):
            first_cage_population.add_busy_dams_batch(DamAvailabilityBatch(cur_day + dt.timedelta(days=i), dams))
        assert first_cage_population.free_dams(cur_day + dt.timedelta(days=3)) == target_dams
