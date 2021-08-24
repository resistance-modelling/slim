import datetime as dt

from src.QueueBatches import DamAvailabilityBatch


class TestLicePopulation:
    def test_avail_dams_freed_early(self, first_cage, first_cage_population, cur_day):
        dams, _ = first_cage.do_mating_events()

        first_cage_population.add_busy_dams_batch(DamAvailabilityBatch(cur_day + dt.timedelta(days=1), dams))
        assert all(x == 0 for x in first_cage_population.free_dams(cur_day).values())

    def test_avail_dams_freed_same_day_once(self, first_cage, first_cage_population, cur_day):
        first_cage_population["L5m"] = 1000
        first_cage_population["L5f"] = 1000
        dams, _ = first_cage.do_mating_events()
        target_dams = {('A',): 278,
                       ('a',): 179,
                       ('A', 'a'): 469}

        first_cage_population.add_busy_dams_batch(DamAvailabilityBatch(cur_day + dt.timedelta(days=1), dams))
        assert first_cage_population.free_dams(cur_day + dt.timedelta(days=1)) == target_dams

    def test_avail_dams_freed_same_day_thrice(self, first_cage, first_cage_population, cur_day):
        first_cage_population["L5m"] = 1000
        first_cage_population["L5f"] = 1000
        dams, _ = first_cage.do_mating_events()
        target_dams = {('A',): 834,
                       ('a',): 537,
                       ('A', 'a'): 1407}

        for i in range(3):
            first_cage_population.add_busy_dams_batch(DamAvailabilityBatch(cur_day + dt.timedelta(days=i), dams))
        assert first_cage_population.free_dams(cur_day + dt.timedelta(days=3)) == target_dams
