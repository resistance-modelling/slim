import datetime as dt

import numpy as np
import pytest

from slim.simulation.organisation import OffspringAveragingQueue
from slim.simulation.lice_population import largest_remainder, GenoDistrib


class TestLicePopulation:
    def test_largest_remainder(self):
        x = np.array([1, 2, 3])
        assert np.all(largest_remainder(x) == x)

        x = np.array([0, 0, 0])
        assert np.all(largest_remainder(x) == x)

        x = np.array([1.4, 1.6])
        assert np.all(largest_remainder(x) == [1, 2])

        x = np.array([0.5, 0.5, 1])
        res = largest_remainder(x)
        assert np.all(res == [0.0, 1.0, 1.0]) or np.all(res == [1.0, 0.0, 1.0])

        x = np.array([200.5, 200.5, 1])
        assert np.all(largest_remainder(x) == [200.0, 201.0, 1.0])

        x = np.array([2.45, 2.45, 4.10])
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

        with pytest.raises(AssertionError):
            x = np.array([1.0, -2.0, 3.0])
            assert np.all(largest_remainder(x) == x)

        x = np.array([-5, -8, -13])
        assert np.all(largest_remainder(x) == x)


class TestOffspring:
    def test_append_no_averaging(self, cur_day, sample_offspring_distrib):
        queue = OffspringAveragingQueue(1)

        offspring = {cur_day: sample_offspring_distrib}

        assert queue.average == GenoDistrib()

        queue.append(offspring_per_farm=[offspring])

        assert len(queue) == 1
        assert queue.average == sample_offspring_distrib

        queue.append(offspring_per_farm=[offspring])

        assert len(offspring) == 1
        assert queue.average == sample_offspring_distrib

    def test_append(self, cur_day, sample_offspring_distrib):
        queue = OffspringAveragingQueue(10)

        offspring = {cur_day: sample_offspring_distrib}
        for i in range(10):
            queue.append(offspring_per_farm=[offspring])
            assert len(queue) == i + 1

        assert queue.average == sample_offspring_distrib
        assert len(queue) == 10

        queue.append(offspring_per_farm=[offspring])

        assert len(queue) == 10
        assert queue.average == sample_offspring_distrib
