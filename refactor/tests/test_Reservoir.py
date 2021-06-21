from src.Reservoir import Reservoir


class TestReservoir:

    def test_dist_sums_to_total(self):
        total = 1000
        reservoir = Reservoir(total)

        assert sum(reservoir.lice_population.values()) == total
