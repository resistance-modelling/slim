from src.Reservoir import Reservoir
from src.Config import Config

class TestReservoir:
    def test_dist_sums_to_total(self, farm_config):
        total = 1000
        farm_config.reservoir_num_lice = total
        reservoir = Reservoir(farm_config)

        assert sum(reservoir.lice_population.values()) == total