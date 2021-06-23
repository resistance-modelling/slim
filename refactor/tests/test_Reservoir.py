from src.Reservoir import Reservoir
from src.Config import Config


class TestReservoir:

    def test_dist_sums_to_total(self):
        
        # temporary quickfix before proper fixtures and setup
        cfg = Config("config_data/test.json", None)

        total = 1000
        cfg.reservoir_num_lice = total
        reservoir = Reservoir(cfg)

        assert sum(reservoir.lice_population.values()) == total
