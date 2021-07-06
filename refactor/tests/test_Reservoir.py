from src.Reservoir import Reservoir


class TestReservoir:
    def test_dist_sums_to_total(self, farm_config):
        total = 1000
        farm_config.reservoir_num_lice = total
        reservoir = Reservoir(farm_config)

        assert sum(reservoir.lice_population.values()) == total

    def test_to_csv(self, reservoir):
        reservoir.lice_population = {"L1": 1,
                                     "L2": 2,
                                     "L3": 3,
                                     "L4": 4,
                                     "L5f": 5,
                                     "L5m": 6}
        reservoir.num_fish = 7

        csv_str = reservoir.to_csv()
        csv_list = csv_str.split(", ")
        print(csv_list)

        assert csv_list[0] == "reservoir"
        assert csv_list[1] == "7"
        for i in range(2, 7):
            assert csv_list[i] == str(i - 1)
        assert csv_list[8] == "21"
