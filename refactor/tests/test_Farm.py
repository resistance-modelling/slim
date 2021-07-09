import numpy as np

class TestFarm:
    def test_farm_loads_params(self, farm):
        assert farm.name == 0
        assert len(farm.cages) == 6
        assert farm.loc_x == 190300
        assert farm.loc_y == 665300

    def test_year_temperatures(self, farm):
        tarbert = farm.cfg.farm_data["tarbert"]["temperatures"]
        ardrishaig = farm.cfg.farm_data["ardrishaig"]["temperatures"]
        temps = np.stack([tarbert, ardrishaig])
        min_temps = np.round(np.min(temps, axis=0), 1)
        max_temps = np.round(np.max(temps, axis=0), 1)
        assert all(min_temps <= farm.year_temperatures)
        assert all(farm.year_temperatures <= max_temps)

    def test_farm_update(self, farm):
        # TODO: test integration across **all** cages. Requires further refactoring.
        pass
