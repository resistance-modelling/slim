class TestFarm:
    def test_farm_loads_params(self, farm):
        assert farm.name == 0
        assert len(farm.cages) == 6
        assert farm.loc_x == 190300
        assert farm.loc_y == 665300

    def test_farm_update(self, farm):
        # TODO: test integration across **all** cages. Requires further refactoring.
        pass
