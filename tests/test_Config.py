from slim.simulation.config import Config


class TestConfig:
    def test_Fyne(self):
        config = Config("config_data/config.json", "config_data/Fyne")

    def test_Fyne_complete(self):
        config = Config("config_data/config.json", "config_data/Fyne_complete")

    def test_Linnhe_complete(self):
        config = Config("config_data/config.json", "config_data/Linnhe_complete")
