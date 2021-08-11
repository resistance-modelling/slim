from src.TreatmentTypes import Money

class TestOrganisation:
    def test_organisation_loads(self, organisation):
        assert organisation.name == "Loch Fyne ACME Ltd."
        assert len(organisation.farms) == 2
        assert organisation.capital == Money("180000.00")

    def test_json(self):
        pass

    def test_step(self):
        pass

