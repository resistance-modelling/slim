class TestOrganisation:
    def test_organisation_loads(self, organisation):
        assert organisation.name == "Loch Fyne ACME Ltd."
        assert len(organisation.farms) == 2
