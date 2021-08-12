import datetime as dt
import json
from src.TreatmentTypes import Money


class TestOrganisation:
    def test_organisation_loads(self, organisation):
        assert organisation.name == "Loch Fyne ACME Ltd."
        assert len(organisation.farms) == 2
        assert organisation.capital == Money("180000.00")

    def test_json(self, organisation):
        json_str = organisation.to_json()
        assert isinstance(json.loads(json_str), dict)

    def test_step(self, organisation):
        day = organisation.cfg.start_date
        limit = 10
        for i in range(limit):
            organisation.step(day)
            day += dt.timedelta(days=i)

