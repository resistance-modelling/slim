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
            day += dt.timedelta(days=1)

    def test_intervene_when_needed(self, no_prescheduled_organisation, no_prescheduled_farm, no_prescheduled_cage, cur_day):
        # Cannot apply treatment as threshold is not met
        no_prescheduled_farm.report_sample(cur_day)
        no_prescheduled_organisation.handle_farm_messages(cur_day, no_prescheduled_farm)
        no_prescheduled_farm.handle_events(cur_day) # makes a reporting event
        assert no_prescheduled_cage.treatment_events.qsize() == 0

        no_prescheduled_cage.lice_population["L5f"] = 5000
        future_day = cur_day + dt.timedelta(days=14)
        no_prescheduled_farm.report_sample(future_day)
        no_prescheduled_organisation.handle_farm_messages(future_day, no_prescheduled_farm)
        no_prescheduled_farm.handle_events(future_day) # makes a reporting event
        # Meets the threshold
        assert no_prescheduled_cage.treatment_events.qsize() == 1
