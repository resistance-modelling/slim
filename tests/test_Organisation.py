import datetime as dt
import json

import numpy as np

from slim.simulation.lice_population import GenoDistrib


class TestOrganisation:
    def test_organisation_loads(self, organisation):
        assert organisation.name == "Loch Fyne ACME Ltd."
        assert len(organisation.farms) == 2

    def test_json(self, organisation):
        json_str = organisation.to_json()
        assert isinstance(json.loads(json_str), dict)

    def test_step(self, organisation):
        day = organisation.cfg.start_date
        limit = 10
        for i in range(limit):
            organisation.step(day)
            day += dt.timedelta(days=1)

    def test_get_external_pressure_day0(self, organisation, farm_config):
        number, ratios = organisation.get_external_pressure()
        assert number == farm_config.min_ext_pressure
        assert ratios == organisation.external_pressure_ratios
        assert organisation.external_pressure_ratios == farm_config.initial_genetic_ratios

    def test_update_external_pressure(self, organisation):
        # we witness an increase of nonresistant offspring
        new_offspring = GenoDistrib({('A',): 10, ('a',): 50, ('A','a'): 10})
        old_ratios = organisation.external_pressure_ratios.copy()
        organisation.update_genetic_ratios(new_offspring)
        new_ratios = organisation.external_pressure_ratios.copy()
        assert old_ratios != new_ratios
        assert np.isclose(sum(new_ratios.values()), 1.0)
        assert max(new_ratios.values()) == new_ratios[('a',)]

    def test_update_external_pressure_constant(self, organisation):
        new_offspring = GenoDistrib({('A',): 10, ('a',): 10, ('A', 'a'): 10})
        for i in range(30):
            organisation.update_genetic_ratios(new_offspring)
        new_ratios = organisation.external_pressure_ratios

        # roughly equal
        assert np.isclose(sum(new_ratios.values()), 1.0)
        assert all(new_ratio >= 0.29 for new_ratio in new_ratios.values())

        for i in range(300):
            organisation.update_genetic_ratios(new_offspring)
        new_ratios = organisation.external_pressure_ratios

        # roughly equal
        assert np.isclose(sum(new_ratios.values()), 1.0)
        assert all(new_ratio >= 0.3 for new_ratio in new_ratios.values())

    def test_update_external_pressure_periodic(self, cur_day, organisation):
        phases = np.array([0, np.pi/3, 2*np.pi/3])
        for i in range(900):
            deg = i / 30 * np.pi
            values = np.rint(100*(1.0+np.sin(deg+phases))).tolist()
            offspring = GenoDistrib(dict(zip(GenoDistrib.alleles, values)))
            organisation.offspring_queue.append([{cur_day + dt.timedelta(days=i): offspring}])
            organisation.update_genetic_ratios(organisation.offspring_queue.average)

    def test_intervene_when_needed(self, no_prescheduled_organisation, no_prescheduled_farm, no_prescheduled_cage, cur_day):
        # Cannot apply treatment as threshold is not met
        no_prescheduled_farm._report_sample(cur_day)
        no_prescheduled_organisation.handle_farm_messages(cur_day, no_prescheduled_farm)
        no_prescheduled_farm._handle_events(cur_day) # makes a reporting event
        assert no_prescheduled_cage.treatment_events.qsize() == 0

        no_prescheduled_cage.lice_population["L5f"] = 5000
        future_day = cur_day + dt.timedelta(days=14)
        no_prescheduled_farm._report_sample(future_day)
        no_prescheduled_organisation.handle_farm_messages(future_day, no_prescheduled_farm)
        no_prescheduled_farm._handle_events(future_day) # makes a reporting event
        # Meets the threshold
        assert no_prescheduled_cage.treatment_events.qsize() == 1
