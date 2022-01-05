import pytest
from slim.TreatmentTypes import TreatmentParams, EMB, Thermolicer, Treatment


class TestTreatmentType:
    def test_treatment_enum(self, farm_config):
        assert isinstance(farm_config.get_treatment(Treatment.EMB), EMB)
        assert isinstance(farm_config.get_treatment(Treatment.THERMOLICER), Thermolicer)

class TestThermolicer:
    def test_thermolicer_efficacy_no_lice(self, farm_config, planctonic_only_population):
        temperature = 10
        thermolicer = farm_config.thermolicer
        geno = thermolicer.get_lice_treatment_mortality_rate(planctonic_only_population, temperature)
        geno_list = list(geno.values())

        assert all([v.num_susc == 0 for v in geno_list])

    def test_thermolicer_efficacy(self, farm_config, first_cage_population):
        temperature = 12
        thermolicer = farm_config.thermolicer
        geno = thermolicer.get_lice_treatment_mortality_rate(first_cage_population, temperature)
        geno_list = list(geno.values())

        assert [v.num_susc for v in geno_list] == [22, 40, 18]
        assert [v.mortality_rate == 0.8 for v in geno_list]

        temperature = 10

        geno = thermolicer.get_lice_treatment_mortality_rate(first_cage_population, temperature)
        geno_list = list(geno.values())

        assert [v.num_susc for v in geno_list] == [22, 40, 18]
        assert all([v.mortality_rate == 0.99 for v in geno_list])

    def test_thermolicer_fish_mortality(self, farm_config):
        efficacy_window = 1
        num_natural_death = 5
        num_lice_death = 30
        num_fish = 1000
        temperature = 12
        fish_mass = 1000

        num_mortality_events = num_lice_death + num_natural_death

        treatment = farm_config.thermolicer
        num_deaths = treatment.get_fish_mortality_occurrences(temperature, fish_mass, num_fish, efficacy_window, num_mortality_events)

        assert 25 <= num_deaths <= 30

        fish_mass = 3000
        num_deaths = treatment.get_fish_mortality_occurrences(temperature, fish_mass, num_fish, efficacy_window, num_mortality_events)

        assert 20 <= num_deaths <= 30

        fish_mass = 1000
        temperature = 10
        num_deaths = treatment.get_fish_mortality_occurrences(temperature, fish_mass, num_fish, efficacy_window, num_mortality_events)
        assert 27 <= num_deaths <= 30