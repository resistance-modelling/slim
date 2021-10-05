import copy
import datetime
import datetime as dt
import itertools
import json
from queue import PriorityQueue

import numpy as np
import pytest

from src.Cage import Cage
from src.Config import to_dt
from src.QueueTypes import DamAvailabilityBatch, EggBatch, TravellingEggBatch, TreatmentEvent
from src.TreatmentTypes import GeneticMechanism, Treatment, Money
from src.LicePopulation import GenoDistrib


class TestCage:
    def test_cage_loads_params(self, first_cage):
        assert first_cage.id == 0
        assert first_cage.start_date == to_dt("2017-10-01 00:00:00")
        assert first_cage.num_fish == 4000
        assert first_cage.num_infected_fish == first_cage.get_mean_infected_fish()
        assert first_cage.lice_population == {
            "L1": 150,
            "L2": 0,
            "L3": 30,
            "L4": 30,
            "L5f": 10,
            "L5m": 10
        }

        assert sum(first_cage.lice_population.available_dams.values()) == 10

        for stage in first_cage.lice_stages:
            assert first_cage.lice_population[stage] == sum(first_cage.lice_population.geno_by_lifestage[stage].values())

    def test_cage_json(self, first_cage):
        return_str = str(first_cage)
        imported_cage = json.loads(return_str)
        assert isinstance(imported_cage, dict)

    def test_cage_lice_background_mortality_one_day(self, first_cage):
        # NOTE: this currently relies on Stien's approach.
        # Changing this approach will break things
        dead_lice_dist = first_cage.get_background_lice_mortality()
        dead_lice_dist_np = np.array(list(dead_lice_dist.values()))
        expected_dead_lice = np.array([28, 0, 0, 0, 0, 2])
        assert np.alltrue(dead_lice_dist_np >= 0.0)
        assert np.alltrue(np.isclose(dead_lice_dist_np, expected_dead_lice))

    def test_cage_update_lice_treatment_mortality_no_effect(self, first_farm, first_cage):
        treatment_dates = first_farm.farm_cfg.treatment_starts
        assert(treatment_dates == sorted(treatment_dates))

        # before a 14-day activation period there should be no effect
        for i in range(-14, first_cage.cfg.emb.effect_delay):
            cur_day = treatment_dates[0] + dt.timedelta(days=i)
            mortality_updates, cost = first_cage.get_lice_treatment_mortality(cur_day)
            assert all(geno_rate == 0.0 for rate in mortality_updates.values() for geno_rate in rate.values())
            assert first_cage.last_effective_treatment is None
            assert cost == Money("0.00")

        # Even 5 days after, no effect can occur if the cage has not started yet.
        cur_day = treatment_dates[0] + dt.timedelta(days=5)
        mortality_updates, cost = first_cage.get_lice_treatment_mortality(cur_day)
        assert all(geno_rate == 0.0 for rate in mortality_updates.values() for geno_rate in rate.values())
        assert first_cage.last_effective_treatment is None
        assert cost == Money("0.00")

    def test_cage_update_lice_treatment_mortality(self, first_farm, first_cage):
        treatment_dates = first_farm.farm_cfg.treatment_starts

        # before the first useful day, make sure the cage is aware of being under treatment
        cur_day = treatment_dates[1] + dt.timedelta(days=1)
        first_cage.get_lice_treatment_mortality(cur_day)
        assert first_cage.is_treated(cur_day)

        # first useful day
        cur_day = treatment_dates[1] + dt.timedelta(days=5)
        mortality_updates, cost = first_cage.get_lice_treatment_mortality(cur_day)

        for stage in Cage.lice_stages:
            if stage not in Cage.susceptible_stages:
                assert sum(mortality_updates[stage].values()) == 0

        assert first_cage.last_effective_treatment.affecting_date == cur_day
        assert mortality_updates['L5f'] == {('a',): 3}
        assert mortality_updates['L5m'] == {('a',): 3}
        assert mortality_updates['L4'] == {('A', 'a'): 1, ('a',): 8}
        assert mortality_updates['L3'] == {('A',): 2, ('A', 'a'): 2, ('a',): 8}
        assert first_cage.is_treated(cur_day)

        assert 55000 <= cost <= 60000

    def test_cage_update_lice_treatment_mortality_close_days(self, first_farm, first_cage):
        treatment_dates = first_farm.farm_cfg.treatment_starts
        treatment_event = first_farm.generate_treatment_event(Treatment.emb, treatment_dates[1])

        first_cage.treatment_events = PriorityQueue()
        first_cage.treatment_events.put(treatment_event)

        # first useful day
        cur_day = treatment_dates[1] + dt.timedelta(days=5)
        first_cage.get_lice_treatment_mortality(cur_day)

        # after 10 days there should be noticeable effects
        cur_day = treatment_dates[1] + dt.timedelta(days=10)
        mortality_updates, cost = first_cage.get_lice_treatment_mortality(cur_day)

        assert mortality_updates['L3'] == {('A', 'a'): 2, ('a',): 8}
        assert mortality_updates['L4'] == {('A',): 1, ('A', 'a'): 1, ('a',): 8}
        assert mortality_updates['L5m'] == {('a',): 2}
        assert mortality_updates['L5f'] == {('a',): 2}

        assert cost == 0

        # After a long time, the previous treatment has no longer effect
        cur_day = treatment_dates[1] + dt.timedelta(days=40)
        mortality_updates, cost = first_cage.get_lice_treatment_mortality(cur_day)

        assert all(geno_rate == 0.0 for rate in mortality_updates.values() for geno_rate in rate.values())
        assert cost == 0

    def test_get_stage_ages_respects_constraints(self, first_cage):
        test_num_lice = 1000
        development_days = 15
        min_development_stage = 4
        mean_development_stage = 8

        for _ in range(100):
            stage_ages = first_cage.get_evolution_ages(
                test_num_lice,
                minimum_age=min_development_stage,
                mean=mean_development_stage,
                development_days=development_days
            )

            assert len(stage_ages) == test_num_lice

            assert np.min(stage_ages) >= min_development_stage
            assert np.max(stage_ages) < development_days

            # This test luckily doesn't fail
            assert abs(np.mean(stage_ages) - mean_development_stage) < 1

    def test_get_stage_ages_edge_cases(self, first_cage):
        test_num_lice = 1000
        development_days = 15
        min_development_stages = list(range(10))
        mean_development_stages = list(range(10))

        for minimum, mean in itertools.product(min_development_stages, mean_development_stages):
            if mean <= minimum or minimum == 0:
                with pytest.raises(AssertionError):
                    first_cage.get_evolution_ages(
                        test_num_lice,
                        minimum_age=minimum,
                        mean=mean,
                        development_days=development_days
                    )
            else:
                first_cage.get_evolution_ages(
                    test_num_lice,
                    minimum_age=minimum,
                    mean=mean,
                    development_days=development_days
                )

    def test_get_lice_lifestage(self, first_cage):
        new_l2, new_l4, new_females, new_males = first_cage.get_lice_lifestage(1)

        assert 0 <= new_l2 <= 10
        assert 0 <= new_l4 <= 10
        assert new_females == 1
        assert new_males == 1

    def test_get_lice_lifestage_planctonic_only(self, first_cage, planctonic_only_population):
        first_cage.lice_population = planctonic_only_population

        _, new_l4, new_females, new_males = first_cage.get_lice_lifestage(1)

        assert new_l4 == 0
        assert new_females == 0
        assert new_males == 0

    def test_get_fish_death_by_treatment(self, first_cage, cur_day):
        first_affecting_date = first_cage.treatment_events.queue[0].affecting_date
        days_eloped = (first_affecting_date - cur_day).days

        assert first_cage.get_fish_treatment_mortality(days_eloped, 0, 0) == 0
        assert first_cage.get_fish_treatment_mortality(days_eloped, 100, 30) == 0

        first_cage.get_lice_treatment_mortality(cur_day + dt.timedelta(days_eloped))
        assert first_cage.last_effective_treatment is not None

        num_deaths = first_cage.get_fish_treatment_mortality(days_eloped, 100, 30)

        assert 10 <= num_deaths <= 20

        # A change in lice infestation should not cause much of a concern
        # TODO: the paper says otherwise. Investigate whether this matters in our model
        num_deaths = first_cage.get_fish_treatment_mortality(days_eloped, 500, 30)
        assert 10 <= num_deaths <= 20

        # exactly one year after, the temperature is the same (~13 degrees) but the mass increased to 4.5kg.
        days_eloped = 365

        assert first_cage.get_fish_treatment_mortality(days_eloped, 0, 0) == 0

        num_deaths = first_cage.get_fish_treatment_mortality(days_eloped, 100, 30)

        # We expect a surge in mortality now
        assert 30 <= num_deaths <= 50

        num_deaths = first_cage.get_fish_treatment_mortality(days_eloped, 500, 30)
        assert 30 <= num_deaths <= 50

    def test_get_fish_growth(self, first_cage):
        first_cage.num_fish *= 300
        first_cage.num_infected_fish = first_cage.num_fish // 2
        natural_death, lice_death = first_cage.get_fish_growth(1)

        # basic invariants
        assert natural_death >= 0
        assert lice_death >= 0

        # exact figures
        assert 600 <= natural_death <= 700
        assert 0 <= lice_death <= 5

    def test_get_fish_growth_no_lice(self, first_cage):
        first_cage.num_infected_fish = 0
        for k in first_cage.lice_population:
            first_cage.lice_population[k] = 0

        _, lice_death = first_cage.get_fish_growth(1)

        assert lice_death == 0

    def test_get_infection_rates(self, first_cage):
        first_cage.lice_population["L2"] = 100
        rate, avail_lice = first_cage.get_infection_rates(1)
        assert rate > 0
        assert avail_lice > 0

        assert 0.13 <= rate <= 0.17
        assert avail_lice == 90

    def test_get_infected_fish_no_infection(self, first_cage):
        # TODO: maybe make a fixture of this?
        first_cage.lice_population["L3"] = 0
        first_cage.lice_population["L4"] = 0
        first_cage.lice_population["L5m"] = 0
        first_cage.lice_population["L5f"] = 0

        for stage in first_cage.lice_stages:
            assert first_cage.lice_population[stage] == sum(first_cage.lice_population.geno_by_lifestage[stage].values())

        assert first_cage.get_mean_infected_fish() == 0
        assert first_cage.get_variance_infected_fish(first_cage.num_fish, 0) == 0

    def test_get_infected_fish(self, first_cage):
        assert 70 <= first_cage.get_mean_infected_fish() <= 80

    def test_get_std_infected_fish(self, first_cage):
        infecting = first_cage.get_infecting_population()
        assert first_cage.get_variance_infected_fish(first_cage.num_fish, infecting) > 0

    def test_get_num_matings_no_infection(self, first_cage):
        males = first_cage.lice_population["L5m"]
        first_cage.lice_population["L5m"] = 0
        assert first_cage.get_num_matings() == 0
        for stage in first_cage.lice_stages:
            assert first_cage.lice_population[stage] == sum(first_cage.lice_population.geno_by_lifestage[stage].values())
        first_cage.lice_population["L5m"] = males

        first_cage.lice_population["L5f"] = 0
        assert first_cage.get_num_matings() == 0
        for stage in first_cage.lice_stages:
            assert first_cage.lice_population[stage] == sum(first_cage.lice_population.geno_by_lifestage[stage].values())

    def test_get_num_matings(self, first_cage):
        # only 20 lice out of 4000 aren't that great
        assert first_cage.get_num_matings() == 0

        first_cage.lice_population["L5m"] = 100
        first_cage.lice_population["L5f"] = 100
        assert 70 <= first_cage.get_num_matings() <= 100

        first_cage.lice_population["L5m"] = 1000
        first_cage.lice_population["L5f"] = 1000
        assert 900 <= first_cage.get_num_matings() <= 1000

    def test_update_deltas_no_negative_raise(
        self,
        first_cage,
        first_cage_population,
        null_offspring_distrib,
        null_dams_batch,
        null_treatment_mortality
    ):
        first_cage_population["L3"] = 0
        first_cage_population["L4"] = 0
        first_cage_population["L5m"] = 0
        first_cage_population["L5f"] = 0

        background_mortality = first_cage.get_background_lice_mortality()
        fish_deaths_natural = 0
        fish_deaths_from_lice = 0
        fish_deaths_from_treatment = 0
        new_l2 = 0
        new_l4 = 0
        new_females = 0
        new_males = 0
        new_infections = 0

        reservoir_lice = {"L1": 0, "L2": 0}

        null_hatched_arrivals = null_offspring_distrib

        first_cage.update_deltas(background_mortality, null_treatment_mortality, fish_deaths_natural,
                                 fish_deaths_from_lice, fish_deaths_from_treatment, new_l2, new_l4, new_females,
                                 new_males, new_infections, reservoir_lice, null_dams_batch, null_offspring_distrib,
                                 null_hatched_arrivals)

        for population in first_cage.lice_population.values():
            assert population >= 0

    def test_do_mating_events(self, first_cage, first_cage_population, cur_day):
        # Remove mutation effects...
        old_mutation_rate = first_cage.cfg.geno_mutation_rate
        first_cage.cfg.geno_mutation_rate = 0

        first_cage.lice_population.geno_by_lifestage["L5f"] = GenoDistrib({('A',): 15, ('a',): 15, ('A', 'a'): 15})
        first_cage_population.clear_busy_dams()
        first_cage_population.add_busy_dams_batch(DamAvailabilityBatch(cur_day, GenoDistrib({('A', 'a'): 15})))

        target_eggs = {('a',): 1545.5, ('A',): 763.0, tuple(sorted(('a', 'A'))): 6842.5}
        target_delta_dams = {('A',): 2, ('a',): 4}

        delta_avail_dams, delta_eggs = first_cage.do_mating_events()
        assert delta_eggs == target_eggs
        assert delta_avail_dams == target_delta_dams

        # Reconsider mutation effects...
        first_cage.cfg.geno_mutation_rate = old_mutation_rate

        target_mutated_eggs = {('a',): 1574.5, tuple(sorted(('a', 'A'))): 1537.5, ('A',): 1015.0}

        _, delta_mutated_eggs = first_cage.do_mating_events()
        assert delta_mutated_eggs == target_mutated_eggs

    def test_no_available_sires_do_mating_events(self, first_cage):
        first_cage.lice_population.geno_by_lifestage["L5m"] = GenoDistrib({('A',): 0, ('a',): 0, ('A', 'a'): 0})

        delta_avail_dams, delta_eggs = first_cage.do_mating_events()
        assert not bool(delta_avail_dams)
        assert not bool(delta_eggs)

    def test_generate_eggs_maternal(self, first_cage):
        first_cage.genetic_mechanism = GeneticMechanism.maternal
        sire = 'z'
        dam = 'Z'
        num_matings = 10
        target_eggs = {dam: 2550}
        eggs = first_cage.generate_eggs(sire, dam, num_matings)
        for key in eggs:
            assert key == dam
            assert eggs[key] == target_eggs[key]

    def test_generate_eggs_quantitative(self, first_cage):
        first_cage.cfg.geno_mutation_rate = 0
        first_cage.genetic_mechanism = GeneticMechanism.quantitative
        sire = 0.7
        dam = 0.0
        num_matings = 10
        target_eggs = {0.4: 2550}
        eggs = first_cage.generate_eggs(sire, dam, num_matings)
        for key in eggs:
            assert eggs[key] == target_eggs[key]

        sire = 0.2
        dam = 0.2
        num_matings = 10
        target_eggs = {0.2: 2401}
        eggs = first_cage.generate_eggs(sire, dam, num_matings)
        for key in eggs:
            assert eggs[key] == target_eggs[key]

    def test_generate_eggs_discrete(self, first_cage):
        first_cage.cfg.geno_mutation_rate = 0
        first_cage.genetic_mechanism = GeneticMechanism.discrete

        sire = ('A',)
        dam = ('A',)
        hom_dom_target = {('A',): 1533}

        num_matings = 6
        egg_result_hom_dom = first_cage.generate_eggs(sire, dam, num_matings)
        assert egg_result_hom_dom == hom_dom_target

        sire = ('a',)
        dam = ('a',)
        hom_rec_target = {('a',): 1418}
        egg_result_hom_rec = first_cage.generate_eggs(sire, dam, num_matings)
        assert egg_result_hom_rec == hom_rec_target

        sire = tuple(sorted(('a', 'A')))
        dam = ('a',)
        het_target_sire = {('a',): 778.5, tuple(sorted(('a', 'A'))): 778.5}
        egg_result_het_sire = first_cage.generate_eggs(sire, dam, num_matings)
        assert egg_result_het_sire == het_target_sire

        dam = tuple(sorted(('a', 'A')))
        sire = ('a',)
        het_target_dam = {('a',): 765.0, tuple(sorted(('a', 'A'))): 765.0}
        egg_result_het_dam = first_cage.generate_eggs(sire, dam, num_matings)
        assert egg_result_het_dam == het_target_dam

        dam = tuple(sorted(('a', 'A')))
        sire = tuple(sorted(('a', 'A')))
        het_target = {('A', 'a'): 761.5, ('A',): 380.75, ('a',): 380.75}
        egg_result_het = first_cage.generate_eggs(sire, dam, num_matings)
        assert egg_result_het == het_target

    def test_generate_eggs_bad_mechanism(self, first_cage):
        sire = ('A',)
        dam = ('A',)
        num_matings = 6
        first_cage.genetic_mechanism = "dummy"

        with pytest.raises(Exception):
            first_cage.generate_eggs(sire, dam, num_matings)

    def test_not_enough_dams_select_dams(self, first_cage):

        distrib_dams_available = {('a',): 760, tuple(sorted(('a', 'A'))): 760}
        total_dams = sum(distrib_dams_available.values())
        num_dams = 2*total_dams

        delta_dams = first_cage.select_dams(distrib_dams_available, num_dams)
        for key in delta_dams:
            assert delta_dams[key] == distrib_dams_available[key]
        for key in delta_dams:
            assert distrib_dams_available[key] == delta_dams[key]

    def test_update_step(self, first_cage, cur_day):
        first_cage.update(cur_day, 0)

    def test_get_stage_ages_distrib(self, first_cage):
        size = 5
        max_days = 4
        age_distrib = first_cage.get_stage_ages_distrib("L_dummy", size, max_days)
        assert age_distrib[-1] == 0.0
        assert all(age_distrib[:-1] > 0)

    def test_get_stage_ages_distrib_edge_cases(self, first_cage):
        age_distrib = first_cage.get_stage_ages_distrib("L_dummy", 5, 5)
        assert all(age_distrib > 0)

    def test_get_num_eggs_no_females(self, first_cage):
        first_cage.lice_population["L5f"] = 0
        assert first_cage.get_num_eggs(0) == 0

    def test_get_num_eggs(self, first_cage):
        matings = 6
        assert 1500 <= first_cage.get_num_eggs(matings) <= 1600

    def test_egg_mutation(self, first_cage, sample_offspring_distrib):
        sample_offspring = copy.deepcopy(sample_offspring_distrib)
        mutations = 0.001
        first_cage.mutate(sample_offspring, mutations)

        assert sample_offspring != sample_offspring_distrib
        assert sample_offspring == {('a',): 199, ('A', 'a'): 301, ('A',): 100}

    def test_get_egg_batch_null(self, first_cage, null_offspring_distrib, cur_day):
        egg_batch = first_cage.get_egg_batch(cur_day, null_offspring_distrib)
        assert egg_batch.geno_distrib == null_offspring_distrib

    def test_egg_batch_lt(self, first_cage, null_offspring_distrib, cur_day):
        batch1 = EggBatch(cur_day, null_offspring_distrib)
        batch2 = EggBatch(cur_day + dt.timedelta(days=1), null_offspring_distrib)
        assert batch1 != batch2
        assert batch1 < batch2

    def test_get_egg_batch(self, first_cage, cur_day):
        _, new_eggs = first_cage.do_mating_events()
        target_egg_distrib = GenoDistrib({('A', 'a'): 4644.5, ('A',): 2698.25, ('a',): 1927.25})

        new_egg_batch = first_cage.get_egg_batch(cur_day, new_eggs)
        assert new_egg_batch == EggBatch(
            hatch_date=datetime.datetime(2017, 10, 11, 0, 0),
            geno_distrib=target_egg_distrib)

    def test_avail_batch_lt(self, first_cage, null_offspring_distrib, cur_day):
        batch1 = DamAvailabilityBatch(cur_day, null_offspring_distrib)
        batch2 = DamAvailabilityBatch(cur_day + dt.timedelta(days=1), null_offspring_distrib)
        assert batch1 != batch2
        assert batch1 < batch2

    def test_get_egg_batch_across_time(self, first_cage):
        egg_offspring = {
            ('A',): 10,
            ('a',): 10,
            ('A', 'a'): 10,
        }

        # October
        cur_day = to_dt("2017-10-01 00:00:00")
        egg_batch = first_cage.get_egg_batch(cur_day, egg_offspring)
        assert (egg_batch.hatch_date - cur_day).days == 3
        assert egg_batch.geno_distrib == egg_offspring

        # August
        cur_day = to_dt("2017-08-01 00:00:00")
        egg_batch = first_cage.get_egg_batch(cur_day, egg_offspring)
        assert (egg_batch.hatch_date - cur_day).days == 7
        assert egg_batch.geno_distrib == egg_offspring

        # February
        cur_day = to_dt("2017-02-01 00:00:00")
        egg_batch = first_cage.get_egg_batch(cur_day, egg_offspring)
        assert (egg_batch.hatch_date - cur_day).days == 17
        assert egg_batch.geno_distrib == egg_offspring

    def test_create_offspring_early(self, first_cage, cur_day, null_offspring_distrib):
        egg_offspring = {
            ('A',): 10,
            ('a',): 10,
            ('A', 'a'): 10,
        }

        egg_batch = first_cage.get_egg_batch(cur_day, egg_offspring)

        first_cage.hatching_events.put(egg_batch)
        assert first_cage.create_offspring(cur_day) == null_offspring_distrib

    def test_create_offspring_same_day(self, first_cage, cur_day):
        egg_offspring = GenoDistrib({
            ('A',): 10,
            ('a',): 10,
            ('A', 'a'): 10,
        })

        for i in range(3):
            first_cage.hatching_events.put(EggBatch(cur_day+dt.timedelta(days=i), egg_offspring))

        offspring = first_cage.create_offspring(cur_day + dt.timedelta(days=3))
        for val in offspring.values():
            assert val == 30

    def test_promote_population_invalid(self, first_cage):
        with pytest.raises(ValueError):
            first_cage.promote_population("L3", "L4", 100)

    def test_promote_population(self, first_cage):
        first_cage.lice_population.geno_by_lifestage["L3"] = GenoDistrib({('A',): 1000, ('a',): 1000, ('A', 'a'): 1000})
        first_cage.lice_population.geno_by_lifestage["L4"] = GenoDistrib({('A',): 500, ('a',): 500, ('A', 'a'): 500})

        old_L3 = copy.deepcopy(first_cage.lice_population.geno_by_lifestage["L3"])
        old_L4 = copy.deepcopy(first_cage.lice_population.geno_by_lifestage["L4"])

        leaving_lice = 100  # new males and females combined that leave L4 to enter L5{m,f}
        entering_lice = 200  # L3 lice that enter L4

        first_cage.promote_population("L3", "L4", leaving_lice, entering_lice)

        new_L4 = first_cage.lice_population.geno_by_lifestage["L4"]

        target_population = sum(old_L4.values()) + entering_lice - leaving_lice
        target_geno = {('A',): 532, ('a',): 534, ('A', 'a'): 534}

        assert new_L4 == target_geno
        assert sum(new_L4.values()) == sum(old_L4.values()) + entering_lice - leaving_lice
        assert first_cage.lice_population["L4"] == target_population

        # L3 must be unchanged
        assert first_cage.lice_population.geno_by_lifestage["L3"] == old_L3

    def test_promote_population_offspring(self, first_cage):
        offspring_distrib = GenoDistrib({('A',): 500, ('a',): 500, ('A', 'a'): 500})
        new_L2 = 10
        old_L1 = copy.deepcopy(first_cage.lice_population.geno_by_lifestage["L1"])
        first_cage.promote_population(offspring_distrib, "L1", new_L2)
        new_L1 = first_cage.lice_population.geno_by_lifestage["L1"]

        target_population = sum(old_L1.values()) + sum(offspring_distrib.values()) - new_L2
        target_geno = {('A',): 535, ('a',): 535, ('A', 'a'): 570}

        assert new_L1 != old_L1
        assert sum(new_L1.values()) == target_population
        assert first_cage.lice_population["L1"] == target_population
        assert new_L1 == target_geno

    def test_get_arrivals(self, first_cage, cur_day, sample_offspring_distrib):
        hatch_date_unhatched = cur_day + dt.timedelta(5)
        unhatched_batch = TravellingEggBatch(cur_day, hatch_date_unhatched, sample_offspring_distrib)
        first_cage.arrival_events.put(unhatched_batch)

        hatched_at_travel_dist = GenoDistrib({
                                    ('A',): 10,
                                    ('a',): 10,
                                    ('A', 'a'): 10,
                                })
        hatch_date_hatched = cur_day - dt.timedelta(1)
        hatched_batch = TravellingEggBatch(cur_day, hatch_date_hatched, hatched_at_travel_dist)
        first_cage.arrival_events.put(hatched_batch)

        hatched_dist = first_cage.get_arrivals(cur_day)

        assert first_cage.hatching_events.qsize() == 1
        assert first_cage.hatching_events.get() == EggBatch(hatch_date_unhatched, sample_offspring_distrib)
        assert hatched_dist == hatched_at_travel_dist

    def test_get_arrivals_empty_queue(self, first_cage, cur_day):
        hatched_dist = first_cage.get_arrivals(cur_day)
        assert hatched_dist == {}

    def test_get_arrivals_no_arrivals(self, first_cage, cur_day, sample_offspring_distrib):
        arrival_date = cur_day + dt.timedelta(1)
        batch = TravellingEggBatch(arrival_date, arrival_date, sample_offspring_distrib)
        first_cage.arrival_events.put(batch)
        hatched_dist = first_cage.get_arrivals(cur_day)

        assert hatched_dist == {}
        assert first_cage.arrival_events.qsize() == 1
        assert first_cage.hatching_events.qsize() == 0

    def test_update_arrivals(self, first_cage, sample_offspring_distrib, cur_day):
        arrival_dict = {
            cur_day + dt.timedelta(5): {
                        ('A',): 100,
                        ('a',): 100,
                        ('A', 'a'): 100,
                     }
        }

        arrival_date = dt.timedelta(2)
        first_cage.update_arrivals(arrival_dict, arrival_date)

        assert first_cage.arrival_events.qsize() == 1

    def test_update_arrivals_empty_geno_dict(self, first_cage, sample_offspring_distrib, cur_day):
        arrival_dict = {
            cur_day + dt.timedelta(5): {
                        ('A',): 0,
                        ('a',): 0,
                        ('A', 'a'): 0,
                     }
        }

        arrival_date = dt.timedelta(2)
        first_cage.update_arrivals(arrival_dict, arrival_date)

        assert first_cage.arrival_events.qsize() == 0

    def test_get_reservoir_lice(self, first_cage):
        pressure = 100
        dist = first_cage.get_reservoir_lice(pressure)

        assert sum(dist.values()) == pressure
        for value in dist.values():
            assert value >= 0

    def test_get_reservoir_lice_no_pressure(self, first_cage):
        assert first_cage.get_reservoir_lice(0) == {"L1": 0, "L2": 0}

    def test_dying_lice_from_dead_no_dead_fish(self, first_cage):
        assert first_cage.get_dying_lice_from_dead_fish(0) == {}

    def test_dying_lice_from_dead_no_lice(self, first_cage):
        for stage in Cage.susceptible_stages:
            first_cage.lice_population[stage] = 0

        for fish in [0, 1, 10, 1000]:
            assert first_cage.get_dying_lice_from_dead_fish(fish) == {}

    def test_dying_lice_from_dead_fish(self, first_cage):
        dead_fish = 5
        dead_lice_target = {
            'L3': 2,
            'L4': 1,
            'L5m': 1,
        }

        dead_lice = first_cage.get_dying_lice_from_dead_fish(dead_fish)
        assert dead_lice == dead_lice_target

        # An increase in population should cause a proportional number of deaths
        for stage in Cage.susceptible_stages:
            first_cage.lice_population[stage] *= 100

        rate = first_cage.get_mean_infected_fish() / first_cage.get_infecting_population()
        dead_lice_target = {
            'L3': 190,
            'L4': 123,
            'L5m': 19,
            'L5f': 63
        }

        dead_lice = first_cage.get_dying_lice_from_dead_fish(dead_fish)
        assert dead_lice == dead_lice_target

        # Similarly, an increase in dead_fish should cause the same effect

        for stage in Cage.susceptible_stages:
            first_cage.lice_population[stage] //= 100

        dead_fish *= 100

        dead_lice = first_cage.get_dying_lice_from_dead_fish(dead_fish)
        assert dead_lice == dead_lice_target

    def test_update_step_before_start_date_two_days(self, first_cage, planctonic_only_population):
        cur_date = first_cage.start_date - dt.timedelta(5)
        first_cage.lice_population = planctonic_only_population

        hatch_date = cur_date + dt.timedelta(1)
        geno_hatch = GenoDistrib({("a",): 10, ("A", "a"): 0, ("A",): 0})
        first_cage.arrival_events.put(TravellingEggBatch(cur_date, hatch_date, geno_hatch))

        pressure = 10

        offspring, hatch_date, cost = first_cage.update(cur_date, pressure)

        assert offspring == {}
        assert hatch_date is None
        assert all(first_cage.lice_population[susceptible_stage] == 0 for susceptible_stage in Cage.susceptible_stages)
        assert first_cage.num_fish == 4000
        assert first_cage.num_infected_fish == 0
        assert first_cage.arrival_events.qsize() == 0
        assert first_cage.hatching_events.qsize() == 1

        assert 150 <= cost <= 200

        cur_date += dt.timedelta(1)
        offspring, hatch_date, cost = first_cage.update(cur_date, pressure)

        assert offspring == {}
        assert hatch_date is None
        assert all(first_cage.lice_population[susceptible_stage] == 0 for susceptible_stage in Cage.susceptible_stages)
        assert first_cage.num_fish == 4000
        assert first_cage.num_infected_fish == 0
        assert first_cage.arrival_events.qsize() == 0
        assert first_cage.hatching_events.qsize() == 0
        assert 150 <= cost <= 200

    def test_update_step_before_start_date_no_deaths(self, first_cage, planctonic_only_population):
        cur_date = first_cage.start_date - dt.timedelta(5)
        first_cage.lice_population = planctonic_only_population

        hatch_date = cur_date
        geno_hatch = GenoDistrib({("a",): 10, ("A", "a"): 0, ("A",): 0})
        first_cage.hatching_events.put(EggBatch(hatch_date, geno_hatch))
        pressure = 10

        population_before = sum(first_cage.lice_population.values())
        inflow = pressure + sum(geno_hatch.values())

        # set mortality to 0
        first_cage.cfg.background_lice_mortality_rates = {key: 0 for key in first_cage.lice_population}

        offspring, hatch_date, cost = first_cage.update(cur_date, pressure)

        assert offspring == {}
        assert hatch_date is None
        assert all(first_cage.lice_population[susceptible_stage] == 0 for susceptible_stage in Cage.susceptible_stages)
        assert first_cage.num_fish == 4000
        assert first_cage.num_infected_fish == 0

        current_population = sum(first_cage.lice_population.values())
        assert current_population == population_before + inflow
        assert cost > 0

    def test_update_step_before_start_date_only_deaths(self, first_cage, planctonic_only_population):
        cur_date = first_cage.start_date - dt.timedelta(5)
        first_cage.lice_population = planctonic_only_population

        population_before = sum(first_cage.lice_population.values())

        # make sure there will be deaths
        first_cage.cfg.background_lice_mortality_rates = {key: 1 for key in first_cage.lice_population}
        pressure = 0

        offspring, hatch_date, cost = first_cage.update(cur_date, pressure)

        assert offspring == {}
        assert hatch_date is None
        assert all(first_cage.lice_population[susceptible_stage] == 0 for susceptible_stage in Cage.susceptible_stages)
        assert first_cage.num_fish == 4000
        assert first_cage.num_infected_fish == 0

        current_population = sum(first_cage.lice_population.values())
        assert current_population < population_before
        assert cost > 0

    def test_fallowing(self, first_cage):
        assert not first_cage.is_fallowing
        first_cage.fallow()
        assert first_cage.num_fish == 0
        assert first_cage.num_infected_fish == 0
        assert first_cage.is_fallowing

    def test_update_dying_busy_dams(self,
        first_cage,
        first_cage_population,
        null_offspring_distrib,
        null_dams_batch,
        sample_treatment_mortality,
        sample_offspring_distrib,
        cur_day
    ):
        first_cage_population["L5f"] = first_cage_population["L5f"] * 100
        first_cage_population["L5m"] = first_cage_population["L5m"] * 100
        sample_treatment_mortality["L5f"] = sample_treatment_mortality["L5f"] * 10

        dams, _ = first_cage.do_mating_events()
        first_cage_population.add_busy_dams_batch(DamAvailabilityBatch(cur_day + dt.timedelta(days=1), dams))

        background_mortality = first_cage.get_background_lice_mortality()
        fish_deaths_natural = 0
        fish_deaths_from_lice = 0
        fish_deaths_from_treatment = 0
        new_l2 = 0
        new_l4 = 0
        new_females = 20
        new_males = 0
        new_infections = 0

        reservoir_lice = {"L1": 0, "L2": 0}

        null_hatched_arrivals = null_offspring_distrib
        null_returned_dams = null_offspring_distrib

        old_busy_dams = first_cage_population.busy_dams.copy()
        old_busy_dams_gross = old_busy_dams.gross

        first_cage.update_deltas(background_mortality, sample_treatment_mortality, fish_deaths_natural,
                                 fish_deaths_from_lice, fish_deaths_from_treatment, new_l2, new_l4, new_females,
                                 new_males, new_infections, reservoir_lice, null_dams_batch, null_offspring_distrib,
                                 null_hatched_arrivals)

        assert first_cage.lice_population["L5f"] < 1000
        assert first_cage.lice_population.busy_dams.gross < old_busy_dams_gross
        assert first_cage.lice_population.busy_dams <= old_busy_dams
        assert 800 <= first_cage.lice_population.busy_dams.gross <= 840