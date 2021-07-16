import datetime
import datetime as dt
import itertools

import numpy as np
import pytest

import json

from src.Config import to_dt
from src.Cage import EggBatch, TravellingEggBatch


class TestCage:
    def test_cage_loads_params(self, first_cage):
        assert first_cage.id == 0
        assert first_cage.start_date == to_dt("2017-10-01 00:00:00")
        assert first_cage.num_fish == 4000
        assert first_cage.num_infected_fish == first_cage.get_mean_infected_fish()
        # this will likely break
        assert first_cage.lice_population == {
            "L1": 150,
            "L2": 0,
            "L3": 30,
            "L4": 30,
            "L5f": 10,
            "L5m": 10
        }

    def test_cage_json(self, first_cage):
        return_str = str(first_cage)
        imported_cage = json.loads(return_str)
        assert isinstance(imported_cage, dict)

    def test_cage_lice_background_mortality_one_day(self, first_cage):
        # NOTE: this currently relies on Stien's approach.
        # Changing this approach will break things
        first_cage.cfg.tau = 1
        dead_lice_dist = first_cage.get_background_lice_mortality(first_cage.lice_population)
        dead_lice_dist_np = np.array(list(dead_lice_dist.values()))
        expected_dead_lice = np.array([26, 0, 0, 2, 0, 2])
        assert np.alltrue(dead_lice_dist_np >= 0.0)
        assert np.alltrue(np.isclose(dead_lice_dist_np, expected_dead_lice))

    def test_cage_update_lice_treatment_mortality_no_effect(self, farm, first_cage):
        treatment_dates = farm.treatment_dates
        assert(treatment_dates == sorted(treatment_dates))

        # before a 14-day activation period there should be no effect
        for i in range(-14, first_cage.cfg.delay_EMB):
            cur_day = treatment_dates[0] + dt.timedelta(days=i)
            mortality_updates = first_cage.get_lice_treatment_mortality(cur_day)
            assert all(rate == 0 for rate in mortality_updates.values())

    def test_cage_update_lice_treatment_mortality(self, farm, first_cage):
        # TODO: this does not take into account water temperature!
        treatment_dates = farm.treatment_dates

        # first useful day
        cur_day = treatment_dates[0] + dt.timedelta(days=5)
        # We've got an unlucky seed - it will sample 0 the first time so we are not interested
        mortality_updates = first_cage.get_lice_treatment_mortality(cur_day)
        mortality_updates = first_cage.get_lice_treatment_mortality(cur_day)

        assert mortality_updates == {
            "L1": 0,
            "L2": 0,
            "L3": 1,
            "L4": 1,
            "L5m": 0,
            "L5f": 0
        }

    def test_get_stage_ages_respects_constraints(self, first_cage):
        test_num_lice = 1000
        development_days = 15
        min_development_stage = 4
        mean_development_stage = 8

        for i in range(100):
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

        assert new_l2 == 0
        assert new_l4 == 1 and first_cage.lice_population["L3"] > 0
        assert new_males == 2 and first_cage.lice_population["L4"] > 0
        assert new_females == 2

    def test_get_fish_growth(self, first_cage):
        first_cage.num_fish *= 300
        first_cage.num_infected_fish = first_cage.num_fish // 3
        natural_death, lice_death = first_cage.get_fish_growth(1, 1)

        # basic invariants
        assert natural_death > 0
        assert lice_death > 0

        # exact figures
        assert natural_death == 688
        assert lice_death == 4

    def test_get_fish_growth_no_lice(self, first_cage):
        first_cage.num_infected_fish = 0
        for k in first_cage.lice_population:
            first_cage.lice_population[k] = 0

        _, lice_death = first_cage.get_fish_growth(1, 1)

    def test_get_infection_rates(self, first_cage):
        first_cage.lice_population["L2"] = 100
        rate, avail_lice = first_cage.get_infection_rates(1)
        assert rate > 0
        assert avail_lice > 0

        assert np.isclose(rate, 0.16665658047288034)
        assert avail_lice == 90

    def test_do_infection_events(self, first_cage):
        first_cage.lice_population["L2"] = 100
        num_infected_fish = first_cage.do_infection_events(1)

        assert num_infected_fish > 0
        assert num_infected_fish == 15

    def test_get_infected_fish_noinfection(self, first_cage):
        # TODO: maybe make a fixture of this?
        first_cage.lice_population["L3"] = 0
        first_cage.lice_population["L4"] = 0
        first_cage.lice_population["L5m"] = 0
        first_cage.lice_population["L5f"] = 0

        assert first_cage.get_mean_infected_fish() == 0
        assert first_cage.get_variance_infected_fish() == 0

    def test_get_infected_fish(self, first_cage):
        assert first_cage.get_mean_infected_fish() == int(4000 * (1 - (3999 / 4000) ** 80))
        assert 70 <= first_cage.get_variance_infected_fish() <= 80

    def test_get_std_infected_fish(self, first_cage):
        assert first_cage.get_variance_infected_fish() > 0

    def test_get_num_matings_no_infection(self, first_cage):
        males = first_cage.lice_population["L5m"]
        first_cage.lice_population["L5m"] = 0
        assert first_cage.get_num_matings() == 0
        first_cage.lice_population["L5m"] = males

        first_cage.lice_population["L5f"] = 0
        assert first_cage.get_num_matings() == 0

    def test_get_num_matings(self, first_cage):
        first_cage.lice_population["L5m"] = first_cage.lice_population["L5f"]
        assert 1 <= first_cage.get_num_matings() <= 10

    def test_update_deltas_no_negative_raise(self, first_cage, null_offspring_distrib):
        first_cage.lice_population["L3"] = 0
        first_cage.lice_population["L4"] = 0
        first_cage.lice_population["L5m"] = 0
        first_cage.lice_population["L5f"] = 0

        background_mortality = first_cage.get_background_lice_mortality(first_cage.lice_population)
        treatment_mortality = {"L1": 0, "L2": 0, "L3": 10, "L4": 10, "L5m": 20, "L5f": 30}
        fish_deaths_natural = 0
        fish_deaths_from_lice = 0
        new_l2 = 0
        new_l4 = 0
        new_females = 0
        new_males = 0
        new_infections = 0
        reservoir_lice = {"L1": 0, "L2": 0}
        delta_avail_dams = {"L1": {('A',): 0, ('a',): 0, ('A', 'a'): 0},
                            "L2": {('A',): 0, ('a',): 0, ('A', 'a'): 0},
                            "L3": {('A',): 0, ('a',): 0, ('A', 'a'): 0},
                            "L4": {('A',): 0, ('a',): 0, ('A', 'a'): 0},
                            "L5m": {('A',): 0, ('a',): 0, ('A', 'a'): 0},
                            "L5f": {('A',): 0, ('a',): 0, ('A', 'a'): 0}}
        delta_eggs = {('A',): 0, ('a',): 0, ('A', 'a'): 0}
        null_hatched_arrivals = null_offspring_distrib

        first_cage.update_deltas(background_mortality,
                                 treatment_mortality,
                                 fish_deaths_natural,
                                 fish_deaths_from_lice,
                                 new_l2, new_l4, new_females, new_males,
                                 new_infections, reservoir_lice,
                                 delta_avail_dams, delta_eggs,
                                 null_offspring_distrib,
                                 null_hatched_arrivals)

        for population in first_cage.lice_population.values():
            assert population >= 0

    def test_do_mating_events(self, first_cage):
        first_cage.geno_by_lifestage['L5m'] = {('A',): 5, ('a',): 5, ('A', 'a'): 5}
        first_cage.available_dams = {('A',): 10}
        target_eggs = {('A',): 5431.0, tuple(sorted(('a', 'A'))): 3868.0}
        target_delta_dams = {('A',): 6}

        delta_avail_dams, delta_eggs = first_cage.do_mating_events()
        for key in delta_eggs:
            assert delta_eggs[key] == target_eggs[key]
        for key in delta_avail_dams:
            assert delta_avail_dams[key] == target_delta_dams[key]


    def test_no_available_sires_do_mating_events(self, first_cage):
        first_cage.geno_by_lifestage['L5m'] = {('A',): 0, ('a',): 0, ('A', 'a'): 0}

        delta_avail_dams, delta_eggs = first_cage.do_mating_events()
        assert not bool(delta_avail_dams)
        assert not bool(delta_eggs)

    def test_generate_eggs_quantitative(self, first_cage):
        sire = 0.7
        dam = 0.0
        num_matings = 10
        target_eggs = {0.4: 2537}
        eggs = first_cage.generate_eggs(sire, dam, 'quantitative', num_matings)
        for key in eggs:
            assert eggs[key] == target_eggs[key]

        sire = 0.2
        dam = 0.2
        num_matings = 10
        target_eggs = {0.2: 2545}
        eggs = first_cage.generate_eggs(sire, dam, 'quantitative', num_matings)
        for key in eggs:
            assert eggs[key] == target_eggs[key]

    def test_generate_eggs_discrete(self, first_cage):
        breeding_method = 'discrete'

        sire = ('A',)
        dam = ('A',)
        hom_dom_target = {('A',): 1574}

        num_matings = first_cage.get_num_matings()
        egg_result_hom_dom = first_cage.generate_eggs(sire, dam, breeding_method, num_matings)
        for geno in egg_result_hom_dom:
            assert egg_result_hom_dom[geno] == hom_dom_target[geno]

        sire = ('a',)
        dam = ('a',)
        hom_rec_target = {('a',): 1505}
        egg_result_hom_rec = first_cage.generate_eggs(sire, dam, breeding_method, num_matings)
        for geno in egg_result_hom_rec:
            assert egg_result_hom_rec[geno] == hom_rec_target[geno]

        sire = tuple(sorted(('a', 'A')))
        dam = ('a',)
        het_target_sire = {('a',): 760.5, tuple(sorted(('a', 'A'))): 760.5}
        egg_result_het_sire = first_cage.generate_eggs(sire, dam, breeding_method, num_matings)
        for geno in egg_result_het_sire:
            assert egg_result_het_sire[geno] == het_target_sire[geno]

        dam = tuple(sorted(('a', 'A')))
        sire = ('a',)
        het_target_dam = {('a',): 794.0, tuple(sorted(('a', 'A'))): 794.0}
        egg_result_het_dam = first_cage.generate_eggs(sire, dam, breeding_method, num_matings)
        for geno in egg_result_het_dam:
            assert egg_result_het_dam[geno] == het_target_dam[geno]

        dam = tuple(sorted(('a', 'A')))
        sire = tuple(sorted(('a', 'A')))
        het_target = {('A', 'a'): 727.0, ('A',): 363.5, ('a',): 363.5}
        egg_result_het = first_cage.generate_eggs(sire, dam, breeding_method, num_matings)
        for geno in egg_result_het:
            assert egg_result_het[geno] == het_target[geno]

    def test_not_enough_dams_select_dams(self, first_cage):

        distrib_dams_available = {('a',): 760, tuple(sorted(('a', 'A'))): 760}
        total_dams = sum(distrib_dams_available.values())
        num_dams = 2*total_dams

        delta_dams = first_cage.select_dams(distrib_dams_available, num_dams)
        for key in delta_dams:
            assert delta_dams[key] == distrib_dams_available[key]
        for key in delta_dams:
            assert  distrib_dams_available[key] == delta_dams[key]



    def test_update_step(self, first_cage, cur_day):
        first_cage.update(cur_day, 1, 0)

    def test_to_csv(self, first_cage):
        first_cage.lice_population = {"L1": 1,
                                      "L2": 2,
                                      "L3": 3,
                                      "L4": 4,
                                      "L5f": 5,
                                      "L5m": 6}
        first_cage.num_fish = 7
        first_cage.id = 0

        csv_str = first_cage.to_csv()
        csv_list = csv_str.split(", ")
        print(csv_list)

        assert csv_list[0] == "0"
        assert csv_list[1] == "7"
        for i in range(2, 7):
            assert csv_list[i] == str(i - 1)
        assert csv_list[8] == "21"

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
        first_cage.lice_population['L5f'] = 0
        assert first_cage.get_num_eggs(0) == 0

    def test_get_num_eggs(self, first_cage):
        first_cage.lice_population['L5m'] = first_cage.lice_population['L5f']

        matings = first_cage.get_num_matings()
        assert 1500 <= first_cage.get_num_eggs(matings) <= 1600

    def test_get_egg_batch_null(self, first_cage, null_offspring_distrib, cur_day):
        egg_distrib, hatch_date = first_cage.get_egg_batch(cur_day, null_offspring_distrib)
        assert egg_distrib == null_offspring_distrib

    def test_egg_batch_lt(self, first_cage, null_offspring_distrib, cur_day):
        batch1 = EggBatch(cur_day, null_offspring_distrib)
        batch2 = EggBatch(cur_day + dt.timedelta(days=1), null_offspring_distrib)
        assert batch1 != batch2
        assert batch1 < batch2

    def test_get_egg_batch(self, first_cage, cur_day):
        busy_dams, new_eggs = first_cage.do_mating_events()
        target_egg_distrib = {('A', 'a'): 4644.5, ('A',): 2698.25, ('a',): 1927.25}

        egg_distrib, hatch_date = first_cage.get_egg_batch(cur_day, new_eggs)
        assert egg_distrib == target_egg_distrib
        assert hatch_date == datetime.datetime(2017, 10, 6, 0, 0)

    def test_get_egg_batch_across_time(self, first_cage):
        egg_offspring = {
            ('A',): 10,
            ('a',): 10,
            ('A', 'a'): 10,
        }

        # October
        cur_day = to_dt("2017-10-01 00:00:00")
        egg_distrib, hatch_date = first_cage.get_egg_batch(cur_day, egg_offspring)
        assert (hatch_date - cur_day).days == 11
        assert egg_distrib == egg_offspring

        # August
        cur_day = to_dt("2017-08-01 00:00:00")
        egg_distrib, hatch_date = first_cage.get_egg_batch(cur_day, egg_offspring)
        assert (hatch_date - cur_day).days == 3
        assert egg_distrib == egg_offspring

        # February
        cur_day = to_dt("2017-02-01 00:00:00")
        egg_distrib, hatch_date = first_cage.get_egg_batch(cur_day, egg_offspring)
        assert (hatch_date - cur_day).days == 12
        assert egg_distrib == egg_offspring

    def test_create_offspring_early(self, first_cage, cur_day, null_offspring_distrib):
        egg_offspring = {
            ('A',): 10,
            ('a',): 10,
            ('A', 'a'): 10,
        }

        egg_distrib, hatch_date = first_cage.get_egg_batch(cur_day, egg_offspring)

        first_cage.hatching_events.put(EggBatch(hatch_date, egg_distrib))
        assert first_cage.create_offspring(cur_day) == null_offspring_distrib

    def test_create_offspring_same_day(self, first_cage, cur_day):
        egg_offspring = {
            ('A',): 10,
            ('a',): 10,
            ('A', 'a'): 10,
        }

        for i in range(3):
            first_cage.hatching_events.put(EggBatch(cur_day+dt.timedelta(days=i), egg_offspring))

        offspring = first_cage.create_offspring(cur_day + dt.timedelta(days=3))
        for val in offspring.values():
            assert val == 30

    def test_get_arrivals(self, first_cage, cur_day, sample_offspring_distrib):
        hatch_date_unhatched = cur_day + dt.timedelta(5)
        unhatched_batch = TravellingEggBatch(cur_day, hatch_date_unhatched, sample_offspring_distrib)
        first_cage.arrival_events.put(unhatched_batch)

        hatched_at_travel_dist = {
                                    ('A',): 10,
                                    ('a',): 10,
                                    ('A', 'a'): 10,
                                }
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
