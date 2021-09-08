from __future__ import annotations

from collections import Counter
import copy
import datetime as dt
import json
import math
from functools import singledispatch
from queue import PriorityQueue
from typing import Union, Optional, Tuple, cast, TYPE_CHECKING

import iteround
import numpy as np
from scipy import stats

from src.Config import Config
from src.TreatmentTypes import Treatment, GeneticMechanism, HeterozygousResistance, Money
from src.LicePopulation import (Allele, Alleles, GenoDistrib, GrossLiceDistrib,
                                LicePopulation, GenoTreatmentDistrib, GenoTreatmentValue, GenoLifeStageDistrib,
                                QuantitativeGenoDistrib, GenericGenoDistrib)
from src.QueueTypes import DamAvailabilityBatch, EggBatch, TravellingEggBatch, TreatmentEvent, pop_from_queue
from src.JSONEncoders import CustomFarmEncoder

if TYPE_CHECKING:  # pragma: no cover
    from src.Farm import Farm, GenoDistribByHatchDate

OptionalDamBatch = Optional[DamAvailabilityBatch]
OptionalEggBatch = Optional[EggBatch]


class Cage:
    """
    Fish cages contain the fish.
    """

    lice_stages = ["L1", "L2", "L3", "L4", "L5f", "L5m"]
    susceptible_stages = lice_stages[2:]
    pathogenic_stages = lice_stages[2:]

    def __init__(self, cage_id: int, cfg: Config, farm: Farm,
                 initial_lice_pop: Optional[GrossLiceDistrib] = None):
        """
        Create a cage on a farm
        :param cage_id: the label (id) of the cage within the farm
        :param cfg: the farm configuration
        :param farm: a Farm object
        :param initial_lice_pop: if provided, overrides default generated lice population
        """

        self.cfg = cfg
        self.logger = cfg.logger
        self.id = cage_id

        self.farm_id = farm.name
        self.start_date = cfg.farms[self.farm_id].cages_start[cage_id]
        self.date = cfg.start_date

        # TODO: update with calculations
        if initial_lice_pop is None:
            lice_population = {"L1": cfg.ext_pressure, "L2": 0, "L3": 0, "L4": 0, "L5f": 0,
                               "L5m": 0}
        else:
            lice_population = initial_lice_pop

        self.farm = farm

        self.egg_genotypes = GenoDistrib()
        # TODO/Question: what's the best way to deal with having multiple possible genetic schemes?
        # TODO/Question: I suppose some of this initial genotype information ought to come from the config file
        # TODO/Question: the genetic mechanism will be the same for all lice in a simulation, so should it live in the driver?
        self.genetic_mechanism = self.cfg.genetic_mechanism

        geno_by_lifestage = {stage: GenoDistrib(self.cfg.genetic_ratios).normalise_to(lice_population[stage])
                             for stage in lice_population}

        self.lice_population = LicePopulation(lice_population, geno_by_lifestage, self.cfg)

        self.num_fish = cfg.farms[self.farm_id].num_fish
        self.num_infected_fish = self.get_mean_infected_fish()

        self.hatching_events = PriorityQueue()  # type: PriorityQueue[EggBatch]
        self.arrival_events = PriorityQueue()  # type: PriorityQueue[TravellingEggBatch]
        self.treatment_events = PriorityQueue()  # type: PriorityQueue[TreatmentEvent]

        self.last_effective_treatment = None  # type: Optional[TreatmentEvent]

    def to_json_dict(self):
        """
        Create a JSON-serialisable dictionary version of a cage.
        """
        filtered_vars = vars(self).copy()

        del filtered_vars["farm"]
        del filtered_vars["logger"]
        del filtered_vars["cfg"]

        # May want to improve these or change them if we change the representation for genotype distribs
        # TODO: these below can be probably moved to a proper encoder
        filtered_vars["egg_genotypes"] = {str(key): val for key, val in filtered_vars["egg_genotypes"].items()}
        filtered_vars["geno_by_lifestage"] = {str(key): str(val) for key, val in
                                              self.lice_population.geno_by_lifestage.items()}
        filtered_vars["genetic_mechanism"] = str(filtered_vars["genetic_mechanism"])[len("GeneticMechanism."):]

        return filtered_vars

    def __str__(self):
        """
        Get a human readable string representation of the cage in json form.
        :return: a description of the cage
        """

        return json.dumps(self.to_json_dict(), cls=CustomFarmEncoder, indent=4)

    def update(self, cur_date: dt.datetime, pressure: int) -> Tuple[GenoDistrib, Optional[dt.datetime], Money]:
        """Update the cage at the current time step.
        :param cur_date: Current date of simulation
        :param pressure: External pressure, planctonic lice coming from the reservoir
        :return: Tuple (egg genotype distribution, hatching date, cost)
        """

        if cur_date >= self.start_date:
            self.logger.debug("\tUpdating farm {} / cage {}".format(self.farm_id, self.id))
            self.logger.debug("\t\tinitial fish population = {}".format(self.num_fish))
        else:
            self.logger.debug("\tUpdating farm {} / cage {} (non-operational)".format(self.farm_id, self.id))

        self.logger.debug("\t\tinitial lice population = {}".format(self.lice_population))

        # Background lice mortality events
        dead_lice_dist = self.get_background_lice_mortality()

        # Development events
        new_L2, new_L4, new_females, new_males = self.get_lice_lifestage(cur_date.month)

        # Lice coming from other cages and farms
        # NOTE: arrivals first then hatching
        hatched_arrivals_dist = self.get_arrivals(cur_date)

        # Egg hatching
        new_offspring_distrib = self.create_offspring(cur_date)

        # Lice coming from reservoir
        lice_from_reservoir = self.get_reservoir_lice(pressure)

        if cur_date < self.start_date or self.is_fallowing:
            # Values that are not used before the start date
            treatment_mortality = self.lice_population.get_empty_geno_distrib()
            fish_deaths_natural, fish_deaths_from_lice = 0, 0
            num_infection_events = 0
            avail_dams_batch = None  # type: OptionalDamBatch
            new_egg_batch = None  # type: OptionalEggBatch
            cost = self.cfg.monthly_cost / 28

        else:
            # Events that happen when cage is populated with fish
            # (after start date)
            days_since_start = (cur_date - self.date).days

            # Treatment mortality events
            treatment_mortality, cost = self.get_lice_treatment_mortality(cur_date)

            # Fish growth and death
            fish_deaths_natural, fish_deaths_from_lice = self.get_fish_growth(days_since_start)

            # Infection events
            num_infection_events = self.do_infection_events(cur_date, days_since_start)

            # Mating events that create eggs
            delta_avail_dams, delta_eggs = self.do_mating_events()
            avail_dams_batch = DamAvailabilityBatch(cur_date + dt.timedelta(days=self.cfg.dam_unavailability),
                                                    delta_avail_dams)

            new_egg_batch = self.get_egg_batch(cur_date, delta_eggs)

        fish_deaths_from_treatment = self.get_fish_treatment_mortality(
            (cur_date - self.start_date).days,
            fish_deaths_from_lice,
            fish_deaths_natural
        )

        self.update_deltas(
            dead_lice_dist,
            treatment_mortality,
            fish_deaths_natural,
            fish_deaths_from_lice,
            fish_deaths_from_treatment,
            new_L2,
            new_L4,
            new_females,
            new_males,
            num_infection_events,
            lice_from_reservoir,
            avail_dams_batch,
            new_offspring_distrib,
            hatched_arrivals_dist
        )

        self.logger.debug("\t\tfinal lice population= {}".format(self.lice_population))

        egg_distrib = GenoDistrib()
        hatch_date = None  # type: Optional[dt.datetime]
        if cur_date >= self.start_date and new_egg_batch:
            self.logger.debug("\t\tfinal fish population = {}".format(self.num_fish))
            egg_distrib = new_egg_batch.geno_distrib
            hatch_date = new_egg_batch.hatch_date
            self.logger.debug("\t\tlice offspring = {}".format(sum(egg_distrib.values())))

        return egg_distrib, hatch_date, cost

    def get_lice_treatment_mortality_rate(self, cur_date: dt.datetime) -> GenoTreatmentDistrib:
        num_susc_per_geno = sum(self.lice_population.geno_by_lifestage.values(), GenericGenoDistrib())

        geno_treatment_distrib = {geno: GenoTreatmentValue(0.0, 0) for geno in num_susc_per_geno}

        # if no treatment has been applied check if the previous treatment is still effective

        def cts(event):
            nonlocal self
            self.last_effective_treatment = event

        pop_from_queue(self.treatment_events, cur_date, cts)

        if self.last_effective_treatment is None or \
            self.last_effective_treatment.affecting_date > cur_date or \
                cur_date > self.last_effective_treatment.treatment_window:
            return geno_treatment_distrib

        treatment_type = self.last_effective_treatment.treatment_type

        self.logger.debug("\t\ttreating farm {}/cage {} on date {}".format(self.farm_id,
                                                                           self.id, cur_date))

        if treatment_type == Treatment.emb:
            # For now, assume a simple heterozygous distribution with a mere geometric distribution
            for geno, num_susc in num_susc_per_geno.items():
                trait = self.get_allele_heterozygous_trait(geno)
                susceptibility_factor = 1.0 - self.cfg.emb.pheno_resistance[trait]
                geno_treatment_distrib[geno] = GenoTreatmentValue(susceptibility_factor, cast(int, num_susc))
        else:
            raise NotImplementedError("Only EMB treatment is supported")

        return geno_treatment_distrib

    @staticmethod
    def get_allele_heterozygous_trait(alleles: Alleles):
        """
        Get the allele heterozygous type
        """
        # should we move this?
        if 'A' in alleles:
            if 'a' in alleles:
                trait = HeterozygousResistance.incompletely_dominant
            else:
                trait = HeterozygousResistance.dominant
        else:
            trait = HeterozygousResistance.recessive
        return trait

    def get_lice_treatment_mortality(self, cur_date) -> Tuple[GenoLifeStageDistrib, Money]:
        """
        Calculate the number of lice in each stage killed by treatment.
        """

        dead_lice_dist = self.lice_population.get_empty_geno_distrib()

        dead_mortality_distrib = self.get_lice_treatment_mortality_rate(cur_date)

        cost = Money("0.00")

        for geno, (mortality_rate, num_susc) in dead_mortality_distrib.items():
            if mortality_rate > 0:
                num_dead_lice = self.cfg.rng.poisson(mortality_rate * num_susc)
                num_dead_lice = min(num_dead_lice, num_susc)

                # Now we need to decide how many lice from each stage die,
                #   the algorithm is to label each louse  1...num_susc
                #   assign each of these a probability of dying as (phenoEMB)/np.sum(phenoEMB)
                #   randomly pick lice according to this probability distribution
                #        now we need to find out which stages these lice are in by calculating the
                #        cumulative sum of the lice in each stage and finding out how many of our
                #        dead lice falls into this bin.
                dead_lice = self.cfg.rng.choice(range(num_susc), num_dead_lice, replace=False).tolist()
                total_so_far = 0
                for stage in self.susceptible_stages:
                    available_in_stage = self.lice_population.geno_by_lifestage[stage][geno]
                    num_dead = len([x for x in dead_lice if total_so_far <= x <
                                    (total_so_far + available_in_stage)])
                    total_so_far += available_in_stage
                    if num_dead > 0:
                        dead_lice_dist[stage][geno] = num_dead

                self.logger.debug("\t\tdistribution of dead lice on farm {}/cage {} = {}"
                                  .format(self.farm_id, self.id, dead_lice_dist))

        # Compute cost
        if self.last_effective_treatment is not None and self.last_effective_treatment.end_application_date > cur_date:
            treatment_type = self.last_effective_treatment.treatment_type
            treatment_cfg = self.cfg.get_treatment(treatment_type)
            cage_days = (cur_date - self.start_date).days
            cost = treatment_cfg.price_per_kg * int(self.average_fish_mass(cage_days))

        return dead_lice_dist, cost

    def get_stage_ages_distrib(self, stage: str, size=15, stage_age_max_days=None):
        """
        Create an age distribution (in days) for the sea lice within a lifecycle stage.
        In absence of further data or constraints, we simply assume it's a uniform distribution
        """

        # NOTE: no data is available for L5 stages. We assume for simplicity they die after 30-ish days
        if stage_age_max_days is None:
            stage_age_max_days = min(self.cfg.stage_age_evolutions[stage], size + 1)
        p = np.zeros((size,))
        p[:stage_age_max_days] = 1
        return p / np.sum(p)

    def get_evolution_ages(self, size: int, minimum_age: int, mean: int, development_days=25):
        """
        Create an age distribution (in days) for the sea lice within a lifecycle stage.
        TODO: This actually computes the evolution ages.
        :param size the number of lice to consider
        :param minimum_age the minimum age before an evolution is allowed
        :param mean the mean of the distribution. mean must be bigger than min.
        :param development_days the maximum age to consider
        :return a size-long array of ages (in days)
        """

        # Create a shifted poisson distribution,
        k = mean - minimum_age
        max_quantile = development_days - minimum_age

        assert minimum_age > 0, "min must be positive"
        assert k > 0, "mean must be greater than min."

        p = stats.poisson.pmf(range(max_quantile), k)
        p = p / np.sum(p)  # probs need to add up to one
        return self.cfg.rng.choice(range(max_quantile), size, p=p) + minimum_age

    def get_lice_lifestage(self, cur_month) -> Tuple[int, int, int, int]:
        """
        Move lice between lifecycle stages.
        See Section 2.1 of Aldrin et al. (2017)

        :param cur_month: the current month
        :returns a tuple (new_l2, new_l4, new_l5f, new_l5m)
        """
        self.logger.debug("\t\tupdating lice lifecycle stages")

        def dev_times(del_p: float, del_m10: float, del_s: float, temp_c: float, ages: np.ndarray):
            """
            Probability of developing after n_days days in a stage as given by Aldrin et al 2017
            See section 2.2.4 of Aldrin et al. (2017)
            :param ages: stage-age
            :param del_p: power transformation constant on temp_c
            :param del_m10: 10 °C median reference value
            :param del_s: fitted Weibull shape parameter
            :param temp_c: average temperature in °C
            :return: expected development rate
            """
            epsilon = 1e-30
            del_m = del_m10 * (10 / temp_c) ** del_p

            unbounded = math.log(2) * del_s * ages ** (del_s - 1) * del_m ** (-del_s)
            unbounded = np.clip(unbounded, np.float64(epsilon), np.float64(1.0))
            return unbounded.astype(np.float64)

        lice_dist = {}
        ave_temp = self.farm.year_temperatures[cur_month - 1]

        # L4 -> L5
        # TODO these blocks look like the same?

        num_lice = self.lice_population["L4"]
        stage_ages = self.get_evolution_ages(num_lice, minimum_age=10, mean=15)

        l4_to_l5 = dev_times(self.cfg.delta_p["L4"], self.cfg.delta_m10["L4"], self.cfg.delta_s["L4"],
                             ave_temp, stage_ages)
        num_to_move = min(self.cfg.rng.poisson(np.sum(l4_to_l5)), num_lice)
        new_females = int(self.cfg.rng.choice([math.floor(num_to_move / 2.0), math.ceil(num_to_move / 2.0)]))
        new_males = (num_to_move - new_females)

        lice_dist["L5f"] = new_females
        lice_dist["L5m"] = (num_to_move - new_females)

        # L3 -> L4
        num_lice = self.lice_population["L3"]
        stage_ages = self.get_evolution_ages(num_lice, minimum_age=15, mean=18)
        l3_to_l4 = dev_times(self.cfg.delta_p["L3"], self.cfg.delta_m10["L3"], self.cfg.delta_s["L3"],
                             ave_temp, stage_ages)
        num_to_move = min(self.cfg.rng.poisson(np.sum(l3_to_l4)), num_lice)
        new_L4 = num_to_move

        lice_dist["L4"] = num_to_move

        # L2 -> L3
        # This is done in do_infection_events()

        # L1 -> L2
        num_lice = self.lice_population["L2"]
        stage_ages = self.get_evolution_ages(num_lice, minimum_age=3, mean=4)
        l1_to_l2 = dev_times(self.cfg.delta_p["L1"], self.cfg.delta_m10["L1"], self.cfg.delta_s["L1"],
                             ave_temp, stage_ages)
        num_to_move = min(self.cfg.rng.poisson(np.sum(l1_to_l2)), num_lice)
        new_L2 = num_to_move

        lice_dist["L2"] = num_to_move

        self.logger.debug("\t\t\tdistribution of new lice lifecycle stages on farm {}/cage {} = {}"
                          .format(self.farm_id, self.id, lice_dist))

        return new_L2, new_L4, new_females, new_males

    def get_fish_treatment_mortality(
            self,
            days_since_start: int,
            fish_lice_deaths: int,
            fish_backgroud_deaths: int) -> int:
        """
        Get fish mortality due to treatment. Mortality due to treatment is defined in terms of
        point percentage increases, thus we can only account for "excess deaths".

        :param days_since_start the number of days since the beginning of the simulation
        :param fish_lice_deaths the number
        """
        # See surveys/overton_treatment_mortalities.py for an explanation on what is going on
        mortality_events = fish_lice_deaths + fish_backgroud_deaths

        if mortality_events == 0:
            return 0
        if self.last_effective_treatment is None:
            return 0

        efficacy_window = self.last_effective_treatment.effectiveness_duration_days

        cur_date = self.start_date + dt.timedelta(days=days_since_start)
        temperature = self.farm.year_temperatures[cur_date.month-1]
        mortality_events_pp = 100 * mortality_events / self.num_fish
        fish_mass = self.average_fish_mass(days_since_start)

        last_treatment_params = self.cfg.get_treatment(self.last_effective_treatment.treatment_type)
        predicted_pp_increase = last_treatment_params.get_mortality_pp_increase(temperature, fish_mass)

        predicted_deaths = (predicted_pp_increase + mortality_events_pp) * self.num_fish / 100 - mortality_events
        predicted_deaths /= efficacy_window

        treatment_mortalities_occurrences = self.cfg.rng.poisson(predicted_deaths)

        return treatment_mortalities_occurrences

    def get_fish_growth(self, days_since_start) -> Tuple[int, int]:
        """
        Get the number of fish that get killed either naturally or by lice.
        :param days_since_start: the number of days since the beginning.
        :returns a tuple (natural_deaths, lice_induced_deaths, lice_deaths)
        """

        self.logger.debug("\t\tupdating fish population")

        def fb_mort(days):
            """
            Fish background mortality rate, decreasing as in Soares et al 2011

            Fish death rate: constant background daily rate 0.00057 based on
            www.gov.scot/Resource/0052/00524803.pdf
            multiplied by lice coefficient see surv.py (compare to threshold of 0.75 lice/g fish)

            TODO: what should we do with the formulae?

            :param days: number of days elapsed
            :return: fish background mortality rate
            """
            return 0.00057  # (1000 + (days - 700)**2)/490000000

        # Apply a sigmoid based on the number of lice per fish
        pathogenic_lice = sum([self.lice_population[stage] for stage in self.pathogenic_stages])
        if self.num_infected_fish > 0:
            lice_per_host_mass = pathogenic_lice / (self.num_infected_fish * self.average_fish_mass(days_since_start))
        else:
            lice_per_host_mass = 0.0
        prob_lice_death = 1 / (1 + math.exp(-self.cfg.fish_mortality_k *
                                            (lice_per_host_mass - self.cfg.fish_mortality_center)))

        ebf_death = fb_mort(days_since_start) * self.num_fish
        elf_death = self.num_infected_fish * prob_lice_death
        fish_deaths_natural = self.cfg.rng.poisson(ebf_death)
        fish_deaths_from_lice = self.cfg.rng.poisson(elf_death)

        self.logger.debug("\t\t\tnumber of background fish death {}, from lice {}"
                          .format(fish_deaths_natural, fish_deaths_from_lice))

        return fish_deaths_natural, fish_deaths_from_lice

    def compute_eta_aldrin(self, num_fish_in_farm, days):
        return self.cfg.infection_main_delta + math.log(num_fish_in_farm) + self.cfg.infection_weight_delta * \
               (math.log(self.average_fish_mass(days)) - self.cfg.delta_expectation_weight_log)

    def get_infection_rates(self, days) -> Tuple[float, int]:
        """
        Compute the number of lice that can infect and what their infection rate (number per fish) is

        :param days: the amount of time it takes
        :returns a pair (Einf, num_avail_lice)
        """
        # Perhaps we can have a distribution which can change per day (the mean/median increaseѕ?
        # but at what point does the distribution mean decrease)./
        age_distrib = self.get_stage_ages_distrib("L2")
        num_avail_lice = round(self.lice_population["L2"] * np.sum(age_distrib[1:]))
        if num_avail_lice > 0:
            num_fish_in_farm = self.farm.num_fish

            # TODO: this has O(c^2) complexity
            etas = np.array([c.compute_eta_aldrin(num_fish_in_farm, days) for c in self.farm.cages])
            Einf = math.exp(etas[self.id]) / (1 + np.sum(np.exp(etas)))

            return Einf, num_avail_lice

        return 0.0, num_avail_lice

    def do_infection_events(self, cur_date: dt.datetime, days: int) -> int:
        """Infect fish in this cage if the sea lice are in stage L2 and at least 1 day old

        :param cur_date: current date of simulation
        :param days: days the number of days elapsed
        :return: number of evolving lice, or equivalently the new number of infections
        """
        Einf, num_avail_lice = self.get_infection_rates(days)

        if Einf == 0:
            return 0

        expected_events = Einf * num_avail_lice

        inf_events = self.cfg.rng.poisson(expected_events)

        return min(inf_events, num_avail_lice)

    def get_infecting_population(self, *args) -> int:
        if args is None or len(args) == 0:
            infective_stages = ["L3", "L4", "L5m", "L5f"]
        else:
            infective_stages = list(args)
        return sum(self.lice_population[stage] for stage in infective_stages)

    def get_mean_infected_fish(self, *args) -> int:
        """
        Get the average number of infected fish
        :param *args the stages to consider (optional, by default all stages from the third onward are taken into account)
        :returns the number of infected fish
        """
        attached_lice = self.get_infecting_population(*args)

        # see: https://stats.stackexchange.com/a/296053
        num_infected_fish = int(self.num_fish * (1 - ((self.num_fish - 1) / self.num_fish) ** attached_lice))
        return num_infected_fish

    def get_variance_infected_fish(self, n: int, k: int) -> float:
        # Rationale: assuming that we generate N bins that sum to K, we can model this as a multinomial distribution
        # where all p_i are the same. Therefore, the variance of each bin is k*(n-1)/(n**2)
        # This is still incorrect but the error should be relatively small for now

        probs = np.full(n, 1 / n)
        bins = self.cfg.rng.multinomial(k, probs)

        return float(np.var(bins) * k)

    def get_num_matings(self) -> int:
        """
        Get the number of matings. Implement Cox's approach assuming an unbiased sex distribution
        """

        # Background: AF and AM are randomly assigned to fish according to a negative multinomial distribution.
        # What we want is determining what is the expected likelihood there is _at least_ one AF and _at
        # least_ one AM on the same fish.

        males = self.lice_population["L5m"]
        females = sum(self.lice_population.available_dams.values())

        if males == 0 or females == 0:
            return 0

        # VMR: variance-mean ratio; VMR = m/k + 1 -> k = m / (VMR - 1) with k being an "aggregation factor"
        mean = (males + females) / self.get_mean_infected_fish("L5m", "L5f")

        n = self.num_fish
        k = self.get_infecting_population("L5m", "L5f")
        variance = self.get_variance_infected_fish(n, k)
        vmr = variance / mean
        if vmr <= 1.0:
            return 0
        aggregation_factor = mean / (vmr - 1)

        prob_matching = 1 - (1 + males / ((males + females) * aggregation_factor)) ** (-1 - aggregation_factor)
        # TODO: using a poisson distribution can lead to a high std for high prob*females
        return int(np.clip(self.cfg.rng.poisson(prob_matching * females), np.int32(0), np.int32(min(males, females))))

    def do_mating_events(self) -> Tuple[GenericGenoDistrib, GenericGenoDistrib]:
        """
        will generate two deltas:  one to add to unavailable dams and subtract from available dams, one to add to eggs
        assume males don't become unavailable? in this case we don't need a delta for sires
        :returns a pair (delta_dams, new_eggs)

        TODO: deal properly with fewer individuals than matings required
        TODO: right now discrete mode is hard-coded, once we have quantitative egg generation implemented, need to add a switch
        TODO: current date is required by get_num_eggs as of now, but the relationship between extrusion and temperature is not clear
        """

        delta_eggs = GenericGenoDistrib()
        num_matings = self.get_num_matings()

        distrib_sire_available = self.lice_population.geno_by_lifestage["L5m"]
        distrib_dam_available = self.lice_population.available_dams

        delta_dams = self.select_dams(distrib_dam_available, num_matings)

        if sum(distrib_sire_available.values()) == 0 or sum(distrib_dam_available.values()) == 0:
            return GenericGenoDistrib(), GenericGenoDistrib()

        # TODO - need to add dealing with fewer males/females than the number of matings

        for dam_geno in delta_dams:
            for _ in range(int(delta_dams[dam_geno])):
                sire_geno = self.choose_from_distrib(distrib_sire_available)
                new_eggs = self.generate_eggs(sire_geno, dam_geno, num_matings)
                delta_eggs += new_eggs

        return delta_dams, delta_eggs

    def generate_eggs(
            self,
            sire: Union[Alleles, np.ndarray],
            dam: Union[Alleles, np.ndarray],
            num_matings: int
    ) -> GenericGenoDistrib:
        """
        Generate the eggs with a given genomic distribution

        TODO: doesn't do anything sensible re: integer/real numbers of offspring
        TODO: do we still want to keep the mechanism?
        :param sire the genomics of the sires
        :param dam the genomics of the dams
        :param num_matings the number of matings
        :returns a distribution on the number of generated eggs according to the distribution
        """

        number_eggs = self.get_num_eggs(num_matings)

        # TODO: since the genetic mechanism cannot change one could try to cache this

        if self.genetic_mechanism == GeneticMechanism.discrete:
            sire, dam = cast(Alleles, sire), cast(Alleles, dam)
            geno_eggs = self.generate_eggs_discrete(sire, dam, number_eggs)

        elif self.genetic_mechanism == GeneticMechanism.quantitative:
            sire, dam = cast(np.ndarray, sire), cast(np.ndarray, dam)
            geno_eggs = self.generate_eggs_quantitative(sire, dam, number_eggs)

        elif self.genetic_mechanism == GeneticMechanism.maternal:
            geno_eggs = self.generate_eggs_maternal(dam, number_eggs)

        else:
            raise Exception("Genetic mechanism must be 'maternal', 'quantitative' or 'discrete' - '{}' given".format(
                self.genetic_mechanism))

        if self.genetic_mechanism != GeneticMechanism.quantitative:
            self.mutate(geno_eggs, mutation_rate=self.cfg.geno_mutation_rate)
        return geno_eggs

    def generate_eggs_discrete(self, sire: Alleles, dam: Alleles, number_eggs: int) -> GenoDistrib:
        """Get number of eggs based on discrete genetic mechanism.

        If we're in the discrete 2-gene setting, assume for now that genotypes are tuples -
        so in a A/a genetic system, genotypes could be ('A'), ('a'), or ('A', 'a')
        right now number of offspring with each genotype are deterministic, and might be
        missing one (we should update to add jitter in future, but this is a reasonable approx)

        :param sire: the genomics of the sires
        :param dam: the genomics of the dams
        :param number_eggs: the number of eggs produced
        :return: genomics distribution of eggs produced
        """

        eggs_generated = GenoDistrib()
        if len(sire) == 1 and len(dam) == 1:
            eggs_generated[self.get_geno_name(sire[0], dam[0])] = float(number_eggs)
        elif len(sire) == 2 and len(dam) == 1:
            eggs_generated[self.get_geno_name(sire[0], dam[0])] = number_eggs / 2
            eggs_generated[self.get_geno_name(sire[1], dam[0])] = number_eggs / 2
        elif len(sire) == 1 and len(dam) == 2:
            eggs_generated[self.get_geno_name(sire[0], dam[0])] = number_eggs / 2
            eggs_generated[self.get_geno_name(sire[0], dam[1])] = number_eggs / 2
        else:
            # drawing both from the sire in the first case ensures heterozygotes
            # but is a bit hacky.
            eggs_generated[self.get_geno_name(sire[0], sire[1])] = number_eggs / 2
            # and the below gets us our two types of homozygotes
            eggs_generated[self.get_geno_name(sire[0], dam[0])] = number_eggs / 4
            eggs_generated[self.get_geno_name(sire[1], dam[1])] = number_eggs / 4

        return eggs_generated

    def get_geno_name(self, sire_geno: Allele, dam_geno: Allele) -> Alleles:
        """Create name of the genotype based on parents alleles.

        :param sire_geno: the allele of the sires
        :param dam_geno: the allele of the sires
        :return: the genomics of the offspring
        """
        return tuple(sorted({sire_geno, dam_geno}))

    def generate_eggs_quantitative(self, sire: np.ndarray, dam: np.ndarray, number_eggs: int) -> QuantitativeGenoDistrib:
        """Get number of eggs based on quantitative genetic mechanism.

        Additive genes, assume genetic state for an individual looks like a number between 0 and 1.
        because we're only dealing with the heritable part here don't need to do any of the comparison
        to population mean or impact of heritability, etc - that would appear in the code dealing with treatment
        so we could use just the mid-parent value for this genetic recording for children
        as with the discrete genetic model, this will be deterministic for now

        :param sire: the genomics of the sires
        :param dam: the genomics of the dams
        :param number_eggs: the number of eggs produced
        :return: genomics distribution of eggs produced
        """
        mid_parent = float(np.round((sire + dam) / 2, 1).item())
        return QuantitativeGenoDistrib({mid_parent: number_eggs})

    @staticmethod
    def generate_eggs_maternal(dam: Union[Alleles, np.ndarray], number_eggs: int) -> GenoDistrib:
        """Get number of eggs based on maternal genetic mechanism.

        Maternal-only inheritance - all eggs have mother's genotype.

        :param dam: the genomics of the dams
        :param number_eggs: the number of eggs produced
        :return: genomics distribution of eggs produced
        """
        return GenoDistrib({dam: number_eggs})

    def mutate(self, eggs: GenoDistrib, mutation_rate: float):
        """
        Mutate the genotype distribution

        :param eggs: the genotype distribution of the newly produced eggs
        :param mutation_rate: the rate of mutation with respect to the number of eggs.
        """
        if mutation_rate == 0:
            return

        mutations = self.cfg.rng.poisson(mutation_rate * sum(eggs.values()))

        # generate a "swap" matrix
        # rationale: since ('a',) actually represents a pair of genes ('a', 'a')
        # there are only three directions: R->ID, ID->D, ID->R, D->ID. Note that
        # R->D or D->R are impossible via a single mutation.
        # Self-mutations are ignored.
        # To model this, we create a "masking" swapping matrix and force to 0 masked entries
        # and make sure they cannot be selected.

        alleles = [('a',), ('A',), ('A', 'a')]
        n = len(alleles)
        mask_matrix = np.array([[0, 0, 1],
                                [0, 0, 1],
                                [1, 1, 0]])

        p = mask_matrix.flatten() / np.sum(mask_matrix)

        swap_matrix = self.cfg.rng.multinomial(mutations, p).reshape(n, n)

        for idx, allele in enumerate(alleles):
            if allele not in eggs:
                eggs[allele] = 0
            eggs[allele] += np.sum(swap_matrix[idx, :]) - np.sum(swap_matrix[:, idx])
            if eggs[allele] == 0:
                del eggs[allele]

    def select_dams(self, distrib_dams_available: GenoDistrib, num_dams: int) -> GenoDistrib:
        """
        From a geno distribution of eligible dams sample a given number of dams.

        :param distrib_dams_available: the starting dam genomic distribution
        :param num_dams: the wished number of dams to sample
        """
        # TODO: this function is flexible enough to be renamed and used elsewhere
        delta_dams_selected = GenoDistrib()
        copy_dams_avail = copy.deepcopy(distrib_dams_available)
        if sum(distrib_dams_available.values()) <= num_dams:
            return copy_dams_avail

        # TODO: make choose_from_distrib sample N elements without replacement at once rather than looping
        # Use the same trick used in treatment mortality, or otherwise use a hypergeometric distrib?

        for _ in range(num_dams):
            this_dam = self.choose_from_distrib(copy_dams_avail)
            copy_dams_avail[this_dam] -= 1
            if this_dam not in delta_dams_selected:
                delta_dams_selected[this_dam] = 0
            delta_dams_selected[this_dam] += 1

        return delta_dams_selected

    def choose_from_distrib(self, distrib: GenoDistrib) -> Alleles:
        distrib_values = np.array(list(distrib.values()))
        keys = list(distrib.keys())
        choice_ix = self.cfg.rng.choice(range(len(keys)), p=distrib_values / np.sum(distrib_values))

        return tuple(keys[choice_ix])

    def get_num_eggs(self, mated_females) -> int:
        """
        Get the number of new eggs
        :param mated_females the number of mated females that reproduce
        :returns the number of eggs produced
        """

        # See Aldrin et al. 2017, §2.2.6
        age_distrib = self.get_stage_ages_distrib("L5f")
        age_range = np.arange(1, len(age_distrib) + 1)

        # TODO: keep track of free females before calculating mating
        mated_females_distrib = mated_females * age_distrib
        eggs = self.cfg.reproduction_eggs_first_extruded * \
               (age_range ** self.cfg.reproduction_age_dependence) * mated_females_distrib

        return self.cfg.rng.poisson(np.round(np.sum(eggs)))
        # TODO: We are deprecating this. Need to investigate if temperature data is useful. See #46
        # ave_temp = self.farm.year_temperatures[cur_month - 1]
        # temperature_factor = self.cfg.delta_m10["L0"] * (10 / ave_temp) ** self.cfg.delta_p["L0"]

        # reproduction_rates = self.cfg.reproduction_eggs_first_extruded * \
        #                     (age_range ** self.cfg.reproduction_age_dependence) / (temperature_factor + 1)

        # return self.cfg.rng.poisson(np.round(np.sum(reproduction_rates * mated_females_distrib)))

    def get_egg_batch(self, cur_date: dt.datetime, egg_distrib: GenoDistrib) -> EggBatch:
        """
        Get the expected arrival date of an egg batch
        :param cur_date the current time
        :param egg_distrib the egg distribution
        :returns EggBatch representing egg distribution and expected hatching date
        """

        # We use Stien et al (2005)'s regressed formula
        # τE= [β1/(T – 10 + β1β2)]**2  (see equation 8)
        # where β2**(-2) is the average temperature centered at around 10 degrees
        # and β1 is a shaping factor. This function is formally known as Belehrádek’s function
        cur_month = cur_date.month
        ave_temp = self.farm.year_temperatures[cur_month - 1]

        beta_1 = self.cfg.delta_m10["L0"]
        beta_2 = self.cfg.delta_p["L0"]
        expected_time = (self.cfg.delta_m10["L0"] / (ave_temp - 10 + beta_1 * beta_2)) ** 2
        expected_hatching_date = cur_date + dt.timedelta(self.cfg.rng.poisson(expected_time))
        return EggBatch(expected_hatching_date, egg_distrib)

    def create_offspring(self, cur_time: dt.datetime) -> GenoDistrib:
        """
        Hatch the eggs from the event queue
        :param cur_time the current time
        :returns a delta egg genomics
        """

        delta_egg_offspring = GenoDistrib({geno: 0 for geno in self.cfg.genetic_ratios})

        def cts(hatching_event: EggBatch):
            nonlocal delta_egg_offspring
            delta_egg_offspring += hatching_event.geno_distrib

        pop_from_queue(self.hatching_events, cur_time, cts)
        return delta_egg_offspring

    def get_arrivals(self, cur_date: dt.datetime) -> GenoDistrib:
        """Process the arrivals queue.
        :param cur_date: Current date of simulation
        :return: Genotype distribution of eggs hatched in travel
        """

        hatched_dist = GenoDistrib()

        # check queue for arrivals at current date
        def cts(batch: TravellingEggBatch):
            nonlocal self
            nonlocal hatched_dist

            # if the hatch date is after current date, add to stationary egg queue
            if batch.hatch_date >= cur_date:
                stationary_batch = EggBatch(batch.hatch_date, batch.geno_distrib)
                self.hatching_events.put(stationary_batch)

            # otherwise the egg has hatched during travel, so determine the life stage
            # and add to population
            else:
                # TODO: determine life stage it arrives at; assumes all are L1 for now
                hatched_dist = batch.geno_distrib

        pop_from_queue(self.arrival_events, cur_date, cts)

        return hatched_dist

    def get_dying_lice_from_dead_fish(self, num_dead_fish: int) -> GrossLiceDistrib:
        """
        Get the number of lice that die when fish die.
        :param num_dead_fish: the number of dead fish
        :returns a gross distribution
        """

        # Note: no paper actually provides a clear guidance on this.
        # This is mere speculation.
        # Basically, only consider PA and A males (for L4m = 0.5*L4)
        # as potentially able to survive and find a new host.
        # Only a fixed proportion can survive.

        if self.get_infecting_population() == 0 or self.num_infected_fish == 0:
            return {}

        affected_lice_gross = round(self.get_infecting_population() *
                                    num_dead_fish / self.num_infected_fish)
        infecting_lice = self.get_infecting_population()

        affected_lice = {stage: self.lice_population[stage] / infecting_lice * affected_lice_gross
                         for stage in self.susceptible_stages}

        affected_lice = Counter(iteround.saferound(affected_lice, 0))

        surviving_lice = Counter(iteround.saferound({
            'L4': self.lice_population['L4'] / (2*infecting_lice) *
                  affected_lice_gross * self.cfg.male_detachment_rate,
            'L5m': self.lice_population['L5m'] / infecting_lice *
                   affected_lice_gross * self.cfg.male_detachment_rate,
        }, 0))  # type: Counter[str]

        dying_lice = affected_lice - surviving_lice

        return {k: int(v) for k, v in dying_lice.items() if v > 0}

    def promote_population(
            self,
            prev_stage: Union[str, dict],
            cur_stage: str,
            leaving_lice: int,
            entering_lice: Optional[int] = None
    ):
        """
        Promote the population by stage and respect the genotypes
        :param prev_stage the lice stage from which cur_stage evolves
        :param cur_stage the lice stage that is about to evolve
        :param leaving_lice the number of lice in the cur_stage=>next_stage progression
        :param entering_lice the number of lice in the prev_stage=>cur_stage progression.
               If prev_stage is a a string, entering_lice must be an int
        """
        if isinstance(prev_stage, str):
            if entering_lice is not None:
                prev_stage_geno = self.lice_population.geno_by_lifestage[prev_stage]
                entering_geno_distrib = prev_stage_geno.normalise_to(entering_lice)
            else:
                raise ValueError("entering_lice must be an int when prev_stage is a str")
        else:
            entering_geno_distrib = prev_stage
        cur_stage_geno = self.lice_population.geno_by_lifestage[cur_stage]

        leaving_geno_distrib = cur_stage_geno.normalise_to(leaving_lice)
        cur_stage_geno = cur_stage_geno + entering_geno_distrib - leaving_geno_distrib

        # update gross population. This is a bit hairy but I cannot think of anything simpler.
        self.lice_population.geno_by_lifestage[cur_stage] = cur_stage_geno

    def update_deltas(
            self,
            dead_lice_dist: GrossLiceDistrib,
            treatment_mortality: GenoLifeStageDistrib,
            fish_deaths_natural: int,
            fish_deaths_from_lice: int,
            fish_deaths_from_treatment: int,
            new_L2: int,
            new_L4: int,
            new_females: int,
            new_males: int,
            new_infections: int,
            lice_from_reservoir: GrossLiceDistrib,
            delta_dams_batch: OptionalDamBatch,
            new_offspring_distrib: GenericGenoDistrib,
            hatched_arrivals_dist: GenericGenoDistrib
    ):
        """Update the number of fish and the lice in each life stage
        :param dead_lice_dist the number of dead lice due to background death (as a distribution)
        :param treatment_mortality the distribution of genotypes being affected by treatment
        :param fish_deaths_natural the number of natural fish death events
        :param fish_deaths_from_lice the number of lice-induced fish death events
        :param fish_deaths_from_treatment the number of treatment-induced fish death events
        :param new_L2 number of new L2 fish
        :param new_L4 number of new L4 fish
        :param new_females number of new adult females
        :param new_males number of new adult males
        :param new_infections the number of new infections (i.e. progressions from L2 to L3)
        :param lice_from_reservoir the number of lice taken from the reservoir
        :param delta_dams_batch the genotypes of now-unavailable females in batch events
        :param new_offspring_distrib the new offspring obtained from hatching and migrations
        :param hatched_arrivals_dist: new offspring obtained from arrivals
        """

        # Update dead_lice_dist to include fish-caused death as well
        dead_lice_by_fish_death = self.get_dying_lice_from_dead_fish(
            fish_deaths_natural + fish_deaths_from_lice)
        for affected_stage, reduction in dead_lice_by_fish_death.items():
            dead_lice_dist[affected_stage] += reduction

        for stage in self.lice_population:
            # update background mortality
            bg_delta = self.lice_population[stage] - dead_lice_dist[stage]
            self.lice_population[stage] = max(0, bg_delta)

            # update population due to treatment
            # TODO: __isub__ here is broken
            self.lice_population.geno_by_lifestage[stage] = self.lice_population.geno_by_lifestage[stage] - treatment_mortality[stage]

        self.lice_population.remove_negatives()

        self.promote_population("L4", "L5m", 0, new_males)
        self.promote_population("L4", "L5f", 0, new_females)
        self.promote_population("L3", "L4", new_males + new_females, new_L4)
        self.promote_population("L2", "L3", new_L4, new_infections)
        self.promote_population("L1", "L2", new_infections, new_L2 + lice_from_reservoir["L2"])

        self.promote_population(new_offspring_distrib, "L1", new_L2, None)
        self.promote_population(hatched_arrivals_dist, "L1", 0, None)

        self.lice_population.remove_negatives()

        # in absence of wildlife genotype, simply upgrade accordingly
        # TODO: switch to generic genotype distribs?
        self.lice_population["L1"] += lice_from_reservoir["L1"]

        if delta_dams_batch:
            self.lice_population.add_busy_dams_batch(delta_dams_batch)

        self.num_fish -= (fish_deaths_natural + fish_deaths_from_lice + fish_deaths_from_treatment)
        if self.num_fish < 0:
            self.num_fish = 0

        # treatment may kill some lice attached to the fish, thus update at the very end
        self.num_infected_fish = self.get_mean_infected_fish()

    # TODO: update arrivals dict type
    def update_arrivals(self, arrivals_dict: GenoDistribByHatchDate, arrival_date: dt.datetime):
        """Update the arrivals queue

        :param arrivals_dict: List of dictionaries of genotype distributions based on hatch date
        :param arrival_date: Arrival date at this cage
        """

        for hatch_date in arrivals_dict:

            # skip if there are no eggs in the dictionary
            if sum(arrivals_dict[hatch_date].values()) == 0:
                continue

            # create new travelling batch and update the queue
            batch = TravellingEggBatch(arrival_date, hatch_date, arrivals_dict[hatch_date])
            self.arrival_events.put(batch)

    def get_background_lice_mortality(self) -> GrossLiceDistrib:
        """
        Background death in a stage (remove entry) -> rate = number of
        individuals in stage*stage rate (nauplii 0.17/d, copepods 0.22/d,
        pre-adult female 0.05, pre-adult male ... Stien et al 2005)

        :returns the current background mortality. The return value is genotype-agnostic
        """
        lice_mortality_rates = self.cfg.background_lice_mortality_rates
        lice_population = self.lice_population

        dead_lice_dist = {}
        for stage in lice_population:
            mortality_rate = lice_population[stage] * lice_mortality_rates[stage]
            mortality = min(self.cfg.rng.poisson(mortality_rate), lice_population[stage])  # type: int
            dead_lice_dist[stage] = mortality

        self.logger.debug("\t\tbackground mortality distribution of dead lice = {}".format(dead_lice_dist))
        return dead_lice_dist

    def average_fish_mass(self, days):
        """
        Average fish mass.
        """
        smolt_params = self.cfg.smolt_mass_params
        return smolt_params.max_mass / (1 + math.exp(smolt_params.skewness * (days - smolt_params.x_shift)))

    def get_reservoir_lice(self, pressure: int) -> GrossLiceDistrib:
        """Get distribution of lice coming from the reservoir

        :param pressure: External pressure
        :return: Distribution of lice in L1 and L2
        """

        if pressure == 0:
            return {"L1": 0, "L2": 0}

        num_L1 = self.cfg.rng.integers(low=0, high=pressure, size=1)[0]
        new_lice_dist = {"L1": num_L1, "L2": pressure - num_L1}
        self.logger.debug("\t\tdistribution of new lice from reservoir = {}".format(new_lice_dist))
        return new_lice_dist

    def fallow(self):
        """Put the cage in a fallowing state.

        Implications:
        1. All the fish would be removed.
        2. L3/L4/L5 would therefore disappear as they are attached to fish.
        3. Dam waiting and treatment queues will be flushed.
        """

        for stage in self.pathogenic_stages:
            self.lice_population[stage] = 0

        self.num_infected_fish = self.num_fish = 0
        self.treatment_events = PriorityQueue()

    @property
    def is_fallowing(self):
        return self.num_fish == 0

    def is_treated(self, cur_date):
        if self.last_effective_treatment is not None:
            treatment = self.last_effective_treatment
            if cur_date <= treatment.treatment_window:
                return True

        return False

    @property
    def aggregation_rate(self):
        """The aggregation rate is the number of lice over the total number of fish.
        Elsewhere it is referred to as infection rate, but here "infection rate" only refers to host fish.

        :returns the aggregation rate"""
        return sum(self.lice_population.values())/self.num_fish if self.num_fish > 0 else 0.0
