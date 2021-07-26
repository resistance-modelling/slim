from __future__ import annotations

import copy
import datetime as dt
import json
import logging
import math
from functools import singledispatch
from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Union, Optional, Dict, Tuple, cast, MutableMapping, TYPE_CHECKING, NamedTuple

import numpy as np
from scipy import stats

from src.Config import Config, Treatment, GeneticMechanism, HeterozygousResistance

if TYPE_CHECKING:
    from src.Farm import Farm

# TYPE ANNOTATIONS
# TODO: move these annotations to another file?

LifeStage = str
Allele = str
Alleles = Tuple[Allele, ...]
GenoDistrib = Dict[Alleles, Union[int, float]]
GenoLifeStageDistrib = Dict[LifeStage, GenoDistrib]
GrossLiceDistrib = Dict[LifeStage, int]

class GenoTreatmentValue(NamedTuple):
    mortality_rate: float
    pheno_emb: np.ndarray
    num_susc: int

GenoTreatmentDistrib = Dict[Alleles, GenoTreatmentValue]



@dataclass(order=True)
class EggBatch:
    hatch_date: dt.datetime
    geno_distrib: GenoDistrib = field(compare=False)


@dataclass(order=True)
class TravellingEggBatch:
    arrival_date: dt.datetime
    hatch_date: dt.datetime = field(compare=False)
    geno_distrib: GenoDistrib = field(compare=False)


@dataclass(order=True)
class DamAvailabilityBatch:
    availability_date: dt.datetime # expected return time
    geno_distrib: dict = field(compare=False)

# See https://stackoverflow.com/a/7760938
class LicePopulation(dict, MutableMapping[LifeStage, int]):
    """
    Wrapper to keep the global population and genotype information updated
    This is definitely a convoluted way to do this, but I wanted to memoise as much as possible.
    """
    def __init__(self, initial_population: GrossLiceDistrib, geno_data: GenoLifeStageDistrib, logger: logging.Logger):
        super().__init__()
        self.geno_by_lifestage = GenotypePopulation(self, geno_data)
        self._available_dams = copy.deepcopy(self.geno_by_lifestage["L5f"])
        self.logger = logger
        for k, v in initial_population.items():
            super().__setitem__(k, v)

    def __setitem__(self, stage: LifeStage, value: int):
        # If one attempts to make gross modifications to the population these will be repartitioned according to the
        # current genotype information.
        if sum(self.geno_by_lifestage[stage].values()) == 0:
            self.logger.warning(f"Trying to initialise population {stage} with null genotype distribution. Using default genotype information.")
            self.geno_by_lifestage.raw_update_value(stage, Cage.multiply_distrib(Cage.generic_discrete_props, value))
        else:
            self.geno_by_lifestage.raw_update_value(stage, Cage.multiply_distrib(self.geno_by_lifestage[stage], value))
        if stage == "L5f":
            self._available_dams = Cage.multiply_distrib(self._available_dams, value)
        super().__setitem__(stage, value)

    def raw_update_value(self, stage: LifeStage, value: int):
        super().__setitem__(stage, value)

    @property
    def available_dams(self):
        return self._available_dams

    @available_dams.setter
    def available_dams(self, new_value: GenoDistrib):
        for geno in new_value:
            assert self.geno_by_lifestage["L5f"][geno] >= new_value[geno], \
                f"current population geno {geno}:{self.geno_by_lifestage['L5f'][geno]} is smaller than new value geno {new_value[geno]}"

        self._available_dams = new_value


class GenotypePopulation(dict, MutableMapping[LifeStage, GenoDistrib]):
    def __init__(self, gross_lice_population: LicePopulation, geno_data: GenoLifeStageDistrib):
        super().__init__()
        self._lice_population = gross_lice_population
        for k, v in geno_data.items():
            super().__setitem__(k, v)

    def __setitem__(self, stage: LifeStage, value: GenoDistrib):
        # update the value and the gross population accordingly
        super().__setitem__(stage, value)
        self._lice_population.raw_update_value(stage, sum(value.values()))

    def raw_update_value(self, stage: LifeStage, value: GenoDistrib):
        super().__setitem__(stage, value)


class Cage:
    """
    Fish cages contain the fish.
    """

    lice_stages = ["L1", "L2", "L3", "L4", "L5f", "L5m"]
    susceptible_stages = lice_stages[2:]

    generic_discrete_props = {('A',): 0.25, ('a',): 0.25, ('A', 'a'): 0.5}  # type: GenoDistrib

    # TODO: annotating the farm here causes import issues
    def __init__(self, cage_id: int, cfg: Config, farm: Farm):
        """
        Create a cage on a farm
        :param cage_id: the label (id) of the cage within the farm
        :param cfg the farm configuration
        :param farm a Farm object
        """

        self.cfg = cfg
        self.logger = cfg.logger
        self.id = cage_id

        self.farm_id = farm.name
        self.start_date = cfg.farms[self.farm_id].cages_start[cage_id]
        self.date = cfg.start_date

        # TODO: update with calculations
        lice_population = {"L1": cfg.ext_pressure, "L2": 0, "L3": 30, "L4": 30, "L5f": 10, "L5m": 10}   # type: GrossLiceDistrib

        self.farm = farm

        self.egg_genotypes = {}  # type: GenoDistrib
        # TODO/Question: what's the best way to deal with having multiple possible genetic schemes?
        # TODO/Question: I suppose some of this initial genotype information ought to come from the config file
        # TODO/Question: the genetic mechanism will be the same for all lice in a simulation, so should it live in the driver?
        # for now I've hard-coded in one mechanism in this setup, and a particular genotype starting point. Should probably be from a config file?
        self.genetic_mechanism = self.cfg.genetic_mechanism

        geno_by_lifestage = {stage: self.multiply_distrib(self.generic_discrete_props, lice_population[stage])
                             for stage in lice_population}

        self.lice_population = LicePopulation(lice_population, geno_by_lifestage, self.logger)

        self.num_fish = cfg.farms[self.farm_id].num_fish
        self.num_infected_fish = self.get_mean_infected_fish()

        self.hatching_events = PriorityQueue()  # type: PriorityQueue[EggBatch]
        self.busy_dams = PriorityQueue()  # type: PriorityQueue[DamAvailabilityBatch]
        self.arrival_events = PriorityQueue()  # type: PriorityQueue[TravellingEggBatch]

    def __str__(self):
        """
        Get a human readable string representation of the cage in json form.
        :return: a description of the cage
        """

        filtered_vars = vars(self).copy()
        del filtered_vars["farm"]
        del filtered_vars["logger"]
        del filtered_vars["cfg"]
        for k in filtered_vars:
            if isinstance(filtered_vars[k], dt.datetime):
                filtered_vars[k] = filtered_vars[k].strftime("%Y-%m-%d %H:%M:%S")

        # May want to improve these or change them if we change the representation for genotype distribs
        filtered_vars["egg_genotypes"] = {str(key): val for key, val in filtered_vars["egg_genotypes"].items()}
        filtered_vars["geno_by_lifestage"] = {str(key): str(val) for key, val in self.lice_population.geno_by_lifestage.items()}
        filtered_vars["hatching_events"] = sorted(list(self.hatching_events.queue))
        filtered_vars["busy_dams"] = sorted(list(self.busy_dams.queue))
        filtered_vars["arrival_events"] = sorted(list(self.arrival_events.queue))

        return json.dumps(filtered_vars, indent=4)

    def get_empty_geno_distrib(self) -> GenoDistrib:
        # A little factory method to get empty genos
        genos = self.lice_population.geno_by_lifestage["L1"].keys()
        return {geno: 0 for geno in genos}

    def update(self, cur_date: dt.datetime, step_size: int, pressure: int) -> tuple:
        """Update the cage at the current time step.
an
        :param cur_date: Current date of simulation
        :param step_size: Step size
        :param pressure: External pressure, planctonic lice coming from
        the reservoir
        :return: Tuple consisting of egg genotype distribution and hatching
        date
        """

        self.logger.debug(f"\tUpdating farm {self.farm_id} / cage {self.id}")
        self.logger.debug(f"\t\tinitial lice population = {self.lice_population}")
        self.logger.debug(f"\t\tinitial fish population = {self.num_fish}")

        days_since_start = (cur_date - self.date).days

        # Background lice mortality events
        dead_lice_dist = self.get_background_lice_mortality(self.lice_population)

        # Treatment mortality events
        treatment_mortality = self.get_lice_treatment_mortality_old(cur_date)

        # Development events
        new_L2, new_L4, new_females, new_males = self.get_lice_lifestage(cur_date.month)

        # Fish growth and death
        fish_deaths_natural, fish_deaths_from_lice = self.get_fish_growth(days_since_start, step_size)

        # Infection events
        num_infection_events = self.do_infection_events(days_since_start)

        # Mating events that create eggs
        delta_avail_dams, delta_eggs = self.do_mating_events()
        avail_dams_batch = DamAvailabilityBatch(cur_date + dt.timedelta(days=self.cfg.dam_unavailability),
                                                delta_avail_dams)

        new_egg_batch = self.get_egg_batch(cur_date, delta_eggs)

        # Lice coming from other cages and farms
        # NOTE: arrivals first then hatching
        hatched_arrivals_dist = self.get_arrivals(cur_date)

        # Egg hatching
        new_offspring_distrib = self.create_offspring(cur_date)

        # Restore lice availability
        returned_dams = self.free_dams(cur_date)

        # Lice coming from reservoir
        lice_from_reservoir = self.get_reservoir_lice(pressure)

        self.update_deltas(
            dead_lice_dist,
            treatment_mortality,
            fish_deaths_natural,
            fish_deaths_from_lice,
            new_L2,
            new_L4,
            new_females,
            new_males,
            num_infection_events,
            lice_from_reservoir,
            avail_dams_batch,
            new_egg_batch,
            new_offspring_distrib,
            returned_dams,
            hatched_arrivals_dist
        )

        self.logger.debug("\t\tfinal fish population = {}".format(self.num_fish))
        self.logger.debug("\t\tfinal lice population= {}".format(self.lice_population))
        egg_distrib = new_egg_batch.geno_distrib
        hatch_date = new_egg_batch.hatch_date
        self.logger.debug("\t\tlice offspring = {}".format(sum(egg_distrib.values())))

        return egg_distrib, hatch_date

    def get_lice_treatment_mortality_rate(self, cur_date: dt.datetime) -> GenoTreatmentDistrib:
        num_susc_per_geno = {}

        for stage in self.susceptible_stages:
            self.update_distrib_discrete_add(self.lice_population.geno_by_lifestage[stage], num_susc_per_geno)

        geno_treatment_distrib = {geno: (0.0, np.array([]), 0) for geno in num_susc_per_geno}

        treatment = self.farm.farm_cfg.treatment_type

        # TODO: replace this with an enum
        if treatment == Treatment.emb:

            # TODO: take temperatures into account? See #22
            # NOTE: some treatments (e.g. H2O2) are temperature-independent
            if cur_date - dt.timedelta(days=self.cfg.delay_EMB) in self.cfg.farms[self.farm_id].treatment_dates:
                self.logger.debug("\t\ttreating farm {}/cage {} on date {}".format(self.farm_id,
                                                                                   self.id, cur_date))

                # For now, assume a simple heterozygous distribution with a mere linear resistance factor
                for geno, num_susc in num_susc_per_geno.items():
                    if 'A' in geno:
                        if 'a' in geno:
                            trait = HeterozygousResistance.incompletely_dominant
                        else:
                            trait = HeterozygousResistance.dominant
                    else:
                        trait = HeterozygousResistance.recessive

                    susceptibility_factor = 1.0 - self.cfg.pheno_resistance[treatment][trait]
                    # model the resistance of each lice in the susceptible stages (phenoEMB) and estimate
                    # the mortality rate due to treatment (ETmort).
                    f_meanEMB = self.cfg.f_meanEMB * susceptibility_factor
                    f_sigEMB = self.cfg.f_sigEMB * susceptibility_factor
                    pheno_emb = self.cfg.rng.normal(f_meanEMB, f_sigEMB, num_susc) \
                                + self.cfg.rng.normal(self.cfg.env_meanEMB, self.cfg.env_sigEMB, num_susc)
                    pheno_emb = 1 / (1 + np.exp(pheno_emb))
                    mortality_rate = sum(pheno_emb) * self.cfg.EMBmort
                    geno_treatment_distrib[geno] = (mortality_rate, pheno_emb, num_susc)
        else:
            raise NotImplemented("Only EMB treatment is supported")

        return geno_treatment_distrib


    def get_lice_treatment_mortality_rate_old(self, cur_date):
        """
        Compute the mortality rate due to chemotherapeutic treatment (See Aldrin et al, 2017, §2.2.3)
        """
        num_susc = sum(self.lice_population[x] for x in self.susceptible_stages)

        # TODO: take temperatures into account? See #22
        if cur_date - dt.timedelta(days=self.cfg.delay_EMB) in self.cfg.farms[self.farm_id].treatment_dates:
            self.logger.debug("\t\ttreating farm {}/cage {} on date {}".format(self.farm_id,
                                                                               self.id, cur_date))
            # number of lice in those stages that are susceptible to Emamectin Benzoate (i.e.
            # those L3 or above)
            # we assume the mortality rate is the same across all stages and ages, but this may change in the future
            # (or with different chemicals)

            # model the resistance of each lice in the susceptible stages (phenoEMB) and estimate
            # the mortality rate due to treatment (ETmort).
            pheno_emb = self.cfg.rng.normal(self.cfg.f_meanEMB, self.cfg.f_sigEMB, num_susc) \
                        + self.cfg.rng.normal(self.cfg.env_meanEMB, self.cfg.env_sigEMB, num_susc)
            pheno_emb = 1 / (1 + np.exp(pheno_emb))
            mortality_rate = sum(pheno_emb) * self.cfg.EMBmort
            return mortality_rate, pheno_emb, num_susc

        else:
            return 0, 0, num_susc

    def get_lice_treatment_mortality_old(self, cur_date):
        """
        Calculate the number of lice in each stage killed by treatment.
        """

        dead_lice_dist = {stage: 0 for stage in self.lice_stages}

        mortality_rate, pheno_emb, num_susc = self.get_lice_treatment_mortality_rate_old(cur_date)

        if mortality_rate > 0:
            num_dead_lice = self.cfg.rng.poisson(mortality_rate)
            num_dead_lice = min(num_dead_lice, num_susc)

            # Now we need to decide how many lice from each stage die,
            #   the algorithm is to label each louse  1...num_susc
            #   assign each of these a probability of dying as (phenoEMB)/np.sum(phenoEMB)
            #   randomly pick lice according to this probability distribution
            #        now we need to find out which stages these lice are in by calculating the
            #        cumulative sum of the lice in each stage and finding out how many of our
            #        dead lice falls into this bin.
            p = (pheno_emb) / np.sum(pheno_emb)
            dead_lice = self.cfg.rng.choice(range(num_susc), num_dead_lice, p=p,
                                         replace=False).tolist()
            total_so_far = 0
            for stage in self.susceptible_stages:
                num_dead = len([x for x in dead_lice if total_so_far <= x <
                                (total_so_far + self.lice_population[stage])])
                total_so_far += self.lice_population[stage]
                if num_dead > 0:
                    dead_lice_dist[stage] = num_dead

            self.logger.debug("\t\tdistribution of dead lice on farm {}/cage {} = {}"
                              .format(self.farm_id, self.id, dead_lice_dist))

            assert num_dead_lice == sum(list(dead_lice_dist.values()))
        return dead_lice_dist

    def get_lice_treatment_mortality(self, cur_date) -> GenoLifeStageDistrib:
        """
        Calculate the number of lice in each stage killed by treatment.
        """

        dead_lice_dist = {stage: self.get_empty_geno_distrib() for stage in self.lice_stages}

        dead_mortality_distrib = self.get_lice_treatment_mortality_rate(cur_date)

        for geno, (mortality_rate, pheno_emb, num_susc) in dead_mortality_distrib.items():
            if mortality_rate > 0:
                num_dead_lice = self.cfg.rng.poisson(mortality_rate)
                num_dead_lice = min(num_dead_lice, num_susc)

                # Now we need to decide how many lice from each stage die,
                #   the algorithm is to label each louse  1...num_susc
                #   assign each of these a probability of dying as (phenoEMB)/np.sum(phenoEMB)
                #   randomly pick lice according to this probability distribution
                #        now we need to find out which stages these lice are in by calculating the
                #        cumulative sum of the lice in each stage and finding out how many of our
                #        dead lice falls into this bin.
                p = (pheno_emb) / np.sum(pheno_emb)
                dead_lice = self.cfg.rng.choice(range(num_susc), num_dead_lice, p=p,
                                             replace=False).tolist()
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

            return dead_lice_dist


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

    def get_lice_lifestage(self, cur_month):
        """
        Move lice between lifecycle stages.
        See Section 2.1 of Aldrin et al. (2017)
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
        new_females = self.cfg.rng.choice([math.floor(num_to_move / 2.0), math.ceil(num_to_move / 2.0)])
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

    def get_fish_growth(self, days, step_size):
        """
        Get the new number of fish after a step size.
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
        pathogenic_lice = sum([self.lice_population[stage] for stage in self.susceptible_stages])
        if self.num_infected_fish > 0:
            lice_per_host_mass = pathogenic_lice / (self.num_infected_fish * self.fish_growth_rate(days))
        else:
            lice_per_host_mass = 0.0
        prob_lice_death = 1 / (1 + math.exp(-self.cfg.fish_mortality_k *
                                            (lice_per_host_mass - self.cfg.fish_mortality_center)))

        ebf_death = fb_mort(days) * step_size * self.num_fish
        elf_death = self.num_infected_fish * step_size * prob_lice_death
        fish_deaths_natural = self.cfg.rng.poisson(ebf_death)
        fish_deaths_from_lice = self.cfg.rng.poisson(elf_death)

        self.logger.debug("\t\t\tnumber of background fish death {}, from lice {}"
                          .format(fish_deaths_natural, fish_deaths_from_lice))

        return fish_deaths_natural, fish_deaths_from_lice

    def compute_eta_aldrin(self, num_fish_in_farm, days):
        return self.cfg.infection_main_delta + math.log(num_fish_in_farm) + self.cfg.infection_weight_delta * \
               (math.log(self.fish_growth_rate(days)) - self.cfg.delta_expectation_weight_log)

    def get_infection_rates(self, days) -> Tuple[float, int]:
        """
        Compute the number of lice that can infect and what their infection rate (number per fish) is

        :param days the amount of time it takes
        :returns a pair (Einf, num_avail_lice)
        """
        # Perhaps we can have a distribution which can change per day (the mean/median increaseѕ?
        # but at what point does the distribution mean decrease)./
        age_distrib = self.get_stage_ages_distrib("L2")
        num_avail_lice = round(self.lice_population["L2"] * np.sum(age_distrib[1:]))
        if num_avail_lice > 0:
            num_fish_in_farm = sum([c.num_fish for c in self.farm.cages])

            # TODO: this has O(c^2) complexity
            etas = np.array([c.compute_eta_aldrin(num_fish_in_farm, days) for c in self.farm.cages])
            Einf = math.exp(etas[self.id]) / (1 + np.sum(np.exp(etas)))

            return Einf, num_avail_lice

        return 0.0, num_avail_lice

    def do_infection_events(self, days) -> int:
        """
        Infect fish in this cage if the sea lice are in stage L2 and at least 1 day old

        :param days the number of days elapsed
        :returns number of evolving lice, or equivalently the new number of infections
        """
        Einf, num_avail_lice = self.get_infection_rates(days)

        if Einf == 0:
            return 0

        inf_events = self.cfg.rng.poisson(Einf * num_avail_lice)
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

        probs = np.full(n, 1/n)
        bins = self.cfg.rng.multinomial(k, probs)

        return np.var(bins)*k

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
        vmr = variance/mean
        if vmr <= 1.0:
            return 0
        aggregation_factor = mean / (vmr - 1)

        prob_matching = 1 - (1 + males/((males + females)*aggregation_factor)) ** (-1-aggregation_factor)
        # TODO: using a poisson distribution can lead to a high std for high prob*females
        return int(np.clip(self.cfg.rng.poisson(prob_matching * females), np.int32(0), np.int32(min(males, females))))

    def do_mating_events(self) -> Tuple[GenoDistrib, GenoDistrib]:
        """
        will generate two deltas:  one to add to unavailable dams and subtract from available dams, one to add to eggs
        assume males don't become unavailable? in this case we don't need a delta for sires
        :returns a pair (delta_dams, new_eggs)

        TODO: deal properly with fewer individuals than matings required
        TODO: right now discrete mode is hard-coded, once we have quantitative egg generation implemented, need to add a switch
        TODO: current date is required by get_num_eggs as of now, but the relationship between extrusion and temperature is not clear
        """

        delta_eggs = {}  # type: GenoDistrib
        num_matings = self.get_num_matings()

        distrib_sire_available = self.lice_population.geno_by_lifestage["L5m"]
        distrib_dam_available = self.lice_population.available_dams

        delta_dams = self.select_dams(distrib_dam_available, num_matings)

        if sum(distrib_sire_available.values()) == 0 or sum(distrib_dam_available.values()) == 0:
            return {}, {}

        # TODO - need to add dealing with fewer males/females than the number of matings

        for dam_geno in delta_dams:
            for _ in range(int(delta_dams[dam_geno])):
                sire_geno = self.choose_from_distrib(distrib_sire_available)
                new_eggs = self.generate_eggs(sire_geno, dam_geno, num_matings)
                self.update_distrib_discrete_add(new_eggs, delta_eggs)

        return delta_dams, delta_eggs

    def generate_eggs(
            self,
            sire: Union[Alleles, np.ndarray],
            dam: Union[Alleles, np.ndarray],
            num_matings: int
    ) -> GenoDistrib:
        """
        Generate the eggs with a given genomic distribution
        If we're in the discrete 2-gene setting, assume for now that genotypes are tuples - so in a A/a genetic system, genotypes
        could be ('A'), ('a'), or ('A', 'a')
        right now number of offspring with each genotype are deterministic, and might be missing one (we should update to add jitter in future,
        but this is a reasonable approx)
        TODO: doesn't do anything sensible re: integer/real numbers of offspring
        :param sire the genomics of the sires
        :param dam the genomics of the dams
        :param num_matings the number of matings
        :returns a distribution on the number of generated eggs according to the distribution
        """

        number_eggs = self.get_num_eggs(num_matings)

        eggs_generated = {}
        if self.genetic_mechanism == GeneticMechanism.discrete:

            if len(sire) == 1 and len(dam) == 1:
                eggs_generated[tuple(sorted(tuple({sire[0], dam[0]})))] = float(number_eggs)
            elif len(sire) == 2 and len(dam) == 1:
                eggs_generated[tuple(sorted(tuple({sire[0], dam[0]})))] = number_eggs / 2
                eggs_generated[tuple(sorted(tuple({sire[1], dam[0]})))] = number_eggs / 2
            elif len(sire) == 1 and len(dam) == 2:
                eggs_generated[tuple(sorted(tuple({sire[0], dam[0]})))] = number_eggs / 2
                eggs_generated[tuple(sorted(tuple({sire[0], dam[1]})))] = number_eggs / 2
            else: #
                # drawing both from the sire in the first case ensures heterozygotes
                # but is a bit hacky.
                eggs_generated[tuple(sorted(tuple({sire[0], sire[1]})))] = number_eggs / 2
                # and the below gets us our two types of homozygotes
                eggs_generated[tuple(sorted(tuple({sire[0], dam[0]})))] = number_eggs / 4
                eggs_generated[tuple(sorted(tuple({sire[1], dam[1]})))] = number_eggs / 4

        elif self.genetic_mechanism == GeneticMechanism.quantitative:
            # additive genes, assume genetic state for an individual looks like a number between 0 and 1.
            # because we're only dealing with the heritable part here don't need to do any of the comparison
            # to population mean or impact of heritability, etc - that would appear in the code dealing with treatment
            # so we could use just the mid-parent value for this genetic recording for children
            # as with the discrete genetic model, this will be deterministic for now
            mid_parent = np.round((sire + dam)/2, 1)
            eggs_generated[mid_parent] = number_eggs

        elif self.genetic_mechanism == GeneticMechanism.maternal:
            # maternal-only inheritance - all eggs have mother's genotype
            eggs_generated[dam] = number_eggs
        else:
            raise Exception("Genetic mechanism must be 'maternal', 'quantitative' or 'discrete' - '{}' given".format(self.genetic_mechanism))

        return eggs_generated

    @staticmethod
    def update_distrib_discrete_add(distrib_delta, distrib):
        """
        I've assumed that both distrib and delta are dictionaries
        and are *not* normalised (that is, they are effectively counts)
        I've also assumed that we never want a count below zero
        Code is naive - interpret as algorithm spec
        combine these two functions with a negation of a dict?
        """

        for geno in distrib_delta:
            if geno not in distrib:
                distrib[geno] = 0
            distrib[geno] += distrib_delta[geno]

    @staticmethod
    def update_distrib_discrete_subtract(distrib_delta, distrib):
        for geno in distrib:
            if geno in distrib_delta:
                distrib[geno] -= distrib_delta[geno]
            if distrib[geno] < 0:
                distrib[geno] = 0

    @staticmethod
    def multiply_distrib(distrib: GenoDistrib, population: np.int64):
        keys = distrib.keys()
        values = list(distrib.values())
        np_values = np.array(values)
        if np.sum(np_values) == 0 or population == 0:
            return dict(zip(keys, map(int, np.zeros_like(np_values))))
        np_values = np_values * population / np.sum(np_values)
        np_values = np_values.round().astype(np.int32)
        # correct casting errors
        np_values[-1] += population - np.sum(np_values)

        return dict(zip(keys, map(int, np_values)))

    def select_dams(self, distrib_dams_available: GenoDistrib, num_dams: int):
        """
        Assumes the usual dictionary representation of number
        of dams - genotype:number_available
        function separated from other breeding processes to make it easier to come back and optimise
        TODO: there must be a faster way to do this. 
        returns a dictionary in the same format giving genotype:number_selected
        if the num_dams exceeds number available, gives back all the dams
        """
        delta_dams_selected = {}
        copy_dams_avail = copy.deepcopy(distrib_dams_available)
        if sum(distrib_dams_available.values()) <= num_dams:
            return copy_dams_avail

        for _ in range(num_dams):
            this_dam = self.choose_from_distrib(copy_dams_avail)
            copy_dams_avail[this_dam] -= 1
            if this_dam not in delta_dams_selected:
                delta_dams_selected[this_dam] = 0
            delta_dams_selected[this_dam] += 1

        return delta_dams_selected

    def choose_from_distrib(self, distrib: GenoDistrib) -> Alleles:
        distrib_values = np.array(list(distrib.values()))
        return tuple(self.cfg.rng.choice(list(distrib.keys()), p=distrib_values / np.sum(distrib_values)))

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
        eggs = self.cfg.reproduction_eggs_first_extruded *\
            (age_range ** self.cfg.reproduction_age_dependence) * mated_females_distrib

        return self.cfg.rng.poisson(np.round(np.sum(eggs)))
        # TODO: We are deprecating this. Need to investigate if temperature data is useful. See #46
        # ave_temp = self.farm.year_temperatures[cur_month - 1]
        #temperature_factor = self.cfg.delta_m10["L0"] * (10 / ave_temp) ** self.cfg.delta_p["L0"]

        #reproduction_rates = self.cfg.reproduction_eggs_first_extruded * \
        #                     (age_range ** self.cfg.reproduction_age_dependence) / (temperature_factor + 1)

        #return self.cfg.rng.poisson(np.round(np.sum(reproduction_rates * mated_females_distrib)))

    def get_egg_batch(self, cur_date: dt.datetime, egg_distrib: dict) -> EggBatch:
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
        expected_time = (self.cfg.delta_m10["L0"] / (ave_temp - 10 + beta_1 * beta_2))**2
        expected_hatching_date = cur_date + dt.timedelta(self.cfg.rng.poisson(expected_time))
        return EggBatch(expected_hatching_date, egg_distrib)

    @staticmethod
    def pop_from_queue(queue: PriorityQueue, cur_time: dt.datetime, output_geno_distrib: dict):
        # process all the due events
        # Note: queues miss a "peek" method, and this line relies on an implementation detail.


        @singledispatch
        def access_time_lt(_peek_element, _cur_time: dt.datetime):
            pass

        @access_time_lt.register
        def _(arg: EggBatch, _cur_time: dt.datetime):
            return arg.hatch_date <= _cur_time

        @access_time_lt.register
        def _(arg: TravellingEggBatch, _cur_time: dt.datetime):
            return arg.hatch_date <= _cur_time

        @access_time_lt.register
        def _(arg: DamAvailabilityBatch, _cur_time: dt.datetime):
            return arg.availability_date <= _cur_time

        while not queue.empty() and access_time_lt(queue.queue[0], cur_time):
            event = queue.get()
            for geno, value in event.geno_distrib.items():
                output_geno_distrib[geno] += value

    def create_offspring(self, cur_time: dt.datetime) -> dict:
        """
        Hatch the eggs from the event queue
        :param cur_time the current time
        :returns a delta egg genomics
        """

        # TODO: this does not take into account f2f-c2c movements
        delta_egg_offspring = {}
        for geno in self.lice_population.geno_by_lifestage['L5f']:
            delta_egg_offspring[geno] = 0

        self.pop_from_queue(self.hatching_events, cur_time, delta_egg_offspring)

        return delta_egg_offspring

    def get_arrivals(self, cur_date: dt.datetime) -> dict:
        """Process the arrivals queue.
        :param cur_date: Current date of simulation
        :return: Genotype distribution of eggs hatched in travel
        """

        hatched_dist = {}

        # check queue for arrivals at current date
        while not self.arrival_events.empty() and self.arrival_events.queue[0].arrival_date <= cur_date:
            batch = self.arrival_events.get()

            # if the hatch date is after current date, add to stationary egg queue
            if batch.hatch_date >= cur_date:
                stationary_batch = EggBatch(batch.hatch_date, batch.geno_distrib)
                self.hatching_events.put(stationary_batch)

            # otherwise the egg has hatched during travel, so determine the life stage
            # and add to population
            else:
                # TODO: determine life stage it arrives at; assumes all are L1 for now
                hatched_dist = batch.geno_distrib

        return hatched_dist

    def free_dams(self, cur_time):
        """
        Return the number of available dams

        :param cur_time the current time
        :returns the genotype population of dams that return available today
        """
        delta_avail_dams = {}
        for geno in self.lice_population.geno_by_lifestage['L5f']:
            delta_avail_dams[geno] = 0

        self.pop_from_queue(self.busy_dams, cur_time, delta_avail_dams)
        return delta_avail_dams

    def promote_population(
        self,
        prev_stage: Union[str, dict],
        cur_stage: str,
        leaving_lice: np.int64,
        entering_lice: Optional[np.int64] = None
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
            entering_lice = cast(np.int64, entering_lice)
            prev_stage_geno = self.lice_population.geno_by_lifestage[prev_stage]
            entering_geno_distrib = self.multiply_distrib(prev_stage_geno, entering_lice)
        else:
            entering_geno_distrib = prev_stage
        cur_stage_geno = self.lice_population.geno_by_lifestage[cur_stage]

        leaving_geno_distrib = self.multiply_distrib(cur_stage_geno, leaving_lice)

        self.update_distrib_discrete_add(entering_geno_distrib, cur_stage_geno)
        self.update_distrib_discrete_subtract(leaving_geno_distrib, cur_stage_geno)

        # update gross population. This is a bit hairy but I cannot think of anything simpler.
        self.lice_population.geno_by_lifestage[cur_stage] = cur_stage_geno

    def update_deltas(
            self,
            dead_lice_dist: dict,
            treatment_mortality: dict,
            fish_deaths_natural: int,
            fish_deaths_from_lice: int,
            new_L2: int,
            new_L4: int,
            new_females: np.int64,
            new_males: np.int64,
            new_infections: int,
            lice_from_reservoir: dict,
            delta_dams_batch: DamAvailabilityBatch,
            new_egg_batch: EggBatch,
            new_offspring_distrib: dict,
            returned_dams: dict,
            hatched_arrivals_dist: dict
    ):
        """Update the number of fish and the lice in each life stage
        :param dead_lice_dist the number of dead lice due to background death (as a distribution)
        :param treatment_mortality the number of dead lice due to treatment (as a distribution)
        :param fish_deaths_natural the number of natural fish death events
        :param fish_deaths_from_lice the number of lice-induced fish death events
        :param new_L2 number of new L2 fish
        :param new_L4 number of new L4 fish
        :param new_females number of new adult females
        :param new_males number of new adult males
        :param new_infections the number of new infections (i.e. progressions from L2 to L3)
        :param lice_from_reservoir the number of lice taken from the reservoir
        :param delta_dams_batch the genotypes of now-unavailable females in batch events
        :param new_egg_batch the expected hatching time of those egg with a related genomics
        :param new_offspring_distrib the new offspring obtained from hatching and migrations
        :param returned_dams the genotypes of returned dams
        :param hatched_arrivals_dist: new offspring obtained from arrivals
        """

        for stage in self.lice_population:
            # update background mortality
            bg_delta = self.lice_population[stage] - dead_lice_dist[stage]
            self.lice_population[stage] = max(0, bg_delta)

            # update population due to treatment
            num_dead = treatment_mortality.get(stage, 0)
            treatment_delta = self.lice_population[stage] - num_dead
            self.lice_population[stage] = max(0, treatment_delta)

        self.lice_population["L5m"] += new_males
        self.lice_population["L5f"] += new_females

        self.promote_population("L3", "L4", new_males + new_females, new_L4)

        self.promote_population("L2", "L3", new_L4, new_infections)

        self.promote_population("L1", "L2", new_infections, new_L2 + lice_from_reservoir["L2"])

        self.promote_population(new_offspring_distrib, "L1", new_L2, None)
        self.promote_population(hatched_arrivals_dist, "L1", 0, None)

        # in absence of wildlife genotype, simply upgrade accordingly
        # TODO: switch to generic genotype distribs?
        self.lice_population["L1"] += lice_from_reservoir["L1"]

        delta_eggs = new_egg_batch.geno_distrib
        delta_avail_dams = delta_dams_batch.geno_distrib
        self.update_distrib_discrete_subtract(delta_avail_dams, self.lice_population.available_dams)
        self.update_distrib_discrete_add(returned_dams, self.lice_population.available_dams)
        self.update_distrib_discrete_add(delta_eggs, self.egg_genotypes)
        self.update_distrib_discrete_subtract(delta_eggs, new_offspring_distrib)

        self.busy_dams.put(delta_dams_batch)

        #  TODO: remove females that leave L5f by dying from available_dams

        self.num_fish -= fish_deaths_natural
        self.num_fish -= fish_deaths_from_lice

        # treatment may kill some lice attached to the fish, thus update at the very end
        self.num_infected_fish = self.get_mean_infected_fish()

    def update_arrivals(self, arrivals_dict: dict, arrival_date: dt.datetime):
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

    def get_background_lice_mortality(self, lice_population):
        """
        Background death in a stage (remove entry) -> rate = number of
        individuals in stage*stage rate (nauplii 0.17/d, copepods 0.22/d,
        pre-adult female 0.05, pre-adult male ... Stien et al 2005)
        """
        lice_mortality_rates = self.cfg.background_lice_mortality_rates

        dead_lice_dist = {}
        for stage in lice_population:
            mortality_rate = lice_population[stage] * lice_mortality_rates[stage] * self.cfg.tau
            mortality = min(self.cfg.rng.poisson(mortality_rate), lice_population[stage])
            dead_lice_dist[stage] = mortality

        self.logger.debug("\t\tbackground mortality distribution of dead lice = {}".format(dead_lice_dist))
        return dead_lice_dist

    @staticmethod
    def fish_growth_rate(days):
        return 10000/(1 + math.exp(-0.01*(days-475)))

    def to_csv(self):
        """
        Save the contents of this cage as a CSV string
        for writing to a file later.
        """

        data = [str(self.id), str(self.num_fish)]
        data.extend([str(val) for val in self.lice_population.values()])
        data.append(str(sum(self.lice_population.values())))

        return ", ".join(data)

    def get_reservoir_lice(self, pressure):
        """Get distribution of lice coming from the reservoir

        :param pressure: External pressure
        :type pressure: int
        :return: Distribution of lice in L1 and L2
        :rtype: dict
        """

        if pressure == 0:
            return {"L1": 0, "L2": 0}

        num_L1 = self.cfg.rng.integers(low=0, high=pressure, size=1)[0]
        new_lice_dist = {"L1": num_L1, "L2": pressure - num_L1}
        self.logger.debug("\t\tdistribution of new lice from reservoir = {}".format(new_lice_dist))
        return new_lice_dist
