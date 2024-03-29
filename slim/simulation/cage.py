"""
Cages contain almost all of the modelling logic in SLIM. They model single cages (aka pens) of
salmon and lice. In contrast to literature we only assume eggs to be floating between cages.

This module is not meant for external use, and should be relatively self-contained.
"""

from __future__ import annotations

__all__ = ["Cage"]

import datetime as dt
import math
from collections import Counter, defaultdict

from typing import Union, Optional, Tuple, TYPE_CHECKING, Dict, List

import numpy as np

from slim.log import logger, LoggableMixin
from slim.simulation.config import Config
from slim.simulation.lice_population import (
    GenoDistrib,
    GrossLiceDistrib,
    LicePopulation,
    GenoTreatmentDistrib,
    GenoTreatmentValue,
    GenoLifeStageDistrib,
    largest_remainder,
    LifeStage,
    empty_geno_from_cfg,
    from_ratios_rng,
    from_ratios,
    GenoRates,
    from_counts,
    geno_to_alleles,
)
from slim.types.queue import (
    PriorityQueue,
    EggBatch,
    TravellingEggBatch,
    TreatmentEvent,
    pop_from_queue,
)
from slim.types.treatments import (
    GeneticMechanism,
    ChemicalTreatment,
    ThermalTreatment,
    Treatment,
)

if TYPE_CHECKING:  # pragma: no cover
    from slim.simulation.farm import Farm, GenoDistribByHatchDate

OptionalEggBatch = Optional[EggBatch]


class Cage(LoggableMixin):
    """
    The class that contains fish and lice and deals with all their lifecycle logic.

    Avoid instantiating this class directly. Usually a cage belongs to an instance of
    :class:`Farm` which will deal with cage-to-farm lice movements.

    In general, making a cage step resolves to calling the :meth:`update` method every
    day.
    """

    def __init__(
        self,
        cage_id: int,
        cfg: Config,
        farm: Farm,
        initial_lice_pop: Optional[GrossLiceDistrib] = None,
    ):
        """
        :param cage_id: the label (id) of the cage within the farm
        :param cfg: the farm configuration
        :param farm: a Farm object
        :param initial_lice_pop: if provided, overrides default generated lice population
        """
        super().__init__()

        self.cfg = cfg
        self.id = cage_id

        self.farm_id = farm.id_
        self.start_date = cfg.farms[self.farm_id].cages_start[cage_id]
        self.date = cfg.start_date

        # TODO: update with calculations
        if initial_lice_pop is None:
            lice_population = {
                "L1": cfg.min_ext_pressure,
                "L2": 0,
                "L3": 0,
                "L4": 0,
                "L5f": 0,
                "L5m": 0,
            }
        else:
            lice_population = initial_lice_pop

        self.farm = farm

        self.egg_genotypes = empty_geno_from_cfg(self.cfg)
        # TODO/Question: what's the best way to deal with having multiple possible genetic schemes?
        # TODO/Question: I suppose some of this initial genotype information ought to come from the config file
        # TODO/Question: the genetic mechanism will be the same for all lice in a simulation, so should it live in the driver?
        self.genetic_mechanism = self.cfg.genetic_mechanism

        geno_by_lifestage = {
            stage: from_ratios(self.cfg.initial_genetic_ratios, lice_population[stage])
            for stage in lice_population
        }

        self.lice_population = LicePopulation(
            geno_by_lifestage,
            self.cfg.initial_genetic_ratios,
            self.cfg.dam_unavailability,
        )

        self.num_fish = cfg.farms[self.farm_id].num_fish
        self.num_infected_fish = self.get_mean_infected_fish()
        self.num_cleaner = 0

        self.hatching_events: PriorityQueue[EggBatch] = PriorityQueue()
        self.arrival_events: PriorityQueue[TravellingEggBatch] = PriorityQueue()
        self.treatment_events: PriorityQueue[TreatmentEvent] = PriorityQueue()

        self.effective_treatments: List[TreatmentEvent] = []

    def to_json_dict(self):
        """
        Create a JSON-serialisable dictionary version of a cage.
        """
        filtered_vars = vars(self).copy()

        del filtered_vars["farm"]
        del filtered_vars["cfg"]
        del filtered_vars["logged_data"]

        # May want to improve these or change them if we change the representation for genotype distribs
        # Note: JSON encoders do not allow a default() being run on the _keys_ of dictionaries
        # and GenotypePopulation is a dict subclass. The simplest option here is to trivially force to_json_dict()
        # whenever possible,
        filtered_vars["egg_genotypes"] = self.egg_genotypes.to_json_dict()
        filtered_vars["lice_population"] = self.lice_population.to_json_dict()
        filtered_vars["geno_by_lifestage"] = self.lice_population.as_full_dict()
        filtered_vars["genetic_mechanism"] = str(filtered_vars["genetic_mechanism"])[
            len("GeneticMechanism.") :
        ]

        return filtered_vars

    def update(
        self, cur_date: dt.datetime, pressure: int, ext_pressure_ratio: GenoRates
    ) -> Tuple[GenoDistrib, Optional[dt.datetime], float]:
        """Update the cage at the current time step.

        :param cur_date: Current date of simulation
        :param pressure: External pressure, planctonic lice coming from the reservoir
        :param ext_pressure_ratio: The genotype ratio to use for the external pressure
        :return: Tuple (egg genotype distribution, hatching date, cost)
        """

        self.clear_log()

        if cur_date >= self.start_date:
            logger.debug("\tUpdating farm %s / cage %d", self.farm_id, self.id)
            logger.debug("\t\tinitial fish population = %d", self.num_fish)
        else:
            logger.debug(
                "\tUpdating farm %d / cage %d (non-operational)", self.farm_id, self.id
            )

        logger.debug("\t\tinitial lice population = %s", self.lice_population)
        logger.debug("\t\tAdding %s lice from external pressure", pressure)

        # Background lice mortality events
        dead_lice_dist = self.get_background_lice_mortality()

        # Development events
        new_L2, new_L4, new_females, new_males = self.get_lice_lifestage(cur_date)

        # Lice coming from other cages and farms
        # NOTE: arrivals first then hatching
        hatched_arrivals_dist = self.get_arrivals(cur_date)

        # Egg hatching
        new_offspring_distrib = self.create_offspring(cur_date)

        # Cleaner fish
        cleaner_fish_delta = self.get_cleaner_fish_delta()

        # Lice coming from reservoir
        lice_from_reservoir = self.get_reservoir_lice(pressure, ext_pressure_ratio)
        logger.debug(
            "\t\tExternal pressure lice distribution = %s", lice_from_reservoir
        )

        if cur_date < self.start_date or self.is_fallowing:
            # Values that are not used before the start date
            treatment_mortality = self.lice_population.get_empty_geno_distrib(self.cfg)
            fish_deaths_natural, fish_deaths_from_lice = 0, 0
            num_infection_events = 0
            delta_avail_dams = 0
            new_egg_batch: OptionalEggBatch = None
            cost = self.cfg.monthly_cost / 28

        else:
            # Events that happen when cage is populated with fish
            # (after start date)
            days_since_start = (cur_date - self.date).days

            # Treatment mortality events
            treatment_mortality, cost = self.get_lice_treatment_mortality(cur_date)

            # Fish growth and death
            fish_deaths_natural, fish_deaths_from_lice = self.get_fish_growth(
                days_since_start
            )

            # Infection events
            num_infection_events = self.do_infection_events(days_since_start)

            # Mating events that create eggs
            delta_avail_dams, delta_eggs = self.do_mating_events()
            new_egg_batch = self.get_egg_batch(cur_date, delta_eggs)

        fish_deaths_from_treatment = self.get_fish_treatment_mortality(
            (cur_date - self.start_date).days,
            fish_deaths_from_lice,
            fish_deaths_natural,
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
            delta_avail_dams,
            new_offspring_distrib,
            hatched_arrivals_dist,
            cleaner_fish_delta,
        )

        logger.debug("\t\tfinal lice population = %s", self.lice_population)

        egg_distrib = empty_geno_from_cfg(self.cfg)
        hatch_date: Optional[dt.datetime] = None
        if cur_date >= self.start_date and new_egg_batch:
            logger.debug("\t\tfinal fish population = %d", self.num_fish)
            egg_distrib = new_egg_batch.geno_distrib
            hatch_date = new_egg_batch.hatch_date
            logger.debug("\t\tlice offspring = %d", egg_distrib.gross)

        assert egg_distrib.is_positive()
        return egg_distrib, hatch_date, cost

    def get_temperature(self, cur_date: dt.datetime) -> float:
        """Get the cage temperature at this farm and day

        :param cur_date: the current day

        :returns: the temperature (in °C)
        """
        cur_month = cur_date.month
        return self.farm.year_temperatures[cur_month - 1]

    def get_lice_treatment_mortality_rate(
        self, cur_date: dt.datetime
    ) -> GenoTreatmentDistrib:
        """Check if the cage is currently being treated. If yes, calculate the treatment rates.
        Note: this method consumes the internal treatment event queue.

        :param cur_date: the current date
        :returns: the mortality rates broken down by geno data.
        """

        def cts(event):
            nonlocal self
            self.effective_treatments.append(event)

        pop_from_queue(self.treatment_events, cur_date, cts)

        # if no treatment has been applied check if the previous treatment is still effective
        new_effective_treatments = []
        for treatment in self.effective_treatments:
            if cur_date <= treatment.treatment_window:
                new_effective_treatments.append(treatment)

        self.effective_treatments = new_effective_treatments

        treatments = self.effective_treatments
        if self.num_cleaner > 0:
            treatments = treatments + [
                TreatmentEvent(cur_date, Treatment.CLEANERFISH, 0, cur_date, cur_date)
            ]

        # TODO: this is very fragile
        geno_treatment_distribs = defaultdict(lambda: GenoTreatmentValue(0, []))
        ave_temp = self.get_temperature(cur_date)

        for treatment in treatments:
            treatment_type = treatment.treatment_type

            logger.debug(
                "\t\ttreating farm %d/cage %d on date %s",
                self.farm_id,
                self.id,
                cur_date,
            )

            geno_treatment_distrib = self.cfg.get_treatment(
                treatment_type
            ).get_lice_treatment_mortality_rate(ave_temp, self)
            # assumption: all treatments act on different genes
            # TODO: the assumption is correct, but currently the config only uses one gene!
            for k, v in geno_treatment_distrib.items():
                if k in geno_treatment_distribs:
                    # TODO: not all treatments on the same gene have the same susceptible stages!
                    prev = geno_treatment_distribs[k]
                    geno_treatment_distribs[k] = GenoTreatmentValue(
                        prev.mortality_rate + v.mortality_rate, prev.susceptible_stages
                    )
                else:
                    geno_treatment_distribs[k] = v

        return geno_treatment_distribs

    def get_lice_treatment_mortality(
        self, cur_date
    ) -> Tuple[GenoLifeStageDistrib, float]:
        """
        Calculate the number of lice in each stage killed by treatment.

        Note: this method consumes the internal event queue

        :param cur_date: the current date
        :returns: a pair (distribution of dead lice, cost of treatment)
        """

        dead_lice_dist = self.lice_population.get_empty_geno_distrib(self.cfg)

        dead_mortality_distrib = self.get_lice_treatment_mortality_rate(cur_date)

        cost = 0.0

        for geno, (mortality_rate, susc_stages) in dead_mortality_distrib.items():
            if mortality_rate > 0:
                # Compute the number of mortality events, then decide which stages should be affected.
                # To ensure that no stage underflows we use a hypergeometric distribution.

                population_by_stages = np.array(
                    [
                        self.lice_population.geno_by_lifestage[stage][geno]
                        for stage in susc_stages
                    ],
                    dtype=np.int64,
                )
                num_susc = np.sum(population_by_stages)
                num_dead_lice = int(round(mortality_rate * num_susc))
                num_dead_lice = min(num_dead_lice, num_susc)

                dead_lice_nums = (
                    self.cfg.multivariate_hypergeometric(
                        population_by_stages, num_dead_lice
                    )
                    .astype(np.float64)
                    .tolist()
                )
                for stage, dead_lice_num in zip(susc_stages, dead_lice_nums):
                    dead_lice_dist[stage][geno] = dead_lice_num

                logger.debug(
                    "\t\tdistribution of dead lice on farm %d/cage %d = %s",
                    self.farm_id,
                    self.id,
                    dead_lice_dist,
                )

        # Compute cost
        for treatment in self.effective_treatments:
            if treatment.end_application_date <= cur_date:
                continue
            treatment_type = treatment.treatment_type
            treatment_cfg = self.cfg.get_treatment(treatment_type)
            cage_days = (cur_date - self.start_date).days
            if isinstance(treatment_cfg, ChemicalTreatment):
                # SLICE is administered to deliver 50 μg emamectin
                # benzoate/kg fish biomass/day for 7 consecutive days.
                # The suggested feeding rate is 0.5% fish biomass per day.
                cost = (
                    treatment_cfg.price_per_kg
                    * treatment_cfg.dosage_per_fish_kg  # TODO: this value is too small
                    * self.average_fish_mass(cage_days)
                    / 1e3
                )
            elif isinstance(treatment_cfg, ThermalTreatment):
                cost = treatment_cfg.price_per_application

        return dead_lice_dist, cost

    def get_stage_ages_distrib(self, stage: str, temp_c: float = 10.0):
        """
        Create an age distribution (in days) for the sea lice within a lifecycle stage.

        This distribution is computed by using a simplified version of formulae (4), (6), (8)
        in Aldrin et al.: we assume that all lice of a stage m-1 that evolve at stage m
        will be put at a stage-age of 0, and that this is going to be a constant/stable
        amount.
        """

        stage_age_max_days = int(self.cfg.stage_age_evolutions[stage])

        if stage in ("L2", "L5f"):
            # Bogus uniform distribution L5m/L5f follow different schemes.
            # More realistically, this is an exponential/poisson distribution
            return np.full(stage_age_max_days, 1 / stage_age_max_days)

        delta_p = self.cfg.delta_p[stage]
        delta_m10 = self.cfg.delta_m10[stage]
        delta_s = self.cfg.delta_s[stage]

        ages = np.arange(stage_age_max_days)

        weibull_median_rates = self._dev_rates(
            delta_p, delta_m10, delta_s, temp_c, ages
        )

        probas = np.empty_like(ages, dtype=np.float64)
        probas[0] = 1.0
        for i in range(1, stage_age_max_days):
            probas[i] = probas[i - 1] * (1.0 - weibull_median_rates[i - 1])

        return probas / np.sum(probas)

    @staticmethod
    def _dev_rates(
        del_p: float, del_m10: float, del_s: float, temp_c: float, ages: np.ndarray
    ):
        """
        Probability of developing after n_days days in a stage as given by Aldrin et al 2017
        See section 2.2.4 of Aldrin et al. (2017)
        :param del_p: power transformation constant on temp_c
        :param del_m10: 10 °C median reference value
        :param del_s: fitted Weibull shape parameter
        :param temp_c: average temperature in °C
        :param ages: stage-age
        :return: expected development rate
        """
        epsilon = np.float64(1e-30)
        del_m = del_m10 * (10 / temp_c) ** del_p

        unbounded = np.clip(
            np.log(2) * del_s * ages ** (del_s - 1) * del_m ** (-del_s),
            epsilon,
            np.float64(1.0),
        )
        return unbounded

    def get_lice_lifestage(self, cur_date: dt.datetime) -> Tuple[int, int, int, int]:
        """
        Move lice between lifecycle stages.
        See Section 2.1 of Aldrin et al. (2017)

        :param cur_date: the current date
        :returns: a tuple (new_l2, new_l4, new_l5f, new_l5m)
        """
        logger.debug("\t\tupdating lice lifecycle stages")

        def evolve_next_stage(stage: LifeStage, temp_c: float) -> int:
            # TODO: that's a hack
            if stage == "L4":
                return round(
                    self.lice_population["L4"] * self.cfg.lice_development_rates["L4"]
                )

            # weibull params
            del_p = self.cfg.delta_p[stage]
            del_m10 = self.cfg.delta_m10[stage]
            del_s = self.cfg.delta_s[stage]
            num_lice = self.lice_population[stage]
            if num_lice == 0:
                return 0
            max_dev_time = self.cfg.stage_age_evolutions[stage]
            ages_distrib = self.get_stage_ages_distrib(stage, temp_c)
            # TODO:
            # let us sample from 1000 lice
            evolution_rates = self._dev_rates(
                del_p, del_m10, del_s, temp_c, np.arange(max_dev_time)
            )
            average_rates = np.mean(ages_distrib.dot(evolution_rates))

            return round(
                num_lice * average_rates
            )  # int(min(self.cfg.rng.poisson(num_lice * average_rates), num_lice))

        lice_dist = {}
        ave_temp = self.get_temperature(cur_date)

        # L4 -> L5
        l4_to_l5 = evolve_next_stage("L4", ave_temp)
        new_females = int(
            self.cfg.rng.choice([math.floor(l4_to_l5 / 2.0), math.ceil(l4_to_l5 / 2.0)])
        )
        new_males = l4_to_l5 - new_females

        lice_dist["L5f"] = new_females
        lice_dist["L5m"] = new_males

        # L3 -> L4
        new_L4 = lice_dist["L4"] = evolve_next_stage("L3", ave_temp)

        # L2 -> L3 is done in do_infection_events(). Indeed, evolution from L2 to L3 is (virtually) age-independent

        # L1 -> L2
        new_L2 = lice_dist["L2"] = evolve_next_stage("L1", ave_temp)

        logger.debug(
            "\t\t\tdistribution of new lice lifecycle stages on farm %s/cage %s = %s",
            self.farm_id,
            self.id,
            lice_dist,
        )

        return new_L2, new_L4, new_females, new_males

    def get_fish_treatment_mortality(
        self, days_since_start: int, fish_lice_deaths: int, fish_background_deaths: int
    ) -> int:
        """
        Get fish mortality due to treatment. Mortality due to treatment is defined in terms of
        point percentage increases, thus we can only account for "excess deaths".
        If multiple treatments are performed their side effects are additively combined.
        If no treatment is currently in action, no fish deaths are counted.

        :param days_since_start: the number of days since the beginning of the simulation
        :param fish_lice_deaths: the number of fish dead by lice
        :param fish_background_deaths: the number of fish dead by natural reasons

        :returns: number of fish death events
        """
        # See surveys/overton_treatment_mortalities.py for an explanation on what is going on
        mortality_events = fish_lice_deaths + fish_background_deaths

        if mortality_events == 0 or self.num_fish == 0:
            return 0

        cur_date = self.start_date + dt.timedelta(days=days_since_start)
        treatment_mortality_occurrences = 0

        for treatment in self.effective_treatments:
            efficacy_window = treatment.effectiveness_duration_days

            temperature = self.get_temperature(cur_date)
            fish_mass = self.average_fish_mass(days_since_start)

            last_treatment_params = self.cfg.get_treatment(treatment.treatment_type)
            predicted_deaths = last_treatment_params.get_fish_mortality_occurrences(
                temperature, fish_mass, self.num_fish, efficacy_window, mortality_events
            )

            treatment_mortality_occurrences += round(predicted_deaths)

        return treatment_mortality_occurrences

    def get_fish_growth(self, days_since_start) -> Tuple[int, int]:
        """
        Get the number of fish that get killed either naturally or by lice.

        :param: days_since_start: the number of days since the beginning.
        :returns: a tuple (natural_deaths, lice_induced_deaths, lice_deaths)
        """

        logger.debug("\t\tupdating fish population")

        def fb_mort(days):
            """
            Fish background mortality rate, decreasing as in Soares et al 2011

            Fish death rate: constant background daily rate 0.00057 originally based on
            www.gov.scot/Resource/0052/00524803.pdf

            According to Volsett it should be higher (around 0.005) , but that's unlikely.
            However the authors used a trick: assuming that a mortality event takes 5 days one can
            divide the actual mortality rate by 5.
            Thus, 0.005 / 5 ~~ 0.0009-0.001 . It is still quite higher than expected.

            Therefore, we'll stick to this constant.
            TODO: what should we do with the formulae?

            :param days: number of days elapsed
            :return: fish background mortality rate
            """
            # To make things simple, let's assume 0 for now
            return 0
            # return 0.0000057  # (1000 + (days - 700)**2)/490000000

        # Apply a sigmoid based on the number of lice per fish
        pathogenic_lice = sum(
            [self.lice_population[stage] for stage in LicePopulation.pathogenic_stages]
        )
        if self.num_infected_fish > 0:
            lice_per_host_mass = pathogenic_lice / (
                self.num_infected_fish
                * (self.average_fish_mass(days_since_start) / 1e3)
            )
        else:
            lice_per_host_mass = 0.0

        # Calculation taken from Vollken (2019) (see appendix, page 6)
        # Derivation: the formula only provides the surviving odds,
        # from which we extract the probability with the known formula
        # p = e^odds / (1 + e^odds)

        # The daily mortality rate also takes into account a mortality event takes days (in this case, 5).
        exp_odds = math.exp(
            self.cfg.fish_mortality_center
            - self.cfg.fish_mortality_k * lice_per_host_mass
        )

        # TODO: the distribution should be way more skewed towards *high* concentration
        prob_lice_death = 0.001 * (1 - (exp_odds / (1 + exp_odds)))

        ebf_death = fb_mort(days_since_start) * self.num_fish
        elf_death = self.num_infected_fish * prob_lice_death
        fish_deaths_natural = round(ebf_death)  # self.cfg.rng.poisson(ebf_death)
        fish_deaths_from_lice = round(elf_death)  # self.cfg.rng.poisson(elf_death)

        logger.debug(
            "\t\t\tnumber of fish deaths: background = %d, from lice = %d",
            fish_deaths_natural,
            fish_deaths_from_lice,
        )

        return fish_deaths_natural, fish_deaths_from_lice

    def _compute_eta_aldrin(self, num_fish_in_farm, days):
        # TODO: this lacks of stochasticity compared to the paper
        return (
            self.cfg.infection_main_delta
            + math.log(num_fish_in_farm / 1e5)
            + self.cfg.infection_weight_delta * 1
        )  # \
        # (math.log(self.average_fish_mass(days)/1e3) - self.cfg.delta_expectation_weight_log)

    def get_infection_rates(self, days_since_start) -> Tuple[float, int]:
        """
        Compute the number of lice that can infect and what their infection rate (number per fish) is.

        Compared to Aldrin et al. we do not calculate a softmax across cage rates as this would cause multiprocessing
        issues. We still perform

        :param days_since_start: days since the cage has opened

        :returns: a pair (Einf, num_avail_lice)
        """

        # Based on Aldrin et al., but we do not consider a softmax betwwe
        # Perhaps we can have a distribution which can change per day (the mean/median increaseѕ?
        # but at what point does the distribution mean decrease).
        age_distrib = self.get_stage_ages_distrib("L2")
        num_avail_lice = round(self.lice_population["L2"] * np.sum(age_distrib[1:]))
        if num_avail_lice > 0:
            num_fish_in_farm = self.farm.num_fish

            eta = (
                self.cfg.infection_main_delta
                + math.log(num_fish_in_farm / 1e5)
                + self.cfg.infection_weight_delta
                * (
                    math.log(self.average_fish_mass(days_since_start) / 1e3)
                    - self.cfg.delta_expectation_weight_log
                )
            )

            Einf = math.exp(eta) / (1 + math.exp(eta))

            """
            etas = np.array(
                [
                    c.compute_eta_aldrin(num_fish_in_farm, days_since_start)
                    for c in self.farm.cages
                ]
            )
            Einf = math.exp(etas[self.id]) / (1 + np.sum(np.exp(etas)))
            """

            return Einf, num_avail_lice

        return 0.0, num_avail_lice

    def do_infection_events(self, days: int) -> int:
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
        """
        Get the number of lice infecting a population.

        :param \\*args: consider the given stages as infecting (default: CH to AM/AF)
        :returns: gross number of infecting lice
        """
        if args is None or len(args) == 0:
            infective_stages = ["L3", "L4", "L5m", "L5f"]
        else:
            infective_stages = list(args)
        return sum(self.lice_population[stage] for stage in infective_stages)

    def get_mean_infected_fish(self, *args) -> int:
        """
        Get the average number of infected fish.

        :param \\*args: the stages to consider (optional, by default all stages from the third onward are taken into account)

        :returns: the number of infected fish
        """
        if self.num_fish == 0:
            return 0

        attached_lice = self.get_infecting_population(*args)

        # see: https://stats.stackexchange.com/a/296053
        num_infected_fish = int(
            self.num_fish * (1 - ((self.num_fish - 1) / self.num_fish) ** attached_lice)
        )
        return num_infected_fish

    @staticmethod
    def get_variance_infected_fish(num_fish: int, infecting_lice: int) -> float:
        r"""
        Compute the variance of the lice infecting the fish.

        Rationale: assuming that we generate :math:`N` bins that sum to :math:`K`, we can model this as a multinomial distribution
        where all :math:`p_i` are the same. Therefore, the variance of each bin is :math:`k \frac{n-1}{n^2}`

        Because we are considering the total variance of k events at the same time we need to multiply by :math:`k`,
        thus yielding :math:`k^2 \frac{n-1}{n^2}` .

        :param num_fish: the number of fish
        :param infecting_lice: the number of lice attached to the fish
        :returns: the variance
        """

        if num_fish == 0:
            return 0.0
        return infecting_lice**2 * (num_fish - 1) / (num_fish**2)

    def get_num_matings(self) -> int:
        """
        Get the number of matings. Implement Cox (2017)'s approach assuming an unbiased sex distribution.
        """

        # Background: AF and AM are randomly assigned to fish according to a negative multinomial distribution.
        # What we want is determining what is the expected likelihood there is _at least_ one AF and _at
        # least_ one AM on the same fish.

        males = self.lice_population["L5m"]
        females = self.lice_population.available_dams.gross

        if males == 0 or females == 0:
            return 0

        # VMR: variance-mean ratio; VMR = m/k + 1 -> k = m / (VMR - 1) with k being an "aggregation factor"
        mean = (males + females) / self.get_mean_infected_fish("L5m", "L5f_free")

        n = self.num_fish
        k = self.get_infecting_population("L5m", "L5f_free")
        variance = self.get_variance_infected_fish(n, k)
        vmr = variance / mean
        if vmr <= 1.0:
            return 0
        aggregation_factor = mean / (vmr - 1)

        prob_matching = 1 - (1 + males / ((males + females) * aggregation_factor)) ** (
            -1 - aggregation_factor
        )
        # TODO: using a poisson distribution can lead to a high std for high prob*females
        return int(
            np.clip(
                self.cfg.rng.poisson(prob_matching * females),
                np.int64(0),
                np.int64(min(males, females)),
            )
        )

    def do_mating_events(self) -> Tuple[int, GenoDistrib]:
        """
        Will generate two deltas:  one to add to unavailable dams and subtract from available dams, one to add to eggs
        Assume males don't become unavailable? in this case we don't need a delta for sires

        :returns: a pair (mating_dams, new_eggs)
        """

        delta_eggs = empty_geno_from_cfg(self.cfg)
        num_matings = self.get_num_matings()

        distrib_sire_available = self.lice_population.geno_by_lifestage["L5m"]
        distrib_dam_available = self.lice_population.available_dams

        mating_dams = self.select_lice(distrib_dam_available, num_matings)
        mating_sires = self.select_lice(distrib_sire_available, num_matings)

        if distrib_sire_available.gross == 0 or distrib_dam_available.gross == 0:
            return 0, empty_geno_from_cfg(self.cfg)

        num_eggs = self.get_num_eggs(num_matings)
        if self.genetic_mechanism == GeneticMechanism.DISCRETE:
            delta_eggs = self.generate_eggs_discrete_batch(
                mating_sires, mating_dams, num_eggs
            )
        elif self.genetic_mechanism == GeneticMechanism.MATERNAL:
            delta_eggs = self.generate_eggs_maternal_batch(
                distrib_dam_available, num_eggs
            )

        delta_eggs = self.mutate(delta_eggs, mutation_rate=self.cfg.geno_mutation_rate)

        return mating_dams.gross, delta_eggs

    def generate_eggs_discrete_batch(
        self, sire_distrib: GenoDistrib, dam_distrib: GenoDistrib, number_eggs: int
    ) -> GenoDistrib:
        """
        Get number of eggs based on discrete genetic mechanism.

        The algorithm emulates the following scenario: each sire s_i is sampled from sire_distrib
        and will have a given genomic :math:`g_{s_i}` and similarly a dam :math:`d_i` will have a genotype :math:`g_{d_i}`.
        Because the genomic of a lice can only be dominant, recessive or partial dominant then
        the mating will result in one of these happening depending on the usual Mendelevian
        mechanism.
        The algorithm terminates when all sires and dams have been mated.
        Here we emulate such sampling strategy via a O(1) algorithm as follows: we compute
        the probabilities of each combination arising and adopt a multinomial distribution
        in order to achieve a given number of eggs.
        The rationale for the formulae used here is the following:

        - to get an A (dominant) one needs a combination of dominant and dominant allele, or dominant and the right half of partially dominant genes;

        - to get an a (recessive) one does the same as above, but with fully recessive alleles instead;

        - to get a partially dominant one, we use the inclusion-exclusion principle and subtract the cases above from all the possible pairings.

        Together with being fast, this approach naturally models statistical uncertainty compared
        to a perfect Mendelevian split. Unfortunately it does not naturally include mutation
        to non-existent strains (e.g. a mating between pure dominant lice distributions can never
        yield partial dominance or recessive genomics in the offspring).

        :param sire_distrib: the genotype distribution of the sires
        :param dam_distrib: the genotype distribution of the eggs
        :param number_eggs: the total number of eggs

        :returns: the newly sampled eggs as a :class:`GenoDistrib`.
        """

        assert sire_distrib.gross == dam_distrib.gross

        proba_dict = np.empty_like(sire_distrib.values(), dtype=np.float64)

        for gene in range(sire_distrib.num_genes):
            rec, dom, het_dom = geno_to_alleles(gene)
            keys = [dom, rec, het_dom]
            x1, y1, z1 = tuple(sire_distrib[k] for k in keys)
            x2, y2, z2 = tuple(dam_distrib[k] for k in keys)

            N_A = (x1 + 1 / 2 * z1) * (x2 + 1 / 2 * z2)
            N_a = (y1 + 1 / 2 * z1) * (y2 + 1 / 2 * z2)
            denom = sire_distrib.gross * dam_distrib.gross
            N_Aa = denom - N_A - N_a

            if denom == 0:
                return empty_geno_from_cfg(self.cfg)

            # We need to follow GenoDistrib's ordering now

            p = np.array([N_a, N_A, N_Aa]) / denom
            proba_dict[gene] = p

        return from_ratios_rng(number_eggs, proba_dict, self.cfg.rng)

    def generate_eggs_maternal_batch(
        self, dams: GenoDistrib, number_eggs: int
    ) -> GenoDistrib:
        """Get number of eggs based on maternal genetic mechanism.

        Maternal-only inheritance - all eggs have mother's genotype.

        :param dams: the genomics of the dams
        :param number_eggs: the number of eggs produced
        :return: genomics distribution of eggs produced
        """
        if dams.gross == 0:
            return empty_geno_from_cfg(self.cfg)
        return dams.normalise_to(number_eggs)

    def mutate(self, eggs: GenoDistrib, mutation_rate: float) -> GenoDistrib:
        """
        Mutate the genotype distribution

        :param eggs: the genotype distribution of the newly produced eggs
        :param mutation_rate: the rate of mutation with respect to the number of eggs.
        """

        if mutation_rate == 0:
            return eggs

        new_eggs = eggs.copy()
        mutations = self.cfg.rng.poisson(mutation_rate * eggs.gross)
        new_eggs.mutate(mutations)
        return new_eggs

    def select_lice(
        self, distrib_lice_available: GenoDistrib, num_lice: int
    ) -> GenoDistrib:
        """
        From a geno distribution of eligible lice sample a given Genotype distribution

        Note: this is very similar to :meth:`GenoDistrib.normalise_to` but performs an explicit
        sampling.

        TODO: should we integrate this into GenoDistrib class

        :param distrib_lice_available: the starting dam genomic distribution
        :param num_lice: the wished number of dams to sample
        """
        # TODO: this function is flexible enough to be renamed and used elsewhere
        if distrib_lice_available.gross <= num_lice:
            return distrib_lice_available.copy()

        # "we need to select k lice from the given population broken down into different allele
        # bins and subtract" -> "select n balls from the following N_1, ..., N_k bins without
        # replacement -> use a multivariate hypergeometric distribution

        # TODO: Vectorise this

        counts = distrib_lice_available.values()
        lice_as_lists = np.empty_like(counts, np.int64)
        for i in range(len(counts)):
            count = counts[i]
            lice_as_lists[i] = self.cfg.multivariate_hypergeometric(
                count.astype(np.int64), num_lice
            )
        selected_lice = from_counts(lice_as_lists.astype(np.float64), self.cfg)

        return selected_lice

    def get_num_eggs(self, mated_females) -> int:
        """
        Get the number of new eggs

        :param mated_females: the number of mated females that reproduce

        :returns: the number of eggs produced
        """

        # See Aldrin et al. 2017, §2.2.6
        age_distrib = self.get_stage_ages_distrib("L5f")
        age_range = np.arange(1, len(age_distrib) + 1)

        mated_females_distrib = mated_females * age_distrib

        # Hatching time is already covered in get_egg_batch
        eggs = (
            self.cfg.reproduction_eggs_first_extruded
            * (age_range**self.cfg.reproduction_age_dependence)
            * mated_females_distrib
        )

        return int(np.round(np.sum(eggs)))

    def get_egg_batch(
        self, cur_date: dt.datetime, egg_distrib: GenoDistrib
    ) -> EggBatch:
        """
        Get the expected arrival date of an egg batch

        :param cur_date: the current time
        :param egg_distrib: the egg distribution

        :returns: EggBatch representing egg distribution and expected hatching date
        """

        # We use Stien et al (2005)'s regressed formula
        # τE= [β1/(T – 10 + β1β2)]**2  (see equation 8)
        # where β2**(-2) is the average temperature centered at around 10 degrees
        # and β1 is a shaping factor. This function is formally known as Belehrádek’s function
        ave_temp = self.get_temperature(cur_date)

        beta_1 = self.cfg.delta_m10["L0_stien"]
        beta_2 = self.cfg.delta_p["L0"]
        expected_time = (beta_1 / (ave_temp - 10 + beta_1 * beta_2)) ** 2
        expected_hatching_date = cur_date + dt.timedelta(
            self.cfg.rng.poisson(expected_time)
        )
        return EggBatch(expected_hatching_date, egg_distrib)

    def create_offspring(self, cur_time: dt.datetime) -> GenoDistrib:
        """
        Hatch the eggs from the event queue

        :param cur_time: the current time

        :returns: a delta egg genomics
        """

        delta_egg_offspring = empty_geno_from_cfg(self.cfg)

        def cts(hatching_event: EggBatch):
            nonlocal delta_egg_offspring
            delta_egg_offspring.iadd(hatching_event.geno_distrib)

        pop_from_queue(self.hatching_events, cur_time, cts)
        return delta_egg_offspring

    def get_arrivals(self, cur_date: dt.datetime) -> GenoDistrib:
        """Process the arrivals queue.

        :param cur_date: Current date of simulation

        :returns: Genotype distribution of eggs hatched in travel
        """

        hatched_dist = empty_geno_from_cfg(self.cfg)

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
        :returns: a gross distribution
        """

        # Note: no paper actually provides a clear guidance on this.
        # This is mere speculation.
        # Basically, only consider PA and A males (assume L4m = 0.5*L4)
        # as potentially able to survive and find a new host.
        # Only a fixed proportion can survive.

        infecting_lice = self.get_infecting_population()
        infected = self.num_infected_fish
        if infecting_lice * infected == 0:
            return {}

        affected_lice_gross = round(infecting_lice * num_dead_fish / infected)

        affected_lice_quotas = np.array(
            [
                self.lice_population[stage] / infecting_lice * affected_lice_gross
                for stage in LicePopulation.infectious_stages
            ]
        )
        affected_lice_np = largest_remainder(affected_lice_quotas)

        affected_lice = Counter(
            dict(zip(LicePopulation.infectious_stages, affected_lice_np.tolist()))
        )

        surviving_lice_quotas = np.rint(
            np.trunc(
                [
                    self.lice_population["L4"]
                    / (2 * infecting_lice)
                    * affected_lice_gross
                    * self.cfg.male_detachment_rate,
                    self.lice_population["L5m"]
                    / infecting_lice
                    * affected_lice_gross
                    * self.cfg.male_detachment_rate,
                ]
            )
        )
        surviving_lice = Counter(
            dict(zip(["L4", "L5m"], surviving_lice_quotas.tolist()))
        )

        dying_lice = affected_lice - surviving_lice

        dying_lice_distrib = {k: int(v) for k, v in dying_lice.items() if v > 0}
        logger.debug("\t\tLice mortality due to fish mortality: %s", dying_lice_distrib)
        return dying_lice_distrib

    def get_cleaner_fish_delta(self) -> int:
        """
        Call this function before :meth:`get_lice_treatment_mortality()` !
        """
        restock = 0.0
        for treatment in self.current_treatments:
            if treatment == Treatment.CLEANERFISH.value:
                # assume restocking of 5% of Nfish
                restock = self.num_fish * 1 / 100

        return int(
            math.ceil(
                restock - (self.num_cleaner * self.cfg.cleaner_fish.natural_mortality)
            )
        )

    def promote_population(
        self,
        prev_stage: Union[str, GenoDistrib],
        cur_stage: str,
        leaving_lice: int,
        entering_lice: Optional[int] = None,
    ):
        """
        Promote the population by stage and respect the genotypes

        :param prev_stage: the lice stage from which cur_stage evolves
        :param cur_stage: the lice stage that is about to evolve
        :param leaving_lice: the number of lice in the _cur_stage=>next_stage_ progression
        :param entering_lice: the number of lice in the _prev_stage=>cur_stage_ progression. If _prev_stage_ is a a string, _entering_lice_ must be an _int_
        """
        if isinstance(prev_stage, str):
            assert (
                entering_lice is not None
            ), "entering_lice must be an int when prev_stage is a str"
            prev_stage_geno = self.lice_population.geno_by_lifestage[prev_stage]
            entering_geno_distrib = prev_stage_geno.normalise_to(entering_lice)
        else:
            entering_geno_distrib = prev_stage
        cur_stage_geno = self.lice_population.geno_by_lifestage[cur_stage]

        leaving_geno_distrib = cur_stage_geno.normalise_to(leaving_lice)
        cur_stage_geno = cur_stage_geno.add(entering_geno_distrib).sub(
            leaving_geno_distrib
        )

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
        lice_from_reservoir: Dict[LifeStage, GenoDistrib],
        delta_dams_batch: int,
        new_offspring_distrib: GenoDistrib,
        hatched_arrivals_dist: GenoDistrib,
        cleaner_fish_delta: int = 0,
    ):
        """Update the number of fish and the lice in each life stage

        :param dead_lice_dist: the number of dead lice due to background death (as a distribution)
        :param treatment_mortality: the distribution of genotypes being affected by treatment
        :param fish_deaths_natural: the number of natural fish death events
        :param fish_deaths_from_lice: the number of lice-induced fish death events
        :param fish_deaths_from_treatment: the number of treatment-induced fish death events
        :param new_L2: number of new L2 fish
        :param new_L4: number of new L4 fish
        :param new_females: number of new adult females
        :param new_males: number of new adult males
        :param new_infections: the number of new infections (i.e. progressions from L2 to L3)
        :param lice_from_reservoir: the number of lice taken from the reservoir
        :param delta_dams_batch: the genotypes of now-unavailable females in batch events
        :param new_offspring_distrib: the new offspring obtained from hatching and migrations
        :param hatched_arrivals_dist: new offspring obtained from arrivals
        :param cleaner_fish_delta: new cleaner fish
        """

        # Ensure all non-lice-deaths happen
        if delta_dams_batch:
            self.lice_population.add_busy_dams_batch(delta_dams_batch)

        # Update dead_lice_dist to include fish-caused death as well

        dead_lice_by_fish_death = self.get_dying_lice_from_dead_fish(
            min(fish_deaths_natural + fish_deaths_from_lice, self.num_infected_fish)
        )
        for affected_stage, reduction in dead_lice_by_fish_death.items():
            dead_lice_dist[affected_stage] += reduction

        for stage in LicePopulation.lice_stages:
            # update background mortality
            bg_delta = self.lice_population[stage] - dead_lice_dist[stage]
            self.lice_population[stage] = max(0, bg_delta)

            # update population due to treatment
            # TODO: __isub__ here is broken
            self.lice_population.geno_by_lifestage[
                stage
            ] = self.lice_population.geno_by_lifestage[stage].sub(
                treatment_mortality[stage]
            )

        self.lice_population.remove_negatives()

        self.promote_population("L4", "L5m", 0, new_males)
        self.promote_population("L4", "L5f", 0, new_females)
        self.promote_population("L3", "L4", new_males + new_females, new_L4)
        self.promote_population("L2", "L3", new_L4, new_infections)
        self.promote_population("L1", "L2", new_infections, new_L2)

        self.promote_population(new_offspring_distrib, "L1", new_L2, None)
        self.promote_population(hatched_arrivals_dist, "L1", 0, None)

        # in absence of wildlife genotype, simply upgrade accordingly
        self.lice_population.geno_by_lifestage["L2"].iadd(lice_from_reservoir["L2"])
        self.lice_population.geno_by_lifestage["L1"].iadd(lice_from_reservoir["L1"])

        self.lice_population.remove_negatives()

        self.num_fish -= (
            fish_deaths_natural + fish_deaths_from_lice + fish_deaths_from_treatment
        )
        if self.num_fish < 0:
            self.num_fish = 0

        # treatment may kill some lice attached to the fish, thus update at the very end
        self.num_infected_fish = self.get_mean_infected_fish()

        self.num_cleaner += cleaner_fish_delta

    def update_arrivals(
        self, arrivals_dict: GenoDistribByHatchDate, arrival_date: dt.datetime
    ):
        """Update the arrivals queue

        :param arrivals_dict: List of dictionaries of genotype distributions based on hatch date
        :param arrival_date: Arrival date at this cage
        """

        for hatch_date in arrivals_dict:

            # skip if there are no eggs in the dictionary
            if arrivals_dict[hatch_date].gross == 0:
                continue

            # create new travelling batch and update the queue
            batch = TravellingEggBatch(
                arrival_date, hatch_date, arrivals_dict[hatch_date]
            )
            self.arrival_events.put(batch)

    def get_background_lice_mortality(self) -> GrossLiceDistrib:
        """
        Background death in a stage (remove entry) -> rate = number of
        individuals in stage*stage rate (nauplii 0.17/d, copepods 0.22/d,
        pre-adult female 0.05, pre-adult male ... Stien et al 2005)

        :returns: the current background mortality. The return value is genotype-agnostic
        """
        # TODO
        lice_mortality_rates = self.cfg.background_lice_mortality_rates
        lice_population = self.lice_population

        dead_lice_dist = {}
        for stage in LicePopulation.lice_stages:
            mortality_rate = lice_population[stage] * lice_mortality_rates[stage]
            mortality: int = round(
                mortality_rate
            )  # min(self.cfg.rng.poisson(mortality_rate), lice_population[stage])
            dead_lice_dist[stage] = mortality

        logger.debug(
            "\t\tbackground mortality distribution of dead lice = %s", dead_lice_dist
        )
        return dead_lice_dist

    def average_fish_mass(self, days):
        """
        Average fish mass.

        :params days: number of elapsed days

        :returns: the average fish mass (in grams).
        """
        smolt_params = self.cfg.smolt_mass_params
        return smolt_params.max_mass / (
            1 + math.exp(smolt_params.skewness * (days - smolt_params.x_shift))
        )

    def get_reservoir_lice(
        self, pressure: int, external_pressure_ratios: GenoRates
    ) -> Dict[LifeStage, GenoDistrib]:
        """Get distribution of lice coming from the reservoir

        :param pressure: External pressure
        :param external_pressure_ratios: the external pressure ratios to sample from

        :return: Distribution of lice in L1 and L2
        """

        if pressure == 0:
            return {
                "L1": empty_geno_from_cfg(self.cfg),
                "L2": empty_geno_from_cfg(self.cfg),
            }

        new_L1_gross = self.cfg.rng.integers(low=0, high=pressure, size=1)[0]
        new_L2_gross = pressure - new_L1_gross

        new_L1 = from_ratios_rng(new_L1_gross, external_pressure_ratios, self.cfg.rng)
        new_L2 = from_ratios_rng(new_L2_gross, external_pressure_ratios, self.cfg.rng)

        new_lice_dist = {"L1": new_L1, "L2": new_L2}
        logger.debug("\t\tdistribution of new lice from reservoir = %s", new_lice_dist)
        return new_lice_dist

    def fallow(self):
        """Put the cage in a fallowing state.

        Implications:

        1. All the fish would be removed.
        2. L3/L4/L5 would therefore disappear as they are attached to fish.
        3. Dam waiting and treatment queues will be flushed.
        """

        for stage in LicePopulation.infectious_stages:
            self.lice_population[stage] = 0

        self.num_infected_fish = self.num_fish = 0
        self.treatment_events = PriorityQueue()
        self.effective_treatments.clear()

    @property
    def is_fallowing(self):
        """
        True if the cage is fallowing.
        """
        return self.num_fish == 0

    def is_treated(self, treatment_type: Optional[Treatment] = None):
        """
        Check if a farm is treated.

        :param treatment_type: if provided, check if there is a treatment of the given type

        :returns: True if the cage is being treated
        """
        if not treatment_type and len(self.effective_treatments):
            return True
        for treatment in self.effective_treatments:
            if treatment.treatment_type == treatment_type:
                return True

        return False

    @property
    def current_treatments(self):
        """:returns: a list of current treatments"""
        idxs = []
        for event in self.effective_treatments:
            idxs.append(event.treatment_type.value)
        return idxs

    @property
    def aggregation_rate(self):
        """The aggregation rate is the number of lice over the total number of fish.
        Elsewhere, it is referred to as infection rate, but here "infection rate" only refers to host fish.

        :returns: the aggregation rate"""
        return self.lice_population["L5f"] / self.num_fish if self.num_fish > 0 else 0.0
