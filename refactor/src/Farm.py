"""
A Farm is a fundamental agent in this simulation. It has a number of functions:

- controlling individual cages
- managing its own finances
- choose whether to cooperate with other farms belonging to the same organisation

"""

from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
from mypy_extensions import TypedDict

from src import LoggableMixin
from src.Cage import Cage
from src.Config import Config
from src.JSONEncoders import CustomFarmEncoder
from src.LicePopulation import GrossLiceDistrib, GenoDistrib
from src.QueueTypes import *
from src.TreatmentTypes import Money

GenoDistribByHatchDate = Dict[dt.datetime, GenoDistrib]
CageAllocation = List[GenoDistribByHatchDate]
LocationTemps = TypedDict("LocationTemps", {"northing": int, "temperatures": List[float]})


class Farm(LoggableMixin):
    """
    Define a salmon farm containing salmon cages. Over time the salmon in the cages grow and are
    subjected to external infestation pressure from sea lice.
    """

    def __init__(self, name: int, cfg: Config, initial_lice_pop: Optional[GrossLiceDistrib] = None, ):
        """
        Create a farm.
        :param name: the id of the farm.
        :param cfg: the farm configuration
        ::param initial_lice_pop: if provided, overrides default generated lice population
        """
        super().__init__()

        self.cfg = cfg

        farm_cfg = cfg.farms[name]
        self.farm_cfg = farm_cfg
        self.name = name
        self.loc_x = farm_cfg.farm_location[0]
        self.loc_y = farm_cfg.farm_location[1]
        self.start_date = farm_cfg.farm_start
        self.available_treatments = farm_cfg.max_num_treatments
        self.current_capital = self.farm_cfg.start_capital
        self.cages = [Cage(i, cfg, self, initial_lice_pop) for i in range(farm_cfg.n_cages)]  # pytype: disable=wrong-arg-types

        self.year_temperatures = self.initialise_temperatures(cfg.farm_data)

        # TODO: only for testing purposes
        self.preemptively_assign_treatments(self.farm_cfg.treatment_starts)

        # Queues
        self.command_queue: PriorityQueue[FarmCommand] = PriorityQueue() 
        self.farm_to_org: PriorityQueue[FarmResponse] = PriorityQueue() 
        self.__sampling_events: PriorityQueue[SamplingEvent] = PriorityQueue() 

        self.generate_sampling_events()


    def __str__(self):
        """
        Get a human readable string representation of the farm.
        :return: a description of the cage
        """
        cages = ", ".join(str(a) for a in self.cages)
        return f"id: {self.name}, Cages: {cages}"

    def to_json_dict(self, **kwargs):
        filtered_vars = vars(self).copy()
        del filtered_vars["farm_cfg"]
        del filtered_vars["cfg"]
        del filtered_vars["logged_data"]

        filtered_vars.update(kwargs)

        return filtered_vars

    def __repr__(self):
        filtered_vars = self.to_json_dict()
        return json.dumps(filtered_vars, cls=CustomFarmEncoder, indent=4)

    def __eq__(self, other):
        if not isinstance(other, Farm):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.name == other.name

    @property
    def num_fish(self):
        return sum(cage.num_fish for cage in self.cages)

    @property
    def lice_population(self):
        """Return the overall lice population in a farm"""
        return dict(sum([Counter(cage.lice_population.as_dict()) for cage in self.cages], Counter()))

    @property
    def lice_genomics(self):
        """Return the overall lice population indexed by geno distribution and stage."""

        genomics = defaultdict(lambda: GenoDistrib())
        for cage in self.cages:
            for stage, value in cage.lice_population.geno_by_lifestage.as_dict().items():
                genomics[stage] = genomics[stage] + value


        return {k: v.to_json_dict() for k, v in genomics.items()}

    def initialise_temperatures(self, temperatures: Dict[str, LocationTemps]) -> np.ndarray:
        """
        Calculate the mean sea temperature at the northing coordinate of the farm at
        month c_month interpolating data taken from
        www.seatemperature.org
        """

        # TODO: move this in a separate file, e.g. Lake? See #96
        ardrishaig_data = temperatures["ardrishaig"]
        ardrishaig_temps, ardrishaig_northing = np.array(ardrishaig_data["temperatures"]), ardrishaig_data["northing"]
        tarbert_data = temperatures["tarbert"]
        tarbert_temps, tarbert_northing = np.array(tarbert_data["temperatures"]), tarbert_data["northing"]

        degs = (tarbert_temps - ardrishaig_temps) / abs(tarbert_northing - ardrishaig_northing)

        Ndiff = self.loc_y - tarbert_northing
        return np.round(tarbert_temps - Ndiff * degs, 1)

    def generate_treatment_event(self, treatment_type: Treatment, cur_date: dt.datetime
                                 ) -> TreatmentEvent:
        cur_month = cur_date.month
        ave_temp = self.year_temperatures[cur_month - 1]

        treatment_cfg = self.cfg.get_treatment(treatment_type)
        delay = treatment_cfg.effect_delay
        efficacy = treatment_cfg.delay(ave_temp)
        application_period = treatment_cfg.application_period

        return TreatmentEvent(
            cur_date + dt.timedelta(days=delay),
            treatment_type, efficacy,
            cur_date,
            cur_date + dt.timedelta(days=application_period)
        )

    def generate_sampling_events(self):
        spacing = self.farm_cfg.sampling_spacing
        start_date = self.farm_cfg.farm_start
        end_date = self.cfg.end_date

        for days in range(0, (end_date - start_date).days, spacing):
            sampling_event = SamplingEvent(start_date + dt.timedelta(days=days))
            self.__sampling_events.put(sampling_event)

    def preemptively_assign_treatments(self, treatment_dates: List[dt.datetime]):
        """
        Assign a few treatment dates to cages.
        NOTE: Mainly used for testing. May be deprecated when a proper strategy mechanism is in place

        :param treatment_dates: the dates when to apply treatment
        """
        for treatment_date in treatment_dates:
            self.add_treatment(self.farm_cfg.treatment_type, treatment_date)

    def add_treatment(self, treatment_type: Treatment, day: dt.datetime) -> bool:
        """
        Ask to add a treatment. If a treatment was applied too early or if too many treatments
        have been applied so far the request is rejected.

        Note that if **at least** one cage is eligible for treatment and the conditions above
        are still respected this method will still return True. Eligibility depends
        on whether the cage has started already or is fallowing - but that may depend on the type
        of chemical treatment applied. Furthermore, no treatment should be applied on cages that are already
        undergoing a treatment of the same type. This usually means the actual treatment application period
        plus a variable delay period. If no cages are available no treatment can be applied and the function returns
        False.

        :param treatment_type: the treatment type to apply
        :param day: the day when to start applying the treatment
        :returns: whether the treatment has been added to at least one cage or not.
        """

        logger.debug("\t\tFarm {} requests treatment {}".format(self.name, str(treatment_type)))
        if self.available_treatments <= 0:
            return False

        # TODO: no support for treatment combination. See #127
        eligible_cages = [cage for cage in self.cages if not
                          (cage.start_date > day or cage.is_fallowing or cage.is_treated(day))]

        if len(eligible_cages) == 0:
            logger.debug("\t\tTreatment not scheduled as no cages were eligible")
            return False

        event = self.generate_treatment_event(treatment_type, day)

        for cage in eligible_cages:
            cage.treatment_events.put(event)
        self.available_treatments -= 1

        return True

    def ask_for_treatment(self, cur_date: dt.datetime, can_defect=True):
        """
        Ask the farm to perform treatment.
        The farm will thus respond in the following way:
        - choose whether to apply treatment or not (regardless of the actual cage eligibility)
        - if yes, which treatment to apply (according to internal evaluations, e.g. increased lice resistance)
        The farm is not obliged to tell the organisation whether treatment is being performed.

        :param cur_date the current date
        :param can_defect if True, the farm has a choice to not apply treatment
        """

        logger.debug("Asking farm {} to treat".format(self.name))

        # TODO: this is extremely simple.
        p = [self.farm_cfg.defection_proba, 1 - self.farm_cfg.defection_proba]
        want_to_treat = self.cfg.rng.choice([False, True], p=p) if can_defect else True
        self.log("Outcome of the vote: %r", is_treating=want_to_treat)

        if not want_to_treat:
            logger.debug("\tFarm {} refuses to treat".format(self.name))
            return

        # TODO: implement a strategy to pick a treatment of choice
        treatments = list(Treatment)
        picked_treatment = treatments[0]

        self.add_treatment(picked_treatment, cur_date)

    def update(self, cur_date: dt.datetime) -> Tuple[GenoDistribByHatchDate, Money]:
        """Update the status of the farm given the growth of fish and change
        in population of parasites.

        :param cur_date: Current date
        :return: pair of Dictionary of genotype distributions based on hatch date, and cost of the update
        """

        self.clear_log()

        if cur_date >= self.start_date:
            logger.debug("Updating farm {}".format(self.name))
        else:
            logger.debug("Updating farm {} (non-operational)".format(self.name))

        self.handle_events(cur_date)

        # get number of lice from reservoir to be put in each cage
        pressures_per_cage = self.get_cage_pressures()

        total_cost = Money("0.00")

        # collate egg batches by hatch time
        eggs_by_hatch_date: GenoDistribByHatchDate = {} 
        for cage in self.cages:

            # update the cage and collect the offspring info
            egg_distrib, hatch_date, cost = cage.update(cur_date,
                                                        pressures_per_cage[cage.id])

            if hatch_date:
                # update the total offspring info
                if hatch_date in eggs_by_hatch_date:
                    eggs_by_hatch_date[hatch_date] += egg_distrib
                else:
                    eggs_by_hatch_date[hatch_date] = egg_distrib

            total_cost += cost

        self.current_capital -= total_cost

        return eggs_by_hatch_date, total_cost

    def get_cage_pressures(self) -> List[int]:
        """Get external pressure divided into cages
:return: List of values of external pressure for each cage
        """

        if len(self.cages) < 1:
            raise Exception("Farm must have at least one cage.")
        if self.cfg.ext_pressure < 0:
            raise Exception("External pressure cannot be negative.")

        # assume equal chances for each cage
        probs_per_cage = np.full(len(self.cages), 1/len(self.cages))

        return list(self.cfg.rng.multinomial(self.cfg.ext_pressure*len(self.cages),
                                             probs_per_cage,
                                             size=1)[0])

    def get_farm_allocation(self, target_farm: Farm, eggs_by_hatch_date: GenoDistribByHatchDate) -> GenoDistribByHatchDate:
        """Return farm allocation of arrivals, that is a dictionary of genotype distributions based
        on hatch date updated to take into account probability of making it to the target farm.

        The probability accounts for interfarm water movement (currents) as well as lice egg survival.

        :param target_farm: Farm the eggs are travelling to
        :param eggs_by_hatch_date: Dictionary of genotype distributions based
        on hatch date
        :return: Updated dictionary of genotype distributions based
        on hatch date
        """

        # base the new survived arrival dictionary on the offspring one
        farm_allocation = copy.deepcopy(eggs_by_hatch_date)

        for hatch_date, geno_dict in farm_allocation.items():
            for genotype, n in geno_dict.items():

                # get the interfarm travel probability between the two farms
                travel_prob = self.cfg.interfarm_probs[self.name][target_farm.name]

                # calculate number of arrivals based on the probability and total
                # number of offspring
                # NOTE: This works only when the travel probabilities are very low.
                #       Otherwise there is possibility that total number of arrivals
                #       would be higher than total number of offspring.
                arrivals = self.cfg.rng.poisson(travel_prob * n)

                # update the arrival dict
                farm_allocation[hatch_date][genotype] = arrivals

        return farm_allocation

    def get_cage_allocation(self, ncages: int, eggs_by_hatch_date: GenoDistribByHatchDate) -> CageAllocation:
        """Return allocation of eggs for given number of cages.

        :param ncages: Number of bins to allocate to
        :param eggs_by_hatch_date: Dictionary of genotype distributions based on hatch date
        :return: List of dictionaries of genotype distributions based on hatch date per bin
        """

        if ncages < 1:
            raise Exception("Number of bins must be positive.")

        # dummy implmentation - assumes equal probabilities
        # for both intercage and interfarm travel
        # TODO: complete with actual probabilities
        # probs_per_farm = self.cfg.interfarm_probs[self.name]
        probs_per_bin = np.full(ncages, 1 / ncages)

        # preconstruct the data structure
        hatch_list: CageAllocation = [{hatch_date: GenoDistrib() for hatch_date in eggs_by_hatch_date} for n in range(ncages)]
        for hatch_date, geno_dict in eggs_by_hatch_date.items():
            for genotype in geno_dict:
                # generate the bin distribution of this genotype with
                # this hatch date
                genotype_per_bin = self.cfg.rng.multinomial(geno_dict[genotype],
                                                            probs_per_bin,
                                                            size=1)[0]
                # update the info
                for bin_ix, n in enumerate(genotype_per_bin):
                    hatch_list[bin_ix][hatch_date][genotype] = n

        return hatch_list

    def disperse_offspring(self, eggs_by_hatch_date: GenoDistribByHatchDate, farms: List[Farm], cur_date: dt.datetime):
        """Allocate new offspring between the farms and cages.

        Assumes the lice can float freely across a given farm so that
        they are not bound to a single cage while not attached to a fish.

        NOTE: This method is not multiprocessing safe. (why?)

        :param eggs_by_hatch_date: Dictionary of genotype distributions based
        on hatch date
        :param farms: List of Farm objects
        :param cur_date: Current date of the simulation
        """

        logger.debug("\tDispersing total offspring Farm {}".format(self.name))

        for farm in farms:
            if farm.name == self.name:
                logger.debug("\t\tFarm {}:".format(farm.name))
            else:
                logger.debug("\t\tFarm {} (current):".format(farm.name))

            # allocate eggs to cages
            farm_arrivals = self.get_farm_allocation(farm, eggs_by_hatch_date)
            arrivals_per_cage = self.get_cage_allocation(len(farm.cages), farm_arrivals)

            total, by_cage, by_geno_cage = self.get_cage_arrivals_stats(arrivals_per_cage)
            logger.debug("\t\t\tTotal new eggs = {}".format(total))
            logger.debug("\t\t\tPer cage distribution = {}".format(by_cage))
            self.log("\t\t\tPer cage distribution (as geno) = %s", arrivals_per_cage=by_geno_cage)

            # get the arrival time of the egg batch at the allocated
            # destination
            travel_time = self.cfg.rng.poisson(self.cfg.interfarm_times[self.name][farm.name])
            arrival_date = cur_date + dt.timedelta(days=travel_time)

            # update the cages
            for cage in farm.cages:
                cage.update_arrivals(arrivals_per_cage[cage.id], arrival_date)

    @staticmethod
    def get_cage_arrivals_stats(cage_arrivals: CageAllocation) -> Tuple[int, List[int], List[GenoDistrib]]:
        """Get stats about the cage arrivals for logging

        :param cage_arrivals: List of Dictionaries of genotype distributions based
        on hatch date.
        :return: Tuple representing total number of arrivals, arrival
        distribution and genotype distribution by cage
        """

        # Basically ignore the hatch dates and sum up the batches
        geno_by_cage = [cast(GenoDistrib,
                             GenoDistrib.batch_sum(list(hatch_dict.values())))
                   for hatch_dict in cage_arrivals]
        gross_by_cage = [geno.gross for geno in geno_by_cage]
        return sum(gross_by_cage), gross_by_cage, geno_by_cage

    def get_profit(self, cur_date: dt.datetime):
        """
        Get the current mass of fish that can be resold.
        """
        mass_per_cage = [cage.average_fish_mass((cur_date - cage.start_date).days) / 1e3 for cage in self.cages]
        return self.cfg.gain_per_kg * Money(sum(mass_per_cage))

    def handle_events(self, cur_date: dt.datetime):
        def cts_command_queue(command):
            if isinstance(command, SampleRequestCommand):
                self.__sampling_events.put(SamplingEvent(command.request_date))

        pop_from_queue(self.command_queue, cur_date, cts_command_queue)

        self.report_sample(cur_date)

    def report_sample(self, cur_date):
        def cts(_):
            # report the worst across cages
            rate = max(cage.aggregation_rate for cage in self.cages)
            self.farm_to_org.put(SamplingResponse(cur_date, rate))

        pop_from_queue(self.__sampling_events, cur_date, cts)
