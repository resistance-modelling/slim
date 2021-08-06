"""
Defines a Farm class that encapsulates a salmon farm containing several cages.
"""
from __future__ import annotations

import copy
import datetime as dt
from decimal import Decimal
import json
from collections import Counter
from typing import Counter as CounterType
from typing import Dict, List, Optional, Tuple

from mypy_extensions import TypedDict

import numpy as np

from src.Cage import Cage
from src.Config import Config
from src.JSONEncoders import CustomFarmEncoder
from src.LicePopulation import Alleles, GrossLiceDistrib
from src.TreatmentTypes import Treatment
from src.QueueBatches import TreatmentEvent

GenoDistribByHatchDate = Dict[dt.datetime, CounterType[Alleles]]
CageAllocation = List[GenoDistribByHatchDate]
LocationTemps = TypedDict("LocationTemps", {"northing": int, "temperatures": List[float]})


class Farm:
    """
    Define a salmon farm containing salmon cages. Over time the salmon in the cages grow and are
    subjected to external infestation pressure from sea lice.
    """

    def __init__(self, name: int, cfg: Config, initial_lice_pop: Optional[GrossLiceDistrib] = None):
        """
        Create a farm.
        :param name: the id of the farm.
        :param cfg: the farm configuration
        ::param initial_lice_pop: if provided, overrides default generated lice population
        """

        self.logger = cfg.logger
        self.cfg = cfg

        farm_cfg = cfg.farms[name]
        self.farm_cfg = farm_cfg
        self.name = name
        self.loc_x = farm_cfg.farm_location[0]
        self.loc_y = farm_cfg.farm_location[1]
        self.start_date = farm_cfg.farm_start
        # TODO: deprecate this
        self.cages = [Cage(i, cfg, self, initial_lice_pop) for i in range(farm_cfg.n_cages)]  # pytype: disable=wrong-arg-types

        self.year_temperatures = self.initialize_temperatures(cfg.farm_data)

        self.preemptively_assign_treatments(self.farm_cfg.treatment_starts)

    def __str__(self):
        """
        Get a human readable string representation of the farm.
        :return: a description of the cage
        """
        cages = ", ".join(str(a) for a in self.cages)
        return f"id: {self.name}, Cages: {cages}"

    def to_json_dict(self, **kwargs):
        filtered_vars = vars(self).copy()
        del filtered_vars["logger"]
        del filtered_vars["farm_cfg"]
        del filtered_vars["cfg"]
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

    def initialize_temperatures(self, temperatures: Dict[str, LocationTemps]) -> np.ndarray:
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

        # TODO: automatically convert between enum and data
        if treatment_type == Treatment.emb:
            delay = self.cfg.emb.effect_delay
            efficacy = self.cfg.emb.delay(ave_temp)

        return TreatmentEvent(cur_date + dt.timedelta(days=delay), treatment_type, efficacy)

    def preemptively_assign_treatments(self, treatment_dates: List[dt.datetime]):
        """
        Assign a few treatment dates to cages.
        NOTE: Mainly used for testing. May be deprecated when a proper strategy mechanism is in place

        :param treatment_dates: the dates when to apply treatment
        """
        for treatment_date in treatment_dates:
            self.add_treatment(self.farm_cfg.treatment_type, treatment_date)

    def add_treatment(self, treatment_type: Treatment, day: dt.datetime):
        event = self.generate_treatment_event(self.farm_cfg.treatment_type, day)
        for cage in self.cages:
            if cage.start_date <= event.affecting_date:
                cage.treatment_events.put(event)

    def update(self, cur_date: dt.datetime) -> GenoDistribByHatchDate:
        """Update the status of the farm given the growth of fish and change
        in population of parasites.

        :param cur_date: Current date
        :return: Dictionary of genotype distributions based on hatch date
        """

        if cur_date >= self.start_date:
            self.logger.debug("Updating farm {}".format(self.name))
        else:
            self.logger.debug("Updating farm {} (non-operational)".format(self.name))

        # get number of lice from reservoir to be put in each cage
        pressures_per_cage = self.get_cage_pressures()

        # collate egg batches by hatch time
        eggs_by_hatch_date = {}  # type: GenoDistribByHatchDate
        for cage in self.cages:

            # update the cage and collect the offspring info
            egg_distrib, hatch_date = cage.update(cur_date,
                                                  pressures_per_cage[cage.id])

            if hatch_date:
                # update the total offspring info
                if hatch_date in eggs_by_hatch_date:
                    eggs_by_hatch_date[hatch_date] += Counter(egg_distrib)
                else:
                    eggs_by_hatch_date[hatch_date] = Counter(egg_distrib)

        return eggs_by_hatch_date

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

        return list(self.cfg.rng.multinomial(self.cfg.ext_pressure,
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
        hatch_list = [{hatch_date: Counter() for hatch_date in eggs_by_hatch_date} for n in range(ncages)]  # type: CageAllocation
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

        NOTE: This method is not multiprocessing safe.

        :param eggs_by_hatch_date: Dictionary of genotype distributions based
        on hatch date
        :param farms: List of Farm objects
        :param cur_date: Current date of the simulation
        """

        self.logger.debug("\tDispersing total offspring Farm {}".format(self.name))

        for farm in farms:

            if farm.name == self.name:
                self.logger.debug("\t\tFarm {}:".format(farm.name))
            else:
                self.logger.debug("\t\tFarm {} (current):".format(farm.name))

            # allocate eggs to cages
            farm_arrivals = self.get_farm_allocation(farm, eggs_by_hatch_date)
            arrivals_per_cage = self.get_cage_allocation(len(farm.cages), farm_arrivals)

            total, by_cage = self.get_cage_arrivals_stats(arrivals_per_cage)
            self.logger.debug("\t\t\tTotal new eggs = {}".format(total))
            self.logger.debug("\t\t\tPer cage distribution = {}".format(by_cage))

            # get the arrival time of the egg batch at the allocated
            # destination
            travel_time = self.cfg.rng.poisson(self.cfg.interfarm_times[self.name][farm.name])
            arrival_date = cur_date + dt.timedelta(days=travel_time)

            # update the cages
            for cage in farm.cages:
                cage.update_arrivals(arrivals_per_cage[cage.id], arrival_date)

    def get_cage_arrivals_stats(self, cage_arrivals: CageAllocation) -> Tuple[int, List[int]]:
        """Get stats about the cage arrivals for logging

        :param cage_arrivals: Dictionary of genotype distributions based
        on hatch date
        :return: Tuple representing total number of arrivals and arrival
        distribution
        """

        by_cage = []
        for hatch_dict in cage_arrivals:
            cage_total = sum([n for genotype_dict in hatch_dict.values() for n in genotype_dict.values()])
            by_cage.append(cage_total)

        return sum(by_cage), by_cage

    def estimate_treatment_cost(self, treatment: Treatment, cur_day: dt.datetime) -> Decimal:
        """
        Estimate the cost of treatment
        """

        # TODO: convert treatment type into the right cfg
        if treatment == Treatment.EMB:
            cost_per_kg = self.cfg.emb.cost_per_kg
        days_since_start = ((cur_day - cage.start_date).days() for cage in self.cages)
        return sum(cost_per_kg * Cage.fish_growth_rate(cage_days)
                   for cage_days in days_since_start)