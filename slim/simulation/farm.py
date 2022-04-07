"""
A Farm is a fundamental agent in this simulation. It has a number of functions:

- controlling individual cages
- managing its own finances
- choose whether to cooperate with other farms belonging to the same organisation

"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from mypy_extensions import TypedDict
import ray
from ray.util.queue import Queue as RayQueue

from slim import LoggableMixin, logger
from slim.simulation.cage import Cage
from slim.simulation.config import Config
from slim.JSONEncoders import CustomFarmEncoder
from slim.simulation.lice_population import (
    GrossLiceDistrib,
    GenoDistrib,
    GenoDistribDict,
    GenoRates,
    genorates_to_dict,
    empty_geno_from_cfg,
)
from slim.types.queue import *
from slim.types.policies import TREATMENT_NO
from slim.types.treatments import Treatment

GenoDistribByHatchDate = Dict[dt.datetime, GenoDistrib]
CageAllocation = List[GenoDistribByHatchDate]
LocationTemps = TypedDict(
    "LocationTemps", {"northing": int, "temperatures": List[float]}
)

MAX_NUM_CAGES = 20
MAX_NUM_APPLICATIONS = 10


class Farm(LoggableMixin):
    """
    Define a salmon farm containing salmon cages. Over time the salmon in the cages grow and are
    subjected to external infestation pressure from sea lice.
    """

    def __init__(
        self,
        id_: int,
        cfg: Config,
        initial_lice_pop: Optional[GrossLiceDistrib] = None,
    ):
        """
        Create a farm.

        :param id_: the id of the farm.
        :param cfg: the farm configuration
        :param initial_lice_pop: if provided, overrides default generated lice population
        """
        super().__init__()

        self.cfg = cfg

        farm_cfg = cfg.farms[id_]
        self.farm_cfg = farm_cfg
        self.id_ = id_
        self.loc_x = farm_cfg.farm_location[0]
        self.loc_y = farm_cfg.farm_location[1]
        self.start_date = farm_cfg.farm_start
        self.available_treatments = farm_cfg.max_num_treatments
        self.cages = [
            Cage(i, cfg, self, initial_lice_pop) for i in range(farm_cfg.n_cages)
        ]  # pytype: disable=wrong-arg-types

        self.year_temperatures = self._initialise_temperatures(cfg.loch_temperatures)

        # TODO: only for testing purposes
        self._preemptively_assign_treatments(self.farm_cfg.treatment_dates)

        # Queues
        # TODO: command queue and farm_to_org need to be moved away and passed by the org
        self.__sampling_events: PriorityQueue[SamplingEvent] = PriorityQueue()
        self._asked_to_treat = False

        self.generate_sampling_events()

    def clear_flags(self):
        """Call this method before the main update method."""
        self._asked_to_treat = False

    def __str__(self):
        """
        Get a human-readable string representation of the farm.

        :return: a description of the cage
        """
        cages = ", ".join(str(a) for a in self.cages)
        return f"id: {self.id_}, Cages: {cages}"

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

        return self.id_ == other.id_

    @property
    def num_fish(self):
        """Return the number of fish across all cages"""
        return sum(cage.num_fish for cage in self.cages)

    @property
    def lice_population(self):
        """Return the overall lice population in a farm"""
        return dict(
            sum(
                [Counter(cage.lice_population.as_dict()) for cage in self.cages],
                Counter(),
            )
        )

    @property
    def lice_genomics(self):
        """Return the overall lice population indexed by geno distribution and stage."""

        genomics = defaultdict(lambda: empty_geno_from_cfg(self.cfg))
        for cage in self.cages:
            for (
                stage,
                value,
            ) in cage.lice_population.geno_by_lifestage.items():
                genomics[stage] = genomics[stage].add(value)

        return {k: v.to_json_dict() for k, v in genomics.items()}

    def _initialise_temperatures(self, temperatures: np.ndarray) -> np.ndarray:
        """
        Calculate the mean sea temperature at the northing coordinate of the farm at
        month c_month interpolating data taken from
        www.seatemperature.org

        :param temperatures: the array of temperatures from January till december. The expected shape is :math:`(2, n)`.
        :returns: the estimated temperature at this farm location.
        """

        # Schema: 2 rows, first column = northing, remaining 12 columns: temperature starting from Jan
        # We assume the first row has the highest northing

        x_northing, x_temps = temperatures[0][0], temperatures[0][1:]
        y_northing, y_temps = temperatures[1][0], temperatures[1][1:]

        degs = (y_temps - x_temps) / abs(y_northing - x_northing)

        Ndiff = self.loc_y - y_northing
        return np.round(y_temps - Ndiff * degs, 1)

    def generate_treatment_event(
        self, treatment_type: Treatment, cur_date: dt.datetime
    ) -> TreatmentEvent:
        """
        Generate a new treatment event with the correct efficacy based on the given day
        and type.

        :param treatment_type: the type of treatment
        :param cur_date: the current date
        :returns: the treatment event
        """
        cur_month = cur_date.month
        ave_temp = self.year_temperatures[cur_month - 1]

        treatment_cfg = self.cfg.get_treatment(treatment_type)
        delay = treatment_cfg.effect_delay
        efficacy = treatment_cfg.delay(ave_temp)
        application_period = treatment_cfg.application_period

        return TreatmentEvent(
            cur_date + dt.timedelta(days=delay),
            treatment_type,
            efficacy,
            cur_date,
            cur_date + dt.timedelta(days=application_period),
        )

    def generate_sampling_events(self):
        spacing = self.farm_cfg.sampling_spacing
        start_date = self.farm_cfg.farm_start
        end_date = self.cfg.end_date

        for days in range(0, (end_date - start_date).days, spacing):
            sampling_event = SamplingEvent(start_date + dt.timedelta(days=days))
            self.__sampling_events.put(sampling_event)

    def _preemptively_assign_treatments(
        self, scheduled_treatments: List[Tuple[dt.datetime, Treatment]]
    ):
        """
        Assign a few treatment dates to cages.
        NOTE: Mainly used for testing. May be deprecated when a proper strategy mechanism is in place

        :param scheduled_treatments: the dates when to apply treatment
        """
        for scheduled_treatment in scheduled_treatments:
            self.add_treatment(scheduled_treatment[1], scheduled_treatment[0])

    def fallow(self):
        for cage in self.cages:
            cage.fallow()

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
        _False_.

        :param treatment_type: the treatment type to apply
        :param day: the day when to start applying the treatment
        :returns: whether the treatment has been added to at least one cage or not.
        """

        logger.debug(
            "\t\tFarm {} requests treatment {}".format(self.id_, str(treatment_type))
        )
        if self.available_treatments <= 0:
            return False

        # TODO: no support for treatment combination. See #127
        eligible_cages = [
            cage
            for cage in self.cages
            if not (
                cage.start_date > day
                or cage.is_fallowing
                or cage.is_treated(treatment_type)
            )
        ]

        if len(eligible_cages) == 0:
            logger.debug("\t\tTreatment not scheduled as no cages were eligible")
            return False

        event = self.generate_treatment_event(treatment_type, day)

        for cage in eligible_cages:
            cage.treatment_events.put(event)
        self.available_treatments -= 1

        return True

    def ask_for_treatment(self):
        """
        Ask the farm to perform treatment.

        The farm will thus respond in the following way:

        - choose whether to apply treatment or not (regardless of the actual cage eligibility).
        - if yes, which treatment to apply (according to internal evaluations, e.g. increased lice resistance).

        The farm is not obliged to tell the organisation whether treatment is being performed.
        """

        logger.debug("Asking farm {} to treat".format(self.id_))
        self._asked_to_treat = True

    def apply_action(self, cur_date: dt.datetime, action: int):
        """Apply an action

        :param cur_date: the date when the action was issued
        :param action: the action identifier as defined in :mod:`slim.types.Treatments`
        """
        if action < len(Treatment):
            picked_treatment = list(Treatment)[action]
            self.add_treatment(picked_treatment, cur_date)
        elif action == len(Treatment):
            self.fallow()

    def update(
        self,
        cur_date: dt.datetime,
        ext_influx: int,
        ext_pressure_ratios: GenoRates,
        farm_to_org: RayQueue,
    ) -> Tuple[GenoDistribByHatchDate, float]:
        """Update the status of the farm given the growth of fish and change
        in population of parasites. Also distribute the offspring across cages.

        :param cur_date: Current date
        :param ext_influx: the amount of lice that enter a cage
        :param ext_pressure_ratios: the ratio to use for the external pressure
        :param farm_to_org a queue to send a sampling response, if due

        :returns: a pair of (dictionary of genotype distributions based on hatch date, cost of the update)
        """

        # TODO: why not simply returning a possible event as an extra element?

        self.clear_log()

        if cur_date >= self.start_date:
            logger.debug("Updating farm {}".format(self.id_))
        else:
            logger.debug("Updating farm {} (non-operational)".format(self.id_))

        self.log(
            "\tAdding %r new lice from the reservoir", new_reservoir_lice=ext_influx
        )
        self.log(
            "\tReservoir lice genetic ratios: %s",
            new_reservoir_lice_ratios=genorates_to_dict(ext_pressure_ratios),
        )

        self._report_sample(cur_date, farm_to_org)

        # reset the number of treatments
        if ((cur_date - self.start_date).days + 1) % 365 == 0:
            self.available_treatments = self.farm_cfg.max_num_treatments

        # get number of lice from reservoir to be put in each cage
        pressures_per_cage = self.get_cage_pressures(ext_influx)

        total_cost = 0.0

        # collate egg batches by hatch time
        eggs_by_hatch_date: GenoDistribByHatchDate = {}
        eggs_log = empty_geno_from_cfg(self.cfg)

        for cage in self.cages:

            # update the cage and collect the offspring info
            egg_distrib, hatch_date, cost = cage.update(
                cur_date, pressures_per_cage[cage.id], ext_pressure_ratios
            )

            if hatch_date:
                # update the total offspring info
                if hatch_date in eggs_by_hatch_date:
                    eggs_by_hatch_date[hatch_date].iadd(egg_distrib)
                else:
                    eggs_by_hatch_date[hatch_date] = egg_distrib

                eggs_log = eggs_log.add(egg_distrib)

            total_cost += cost

        self.log("\t\tGenerated eggs by farm %d: %s", self.id_, eggs=eggs_log)

        return eggs_by_hatch_date, total_cost

    def get_cage_pressures(self, external_inflow: int) -> List[int]:
        """Get external pressure divided into cages

        :param external_inflow: the total external pressure
        :return: List of values of external pressure for each cage
        """

        assert len(self.cages) >= 1, "Farm must have at least one cage."
        assert external_inflow >= 0, "External pressure cannot be negative."

        # assume equal chances for each cage
        probs_per_cage = np.full(len(self.cages), 1 / len(self.cages))

        return list(
            self.cfg.rng.multinomial(
                external_inflow * len(self.cages), probs_per_cage, size=1
            )[0]
        )

    def get_farm_allocation(
        self, target_farm: Farm, eggs_by_hatch_date: GenoDistribByHatchDate
    ) -> GenoDistribByHatchDate:
        """Return farm allocation of arrivals, that is a dictionary of genotype distributions based
        on hatch date updated to take into account probability of making it to the target farm.

        The probability accounts for interfarm water movement (currents) as well as lice egg survival.

        Note: for efficiency reasons this class *modifies* eggs_by_hatch_date in-place.

        :param target_farm: Farm the eggs are travelling to
        :param eggs_by_hatch_date: Dictionary of genotype distributions based on hatch date

        :return: Updated dictionary of genotype distributions based on hatch date
        """

        farm_allocation = eggs_by_hatch_date.copy()

        for hatch_date, geno_dict in farm_allocation.items():
            # get the interfarm travel probability between the two farms
            travel_prob = self.cfg.interfarm_probs[self.id_][target_farm.id_]
            n = geno_dict.gross
            arrivals = min(self.cfg.rng.poisson(travel_prob * n), n)
            new_geno = geno_dict.normalise_to(arrivals)
            farm_allocation[hatch_date] = new_geno

        return farm_allocation

    def get_cage_allocation(
        self, ncages: int, eggs_by_hatch_date: GenoDistribByHatchDate
    ) -> CageAllocation:
        """Return allocation of eggs for given number of cages.

        :param ncages: Number of bins to allocate to
        :param eggs_by_hatch_date: Dictionary of genotype distributions based on hatch date

        :return: List of dictionaries of genotype distributions based on hatch date per bin
        """

        assert ncages >= 1, "Number of bins must be positive."

        # dummy implementation - assumes equal probabilities
        # for both intercage and interfarm travel
        # TODO: complete with actual probabilities
        # probs_per_farm = self.cfg.interfarm_probs[self.name]
        probs_per_bin = np.full(ncages, 1 / ncages)

        # preconstruct the data structure
        hatch_list: CageAllocation = [
            {
                hatch_date: empty_geno_from_cfg(self.cfg)
                for hatch_date in eggs_by_hatch_date
            }
            for _ in range(ncages)
        ]
        for hatch_date, geno_dict in eggs_by_hatch_date.items():
            for genotype in geno_dict.keys():
                genotype_per_bin = self.cfg.rng.multinomial(
                    geno_dict[genotype], probs_per_bin, size=1
                )[0]
                # update the info
                for bin_ix, n in enumerate(genotype_per_bin):
                    hatch_list[bin_ix][hatch_date][genotype] = n

        return hatch_list

    def disperse_offspring(
        self,
        eggs_by_hatch_date: GenoDistribByHatchDate,
        farms: List[Farm],
        cur_date: dt.datetime,
    ):
        """Allocate new offspring between the farms and cages.

        Assumes the lice can float freely across a given farm so that
        they are not bound to a single cage while not attached to a fish.

        NOTE: This method is not multiprocessing safe. (why?)

        :param eggs_by_hatch_date: Dictionary of genotype distributions based on hatch date
        :param farms: List of Farm objects
        :param cur_date: Current date of the simulation
        """

        logger.debug("\tDispersing total offspring Farm {}".format(self.id_))

        for farm in farms:
            if farm.id_ == self.id_:
                logger.debug("\t\tFarm {} (current):".format(farm.id_))
            else:
                logger.debug("\t\tFarm {}:".format(farm.id_))

            # allocate eggs to cages
            farm_arrivals = self.get_farm_allocation(farm, eggs_by_hatch_date)
            arrivals_per_cage = self.get_cage_allocation(len(farm.cages), farm_arrivals)

            total, by_cage, by_geno_cage = self.get_cage_arrivals_stats(
                arrivals_per_cage
            )
            logger.debug("\t\t\tTotal new eggs = {}".format(total))
            logger.debug("\t\t\tPer cage distribution = {}".format(by_cage))
            self.log(
                "\t\t\tPer cage distribution (as geno) = %s",
                arrivals_per_cage=by_geno_cage,
            )

            # get the arrival time of the egg batch at the allocated
            # destination
            travel_time = self.cfg.rng.poisson(
                self.cfg.interfarm_times[self.id_][farm.id_]
            )
            arrival_date = cur_date + dt.timedelta(days=travel_time)

            # update the cages
            for cage in farm.cages:
                cage.update_arrivals(arrivals_per_cage[cage.id], arrival_date)

    def get_cage_arrivals_stats(
        self,
        cage_arrivals: CageAllocation,
    ) -> Tuple[int, List[int], List[GenoDistrib]]:
        """Get stats about the cage arrivals for logging

        :param cage_arrivals: List of Dictionaries of genotype distributions based on hatch date.

        :return: Tuple representing total number of arrivals, arrival, distribution and genotype distribution by cage
        """

        # Basically ignore the hatch dates and sum up the batches
        geno_by_cage = [
            GenoDistrib.batch_sum(list(hatch_dict.values()))
            if len(hatch_dict.values())
            else empty_geno_from_cfg(self.cfg)
            for hatch_dict in cage_arrivals
        ]
        gross_by_cage = [geno.gross for geno in geno_by_cage]
        return sum(gross_by_cage), gross_by_cage, geno_by_cage

    def get_profit(self, cur_date: dt.datetime) -> float:
        """
        Get the current mass of fish that can be resold.

        :param cur_date: the current day
        :returns: the total profit that can be earned from this farm at the current time
        """
        mass_per_cage = [
            cage.average_fish_mass((cur_date - cage.start_date).days) / 1e3
            for cage in self.cages
        ]
        infection_per_cage = [cage.get_infecting_population() for cage in self.cages]
        infection_rate = sum(infection_per_cage) / sum(mass_per_cage)
        return (
            self.cfg.gain_per_kg - infection_rate * self.cfg.infection_discount
        ) * sum(mass_per_cage)

    @property
    def is_treating(self):
        """
        :returns: a list of applied treatments as binary indices
        """
        current_treatments = set()
        for cage in self.cages:
            current_treatments.update(cage.current_treatments)
        return list(current_treatments)

    def get_gym_space(self):
        """
        :returns: a Gym space for the agent that controls this farm.
        """
        fish_population = np.array([cage.num_fish for cage in self.cages])
        aggregations = np.array([cage.aggregation_rate for cage in self.cages])
        current_treatments = self.is_treating
        current_treatments_np = np.zeros((TREATMENT_NO + 1), dtype=np.int8)
        if len(current_treatments):
            current_treatments_np[current_treatments] = 1

        current_treatments_np[-1] = any(cage.is_fallowing for cage in self.cages)

        return {
            "aggregation": np.pad(
                aggregations, (0, MAX_NUM_CAGES - len(aggregations))
            ).astype(np.float32),
            "fish_population": np.pad(
                fish_population,
                (0, MAX_NUM_CAGES - len(fish_population)),
            ),
            "current_treatments": current_treatments_np,
            "allowed_treatments": self.available_treatments,
            "asked_to_treat": np.array([self._asked_to_treat], dtype=np.int8),
        }

    @property
    def aggregation_rate(self):
        return max(cage.aggregation_rate for cage in self.cages)

    def _report_sample(self, cur_date, farm_to_org):
        def cts(_):
            # report the worst across cages
            farm_to_org.put(SamplingResponse(cur_date, self.aggregation_rate))

        pop_from_queue(self.__sampling_events, cur_date, cts)


@ray.remote
class FarmActor:
    """
    A wrapper for Ray
    """

    def __init__(
        self, id: int, cfg: Config, initial_lice_pop: Optional[GrossLiceDistrib] = None
    ):
        self.farm = Farm(id, cfg, initial_lice_pop)
        # ugly hack: modify the global logger here. TODO: revert to the logging singleton
        # Because this runs on a separate memory space the logger will not be affected.
        global logger
        logger = logging.getLogger(f"SLIM-Farm-{id}")
        logger.setLevel(logging.DEBUG)

    def run(
        self,
        org2farm_q: RayQueue,
        farm2org_step_q: RayQueue,
        farm2org_sample_q: RayQueue,
    ):
        """
        :param org2farm_q: an organisation-to-farm queue to send `FarmCommand` events (currently only Step is supported)
        :param farm2org_step_q: a farm-to-organisation queue to send `FarmResponse` events
        """

        while True:
            # This should be called before the beginning of each day
            command: FarmCommand = org2farm_q.get()
            if isinstance(command, StepCommand):
                cur_date = command.request_date
                action = command.action
                ext_influx = command.ext_influx
                ext_pressure_ratios = command.ext_pressure_ratios
                # Be sure to clear any flags before this point
                # Then we need to make the
                self.farm.apply_action(cur_date, action)
                # TODO: why can't we merge cost and profit?
                eggs, cost = self.farm.update(
                    cur_date, ext_influx, ext_pressure_ratios, farm2org_sample_q
                )
                profit = self.farm.get_profit(cur_date)

                farm2org_step_q.put(StepResponse(cur_date, eggs, profit, cost))

            elif isinstance(command, DisperseCommand):
                self.farm.disperse_offspring(command.request_date, command.offspring)

            elif isinstance(command, AskForTreatmentCommand):
                self.farm.ask_for_treatment()

            elif isinstance(command, ClearFlags):
                self.farm.clear_flags()

            elif isinstance(command, DoneCommand):
                return

    def get_gym_space(self):
        """
        Wrapper to access gym spaces from the inner farm
        """
        return self.farm.get_gym_space()
