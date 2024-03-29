"""
Farms are the fundamental agents in SLIM. They cover a number of roles, including

- controlling individual cages
- managing its own finances
- choose whether to cooperate with other farms belonging to the same organisation

This module thus exports :class:`Farm` but also its ray-friendly version :class:`FarmActor` .
The latter merely serves the purpose of keeping the object alive in an async loop running on
a separate process in multithreaded simulations.


Note FarmActor's documentation cannot generated on sphinx. See https://github.com/ray-project/ray/issues/2658

Please check out its documentation on the corresponding module code.
"""

from __future__ import annotations

__all__ = ["Farm", "FarmActor"]

import copy
import logging
from collections import Counter, defaultdict
from queue import Queue
from typing import Dict, List, Tuple, Optional

import numpy as np
import ray
from mypy_extensions import TypedDict
from numpy.random import SeedSequence, default_rng

# from ray.util.queue import Queue as RayQueue

from slim.log import LoggableMixin, logger
from slim.simulation.cage import Cage
from slim.simulation.config import Config
from slim.simulation.lice_population import (
    GenoDistrib,
    GenoRates,
    GrossLiceDistrib,
    genorates_to_dict,
    empty_geno_from_cfg,
)
from slim.types.queue import *
from slim.types.treatments import Treatment, TREATMENT_NO

GenoDistribByHatchDate = Dict[dt.datetime, GenoDistrib]
CageAllocation = List[GenoDistribByHatchDate]
ArrivalDate = dt.datetime
DispersedOffspring = Tuple[CageAllocation, ArrivalDate]
DispersedOffspringPerFarm = List[Tuple[CageAllocation, ArrivalDate]]
LocationTemps = TypedDict(
    "LocationTemps", {"northing": int, "temperatures": List[float]}
)

MAX_NUM_APPLICATIONS = 24


class Farm(LoggableMixin):
    """
    Define a salmon farm containing salmon cages. Over time the salmon in the cages grow and are
    subjected to external infestation pressure from sea lice.
    A Farm will also provide a constrained Gym space, available with :meth:`get_gym_space`,
    with limited information (by design) for policy usage.

    It's usually preferrable to not instantiate this class directy but rather spawn its farm pool or the organisation
    as farms are required to communicate with each other.

    A farm should be used in two ways:

    **New way**

    * :meth:`apply_action` -> apply an action; this will enque the action to be performed during
      the next update cycle;
    * :meth:`update` -> update the internal state;
    * :meth:`disperse_offspring_v2` -> split the offspring into a list of **cage**-specific allocations.

    **Old way (deprecated, only used in single-threaded mode)**

    * :meth:`apply_action` 
    * :meth:`update`
    * :meth:`disperse_offspring` -> split the offspring into a **farm**-specific list of allocations
    * :meth:`update_arrivals` -> make the lice (offspring or newborn) enter the farm.


    Note: only after dispersing it is safe to check to farm space.
    """

    def __init__(
        self,
        id_: int,
        cfg: Config,
        initial_lice_pop: Optional[GrossLiceDistrib] = None,
    ):
        """
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
        self._reported_aggregation = 0.0

        # fmt: off
        self.cages = [
            Cage(i, cfg, self, initial_lice_pop) # pytype: disable=wrong-arg-types
            for i in range(farm_cfg.n_cages)
        ]
        # fmt: on

        self.year_temperatures = self._initialise_temperatures(cfg.loch_temperatures)

        # TODO: only for testing purposes
        self._preemptively_assign_treatments(self.farm_cfg.treatment_dates)

        # Queues
        self.__sampling_events: PriorityQueue[SamplingEvent] = PriorityQueue()

        self._generate_sampling_events()

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

    def _generate_sampling_events(self):
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
            self.add_treatment(
                scheduled_treatment[1], scheduled_treatment[0], force=True
            )

    def fallow(self):
        """
        Fallow the entire farm. This will in turn fallow all the cages in the farm.
        """
        for cage in self.cages:
            cage.fallow()

    def add_treatment(
        self, treatment_type: Treatment, day: dt.datetime, force=False
    ) -> bool:
        """
        Ask to add a treatment. If a treatment was applied too early or if too many treatments
        have been applied so far the request is rejected.

        Note that if **at least** one cage is eligible for treatment and the conditions above
        are still respected this method will still return True.

        Eligibility depends on the following conditions:

        * the cage should have started by the day treatment is applied
        * the cage should not be fallowing
        * the cage should not be treated by that day with that same type of treatment
        * the lice count should be greater than 1.0 (to prevent spamming)

        If ``force`` is True, the last three constraints will be ignored.

        :param treatment_type: the treatment type to apply
        :param day: the day when to start applying the treatment
        :param force: if True, ignore some eligibility requirement.
        :returns: whether the treatment has been added to at least one cage or not.
        """

        logger.debug("\t\tFarm %d requests treatment %s", self.id_, str(treatment_type))
        if self.available_treatments <= 0:
            return False

        eligible_cages = [
            cage
            for cage in self.cages
            if (
                cage.start_date <= day
                and (
                    force
                    or not (
                        cage.is_fallowing
                        and cage.is_treated(treatment_type)
                        and cage.aggregation_rate < 1.0
                    )
                )
            )
        ]

        if len(eligible_cages) == 0:
            logger.debug("\t\tTreatment not scheduled as no cages were eligible")
            return False

        event = self.generate_treatment_event(treatment_type, day)

        for cage in eligible_cages:
            cage.treatment_events.put(event)

        # Cleaner fish treatments can be applied as many times as possible
        if treatment_type != Treatment.CLEANERFISH and not force:
            self.available_treatments -= 1

        return True

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
    ) -> Tuple[GenoDistribByHatchDate, float]:
        """Update the status of the farm given the growth of fish and change
        in population of parasites. Also distribute the offspring across cages.

        :param cur_date: Current date
        :param ext_influx: the amount of lice that enter a cage
        :param ext_pressure_ratios: the ratio to use for the external pressure

        :returns: a pair of (dictionary of genotype distributions based on hatch date, cost of the update)
        """

        self.clear_log()

        if cur_date >= self.start_date:
            logger.debug("Updating farm %d", self.id_)
        else:
            logger.debug("Updating farm %d (non-operational)", self.id_)

        self.log(
            "\tAdding %r new lice from the reservoir", new_reservoir_lice=ext_influx
        )
        self.log(
            "\tReservoir lice genetic ratios: %s",
            new_reservoir_lice_ratios=genorates_to_dict(ext_pressure_ratios),
        )

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

            if hatch_date and egg_distrib.gross > 0:
                # update the total offspring info
                if hatch_date in eggs_by_hatch_date:
                    eggs_by_hatch_date[hatch_date].iadd(egg_distrib)
                else:
                    eggs_by_hatch_date[hatch_date] = egg_distrib

                eggs_log = eggs_log.add(egg_distrib)

            total_cost += cost

        self.log("\t\tGenerated eggs by farm %d: %s", self.id_, eggs=eggs_log)
        self.log("\t\tPayoff: %f", payoff=(self.get_profit(cur_date) - total_cost))
        self.log("\tNew population: %r", farm_population=self.lice_genomics)
        logger.debug("Total fish population: %d", self.num_fish)
        # self._asked_to_treat = False
        self._report_sample(cur_date)

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
        self, target_farm: int, eggs_by_hatch_date: GenoDistribByHatchDate
    ) -> GenoDistribByHatchDate:
        """Return farm allocation of arrivals, that is a dictionary of genotype distributions based
        on hatch date updated to take into account probability of making it to the target farm.

        The probability accounts for interfarm water movement (currents) as well as lice egg survival.

        Note: for efficiency reasons this class *modifies* eggs_by_hatch_date in-place.

        :param target_farm_id: Farm the eggs are travelling to
        :param eggs_by_hatch_date: Dictionary of genotype distributions based on hatch date

        :return: Updated dictionary of genotype distributions based on hatch date
        """

        farm_allocation = eggs_by_hatch_date.copy()

        for hatch_date, geno_dict in farm_allocation.items():
            # get the interfarm travel probability between the two farms
            travel_prob = self.cfg.interfarm_probs[self.id_][target_farm]
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
        cur_date: dt.datetime,
    ) -> List[DispersedOffspring]:
        """
        DEPRECATED: it shouldn't be within a farm's responbility to calculate cage arrivals
        within other farms.

        Allocate new offspring between the farms and cages.

        Assumes the lice can float freely across a given farm so that
        they are not bound to a single cage while not attached to a fish.

        :param eggs_by_hatch_date: Dictionary of genotype distributions based on hatch date
        :param cur_date: Current date of the simulation
        """

        logger.debug("\tDispersing total offspring Farm %d", self.id_)
        arrivals_per_farm_cage = []

        farms = self.cfg.farms

        for idx, farm in enumerate(farms):
            ncages = farm.n_cages
            logger.debug("\t\tFarm %d%s", idx, " current" if idx == self.id_ else "")

            # allocate eggs to cages
            farm_arrivals = self.get_farm_allocation(idx, eggs_by_hatch_date)
            logger.debug("\t\t\tFarm allocation is %s", farm_arrivals)

            arrivals_per_cage = self.get_cage_allocation(ncages, farm_arrivals)
            logger.debug("\t\t\tCage allocation is %s", arrivals_per_cage)

            total, by_cage, by_geno_cage = self.get_cage_arrivals_stats(
                arrivals_per_cage
            )
            logger.debug("\t\t\tTotal new eggs = %d", total)
            logger.debug("\t\t\tPer cage distribution = %s", str(by_cage))
            self.log(
                "\t\t\tPer cage distribution (as geno) = %s",
                arrivals_per_cage=by_geno_cage,
            )

            # get the arrival time of the egg batch at the allocated
            # destination
            travel_time = self.cfg.rng.poisson(self.cfg.interfarm_times[self.id_][idx])
            arrival_date = cur_date + dt.timedelta(days=travel_time)

            arrivals_per_farm_cage.append((arrivals_per_cage, arrival_date))

            # for cage in self.cages:
            #    cage.update_arrivals(arrivals_per_cage[cage.id], arrival_date)
        logger.debug("Returning %s", arrivals_per_farm_cage)
        return arrivals_per_farm_cage

    def disperse_offspring_v2(
        self,
        eggs_by_hatch_date: GenoDistribByHatchDate,
        cur_date: dt.datetime,
    ) -> DispersedOffspring:
        """Allocate new offspring within the farm

        Assumes the lice can float freely across a given farm so that
        they are not bound to a single cage while not attached to a fish.

        :param eggs_by_hatch_date: Dictionary of genotype distributions based on hatch date
        :param cur_date: Current date of the simulation
        """

        logger.debug("\tDispersing total offspring Farm %d", self.id_)
        idx = self.id_

        ncages = len(self.cages)

        # allocate eggs to cages
        farm_arrivals = self.get_farm_allocation(self.id_, eggs_by_hatch_date)
        logger.debug("\t\t\tFarm allocation is %s", farm_arrivals)

        arrivals_per_cage = self.get_cage_allocation(ncages, farm_arrivals)
        logger.debug("\t\t\tCage allocation is %s", arrivals_per_cage)

        total, by_cage, by_geno_cage = self.get_cage_arrivals_stats(arrivals_per_cage)
        logger.debug("\t\t\tTotal new eggs = %d", total)
        logger.debug("\t\t\tPer cage distribution = %s", str(by_cage))
        self.log(
            "\t\t\tPer cage distribution (as geno) = %s",
            arrivals_per_cage=by_geno_cage,
        )

        # get the arrival time of the egg batch at the allocated
        # destination
        travel_time = self.cfg.rng.poisson(self.cfg.interfarm_times[self.id_][idx])
        arrival_date = cur_date + dt.timedelta(days=travel_time)

        self.distribute_cage_offspring(arrivals_per_cage, arrival_date)

        return arrivals_per_cage, arrival_date

    def update_arrivals(self, arrivals: DispersedOffspring):
        """
        DEPRECATED: only use in single-threaded scenarios.

        :param arrivals: the offspring coming to this lice
        """
        # DEPRECATED
        arrivals_per_cage = arrivals[0]
        arrival_time = arrivals[1]
        for cage, arrival in zip(self.cages, arrivals_per_cage):
            cage.update_arrivals(arrival, arrival_time)

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

    def distribute_cage_offspring(
        self, cage_allocations: CageAllocation, arrival_time: dt.datetime
    ):
        for cage, cage_allocation in zip(self.cages, cage_allocations):
            cage.update_arrivals(cage_allocation, arrival_time)

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

    def get_gym_space(self) -> ObservationSpace:
        """
        :returns: a Gym space for the agent that controls this farm.
        """
        fish_population = (
            np.array([cage.num_fish for cage in self.cages])
            .sum(keepdims=True)
            .astype(np.int64)
        )
        cleaner_fish_pop = np.array([cage.num_cleaner for cage in self.cages]).sum(
            keepdims=True
        )
        aggregation = (
            np.array([cage.aggregation_rate for cage in self.cages])
            .mean(keepdims=True)
            .astype(np.float32)
        )
        reported_aggregation = np.array(
            [self._get_aggregation_rate()], dtype=np.float32
        )
        current_treatments = self.is_treating
        current_treatments_np = np.zeros((TREATMENT_NO + 1), dtype=np.int8)
        if len(current_treatments):
            current_treatments_np[current_treatments] = 1

        current_treatments_np[-1] = any(cage.is_fallowing for cage in self.cages)

        return {
            "aggregation": aggregation,
            "reported_aggregation": reported_aggregation,
            "fish_population": fish_population,
            "cleaner_fish": cleaner_fish_pop,
            "current_treatments": current_treatments_np,
            "allowed_treatments": self.available_treatments,
            "asked_to_treat": np.array([0], dtype=np.int8),
        }

    def _get_aggregation_rate(self):
        """Get the maximum aggregation rate updated to last time it was sampled.
        This operation "consumes" the aggregation rate by setting it to -1 after returning
        the actual value, and is updated once a sampling event occurs.
        This is more efficient than using queues.
        """
        to_return = self._reported_aggregation
        self._reported_aggregation = 0.0
        return to_return

    def _report_sample(self, cur_date):
        def cts(_):
            # report the worst across cages
            self._reported_aggregation = max(
                cage.aggregation_rate for cage in self.cages
            )

        pop_from_queue(self.__sampling_events, cur_date, cts)


@ray.remote
class FarmActor:
    """
    A Ray-based wrapper for farms. When a FarmActor is instantiated, a new process will be
    started and kept alive with its own separate process space containing a number of :class:`Farm`
    objects

    Usage: instantiate all your farm actors will all the farms, register all your farm pools (required
    for lock-free dispersion), then just call step for each day.

    >>> # Spawn 2 farm actors, each managing two farms each
    >>> farm_ids = [0, 1, 2, 3]
    >>> cfg = Config(...) ; rngs = np.random.SeedSequence(0).spawn(2)
    >>> farm_actors = [FarmActor.remote(farm_ids[:2], rngs[0]),
    >>>                FarmActor.remote(farm_ids[2:], rngs[1])]
    >>> ray.get(farm_actors[0].register_farm_pool.remote(farm_actors))
    >>> ray.get(farm_actors[1].register_farm_pool.remote(farm_actors))
    >>> # now you can simply iterate for every day of your simulation
    >>> for day in range(T):
    >>>     res1, res2 = ray.get([farm_actors[i].step.remote(...) for i in range(2)]
    >>>     # res1 will be a StepResponse object

    Users are recommended to use a multi-processing farm pool rather than using this directly.

    **Implementation details**:

    * Ray Queues are expensive thus we opted for a lock-less, async-based approach;
      due to ray's limitations you cannot use FarmActors in local mode.
    * Each actor uses a producer/consumer approach: each farm actor will produce as many
      batches as the number of actors, and will be produced directly to the recipient
      space. If a farm crashes synchronisation issues may arise due to starvation.
    """

    def __init__(
        self,
        ids: List[int],
        cfg: Config,
        rng: SeedSequence,
        initial_lice_pop: Optional[GrossLiceDistrib] = None,
    ):
        """
        :param ids: the farm ids to spawn
        :param cfg: the configuration (farm info will be taken from here)
        :param rng: a rng. Make sure no two actors share the same RNG
        :param initial_lice_pop: Initial farm configuration (optional)
        """
        # ugly hack: modify the global logger here.
        # Because this runs on a separate memory space the logger will not be affected.
        # NOTE: This is undocumented.
        if ray.worker and ray.worker.global_worker.mode != ray.LOCAL_MODE:
            global logger
            logger = logging.getLogger(f"SLIM-Farm-{ids}")
            logging.basicConfig(
                level=logging.INFO,
                format="[%(funcName)s@%(filename)s:%(lineno)s]%(levelname)s: %(message)s",
            )
            # for some reason logger.setLevel is not enough...
        else:
            # Avoid sharing the same cfg instance
            cfg = copy.deepcopy(cfg)

        cfg.rng = default_rng(rng)

        self.ids = ids
        self.cfg = cfg
        self.farms = [Farm(id_, cfg, initial_lice_pop) for id_ in ids]
        self.allocation_queue = Queue()
        self._batch_pool: np.ndarray = np.array([])
        self._farm_pool: List[FarmActor] = []

    def register_farm_pool(self, farm_pool: List[FarmActor], batch_pool: np.ndarray):
        """
        Register the farm pool. You **must** call this method before stepping.

        :param farm_pool: a list of :class:`FarmActor`
        :param batch_pool: an array of farm ids owned for each actor.
        """
        self._farm_pool = farm_pool
        self._batch_pool = batch_pool

    def _select(self, id_):
        return next(farm for farm in self.farms if farm.id_ == id_)

    def _find_in_batch(self, idx):
        # Gives the farm actor descriptor containing the farm idx
        return next(i for i, v in enumerate(self._batch_pool) if idx in v)

    def _disperse_offspring(
        self, cur_date, eggs_per_farm: List[GenoDistribByHatchDate]
    ):
        nfarms = self.cfg.nfarms
        for eggs in eggs_per_farm:
            for idx in range(nfarms):
                # trick to save a queue operation
                if idx in self.ids:
                    self._select(idx).disperse_offspring_v2(eggs, cur_date)
                else:
                    self._farm_pool[self._find_in_batch(idx)]._produce.remote(idx, eggs)

        # Each farm i will produce an update for the j-th farm, except when i=j
        # thus there are N-1 farms updates to pop every day
        # if batching is enabled the same queues are recycled.
        for i in range(nfarms - 1):
            farm_id, eggs = self.allocation_queue.get()
            self._select(farm_id).disperse_offspring_v2(eggs, cur_date)

    def _produce(self, idx, eggs):
        self.allocation_queue.put((idx, eggs))

    def step(
        self,
        payload: Dict[int, Tuple[dt.datetime, int, int, GenoRates]],
    ) -> Dict[int, StepResponse]:
        """
        Step the simulation for this farm.

        :param payload: a dictionary (farm_id -> (cur_day, action, ext_influx, genetic ratios))

        :returns: a :class:`StepResponse` with the new gym space.
        """
        temp = {}
        to_return = {}
        eggs_per_farm = []

        for farm in self.farms:
            command = payload[farm.id_]
            cur_date, action, ext_influx, ext_pressure_ratios = command
            # Be sure to clear any flags before this point
            # Then we need to make the
            farm.apply_action(cur_date, action)
            # TODO: why can't we merge cost and profit?
            eggs, cost = farm.update(cur_date, ext_influx, ext_pressure_ratios)
            eggs_per_farm.append(eggs)

            if len(eggs) > 0:
                final_eggs = GenoDistrib.batch_sum(list(eggs.values()))
            else:
                final_eggs = empty_geno_from_cfg(self.cfg)

            profit = farm.get_profit(cur_date)

            # TODO: we could likely just return the final egg distribution...
            temp[farm.id_] = (final_eggs, profit, cost)

        self._disperse_offspring(cur_date, eggs_per_farm)

        for farm in self.farms:
            to_return[farm.id_] = StepResponse(
                *temp[farm.id_], farm.get_gym_space(), farm.logged_data
            )

        return to_return
