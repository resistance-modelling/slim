"""
This module provides the main entry point to any simulation task.
"""

from __future__ import annotations

__all__ = ["Organisation"]

import datetime as dt
import json

from typing import List, Tuple, TYPE_CHECKING, cast, Union, Any, ChainMap

import numpy as np
import ray
from line_profiler_pycharm import profile
from ray.util.queue import Queue as RayQueue, Empty
from typeguard import check_type

from .farm import (
    Farm,
    FarmActor,
    MAX_NUM_CAGES,
    CageAllocation,
    DispersedOffspringPerFarm,
)
from slim.JSONEncoders import CustomFarmEncoder
from .lice_population import (
    GenoDistrib,
    empty_geno_from_cfg,
    geno_config_to_matrix,
    GenoRates,
)

from slim.types.queue import *

from slim.types.policies import no_observation

if TYPE_CHECKING:
    from .config import Config
    from .farm import GenoDistribByHatchDate
    from slim.types.policies import SAMPLED_ACTIONS, ObservationSpace, SimulatorSpace


class Organisation:
    """
    An organisation is a cooperative of :class:`.farm.Farm` s. At every time step farms
    handle the population update logic and produce a number of offspring, which the
    Organisation is supposed to handle.

    Furthermore, farms regularly send messages to
    their farms about their statuses. An organisation can recommend farms to apply treatment
    if one of those has surpassed critical levels (see :meth:`handle_farm_messages`).

    Ultimately, farm updates are used to recompute the external pressure.
    """

    def __init__(self, cfg: Config, *args):
        """
        :param cfg: a Configuration
        :param \\*args: other constructing parameters passed to the underlying :class:`.farm.Farm` s.
        """
        self.name: str = cfg.name
        self.cfg = cfg
        self.extra_farm_args = args

        self.reset()

    def update_genetic_ratios(self, offspring: GenoDistrib):
        """
        Update the genetic ratios after an offspring update.

        :param offspring: the offspring update
        """

        # Because the Dirichlet distribution is the prior of the multinomial distribution
        # (see: https://en.wikipedia.org/wiki/Conjugate_prior#When_likelihood_function_is_a_discrete_distribution )
        # we can use a simple bayesian approach
        if offspring.gross > 0:
            self.genetic_ratios = (
                self.genetic_ratios
                + (offspring.mul_by_scalar(1 / offspring.gross).values())
                * self.cfg.genetic_learning_rate
            )

        # TODO: vectorise this
        alphas = np.empty_like(self.genetic_ratios)
        for i in range(len(self.genetic_ratios)):
            alphas[i] = self.cfg.rng.dirichlet(self.genetic_ratios[i])

        self.external_pressure_ratios = alphas

    def get_external_pressure(self) -> Tuple[int, GenoRates]:
        """
        Get the external pressure. Callers of this function should then invoke
        some distribution to sample the obtained number of lice that respects the probabilities.

        For example:

        >>> org = Organisation(...)
        >>> n, p = org.get_external_pressure()
        >>> new_lice = lice_population.from_ratios(p, n)

        :returns: a pair (number of new lice from reservoir, the ratios to sample from)
        """
        number = int(
            self.averaged_offspring.gross
            * self.cfg.reservoir_offspring_integration_ratio
            + self.cfg.min_ext_pressure
        )
        ratios = self.external_pressure_ratios

        return number, ratios

    def _broadcast(
        self,
        f: Callable[[FarmActor, Optional[dict], Optional[Any]], Any],
        vals: Optional[Dict[int, Any]] = None,
        *args,
    ):
        # mimic an actor pool API, but works on dicts
        future = []
        if vals:
            val_ref = ray.put(vals)
            for farm in self._farm_actors:
                future.append(f(farm, val_ref, *args))
        else:
            for farm in self._farm_actors:
                future.append(f(farm, *args))

        res = ray.get(future)
        to_return = {}
        for r in res:
            if r:
                to_return.update(r)

        return to_return

    @profile
    def step(self, cur_date, actions: SAMPLED_ACTIONS) -> Dict[int, float]:
        """
        Perform an update across all farms.
        After that, some offspring will be distributed into the farms while others will be dispersed into
        the reservoir, thus changing the global external pressure.

        :param cur_date: the current date
        :param actions: if given pass the action to the policy.
        :returns: the cumulated reward from all the farm updates.
        """

        if not self.is_running:
            self.start()

        # update the farms and get the offspring
        # TODO: the "clean" command is pretty useless and must be removed...
        self._broadcast(lambda f: f.clear_flags.remote())
        self.handle_aggregation_rate()

        offspring_per_farm: Dict[int, GenoDistrib] = {}
        payoffs: Dict[int, float] = {}
        spaces: Dict[int, ObservationSpace] = {}

        params = {
            i: (cur_date, action, *self.get_external_pressure())
            for i, action in enumerate(actions)
        }
        results: Dict[
            int, Tuple[GenoDistrib, float, float, ObservationSpace]
        ] = self._broadcast(
            lambda f, v: f.step.remote(v), params
        )

        for k, v in results.items():
            offspring_per_farm.update({k: v[0]})
            payoffs.update({k: v[1] - v[2]})
            spaces.update({k: v[3]})

        # once all the offspring is collected
        # it can be dispersed (interfarm and intercage movement)
        # note: this is done on organisation level because it requires access to
        # other farms - and will allow multiprocessing of the main update

        self.update_offspring_average(offspring_per_farm)
        self.update_genetic_ratios(self.averaged_offspring)

        self._gym_space = {f"farm_{i}": space for i, space in spaces.items()}

        return payoffs

    def update_offspring_average(self, offspring_per_farm: Dict[int, GenoDistrib]):
        t = self.cfg.reservoir_offspring_average

        total_offsprings = GenoDistrib.batch_sum(list(offspring_per_farm.values()))

        self.averaged_offspring = self.averaged_offspring.mul_by_scalar(
            (t - 1) / t
        ).add(total_offsprings.mul_by_scalar(1 / t))

    def to_json(self, **kwargs):
        json_dict = kwargs.copy()
        json_dict.update(self.to_json_dict())
        return json.dumps(json_dict, cls=CustomFarmEncoder, indent=4)

    def to_json_dict(self):
        return {"name": self.name}

    def __repr__(self):
        return json.dumps(self.to_json_dict(), cls=CustomFarmEncoder, indent=4)

    def start(self):
        # Note: pickling won't be fun.
        cfg = self.cfg
        nfarms = cfg.nfarms
        farms_per_process = cfg.farms_per_process

        self._farm_actors = []
        self._offspring_queues = {}

        if farms_per_process == 1:
            batch_pools = np.arange(0, nfarms).reshape((-1, 1))
        else:
            batch_pools = np.array_split(np.arange(0, nfarms), nfarms // farms_per_process)


        for pool in batch_pools:
            q = RayQueue()
            for id_ in pool:
                self._offspring_queues[id_] = q

        for actor_idx, farm_ids in enumerate(batch_pools):
            # concurrency is needed to call gym space.
            # TODO: we could likely add another event in the command queue or merely have the
            #  gym space returned and cached after a step command...
            self._farm_actors.append(
                FarmActor.options(
                    name=f"FarmActor-{actor_idx}"
                ).remote(farm_ids, cfg, self._offspring_queues, *self.extra_farm_args)
            )


    def stop(self):
        print("Stopping farms")
        self._farm_actors = []  # out-of-scope actors will be purged

    @property
    def is_running(self):
        return hasattr(self, "_farm_actors") and len(self._farm_actors) > 0

    def reset(self):
        cfg = self.cfg

        if self.is_running:
            self.stop()

        nfarms = self.cfg.nfarms
        names = [f"farm_{i}" for i in range(nfarms)]
        self._farm_actors: List[FarmActor] = []
        self._offspring_queues: List[RayQueue] = []
        self._gym_space: SimulatorSpace = {
            farm_name: no_observation(MAX_NUM_CAGES) for farm_name in names
        }
        self.genetic_ratios = geno_config_to_matrix(cfg.initial_genetic_ratios)
        self.external_pressure_ratios = geno_config_to_matrix(
            cfg.initial_genetic_ratios
        )
        self.averaged_offspring = empty_geno_from_cfg(cfg)

    @property
    def get_gym_space(self) -> SimulatorSpace:
        """
        Get a gym space for all the agents
        """

        return self._gym_space

    def handle_aggregation_rate(self):
        """
        Handle the messages sent in the farm-to-organisation queue.
        Currently, only one type of message is implemented: :class:`slim.types.QueueTypes.SamplingResponse`.

        If the lice aggregation rate detected and reported on that farm exceeds the configured threshold
        then all the farms will be asked to treat.

        :param cur_date: the current day
        :param farm: the farm that sent the message
        """

        to_broadcast = False
        for space in self.get_gym_space.values():
            if np.max(space["aggregation"]) >= self.cfg.aggregation_rate_threshold:
                to_broadcast = True

        if to_broadcast:
            self._broadcast(lambda f: f.ask_for_treatment.remote())
