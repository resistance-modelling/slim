"""
This module provides the main entry point to any simulation task.
"""

from __future__ import annotations

__all__ = ["Organisation"]

import logging
from abc import ABC, abstractmethod
import datetime as dt
import json

from typing import TYPE_CHECKING

import numpy as np
import ray
from numpy.random import SeedSequence
from ray.util.queue import Queue as RayQueue, Empty

from .farm import (
    FarmActor,
    MAX_NUM_CAGES,
    Farm,
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
    from typing import Dict
    from typing import List, Tuple, TYPE_CHECKING, Any, Dict, Callable, Optional
    from .config import Config
    from .farm import GenoDistribByHatchDate
    from slim.types.policies import SAMPLED_ACTIONS, ObservationSpace, SimulatorSpace


class FarmPool(ABC):
    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def step(
        self,
        cur_date: dt.datetime,
        actions: SAMPLED_ACTIONS,
        ext_influx: int,
        ext_pressure: GenoRates,
    ):
        pass

    def handle_aggregation_rate(self, spaces: SimulatorSpace, threshold: float) -> bool:
        to_broadcast = False
        for space in spaces.values():
            if space["reported_aggregation"].item() >= threshold:
                to_broadcast = True

        return to_broadcast


class SingleProcFarmPool(FarmPool):
    # A single-process farm pool. Useful for debugging
    def __init__(self, cfg: Config, *args):
        nfarms = cfg.nfarms
        self.threshold = cfg.aggregation_rate_threshold
        self.cfg = cfg
        self._farms = [Farm(i, cfg, *args) for i in range(nfarms)]

    def __getitem__(self, i: int):
        # for testing only
        return self._farms[i]

    def __len__(self):
        return len(self._farms)

    def start(self):
        pass

    def stop(self):
        pass

    def step(
        self,
        cur_date: dt.datetime,
        actions: SAMPLED_ACTIONS,
        ext_influx: int,
        ext_pressure: GenoRates,
    ):
        for farm, action in zip(self._farms, actions):
            farm.apply_action(cur_date, action)

        offspring_dict = {}
        payoffs = {}
        for farm in self._farms:
            offspring, cost = farm.update(cur_date, ext_influx, ext_pressure)
            offspring_dict[farm.id_] = offspring
            # TODO: take into account other types of disadvantages, not just the mere treatment cost
            # e.g. how are environmental factors measured? Are we supposed to add some coefficients here?
            # TODO: if we take current fish population into account then what happens to the infrastructure cost?
            payoffs[farm.id_] = farm.get_profit(cur_date) - cost

        # once all the offspring is collected
        # it can be dispersed (interfarm and intercage movement)
        # note: this is done on organisation level because it requires access to
        # other farms - and will allow multiprocessing of the main update

        for farm_ix, offspring in offspring_dict.items():
            offspring_per_farm = self._farms[farm_ix].disperse_offspring(
                offspring, cur_date
            )
            for dest_farm, offspring in zip(self._farms, offspring_per_farm):
                dest_farm.update_arrivals(offspring)

        total_offspring = {
            i: GenoDistrib.batch_sum(list(offspring_dict[i].values()))
            if len(offspring_dict[i]) > 0
            else empty_geno_from_cfg(self.cfg)
            for i in range(len(self))
        }

        logs = {farm.id_: farm.logged_data for farm in self._farms}

        return total_offspring, payoffs, logs

    @property
    def get_gym_space(self):
        return {f"farm_{i}": self._farms[i].get_gym_space() for i in range(len(self))}

    def reset(self):
        pass


class MultiProcFarmPool(FarmPool):
    # A multi-process farm pool.
    def __init__(self, cfg: Config, *args, **kwargs):
        self.cfg = cfg
        self._extra_farm_args = args

        if not ray.is_initialized():
            ray_props = {}
            for k, v in kwargs:
                if k.startswith("ray_"):
                    ray_props[k[len("ray_") :]] = v

            ray.init(**ray_props)

        self.reset()

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

    def start(self):
        cfg = self.cfg
        nfarms = cfg.nfarms
        farms_per_process = cfg.farms_per_process

        self._farm_actors = []
        self._offspring_queues = {}

        if farms_per_process == 1:
            batch_pools = np.arange(0, nfarms).reshape((-1, 1))
        else:
            batch_pools = np.array_split(
                np.arange(0, nfarms), nfarms // farms_per_process
            )

        for pool in batch_pools:
            q = RayQueue()
            for id_ in pool:
                self._offspring_queues[id_] = q

        rngs = SeedSequence(self.cfg.seed).spawn(len(batch_pools))
        for actor_idx, farm_ids in enumerate(batch_pools):
            self._farm_actors.append(
                FarmActor.options(name=f"FarmActor-{actor_idx}").remote(
                    farm_ids,
                    cfg,
                    rngs[actor_idx],
                    self._offspring_queues,
                    *self._extra_farm_args,
                )
            )

    def stop(self):
        print("Stopping farms")
        self._farm_actors = []  # out-of-scope actors will be purged

    def reset(self):
        if self.is_running:
            self.stop()

        names = [f"farm_{i}" for i in range(self.cfg.nfarms)]
        self._farm_actors: List[FarmActor] = []
        self._offspring_queues: Dict[int, RayQueue] = {}
        self._gym_space = {
            farm_name: no_observation(MAX_NUM_CAGES) for farm_name in names
        }

    @property
    def is_running(self):
        return hasattr(self, "_farm_actors") and len(self._farm_actors) > 0

    @property
    def get_gym_space(self) -> SimulatorSpace:
        """
        Get a gym space for all the agents.
        """

        return self._gym_space

    def step(
        self, cur_date, actions: SAMPLED_ACTIONS, ext_influx, ext_ratios
    ) -> Tuple[Dict[int, GenoDistrib], Dict[int, float], Dict[int, Any]]:
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

        offspring_per_farm: Dict[int, GenoDistrib] = {}
        payoffs: Dict[int, float] = {}
        spaces: Dict[int, ObservationSpace] = {}
        logs = {}

        params = {
            i: (cur_date, action, ext_influx, ext_ratios)
            for i, action in enumerate(actions)
        }
        results: Dict[int, StepResponse] = self._broadcast(
            lambda f, v: f.step.remote(v), params
        )

        for k, v in results.items():
            offspring_per_farm.update({k: v.total_offspring})
            payoffs.update({k: v.profit - v.total_cost})
            spaces.update({k: v.observation_space})
            logs.update({k: v.loggable})

        self._gym_space = {f"farm_{i}": space for i, space in spaces.items()}

        return offspring_per_farm, payoffs, logs


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

    def __init__(self, cfg: Config, *args, **kwargs):
        """
        :param cfg: a Configuration
        :param \\*args: other constructing parameters passed to the underlying :class:`.farm.Farm` s.
        :param ray_address: if using multiprocessing, pass the address to ray
        :param ray_redis_password: if using multiprocessing, pass the redis password to ray.
        """
        self.name: str = cfg.name
        self.cfg = cfg

        if cfg.farms_per_process == -1 or cfg.farms_per_process == cfg.nfarms:
            self.farms = SingleProcFarmPool(cfg, *args)
        else:
            self.farms = MultiProcFarmPool(cfg, *args, **kwargs)

        self._gym_space: SimulatorSpace = {}
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

    def step(
        self, cur_date: dt.datetime, actions: SAMPLED_ACTIONS
    ) -> Tuple[Dict[int, float], Dict[str, Any]]:
        """
        Perform an update across all farms.
        After that, some offspring will be distributed into the farms while others will be dispersed into
        the reservoir, thus changing the global external pressure.

        :param cur_date: the current date
        :param actions: if given pass the action to the policy.
        :returns: the cumulated reward from all the farm updates, and logs generated by all the farms
        """

        offspring_per_farm, payoffs, logs = self.farms.step(
            cur_date, actions, *self.get_external_pressure()
        )
        spaces = self.farms.get_gym_space

        self.update_offspring_average(offspring_per_farm)
        self.update_genetic_ratios(self.averaged_offspring)

        to_treat = self.farms.handle_aggregation_rate(
            spaces, self.cfg.aggregation_rate_threshold
        )
        for space in spaces.values():
            space["asked_to_treat"] = np.array([int(to_treat)], dtype=np.int8)

        self._gym_space = spaces
        logs = {f"farm_{i}": log for i, log in logs.items()}

        return payoffs, logs

    def reset(self):
        self.farms.reset()

        self.genetic_ratios = geno_config_to_matrix(self.cfg.initial_genetic_ratios)
        self.external_pressure_ratios = geno_config_to_matrix(
            self.cfg.initial_genetic_ratios
        )
        self.averaged_offspring = empty_geno_from_cfg(self.cfg)

    def stop(self):
        self.farms.stop()

    @property
    def get_gym_space(self):
        return (
            self._gym_space
        )  # {f"farm_{i}": space for i, space in self._gym_space.items()}

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
