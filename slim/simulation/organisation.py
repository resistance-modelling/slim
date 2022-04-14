"""
This module provides the main entry point to any simulation task.
"""

from __future__ import annotations

__all__ = ["Organisation"]

import datetime as dt
import json

from typing import List, Tuple, TYPE_CHECKING, cast, Union

import numpy as np
import ray
from ray.util.queue import Queue as RayQueue, Empty

from .farm import Farm, FarmActor, MAX_NUM_CAGES
from slim.JSONEncoders import CustomFarmEncoder
from .lice_population import (
    GenoDistrib,
    empty_geno_from_cfg,
    geno_config_to_matrix,
    GenoRates,
)
from slim.types.queue import (
    FarmResponse,
    SamplingResponse,
    ClearFlags,
    StepCommand,
    StepResponse,
    DisperseCommand,
    AskForTreatmentCommand,
    DoneCommand,
    DisperseResponse,
    DistributeCageOffspring,
)

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

    def _send_command(self, farm_idx: int, command: type, *args, **kwargs):
        queue = self._org2farm_queues[farm_idx]
        queue.put(command(*args, **kwargs))

    def _broadcast_command(self, command: type, *args, **kwargs):
        for i in range(len(self._farm_actors)):
            self._send_command(i, command, *args, **kwargs)

    def _receive_outputs(self, farm_idx):
        queue = self._farm2org_step_queues[farm_idx]
        response = queue.get()
        # assert isinstance(response, StepResponse)
        return response

    def _reduce(self) -> List[StepResponse]:
        return [self._receive_outputs(i) for i in range(len(self._farm_actors))]

    def _for(self, command, arg_list: Union[dict, list], *extra_args):
        # The lack of currying is a tad annoying
        # TODO: make this look nicer?
        if isinstance(arg_list, list):
            for farm_idx, elem in enumerate(arg_list):
                if isinstance(elem, tuple):
                    self._send_command(farm_idx, command, *elem, *extra_args)
                else:
                    self._send_command(farm_idx, command, elem, *extra_args)

        else:
            for farm_idx, elem in arg_list.items():
                if isinstance(elem, tuple):
                    self._send_command(farm_idx, command, *elem, *extra_args)
                else:
                    self._send_command(farm_idx, command, elem, *extra_args)

    def _map(self, command, arg_list: Union[dict, list], *extra_args):
        self._for(command, arg_list, *extra_args)
        return self._reduce()

    def _disperse(self, cur_date: dt.datetime, offspring_list):
        param = [(cur_date, offspring) for offspring in offspring_list]
        dispersed = cast(List[DisperseResponse], self._map(DisperseCommand, param))

        allocations_per_farm = {i: (cur_date, []) for i in range(self.cfg.nfarms)}

        for from_farm in dispersed:
            for to_farm_idx, to_farm_cages in enumerate(
                from_farm.arrivals_per_farm_cage
            ):
                allocations_per_farm[to_farm_idx][1].append(to_farm_cages)

        self._for(DistributeCageOffspring, allocations_per_farm)

    def step(self, cur_date, actions: SAMPLED_ACTIONS) -> List[float]:
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
        # self._broadcast_command(ClearFlags, cur_date)
        # for farm_queue in self.farm2org_sample_queues:
        #    self.handle_farm_messages(farm_queue)

        offspring_per_farm = []
        payoffs = []

        params = [(cur_date, action) for action in actions]
        results = self._map(StepCommand, params, *self.get_external_pressure())
        for idx, response in enumerate(results):
            offspring_per_farm.append(response.eggs_by_hatch_date)
            payoffs.append(response.profit - response.total_cost)

        # once all the offspring is collected
        # it can be dispersed (interfarm and intercage movement)
        # note: this is done on organisation level because it requires access to
        # other farms - and will allow multiprocessing of the main update

        # for farm_ix, offspring in offspring_dict.items():
        #    self._send_command(farm_ix, DisperseCommand, cur_date, offspring)

        # TODO: merge egg dispersion with stepping, only perform cage assignments separately...
        self._disperse(cur_date, offspring_per_farm)
        self.update_offspring_average(offspring_per_farm)
        self.update_genetic_ratios(self.averaged_offspring)

        return payoffs

    def update_offspring_average(
        self, offspring_per_farm: List[GenoDistribByHatchDate]
    ):
        t = self.cfg.reservoir_offspring_average

        to_add = []
        for farm_offspring in offspring_per_farm:
            if len(farm_offspring):
                to_add.append(GenoDistrib.batch_sum(list(farm_offspring.values())))

        if len(to_add):
            new_batch = GenoDistrib.batch_sum(to_add)
        else:
            new_batch = empty_geno_from_cfg(self.cfg)

        self.averaged_offspring = self.averaged_offspring.mul_by_scalar(
            (t - 1) / t
        ).add(new_batch.mul_by_scalar(1 / t))

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
        self._org2farm_queues = [RayQueue() for _ in range(cfg.nfarms)]
        # Yes, this is bad. I should replace this with a future for the samples (if any?)
        # or maybe just flip the hierarchy
        self._farm2org_step_queues = [RayQueue() for _ in range(cfg.nfarms)]
        self._farm2org_sample_queues = [RayQueue() for _ in range(cfg.nfarms)]

        self._farm_actors = []

        for i in range(cfg.nfarms):
            # concurrency is needed to call gym space.
            # TODO: we could likely add another event in the command queue...
            self._farm_actors.append(
                FarmActor.options(max_concurrency=2).remote(
                    i, cfg, *self.extra_farm_args
                )
            )

        for actor, producer, consumer_step, consumer_sample in zip(
            self._farm_actors,
            self._org2farm_queues,
            self._farm2org_step_queues,
            self._farm2org_sample_queues,
        ):
            self._futures.append(
                actor.run.remote(producer, consumer_step, consumer_sample)
            )

    def stop(self):
        print("Stopping farms")
        self._broadcast_command(DoneCommand, None)
        ray.get(self._futures)
        # TODO: implement process resuming?
        #  In principle, if the queues are empty reentrancy should be possible right?

    @property
    def is_running(self):
        return hasattr(self, "_farm_actors") and len(self._farm_actors) > 0

    def reset(self):
        cfg = self.cfg

        if self.is_running:
            self.stop()

        self._farm_actors: List[FarmActor] = []
        self._farm2org_sample_queues: List[RayQueue] = []
        self._farm2org_step_queues: List[RayQueue] = []
        self._org2farm_queues: List[RayQueue] = []
        self._futures = []

        self.genetic_ratios = geno_config_to_matrix(cfg.initial_genetic_ratios)
        self.external_pressure_ratios = geno_config_to_matrix(
            cfg.initial_genetic_ratios
        )
        self.averaged_offspring = empty_geno_from_cfg(cfg)

    def get_gym_space(self) -> SimulatorSpace:
        """
        Get a gym space for all the agents
        """
        nfarms = self.cfg.nfarms
        names = [f"farm_{i}" for i in range(nfarms)]
        if self.is_running:
            spaces: List[ObservationSpace] = ray.get(
                [actor.get_gym_space.remote() for actor in self._farm_actors]
            )
            return dict(zip(names, spaces))
        else:
            return {farm_name: no_observation(MAX_NUM_CAGES) for farm_name in names}

    def handle_farm_messages(self, farm_sample_queue: RayQueue):
        """
        Handle the messages sent in the farm-to-organisation queue.
        Currently, only one type of message is implemented: :class:`slim.types.QueueTypes.SamplingResponse`.

        If the lice aggregation rate detected and reported on that farm exceeds the configured threshold
        then all the farms will be asked to treat.

        :param cur_date: the current day
        :param farm: the farm that sent the message
        """

        try:
            while True:
                farm_response = farm_sample_queue.get_nowait()
                assert isinstance(
                    farm_response, SamplingResponse
                ), f"Out of sync! Expected a SamplingResponse, got a {type(farm_response)}"

                if farm_response.detected_rate >= self.cfg.aggregation_rate_threshold:
                    # send a treatment command to everyone
                    self._broadcast_command(AskForTreatmentCommand)

        except Empty:
            print("No sampling events, skipping")
            pass
