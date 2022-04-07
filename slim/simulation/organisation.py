"""
This module provides the main entry point to any simulation task.
"""

from __future__ import annotations

__all__ = ["Organisation"]

import datetime as dt
import json

from typing import List, Tuple, TYPE_CHECKING

import numpy as np
import ray
from ray.util.queue import Queue as RayQueue, Empty

from .farm import Farm, FarmActor
from slim.JSONEncoders import CustomFarmEncoder
from .lice_population import (
    GenoDistrib,
    empty_geno_from_cfg,
    geno_config_to_matrix,
    GenoRates,
)
from slim.types.queue import (
    pop_from_queue,
    FarmResponse,
    SamplingResponse,
    ClearFlags,
    StepCommand,
    StepResponse,
    DisperseCommand,
    AskForTreatmentCommand,
)

if TYPE_CHECKING:
    from .config import Config
    from .farm import GenoDistribByHatchDate
    from slim.types.policies import SAMPLED_ACTIONS


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
        # Note: pickling won't be fun.
        self.org2farm_queues = [RayQueue() for _ in range(cfg.nfarms)]
        # Yes, this is bad. I should replace this with a future for the samples (if any?)
        # or maybe just flip the hierarchy
        self.farm2org_step_queues = [RayQueue() for _ in range(cfg.nfarms)]
        self.farm2org_sample_queues = [RayQueue() for _ in range(cfg.nfarms)]

        self.farm_actors = [
            FarmActor.options(max_concurrency=3).remote(i, cfg, *args)
            for i in range(cfg.nfarms)
        ]
        self.genetic_ratios = geno_config_to_matrix(cfg.initial_genetic_ratios)
        self.external_pressure_ratios = geno_config_to_matrix(
            cfg.initial_genetic_ratios
        )
        self.averaged_offspring = empty_geno_from_cfg(cfg)

        # start the actors
        self._futures = []
        for actor, producer, consumer_step, consumer_sample in zip(
            self.farm_actors,
            self.org2farm_queues,
            self.farm2org_step_queues,
            self.farm2org_sample_queues,
        ):
            self._futures.append(
                actor.run.remote(producer, consumer_step, consumer_sample)
            )

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
        queue = self.org2farm_queues[farm_idx]
        queue.put(command(*args, **kwargs))

    def _broadcast_command(self, command: type, *args, **kwargs):
        for i in range(len(self.farm_actors)):
            self._send_command(i, command, *args, **kwargs)

    def _receive_outputs(self, farm_idx):
        queue = self.farm2org_step_queues[farm_idx]
        response: StepResponse = queue.get(timeout=1.0)
        return response

    def _reduce(self) -> List[StepResponse]:
        return [self._receive_outputs(i) for i in range(len(self.farm_actors))]

    def step(self, cur_date, actions: SAMPLED_ACTIONS) -> List[float]:
        """
        Perform an update across all farms.
        After that, some offspring will be distributed into the farms while others will be dispersed into
        the reservoir, thus changing the global external pressure.

        :param cur_date: the current date
        :param actions: if given pass the action to the policy.
        :returns: the cumulated reward from all the farm updates.
        """

        # update the farms and get the offspring
        self._broadcast_command(ClearFlags, cur_date)
        for farm_queue in self.farm2org_step_queues:
            self.handle_farm_messages(farm_queue)

        offspring_dict = {}
        payoffs = []

        for farm_idx, action in enumerate(actions):
            self._send_command(
                farm_idx, StepCommand, cur_date, action, *self.get_external_pressure()
            )

        results = self._reduce()
        for idx, response in enumerate(results):
            offspring_dict[idx] = response.eggs_by_hatch_date
            payoffs.append(response.profit - response.total_cost)

        # once all the offspring is collected
        # it can be dispersed (interfarm and intercage movement)
        # note: this is done on organisation level because it requires access to
        # other farms - and will allow multiprocessing of the main update

        for farm_ix, offspring in offspring_dict.items():
            self._send_command(farm_ix, DisperseCommand, cur_date, offspring)

        total_offspring = list(offspring_dict.values())
        self.update_offspring_average(total_offspring)
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

    def handle_farm_messages(self, farm_queue: RayQueue):
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
                farm_response = farm_queue.get_nowait()
                assert isinstance(
                    farm_response, SamplingResponse
                ), f"Out of sync! Expected a SamplingResponse, got a {type(farm_response)}"

                if farm_response.detected_rate >= self.cfg.aggregation_rate_threshold:
                    # send a treatment command to everyone
                    self._broadcast_command(AskForTreatmentCommand)

        except Empty:
            pass
