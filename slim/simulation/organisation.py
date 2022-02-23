"""
This module provides the main entry point to any simulation task.
"""

from __future__ import annotations

__all__ = ["Organisation"]

import datetime as dt
import json

from typing import List, Optional, Tuple, Deque, TYPE_CHECKING, Iterator, Union

from .farm import Farm
from slim.JSONEncoders import CustomFarmEncoder
from .lice_population import GenoDistrib
from slim.types.queue import pop_from_queue, FarmResponse, SamplingResponse

if TYPE_CHECKING:
    from .config import Config
    from .lice_population import GenoDistribDict
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
        :param \*args: other constructing parameters passed to the underlying :class:`.farm.Farm` s.
        """
        self.name: str = cfg.name
        self.cfg = cfg
        self.farms = [Farm(i, cfg, *args) for i in range(cfg.nfarms)]
        self.genetic_ratios = GenoDistrib(cfg.initial_genetic_ratios)
        self.external_pressure_ratios = cfg.initial_genetic_ratios.copy()
        self.offspring_queue = OffspringAveragingQueue(
            self.cfg.reservoir_offspring_average
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
                + (offspring * (1 / offspring.gross)) * self.cfg.genetic_learning_rate
            )
        multinomial_probas = self.cfg.rng.dirichlet(
            tuple(self.genetic_ratios.values())
        ).tolist()
        keys = self.genetic_ratios.keys()
        self.external_pressure_ratios = dict(zip(keys, multinomial_probas))

    def get_external_pressure(self) -> Tuple[int, GenoDistribDict]:
        """
        Get the external pressure. Callers of this function should then invoke
        some distribution to sample the obtained number of lice that respects the probabilities.

        For example:

        >>> n, p = org.get_external_pressure()
        >>> new_lice = GenoDistrib.from_ratios(n, p)


        :returns: a pair (number of new lice from reservoir, the ratios to sample from)
        """
        number = (
            self.offspring_queue.offspring_sum.gross
            * self.cfg.reservoir_offspring_integration_ratio
            + self.cfg.min_ext_pressure
        )
        ratios = self.external_pressure_ratios

        return number, ratios

    def step(
        self, cur_date, actions: Optional[SAMPLED_ACTIONS] = None
    ) -> Union[float, List[float]]:
        """
        Perform an update across all farms.
        After that, some offspring will be distributed into the farms while others will be dispersed into
        the reservoir, thus changing the global external pressure.

        :param cur_date: the current date
        :param actions: if given pass the action to the policy.
        :returns: the cumulated reward from all the farm updates.
        """

        # update the farms and get the offspring
        v2 = actions is not None
        for farm in self.farms:
            self.handle_farm_messages(cur_date, farm, v2=v2)
        if v2:
            for farm, action in zip(self.farms, actions):
                farm.apply_action(cur_date, action)

        offspring_dict = {}
        payoffs = []
        for farm in self.farms:
            offspring, cost = farm.update(cur_date, *self.get_external_pressure())
            offspring_dict[farm.id_] = offspring
            # TODO: take into account other types of disadvantages, not just the mere treatment cost
            # e.g. how are environmental factors measured? Are we supposed to add some coefficients here?
            # TODO: if we take current fish population into account then what happens to the infrastructure cost?
            payoffs.append(farm.get_profit(cur_date) - cost)

        # once all of the offspring is collected
        # it can be dispersed (interfarm and intercage movement)
        # note: this is done on organisation level because it requires access to
        # other farms - and will allow multiprocessing of the main update

        for farm_ix, offspring in offspring_dict.items():
            self.farms[farm_ix].disperse_offspring(offspring, self.farms, cur_date)

        total_offspring = list(offspring_dict.values())
        self.offspring_queue.append(total_offspring)

        self.update_genetic_ratios(self.offspring_queue.average)

        if v2:
            return payoffs
        payoff = sum(payoffs)
        return payoff

    def to_json(self, **kwargs):
        json_dict = kwargs.copy()
        json_dict.update(self.to_json_dict())
        return json.dumps(json_dict, cls=CustomFarmEncoder, indent=4)

    def to_json_dict(self):
        return {"name": self.name, "farms": self.farms}

    def __repr__(self):
        return json.dumps(self.to_json_dict(), cls=CustomFarmEncoder, indent=4)

    def handle_farm_messages(self, cur_date: dt.datetime, farm: Farm, v2=False):
        """
        Handle the messages sent in the farm-to-organisation queue.
        Currently, only one type of message is implemented: :class:`slim.types.QueueTypes.SamplingResponse`.

        If the lice aggregation rate detected and reported on that farm exceeds the configured threshold
        then all the farms will be asked to treat, and the offending farm **can not** defect.
        Note however that this does not mean that treatment will be applied anyway due
        to eligibility requirements. See :meth:`.farm.Farm.ask_for_treatment` for details.

        :param cur_date: the current day
        :param farm: the farm that sent the message
        """

        def cts(farm_response: FarmResponse):
            if (
                isinstance(farm_response, SamplingResponse)
                and farm_response.detected_rate >= self.cfg.aggregation_rate_threshold
            ):
                # send a treatment command to everyone
                for other_farm in self.farms:
                    other_farm.ask_for_treatment(
                        cur_date, v2=v2, can_defect=other_farm != farm
                    )

        pop_from_queue(farm.farm_to_org, cur_date, cts)


class OffspringAveragingQueue:
    """Helper class to compute a rolling average"""

    def __init__(self, rolling_average: int):
        """
        :param rolling_average: the maximum length to consider
        """
        self._queue = Deque[GenoDistrib](  # pytype: disable=not-callable
            maxlen=rolling_average
        )
        self.offspring_sum = GenoDistrib()

    def __len__(self):
        return len(self._queue)

    @property
    def rolling_average_factor(self):
        return self._queue.maxlen

    def append(self, offspring_per_farm: List[GenoDistribByHatchDate]):
        """Add an element to our rolling average. This will automatically pop elements on the left."""
        to_add = GenoDistrib()
        for farm_offspring in offspring_per_farm:
            to_add = to_add + GenoDistrib.batch_sum(list(farm_offspring.values()))
        if len(self) == self.rolling_average_factor:
            self.popleft()
        self._queue.append(to_add)
        self.offspring_sum += to_add

    def popleft(self) -> GenoDistrib:
        element = self._queue.popleft()
        self.offspring_sum = self.offspring_sum - element
        return element

    @property
    def average(self):
        return self.offspring_sum * (1 / self.rolling_average_factor)
