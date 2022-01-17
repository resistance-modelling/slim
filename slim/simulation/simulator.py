"""
This module provides the main entry point to any simulation task.
"""

from __future__ import annotations

__all__ = ['Organisation', 'Simulator']

import datetime as dt
import json
import lzma
from pathlib import Path
from typing import List, Optional, Tuple, Deque

import dill as pickle
import pandas as pd
import tqdm

from slim import logger
from slim.simulation.config import Config
from slim.simulation.farm import Farm, GenoDistribByHatchDate
from slim.JSONEncoders import CustomFarmEncoder
from slim.simulation.lice_population import GenoDistrib, GenoDistribDict
from slim.types.QueueTypes import pop_from_queue, FarmResponse, SamplingResponse
from slim.types.TreatmentTypes import Money


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
        self.offspring_queue = OffspringAveragingQueue(self.cfg.reservoir_offspring_average)

    def update_genetic_ratios(self, offspring: GenoDistrib):
        """
        Update the genetic ratios after an offspring update.

        :param offspring: the offspring update
        """

        # Because the Dirichlet distribution is the prior of the multinomial distribution
        # (see: https://en.wikipedia.org/wiki/Conjugate_prior#When_likelihood_function_is_a_discrete_distribution )
        # we can use a simple bayesian approach
        if offspring.gross > 0:
            self.genetic_ratios = self.genetic_ratios + (offspring * (1/offspring.gross))
        multinomial_probas = self.cfg.rng.dirichlet(tuple(self.genetic_ratios.values())).tolist()
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
        number = self.offspring_queue.offspring_sum.gross * \
                 self.cfg.reservoir_offspring_integration_ratio + self.cfg.min_ext_pressure
        ratios = self.external_pressure_ratios

        return number, ratios

    def step(self, cur_date) -> Money:
        """
        Perform an update across all farms.
        After that, some offspring will be distributed into the farms while others will be dispersed into
        the reservoir, thus changing the global external pressure.

        :param cur_date: the current date
        :returns: the cumulated reward from all the farm updates.
        """

        # update the farms and get the offspring
        for farm in self.farms:
            self.handle_farm_messages(cur_date, farm)

        offspring_dict = {}
        payoff = Money()
        for farm in self.farms:
            offspring, cost = farm.update(cur_date, *self.get_external_pressure())
            offspring_dict[farm.name] = offspring
            # TODO: take into account other types of disadvantages, not just the mere treatment cost
            # e.g. how are environmental factors measured? Are we supposed to add some coefficients here?
            # TODO: if we take current fish population into account then what happens to the infrastructure cost?
            payoff += farm.get_profit(cur_date) - cost

        # once all of the offspring is collected
        # it can be dispersed (interfarm and intercage movement)
        # note: this is done on organisation level because it requires access to
        # other farms - and will allow multiprocessing of the main update

        for farm_ix, offspring in offspring_dict.items():
            self.farms[farm_ix].disperse_offspring(offspring, self.farms, cur_date)

        total_offspring = list(offspring_dict.values())
        self.offspring_queue.append(total_offspring)

        self.update_genetic_ratios(self.offspring_queue.average)

        return payoff

    def to_json(self, **kwargs):
        json_dict = kwargs.copy()
        json_dict.update(self.to_json_dict())
        return json.dumps(json_dict, cls=CustomFarmEncoder, indent=4)

    def to_json_dict(self):
        return {
            "name": self.name,
            "farms": self.farms
        }

    def __repr__(self):
        return json.dumps(self.to_json_dict(), cls=CustomFarmEncoder, indent=4)

    def handle_farm_messages(self, cur_date: dt.datetime, farm: Farm):
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
            if isinstance(farm_response, SamplingResponse) and \
                    farm_response.detected_rate >= self.cfg.aggregation_rate_threshold:
                # send a treatment command to everyone
                for other_farm in self.farms:
                    other_farm.ask_for_treatment(cur_date, can_defect=other_farm != farm)

        pop_from_queue(farm.farm_to_org, cur_date, cts)


class Simulator: #pragma: no cover
    """The main entry point of the simulator."""
    def __init__(self, output_dir: Path, sim_id: str, cfg: Config):
        self.output_dir = output_dir
        self.sim_id = sim_id
        self.cfg = cfg
        self.output_dump_path = self.get_simulation_path(output_dir, sim_id)
        self.cur_day = cfg.start_date
        self.organisation = Organisation(cfg)
        self.payoff = Money()

    @staticmethod
    def get_simulation_path(path: Path, sim_id: str):
        if not path.is_dir():
            path.mkdir(parents=True, exist_ok=True)

        return path / f"simulation_data_{sim_id}.pickle.xz"

    @staticmethod
    def reload_all_dump(path: Path, sim_id: str):
        """Reload a simulator"""
        logger.info("Loading from a dump...")
        data_file = Simulator.get_simulation_path(path, sim_id)
        states = []
        times = []
        with open(data_file, "rb") as fp:
            while True:
                try:
                    sim_state: Simulator = pickle.load(fp)
                    states.append(sim_state)
                    times.append(sim_state.cur_day)
                except EOFError:
                    logger.debug("Loaded %d states from the dump", len(states))
                    break

        return states, times

    @staticmethod
    def dump_as_pd(states: List[Simulator], times: List[dt.datetime]) -> pd.DataFrame:
        """
        Convert a dump into a pandas dataframe

        Format: index is (timestamp, farm)
        Columns are: "L1" ... "L5f" (nested dict for now...), "a", "A", "Aa" (ints)

        :param states: a list of states
        :param times: a list of timestamps for each state
        :return: a dataframe as described above
        """

        farm_data = {}
        # farms = states[0].organisation.farms
        for state, time in zip(states, times):
            for farm in state.organisation.farms:
                key = (time, "farm_" + str(farm.name))
                is_treating = all([cage.is_treated(time) for cage in farm.cages])
                farm_data[key] = {
                    **farm.lice_genomics,
                    **farm.logged_data,
                    "num_fish": farm.num_fish,
                    "is_treating": is_treating,
                    "payoff": float(state.payoff),
                }

        dataframe = pd.DataFrame.from_dict(farm_data, orient='index')

        # extract cumulative geno info regardless of the stage
        def aggregate_geno(data):
            data_to_sum = [elem for elem in data if isinstance(elem, dict)]
            return GenoDistrib.batch_sum(data_to_sum, True)

        aggregate_geno_info = dataframe.apply(aggregate_geno, axis=1).apply(pd.Series)
        dataframe["eggs"] = dataframe["eggs"].apply(lambda x: x.gross)

        dataframe = dataframe.join(aggregate_geno_info)

        return dataframe.rename_axis(("timestamp", "farm_name"))

    @staticmethod
    def dump_optimiser_as_pd(states: List[List[Simulator]]):
        # TODO: maybe more things may be valuable to visualise
        payoff_row = []
        proba_row = {}
        for state_row in states:
            payoff = sum([float(state.payoff) for state in state_row])/len(state_row)
            farm_cfgs = state_row[0].cfg.farms
            farms = state_row[0].organisation.farms
            payoff_row.append(payoff)

            for farm_cfg, farm in zip(farm_cfgs, farms):
                proba = farm_cfg.defection_proba
                key = "farm_" + str(farm.name)
                proba_row.setdefault(key, []).append(proba)

        return pd.DataFrame({"payoff": payoff_row, **proba_row})


    @staticmethod
    def reload(path: Path, sim_id: str, timestamp: Optional[dt.datetime] = None, resume_after: Optional[int] = None):
        """Reload a simulator state from a dump at a given time"""
        states, times = Simulator.reload_all_dump(path, sim_id)

        if not (timestamp or resume_after):
            raise ValueError("Resume timestep or range must be provided")

        if resume_after:
            return states[resume_after]

        idx = (timestamp - times[0]).days
        return states[idx]

    @staticmethod
    def reload_from_optimiser(path: Path) -> List[List[Simulator]]:
        """Reload from optimiser output

        :param path: the folder containing the optimiser walk and output.
        :return: a matrix of simulation events generated by the walk
        """
        if not path.is_dir():
            raise NotADirectoryError(f"{path} needs to be a directory to extract from the optimiser")

        with open(path / "params.json") as f:
            params = json.load(f)

        if params["method"] == "annealing":
            avg_iterations = params["average_iterations"]
            walk_iterations = params["walk_iterations"]

            res = []
            prematurely_ended = False
            for step in range(walk_iterations):
                new_step = []
                for it in range(avg_iterations):
                    try:
                        pickle_path = path / f"simulation_data_optimisation_{step}_{it}.pickle.xz"
                        print(pickle_path)
                        with pickle_path.open("rb") as f:
                            new_step.append(pickle.load(f))
                    except FileNotFoundError:
                        # The run is incomplete. We reject it and terminate it prematurely
                        prematurely_ended = True
                        break
                    except EOFError:
                        prematurely_ended = True
                        break
                if prematurely_ended:
                    break
                else:
                    res.append(new_step)
            return res
        else:
            raise NotImplementedError()

    def run_model(self, resume=False):
        """Perform the simulation by running the model.

        :param path: Path to store the results in
        :param sim_id: Simulation name
        :param cfg: Configuration object holding parameter information.
        :param Organisation: the organisation to work on.
        """
        logger.info("running simulation, saving to %s", self.output_dir)

        # create a file to store the population data from our simulation
        if resume and not self.output_dump_path.exists():
            logger.warning(f"{self.output_dump_path} could not be found! Creating a new log file.")

        if not resume:
            data_file = self.output_dump_path.open(mode="wb")
            lzf_stream = lzma.open(data_file, "wb")

        num_days = (self.cfg.end_date - self.cur_day).days
        for day in tqdm.trange(num_days):
            logger.debug("Current date = %s (%d / %d)", self.cur_day, day, num_days)
            self.payoff += self.organisation.step(self.cur_day)

            # Save the model snapshot either when checkpointing or during the last iteration
            if not resume:
                if (self.cfg.save_rate and (self.cur_day - self.cfg.start_date).days % self.cfg.save_rate == 0) \
                        or day == num_days - 1:
                    pickle.dump(self, lzf_stream)
            self.cur_day += dt.timedelta(days=1)

        if not resume:
            lzf_stream.close()
            data_file.close()


class OffspringAveragingQueue:
    """Helper class to compute a rolling average"""
    def __init__(self, rolling_average: int):
        """
        :param rolling_average: the maximum length to consider
        """
        self._queue = Deque[GenoDistrib](maxlen=rolling_average) # pytype: disable=not-callable
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
