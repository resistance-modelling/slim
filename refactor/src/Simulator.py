from __future__ import annotations
import datetime as dt
import json
from bisect import bisect_left
from pathlib import Path
from typing import Tuple, List, Optional

import dill as pickle
import pandas as pd
from tqdm import tqdm

from src import logger
from src.Config import Config
from src.Farm import Farm
from src.JSONEncoders import CustomFarmEncoder
from src.LicePopulation import GenoDistrib
from src.QueueTypes import pop_from_queue, FarmResponse, SamplingResponse
from src.TreatmentTypes import Money


class Organisation:
    """
    An organisation is a cooperative of `Farm`s.
    """

    def __init__(self, cfg: Config, *args):
        self.name: str = cfg.name
        self.cfg = cfg
        self.farms = [Farm(i, cfg, *args) for i in range(cfg.nfarms)]

    @property
    def capital(self):
        return sum((farm.current_capital for farm in self.farms), Money())

    def step(self, cur_date) -> Money:
        # days = (cur_date - self.cfg.start_date).days

        # update the farms and get the offspring
        for farm in self.farms:
            self.handle_farm_messages(cur_date, farm)

        offspring_dict = {}
        payoff = Money()
        for farm in self.farms:
            offspring, cost = farm.update(cur_date)
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
        def cts(farm_response: FarmResponse):
            if isinstance(farm_response, SamplingResponse) and \
                    farm_response.detected_rate >= self.cfg.aggregation_rate_threshold:
                # send a treatment command to everyone
                for other_farm in self.farms:
                    other_farm.ask_for_treatment(cur_date, can_defect=other_farm != farm)

        pop_from_queue(farm.farm_to_org, cur_date, cts)


class Simulator:

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

        return path / f"simulation_data_{sim_id}.pickle"

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

        :param states a list of states
        :param times a list of timestamps for each state
        :return a dataframe as described above
        """

        farm_data = {}
        # farms = states[0].organisation.farms
        for state, time in zip(states, times):
            for farm in state.organisation.farms:
                key = (time, "farm_" + str(farm.name))
                farm_data[key] = {**farm.lice_genomics, "num_fish": farm.num_fish}

        dataframe = pd.DataFrame.from_dict(farm_data, orient='index')

        # extract cumulative geno info regardless of the stage
        def aggregate_geno(data):
            data_to_sum = [elem for elem in data if isinstance(elem, dict)]
            return GenoDistrib.batch_sum(data_to_sum, True)

        aggregate_geno_info = dataframe.apply(aggregate_geno, axis=1).apply(pd.Series)

        dataframe = dataframe.join(aggregate_geno_info)

        return dataframe.rename_axis(("timestamp", "farm_name"))

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

        num_days = (self.cfg.end_date - self.cur_day).days
        for day in tqdm(range(num_days)):
            logger.debug("Current date = %s (%d / %d)", self.cur_day, day, num_days)
            self.payoff += self.organisation.step(self.cur_day)

            # Save the model snapshot either when checkpointing or during the last iteration
            if not resume:
                if (self.cfg.save_rate and (self.cur_day - self.cfg.start_date).days % self.cfg.save_rate == 0) \
                        or self.cur_day == self.cfg.end_date:
                    pickle.dump(self, data_file)
            self.cur_day += dt.timedelta(days=1)

        if not resume:
            data_file.close()
