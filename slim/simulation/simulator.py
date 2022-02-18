"""
This module provides two important classes:

* a :class:`Simulator` class for standalone runs;
* a :class:`SimulatorEnv` class for RL-based optimisation.

The former is a simple wrapper for the latter so that calling code does not have
to worry about stepping or manually instantiating a policy.
"""

from __future__ import annotations

__all__ = ["Organisation", "Simulator"]

import datetime as dt
import json
import os

import lz4.frame
from pathlib import Path
from typing import List, Optional, Tuple, Iterator

import dill as pickle
import pandas as pd
import tqdm

from slim import logger
from .lice_population import GenoDistrib
from .organisation import Organisation
from slim.types.TreatmentTypes import Money

import functools

import numpy as np
from gym import spaces
from pettingzoo import AECEnv
from slim.types.policies import ACTION_SPACE, CURRENT_TREATMENTS
from .config import Config
from pettingzoo.utils import agent_selector


class SimulatorPZRawEnv(AECEnv):
    """
    The PettingZoo environment. This implements the basic API that any policy expects.

    If you simply want to launch a simulation please just use the :class:`Simulator` class.

    This class models an AEC environment in which each farmer will actively maximise their
    own rewards.

    To better model reality, a farm operator is not omniscient but only has access to:
    - lice aggregation.
    - fish population (per-cage)
    - which treatment(s) are being performed right now
    - if the organisation has asked you to treat, e.g. because someone else is treating as well

    The action space is the following:
    - Nothing
    - Fallow (game over - until production cycles are implemented)
    - Apply 1 out of N treatments

    Typically, all treatments will be considered in use for a few days (or months)
    and a repeated treatment command will be silently ignored.

    TODO: Lice aggregation should only be available once a sampling is performed
    TODO: what about treating in secret?
    TODO: what about hydrodynamics? if yes, should we represent a row and a column representing them?
    """

    metadata = {"render.modes": ["human"], "name": "slim_v0"}

    def __init__(self, output_dir: Path, sim_id: str, cfg: Config):
        super(SimulatorPZRawEnv).__init__()
        self.output_dir = output_dir
        self.sim_id = sim_id
        self.cfg = cfg
        self.cur_day = cfg.start_date
        self.organisation = Organisation(cfg)
        self.payoff = Money()
        self.reset()

        self._action_spaces = {agent: ACTION_SPACE for agent in self.agents}

        # TODO: Question: would it be better to model distinct cage pops?
        self._observation_spaces = {
            i: spaces.Dict(
                {
                    "aggregation": spaces.Box(low=0, high=20),
                    "fish_population": spaces.Box(low=0, high=1e6, shape=(1, 20)),
                    "current_treatments": CURRENT_TREATMENTS,
                    "asked_to_treat": spaces.MultiBinary(1),  # Yes or no
                }
            )
            for i in range(len(self.agents))
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]

    def render(self, mode="human"):
        # TODO: could we invoke pyqt from here?

        pass

    @property
    @functools.lru_cache(maxsize=None)
    def no_observation(self):
        return {
            "aggregation": 0.0,
            "fish_population": np.zeros((20,), dtype=np.int64),
            "current_treatments": np.zeros((self.treatment_no,), dtype=np.int8),
            "asked_to_treat": np.zeros((1,), dtype=np.int8),
        }

    def observe(self, agent):
        return self.observations[agent]

    def step(self, action: spaces.Discrete):
        # the agent id is implicitly given by self.agent_selection
        if self.cur_day >= self.cfg.end_date:
            return  # TODO

        agent = self.agent_selection
        if self.dones[agent]:
            # handles stepping an agent which is already done
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next done agent,  or if there are no more done agents, to the next live agent
            return self._was_done_step(action)
        self._chosen_actions[agent] = action

        if self._agent_selector.is_last():
            rewards = self.organisation.step(self.cur_day, self._chosen_actions)

    def reset(self):
        self.organisation = Organisation(self.cfg)
        self._agent_to_farm = self._org.farms
        self.possible_agents = list(self._agent_to_farm.keys())
        self.agents = self.possible_agents[:]
        self.dones = {agent: False for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.observations = {agent: self.no_observation for agent in self.agents}

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self._chosen_actions = [-1] * len(self.agents)


class Simulator:  # pragma: no cover
    """
    The main entry point of the simulator.
    This class provides the main loop of the simulation and is typically used by the driver
    when extracting simulation data.
    Furthermore, the class allows the user to perform experience replays by resuming
    snapshots.

    TODO: merge Simulator and SimulatorPZRawEnv.
    """

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

        return path / f"simulation_data_{sim_id}.pickle.lz4"

    @staticmethod
    def reload_all_dump(
        path: Path, sim_id: str
    ) -> Iterator[Tuple[Simulator, dt.datetime]]:
        """Reload a simulator.

        :param path: the folder containing the artifact
        :param sim_id: the simulation id

        :returns: an iterator of pairs (sim_state, time)
        """
        logger.info("Loading from a dump...")
        data_file = Simulator.get_simulation_path(path, sim_id)
        with lz4.frame.open(
            data_file, "rb", compression_level=lz4.frame.COMPRESSIONLEVEL_MINHC
        ) as fp:
            while True:
                try:
                    sim_state: Simulator = pickle.load(fp)
                    yield sim_state, sim_state.cur_day
                except EOFError:
                    break

    @staticmethod
    def dump_as_dataframe(
        states_times_it: Iterator[Tuple[Simulator, dt.datetime]]
    ) -> Tuple[pd.DataFrame, List[dt.datetime], Config]:
        """
        Convert a dump into a pandas dataframe

        Format: index is (timestamp, farm)
        Columns are: "L1" ... "L5f" (nested dict for now...), "a", "A", "Aa" (ints)

        :param states_times_it: a list of states
        :returns: a pair (dataframe, times)
        """

        farm_data = {}
        times = []
        cfg = None
        for state, time in states_times_it:
            times.append(time)
            if cfg is None:
                cfg = state.cfg

            for farm in state.organisation.farms:
                key = (time, "farm_" + str(farm.id_))
                is_treating = all([cage.is_treated(time) for cage in farm.cages])
                farm_data[key] = {
                    **farm.lice_genomics,
                    **farm.logged_data,
                    "num_fish": farm.num_fish,
                    "is_treating": is_treating,
                    "payoff": float(state.payoff),
                }

        dataframe = pd.DataFrame.from_dict(farm_data, orient="index")

        # extract cumulative geno info regardless of the stage
        def aggregate_geno(data):
            data_to_sum = [elem for elem in data if isinstance(elem, dict)]
            return GenoDistrib.batch_sum(data_to_sum, True)

        aggregate_geno_info = dataframe.apply(aggregate_geno, axis=1).apply(pd.Series)
        dataframe["eggs"] = dataframe["eggs"].apply(lambda x: x.gross)

        dataframe = dataframe.join(aggregate_geno_info)

        return dataframe.rename_axis(("timestamp", "farm_id")), times, cfg

    @staticmethod
    def dump_optimiser_as_pd(states: List[List[Simulator]]):
        # TODO: maybe more things may be valuable to visualise
        payoff_row = []
        proba_row = {}
        for state_row in states:
            payoff = sum([float(state.payoff) for state in state_row]) / len(state_row)
            farm_cfgs = state_row[0].cfg.farms
            farms = state_row[0].organisation.farms
            payoff_row.append(payoff)

            for farm_cfg, farm in zip(farm_cfgs, farms):
                proba = farm_cfg.defection_proba
                key = "farm_" + str(farm.id_)
                proba_row.setdefault(key, []).append(proba)

        return pd.DataFrame({"payoff": payoff_row, **proba_row})

    @staticmethod
    def reload(
        path: Path,
        sim_id: str,
        timestamp: Optional[dt.datetime] = None,
        resume_after: Optional[int] = None,
    ):
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
            raise NotADirectoryError(
                f"{path} needs to be a directory to extract from the optimiser"
            )

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
                        pickle_path = (
                            path
                            / f"simulation_data_optimisation_{step}_{it}.pickle.lz4"
                        )
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

    @staticmethod
    def load_counts(cfg: Config) -> pd.DataFrame:
        """Load a lice count, salmon mortality report and salmon survivorship
        estimates.

        :param cfg: the environment configuration
        :returns: the bespoke count
        """
        report = os.path.join(cfg.experiment_id, "report.csv")
        report_df = pd.read_csv(report)
        report_df["date"] = pd.to_datetime(report_df["date"])
        report_df = report_df[report_df["date"] >= cfg.start_date]

        # calculate expected fish population
        # TODO: there are better ways to do this...

        farm_populations = {
            farm.name: farm.n_cages * farm.num_fish for farm in cfg.farms
        }

        def aggregate_mortality(x):
            # It's like np.cumprod but has to deal with fallowing that resets the count...
            res = []
            last_val = 1.0

            for val in x:
                if np.isnan(val):
                    last_val = 1.0
                    res.append(val)
                else:
                    last_val = (1 - 0.01 * val) * last_val
                    res.append(last_val)

            return res

        aggregated_df = report_df.groupby("site_name")[["date", "mortality"]].agg(
            {"date": pd.Series, "mortality": aggregate_mortality}
        )

        dfs = []
        for farm_data in aggregated_df.iterrows():
            farm_name, (dates, mortalities) = farm_data
            dfs.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "survived_fish": (
                            np.array(mortalities) * farm_populations.get(farm_name, 0)
                        ).round(),
                        "site_name": farm_name,
                    }
                )
            )

        new_df = pd.concat(dfs)
        return (
            report_df.set_index(["date", "site_name"])
            .join(new_df.set_index(["date", "site_name"]))
            .reset_index()
        )

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
            logger.warning(
                f"{self.output_dump_path} could not be found! Creating a new log file."
            )

        if not resume:
            data_file = self.output_dump_path.open(mode="wb")
            compressed_stream = lz4.frame.open(
                data_file, "wb", compression_level=lz4.frame.COMPRESSIONLEVEL_MINHC
            )

        num_days = (self.cfg.end_date - self.cur_day).days
        for day in tqdm.trange(num_days):
            logger.debug("Current date = %s (%d / %d)", self.cur_day, day, num_days)
            self.payoff += self.organisation.step(self.cur_day)

            # Save the model snapshot either when checkpointing or during the last iteration
            if not resume:
                if (
                    self.cfg.save_rate
                    and (self.cur_day - self.cfg.start_date).days % self.cfg.save_rate
                    == 0
                ) or day == num_days - 1:
                    pickle.dump(self, compressed_stream)
            self.cur_day += dt.timedelta(days=1)

        if not resume:
            compressed_stream.close()
            data_file.close()
