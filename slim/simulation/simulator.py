"""
This module provides two important classes:

* a :class:`Simulator` class for standalone runs;
* a :func:`get_env` method to generate a Simulator

The former is a simple wrapper for the latter so that calling code does not have
to worry about stepping or manually instantiating a policy.
"""

from __future__ import annotations

__all__ = ["get_env", "Simulator", "SimulatorPZEnv", "BernoullianPolicy"]

import datetime as dt
import json
import os

import lz4.frame
from pathlib import Path
from typing import List, Optional, Tuple, Iterator

import dill as pickle
import pandas as pd
import ray
import tqdm

from slim import logger, LoggableMixin
from .farm import MAX_NUM_CAGES, MAX_NUM_APPLICATIONS
from .lice_population import GenoDistrib, from_dict
from .organisation import Organisation
from slim.types.policies import (
    ACTION_SPACE,
    CURRENT_TREATMENTS,
    ObservationSpace,
    NO_ACTION,
    TREATMENT_NO,
    no_observation,
    get_observation_space_schema,
)
from slim.types.treatments import Treatment

import functools

import numpy as np
from gym import spaces
from pettingzoo import AECEnv
from .config import Config
from pettingzoo.utils import agent_selector, wrappers


def get_env(cfg: Config) -> wrappers.OrderEnforcingWrapper:
    """
    Generate a :class:`SimulatorPZEnv` wrapped inside a PettingZoo wrapper.
    Note that nesting wrappers may make accessing attributes more difficult.

    :param cfg: the config to use
    :returns: the wrapped environment
    """
    env = SimulatorPZEnv(cfg)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide variety of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class SimulatorPZEnv(AECEnv):
    """
    The PettingZoo environment. This implements the basic API that any policy expects.

    If you simply want to launch a simulation please just use the :class:`Simulator` class.
    Also consider using the :func:`get_env` helper rather than using this class directly.

    Environment description

    This class models an AEC environment in which each farmer will actively maximise their
    own rewards.

    To better model reality, a farm operator is not omniscient but only has access to:

    * lice aggregation
    * fish population (per-cage)
    * which treatment(s) are being performed right now
    * if the organisation has asked you to treat, e.g. because someone else is treating as well

    The action space is the following:

    * Nothing
    * Fallow (game over - until production cycles are implemented)
    * Apply 1 out of N treatments

    Typically, all treatments will be considered in use for a few days (or months)
    and a repeated treatment command will be silently ignored.

    """

    # TODO: Lice aggregation should only be available once a sampling is performed
    # TODO: what about treating in secret?
    # TODO: what about hydrodynamics? if yes, should we represent a row and a column representing them?

    metadata = {"render.modes": ["human"], "name": "slim_v0"}

    def __init__(self, cfg: Config):
        """
        :param cfg: the config to use
        """
        super(SimulatorPZEnv).__init__()
        self.cfg = cfg
        self.cur_day = cfg.start_date
        self.organisation = Organisation(cfg)
        self.payoff = 0.0
        self.treatment_no = len(Treatment)
        self.reset()

        self._action_spaces = {agent: ACTION_SPACE for agent in self.agents}

        # TODO: Question: would it be better to model distinct cage pops?
        self._observation_spaces = get_observation_space_schema(
            self.agents, MAX_NUM_APPLICATIONS
        )

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
    def no_observation(self):
        return no_observation(MAX_NUM_CAGES)

    def observe(self, agent):
        return self.observations[agent]

    def step(self, action: spaces.Discrete):
        # the agent id is implicitly given by self.agent_selection
        agent = self.agent_selection
        if self.cur_day >= self.cfg.end_date:
            # handles stepping an agent which is already done
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next done agent,  or if there are no more done agents, to the next live agent
            self.dones[agent] = True
        if self.dones[agent]:
            return self._was_done_step(action)

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0

        self._chosen_actions[agent] = action

        if self._agent_selector.is_last():
            actions = [NO_ACTION] * len(self.possible_agents)
            for k, v in self._chosen_actions.items():
                id = int(k[len("farm_") :])
                actions[id] = v

            payoffs = self.organisation.step(self.cur_day, actions)
            id_to_gym = {i: f"farm_{i}" for i in range(len(self.possible_agents))}

            self.rewards = {id_to_gym[i]: payoff for i, payoff in payoffs.items()}

            self.observations = self.organisation.get_gym_space
            self.cur_day += dt.timedelta(days=1)
        else:
            self._clear_rewards()
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

    def reset(self):
        self.organisation.reset()
        self.organisation = Organisation(self.cfg)
        # Gym/PZ assume the agents are a dict
        self.possible_agents = [f"farm_{i}" for i in range(self.cfg.nfarms)]
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.observations = {agent: self.no_observation for agent in self.agents}

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self._chosen_actions = {agent: -1 for agent in self.agents}

    def stop(self):
        self.organisation.stop()


class BernoullianPolicy:
    """
    Perhaps the simplest policy here.
    Never apply treatment except for when asked by the organisation, and in which case whether to apply
    a treatment with a probability :math:`p` or not (:math:`1-p`). In the first case each treatment
    will be chosen with likelihood :math:`q`.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.proba = [farm_cfg.defection_proba for farm_cfg in cfg.farms]
        n = len(Treatment)
        self.treatment_probas = np.ones(n) / n
        self.treatment_threshold = cfg.aggregation_rate_threshold
        self.seed = cfg.seed
        self.reset()

    def _predict(self, asked_to_treat, agg_rate: float, agent: int):
        if not asked_to_treat and np.any(agg_rate < self.treatment_threshold):
            return NO_ACTION

        p = [self.proba[agent], 1 - self.proba[agent]]

        want_to_treat = self.rng.choice([False, True], p=p)
        logger.debug(f"Outcome of the vote: {want_to_treat}")

        if not want_to_treat:
            logger.debug("\tFarm {} refuses to treat".format(agent))
            return NO_ACTION

        picked_treatment = self.rng.choice(
            np.arange(TREATMENT_NO), p=self.treatment_probas
        )
        return picked_treatment

    def predict(self, observation: ObservationSpace, agent: str):
        if isinstance(observation, spaces.Box):
            raise NotImplementedError("Only dict spaces are supported for now")

        agent_id = int(agent[len("farm_") :])
        asked_to_treat = bool(observation["asked_to_treat"])
        return self._predict(asked_to_treat, observation["aggregation"], agent_id)

    def reset(self):
        self.rng = np.random.default_rng(self.seed)


class UntreatedPolicy:
    """
    A treatment stub that performs no action.
    """

    def predict(self, **_kwargs):
        return NO_ACTION


class MosaicPolicy:
    """
    A simple treatment: as soon as farms receive treatment the farmers apply treatment.
    Treatments are selected in rotation
    """

    def __init__(self, cfg: Config):
        self.last_action = [0] * len(cfg.farms)

    def predict(self, observation: ObservationSpace, agent: str):
        if not observation["asked_to_treat"]:
            return NO_ACTION

        agent_id = int(agent[len("farm_") :])
        action = self.last_action[agent_id]
        self.last_action[agent_id] = (action + 1) % TREATMENT_NO
        return action


class Simulator:  # pragma: no cover
    """
    The main entry point of the simulator.
    This class provides the main loop of the simulation and is typically used by the driver
    when extracting simulation data.
    Furthermore, the class allows the user to perform experience replays by resuming
    snapshots.
    """

    def __init__(self, output_dir: Path, sim_id: str, cfg: Config):
        self.env = get_env(cfg)
        self.output_dir = output_dir
        self.sim_id = sim_id
        self.cfg = cfg

        strategy = self.cfg.treatment_strategy
        # TODO: stable-baselines provides a BasePolicy abstract class
        # make these policies inherit from it.
        if strategy == "untreated":
            self.policy = UntreatedPolicy()
        elif strategy == "bernoulli":
            self.policy = BernoullianPolicy(self.cfg)
        elif strategy == "mosaic":
            self.policy = MosaicPolicy(self.cfg)
        else:
            raise ValueError("Unsupported strategy")

        self.output_dump_path = _get_simulation_path(output_dir, sim_id)
        self.cur_day = cfg.start_date
        self.payoff = 0.0

    def run_model(self, resume=False):
        """Perform the simulation by running the model.

        :param path: Path to store the results in
        :param sim_id: Simulation name
        :param cfg: Configuration object holding parameter information.
        :param Organisation: the organisation to work on.
        """
        if not resume:
            logger.info("running simulation, saving to %s", self.output_dir)
        else:
            logger.info("resuming simulation", self.output_dir)

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

        self.env.reset()

        try:

            num_days = (self.cfg.end_date - self.cur_day).days
            for day in tqdm.trange(num_days):
                logger.debug("Current date = %s (%d / %d)", self.cur_day, day, num_days)
                for agent in self.env.agents:
                    action = self.policy.predict(self.env.observe(agent), agent)
                    self.env.step(action)

                reward = self.env.rewards
                self.payoff += sum(reward.values())

                # Save the model snapshot either when checkpointing or during the last iteration
                if not resume:
                    if (
                        self.cfg.save_rate
                        and (self.cur_day - self.cfg.start_date).days
                        % self.cfg.save_rate
                        == 0
                    ) or day == num_days - 1:
                        pickle.dump(self, compressed_stream)
                self.cur_day += dt.timedelta(days=1)

        except KeyboardInterrupt:
            env = self.env
            while not isinstance(env, SimulatorPZEnv):
                env = env.env
            env.stop()

        finally:
            compressed_stream.close()
            data_file.close()


def _get_simulation_path(path: Path, sim_id: str):  # pragma: no-cover
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)

    return path / f"simulation_data_{sim_id}.pickle.lz4"


def reload_all_dump(
    path: Path, sim_id: str
) -> Iterator[Tuple[Simulator, dt.datetime]]:  # pragma: no cover
    """Reload a simulator.

    :param path: the folder containing the artifact
    :param sim_id: the simulation id

    :returns: an iterator of pairs (sim_state, time)
    """
    logger.info("Loading from a dump...")
    data_file = _get_simulation_path(path, sim_id)
    with lz4.frame.open(
        data_file, "rb", compression_level=lz4.frame.COMPRESSIONLEVEL_MINHC
    ) as fp:
        while True:
            try:
                sim_state: Simulator = pickle.load(fp)
                yield sim_state, sim_state.cur_day
            except EOFError:
                break


def dump_as_dataframe(
    states_times_it: Iterator[Tuple[Simulator, dt.datetime]]
) -> Tuple[pd.DataFrame, List[dt.datetime], Config]:  # pragma: no cover
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

        for farm in state.env.unwrapped.organisation.farms:
            key = (time, "farm_" + str(farm.id_))
            is_treating = farm.is_treating
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
        data_to_sum = [from_dict(elem) for elem in data if isinstance(elem, dict)]
        return GenoDistrib.batch_sum(data_to_sum).to_json_dict()

    aggregate_geno_info = dataframe.apply(aggregate_geno, axis=1).apply(pd.Series)
    dataframe["eggs"] = dataframe["eggs"].apply(lambda x: x.gross)

    dataframe = dataframe.join(aggregate_geno_info)

    return dataframe.rename_axis(("timestamp", "farm_id")), times, cfg


def dump_optimiser_as_pd(states: List[List[Simulator]]):  # pragma: no cover
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


def reload(
    path: Path,
    sim_id: str,
    timestamp: Optional[dt.datetime] = None,
    resume_after: Optional[int] = None,
):  # pragma: no cover
    """Reload a simulator state from a dump at a given time"""
    if not (timestamp or resume_after):
        raise ValueError("Resume timestep or range must be provided")

    first_time = None
    for idx, (state, time) in enumerate(reload_all_dump(path, sim_id)):
        if first_time is None:
            first_time = time
        if resume_after and idx == resume_after:
            return state
        elif timestamp and timestamp >= time:
            return state

    raise ValueError("Your chosen timestep is too much in the future!")


def reload_from_optimiser(path: Path) -> List[List[Simulator]]:  # pragma: no cover
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
                        path / f"simulation_data_optimisation_{step}_{it}.pickle.lz4"
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


def load_counts(cfg: Config) -> pd.DataFrame:  # pragma: no cover
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

    farm_populations = {farm.name: farm.n_cages * farm.num_fish for farm in cfg.farms}

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
