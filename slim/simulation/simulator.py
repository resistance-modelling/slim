"""
This module provides two important classes:

* a :class:`Simulator` class for standalone runs;
* a :func:`get_env` method to generate a Simulator

The former is a simple wrapper for the latter so that calling code does not have
to worry about stepping or manually instantiating a policy.
"""

from __future__ import annotations

__all__ = [
    "get_env",
    "Simulator",
    "SimulatorPZEnv",
    "BernoullianPolicy",
    "MosaicPolicy",
    "PilotedPolicy",
    "UntreatedPolicy",
    "get_simulation_path",
    "load_counts",
]

import datetime as dt
import functools
import json
import os

# import dill as pickle
import pickle
import sys
import traceback
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, Iterator, Union, cast

import lz4.frame
import ray
import tqdm
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
import pyarrow as pa
import pyarrow.parquet as paq
from ray.exceptions import RayError

from slim.types.policies import (
    ACTION_SPACE,
    no_observation,
    get_observation_space_schema,
)

from .policies import *
from slim.types.treatments import Treatment
from .config import Config
from .farm import MAX_NUM_APPLICATIONS
from .lice_population import GenoDistrib, from_dict
from .organisation import Organisation


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
        self.cur_day: dt.datetime = cfg.start_date
        self.organisation = Organisation(cfg)
        self.payoff = 0.0
        self.treatment_no = len(Treatment)
        self.last_logs = {}
        self.reset()

        self._action_spaces = {agent: ACTION_SPACE for agent in self.agents}

        # TODO: Question: would it be better to model distinct cage pops?
        self._observation_spaces = get_observation_space_schema(
            self.agents, MAX_NUM_APPLICATIONS
        )

        self._fish_pop = []
        self._adult_females_agg = []

    def __cur_day(self):
        return self.cur_day

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
        return no_observation()

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

            payoffs, logs = self.organisation.step(self.cur_day, actions)
            self.last_logs = logs

            id_to_gym = {i: f"farm_{i}" for i in range(len(self.possible_agents))}

            self.rewards = {id_to_gym[i]: payoff for i, payoff in payoffs.items()}

            self.observations = self.organisation.get_gym_space
            self.cur_day += dt.timedelta(days=1)
            self._fish_pop.append(
                {
                    farm: obs["fish_population"].sum()
                    for farm, obs in self.observations.items()
                }
            )
            self._adult_females_agg.append(
                {
                    farm: obs["aggregation"].max()
                    for farm, obs in self.observations.items()
                }
            )
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

        self._fish_pop = []
        self._adult_females_agg = []

    def stop(self):
        self.organisation.stop()


@ray.remote
class DumpingActor:
    """Takes an output log and serialises it"""

    def __init__(self, output_path: Path, sim_id: str, checkpointing=False):
        """
        :param output_path: the output path
        :param sim_id: the simulation id of the current simulation
        """

        self.parquet_path, self.cfg_path, self.checkpoint_path = get_simulation_path(
            output_path, sim_id
        )

        self.log_lists = defaultdict(lambda: [])

        if checkpointing:
            self.sim_state_file = self.checkpoint_path.open(mode="wb")
            self.sim_state_stream = lz4.frame.open(
                self.sim_state_file,
                "wb",
                compression_level=lz4.frame.COMPRESSIONLEVEL_MINHC,
            )

    def save_sim(self, data: bytes):
        """Dump a checkpoint

        :param data: checkpointed data (already pickled into bytes)
        """
        self.sim_state_stream.write(data)

    def _dump(self, logs: dict):
        timestamp = logs.pop("timestamp")

        for k, v in logs.items():
            if k == "farms":
                for agent, farm_value in v.items():
                    farm_pop = farm_value.pop("farm_population")
                    for k2, v2 in farm_value.items():
                        if isinstance(v2, np.ndarray) and len(v2) == 1:
                            farm_value[k2] = v2[0]
                    self.log_lists["timestamp"].append(timestamp)
                    self.log_lists["farm_name"].append(agent)
                    for farm_k, farm_v in farm_pop.items():
                        # This breaks down the genes
                        self.log_lists[farm_k].append(farm_v)
                    for k2, v2 in farm_value.items():
                        if k2 == "arrivals_per_cage":
                            v2 = GenoDistrib.batch_sum(v2).to_json_dict()
                        elif k2 == "eggs":
                            v2 = v2.gross
                        self.log_lists[k2].append(v2)
            else:
                self.log_lists[k].append(v)

    # def _flush_stream(self, buffer: BytesIO, stream: lz4.frame.LZ4FrameFile):
    #    stream.write(buffer.getvalue())
    #    buffer.seek(0)
    #    buffer.truncate(0)

    def _flush_dump(self):
        table = pa.Table.from_pydict(self.log_lists)
        if not hasattr(self, "_pqwriter"):
            self._pqwriter = paq.ParquetWriter(
                str(self.parquet_path),
                table.schema,
                compression="zstd",
                compression_level=11,
            )
        self._pqwriter.write_table(table)
        self.log_lists.clear()

    def dump(self, logs: dict):
        """
        Dump daily data.
        """
        self._dump(logs)
        if len(self.log_lists["timestamp"]) > 40:
            self._flush_dump()

    def dump_cfg(self, cfg_as_bytes: bytes):
        """
        Save the configuration. This should be called before serialising the dump.
        """
        # This is available only for bytes to avoid having to serialise/deserialise again via Ray
        with self.cfg_path.open("wb") as f:
            f.write(cfg_as_bytes)

    def teardown(self):
        if len(self.log_lists["timestamp"]) > 0:
            self._flush_dump()
        if hasattr(self, "_pqwriter"):
            self._pqwriter.close()

        if hasattr(self, "sim_state_stream"):
            self.sim_state_stream.close()
            self.sim_state_file.close()


class Simulator:  # pragma: no cover
    """
    The main entry point of the simulator.
    This class provides the main loop of the simulation and is typically used by the driver
    when extracting simulation data.
    Furthermore, the class allows the user to perform experience replays by resuming
    snapshots.
    """

    def __init__(self, output_dir: Path, cfg: Config):
        self.env = get_env(cfg)
        self.output_dir = output_dir
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
            with open(strategy, "rb") as f:
                self.policy = pickle.load(f)

        self.cur_day = cfg.start_date
        self.payoff = 0.0

    def run_model(self, *, resume=False, quiet=False):
        """Perform the simulation by running the model.

        :param resume: if True it will resume the simulation
        :param quiet: if True it will disable tqdm's pretty printing.
        """
        if not resume:
            logger.info(
                "running simulation %s, saving to %s", self.cfg.name, self.output_dir
            )
        else:
            logger.info(
                "resuming simulation (searching %s in %s)",
                self.cfg.name,
                self.output_dir,
            )

        # create a file to store the population data from our simulation
        if not resume:
            serialiser = DumpingActor.remote(self.output_dir, self.cfg.name)
            ray.get(serialiser.dump_cfg.remote(pickle.dumps(self.cfg)))
            self.env.reset()

        try:
            num_days = (self.cfg.end_date - self.cur_day).days
            for day in tqdm.trange(num_days, disable=quiet):
                logger.debug("Current date = %s (%d / %d)", self.cur_day, day, num_days)
                for agent in self.env.agents:
                    action = self.policy.predict(self.env.observe(agent), agent)
                    self.env.step(action)

                reward = self.env.rewards
                self.payoff += sum(reward.values())

                if not resume:
                    if (
                        self.cfg.save_rate
                        and (self.cur_day - self.cfg.start_date).days
                        % self.cfg.save_rate
                        == 0
                    ) or day == num_days - 1:
                        expanded_logs = {"timestamp": self.cur_day, "farms": {}}
                        farm_logs = self.env.unwrapped.last_logs
                        for agent in self.env.possible_agents:
                            expanded_logs["farms"][agent] = {
                                **self.env.observe(agent),
                                **farm_logs[agent],
                            }

                        serialiser.dump.remote(expanded_logs)

                    if self.cfg.checkpoint_rate and (
                        (self.cur_day - self.cfg.start_date).days
                        % self.cfg.checkpoint_rate
                        == 0
                        or day == num_days - 1
                    ):
                        # TODO: Simulator already has a field denoting the day
                        serialiser.save_sim.remote(pickle.dumps((self, self.cur_day)))

                self.cur_day += dt.timedelta(days=1)

        except KeyboardInterrupt:
            pass

        except RayError:
            traceback.print_exc(file=sys.stderr)

        finally:
            if not resume:
                ray.get(serialiser.teardown.remote())
            env = self.env.unwrapped
            env.stop()


def get_simulation_path(path: Path, other: Union[Config, str]):  # pragma: no-cover
    """
    :param path: the output path
    :param other: either the simulation id or a Config (containing such simulation id)

    :returns: a triple (artifact path, config path, checkpoint path)
    """
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)

    sim_id = other.name if isinstance(other, Config) else other

    return (
        path / f"simulation_data_{sim_id}.parquet",
        path / f"config_{sim_id}.pickle",
        path / f"checkpoint_{sim_id}.pickle.lz4",
    )


def parse_artifact(
    path: Path, sim_id: str, checkpoint=False
) -> Tuple[pd.DataFrame, Config]:  # pragma: no cover
    """Reload a simulator.

    :param path: the folder containing the artifact
    :param sim_id: the simulation id
    :param checkpoint: if True, load from the checkpointed simulation state.

    :returns: an iterator. The first row is a ``Config``. After this come the serialised logs (as dict),
    one per serialised day.
    """
    logger.info("Loading from a dump...")
    data_file, cfg_path, pickle_file = get_simulation_path(path, sim_id)
    with cfg_path.open("rb") as f:
        cfg = pickle.load(f)
    return paq.read_table(data_file).to_pandas(), cfg


def load_checkpoint(path, sim_id):
    """
    :param path: the folder containing the artifact
    :param sim_id: the simulation id
    """

    logger.info("Loading from a dump...")
    _, __, pickle_file = get_simulation_path(path, sim_id)

    with lz4.frame.open(
        pickle_file, "rb", compression_level=lz4.frame.COMPRESSIONLEVEL_MINHC
    ) as fp:
        while True:
            try:
                yield pickle.load(fp)
            except EOFError:
                break


def dump_optimiser_as_pd(states: List[List[Simulator]]):  # pragma: no cover
    # TODO: This is currently broken.
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
) -> Simulator:  # pragma: no cover
    """Reload a simulator state from a dump at a given time"""
    if timestamp is None and resume_after is None:
        raise ValueError("Resume timestep or range must be provided")

    first_time = None
    for idx, (state, time) in enumerate(parse_artifact(path, sim_id, True)):
        print(time)
        if first_time is None:
            first_time = time
        if resume_after is not None and (time - first_time).days >= resume_after:
            return state
        elif timestamp is not None and timestamp >= time:
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
