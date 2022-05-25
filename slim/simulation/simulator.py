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
    "load_artifact",
    "dump_as_dataframe",
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
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, Iterator, Union, cast

import lz4.frame
import ray
import tqdm
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from ray.exceptions import RayError

from slim.types.policies import (
    ACTION_SPACE,
    no_observation,
    get_observation_space_schema,
)

from .policies import *
from slim.types.treatments import Treatment
from .config import Config
from .farm import MAX_NUM_CAGES, MAX_NUM_APPLICATIONS
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

    def __init__(self, dump_path: Path, pickled_path: Optional[Path] = None):
        self.data_file = dump_path.open(mode="wb")
        self.compressed_stream = lz4.frame.open(
            self.data_file, "wb", compression_level=lz4.frame.COMPRESSIONLEVEL_MINHC
        )
        self.log_buffer = BytesIO()

        if pickled_path:
            self.sim_buffer = BytesIO()
            self.sim_state_file = pickled_path.open(mode="wb")
            self.sim_state_stream = lz4.frame.open(
                self.sim_state_file,
                "wb",
                compression_level=lz4.frame.COMPRESSIONLEVEL_MINHC,
            )

    def _flush(self, buffer: BytesIO, stream: lz4.frame.LZ4FrameFile):
        stream.write(buffer.getvalue())
        buffer.seek(0)
        buffer.truncate(0)

    def _save(self, data: bytes, buffer: BytesIO, stream: lz4.frame.LZ4FrameFile):
        buffer.write(data)
        # if len(buffer.getvalue()) >= (1 << 19):  # ~ 512 KiB
        self._flush(buffer, stream)

    def dump(self, logs: dict):
        self._save(pickle.dumps(logs), self.log_buffer, self.compressed_stream)

    def save_sim(self, sim_state: bytes):
        self._save(sim_state, self.sim_buffer, self.sim_state_stream)

    def teardown(self):
        self._flush(self.log_buffer, self.compressed_stream)
        self.log_buffer.close()
        self.compressed_stream.close()
        self.data_file.close()
        if hasattr(self, "sim_buffer"):
            self._flush(self.sim_buffer, self.sim_state_stream)
            self.sim_buffer.close()
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
            with open(strategy, "rb") as f:
                self.policy = pickle.load(f)

        self.output_dump_path, self.output_dump_pickle = _get_simulation_path(
            output_dir, sim_id
        )
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
            logger.info("resuming simulation from %s", self.output_dump_pickle)

        # create a file to store the population data from our simulation
        if resume and not self.output_dump_pickle.exists():
            logger.warning(
                f"{self.output_dump_pickle} could not be found! Creating a new log file."
            )

        if not resume:
            if self.cfg.checkpoint_rate:
                serialiser = DumpingActor.remote(
                    self.output_dump_path, self.output_dump_pickle
                )
            else:
                serialiser = DumpingActor.remote(self.output_dump_path)
            serialiser.dump.remote(self.cfg)
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


def _get_simulation_path(path: Path, sim_id: str):  # pragma: no-cover
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)

    return (
        path / f"simulation_data_{sim_id}.pickle.lz4",
        path / f"checkpoint_{sim_id}.pickle.lz4",
    )


def parse_artifact(
    path: Path, sim_id: str, checkpoint=False
) -> Iterator[Union[dict, Config, Simulator]]:  # pragma: no cover
    """Reload a simulator.

    :param path: the folder containing the artifact
    :param sim_id: the simulation id
    :param checkpoint: if True, load from the checkpointed simulation state.

    :returns: an iterator. The first row is a ``Config``. After this come the serialised logs (as dict),
    one per serialised day.
    """
    logger.info("Loading from a dump...")
    data_file, pickle_file = _get_simulation_path(path, sim_id)
    path = pickle_file if checkpoint else data_file
    with lz4.frame.open(
        path, "rb", compression_level=lz4.frame.COMPRESSIONLEVEL_MINHC
    ) as fp:
        while True:
            try:
                yield pickle.load(fp)
            except EOFError:
                break


def dump_as_dataframe(
    states_times_it: Iterator[Union[dict, Config]],
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
    for state in states_times_it:
        if isinstance(state, Config):
            cfg = state
            continue

        time = state["timestamp"]
        times.append(time)

        for agent, values in state["farms"].items():
            key = (time, agent)
            farm_pop = values.pop("farm_population")
            farm_data[key] = {**values, **farm_pop}

    dataframe = pd.DataFrame.from_dict(farm_data, orient="index")

    # extract cumulative geno info regardless of the stage
    def aggregate_geno(data):
        data_to_sum = [from_dict(elem) for elem in data if isinstance(elem, dict)]
        return GenoDistrib.batch_sum(data_to_sum).to_json_dict()

    aggregate_geno_info = dataframe.apply(aggregate_geno, axis=1).apply(pd.Series)
    dataframe["eggs"] = dataframe["eggs"].apply(lambda x: x.gross)

    dataframe = dataframe.join(aggregate_geno_info)

    return dataframe.rename_axis(("timestamp", "farm_id")), times, cast(Config, cfg)


def load_artifact(
    path: Path, sim_id: str
) -> Tuple[pd.DataFrame, List[dt.datetime], Config]:
    """Loads an artifact.
    Combines :func:`parse_artifact` and :func:`.dump_as_dataframe`

    :param path: the path where the output is
    :param sim_id: the simulation name

    :returns: the dataframe to dump
    """
    return dump_as_dataframe(parse_artifact(path, sim_id))


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
