"""
Simulate effects of sea lice population on salmon population in
a given body of water.
See README.md for details.
"""
import argparse
from bisect import bisect_left
import datetime as dt
import json
import logging
import dill as pickle
import sys

from pathlib import Path

from src.Config import Config, to_dt
from src.Organisation import Organisation
from typing import Optional


def create_logger():
    """
    Create a logger that logs to both file (in debug mode) and terminal (info).
    """
    logger = logging.getLogger("SeaLiceManagementGame")
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler("SeaLiceManagementGame.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    term_handler = logging.StreamHandler()
    term_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    term_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(term_handler)
    logger.addHandler(file_handler)

    return logger

class Simulator:
    def __init__(self, output_dir: Path, sim_id: str, cfg: Config):
        self.output_dir = output_dir
        self.sim_id = sim_id
        self.cfg = cfg
        self.output_dump_path = self.get_simulation_path(output_dir, sim_id)
        self.cur_day = cfg.start_date
        self.organisation = Organisation(cfg)

    @staticmethod
    def get_simulation_path(path: Path, sim_id: str):
        return path / f"simulation_data_{sim_id}.pickle"

    @staticmethod
    def reload_all_dump(path: Path, sim_id: str):
        """Reload a simulator"""
        data_file = Simulator.get_simulation_path(path, sim_id)
        states = []
        times = []
        with open(data_file, "rb") as fp:
            try:
                sim_state = pickle.load(fp)  # type: Simulator
                states.append(sim_state)
                times.append(sim_state.cur_day)
            except EOFError:
                pass

        return states, times

    @staticmethod
    def reload(path: Path, sim_id: str, timestep: dt.datetime):
        """Reload a simulator state from a dump at a given time"""
        states, times = Simulator.reload_all_dump(path, sim_id)

        idx = bisect_left(times, timestep)
        return states[idx]

    def run_model(self, resume=False):
        """Perform the simulation by running the model.

        :param path: Path to store the results in
        :param sim_id: Simulation name
        :param cfg: Configuration object holding parameter information.
        :param Organisation: the organisation to work on.
        """
        cfg.logger.info("running simulation, saving to %s", self.output_dir)

        # create a file to store the population data from our simulation
        if resume and not self.output_dump_path.exists():
            cfg.logger.warning(f"{self.output_dump_path} could not be found! Creating a new log file.")

        data_file = (self.output_dump_path).open(mode="wb")

        while self.cur_day <= cfg.end_date:
            cfg.logger.debug("Current date = %s", self.cur_day)
            self.organisation.step(self.cur_day)
            self.cur_day += dt.timedelta(days=1)

            # Save the model snapshot
            if self.cfg.save_rate and (self.cur_day - self.cfg.start_date).days % self.cfg.save_rate == 0:
                pickle.dump(self, data_file)

        data_file.close()

def generate_argparse_from_config(cfg_path: str, simulation_path: str):
    parser = argparse.ArgumentParser(description="Sea lice simulation")

    # TODO: we are parsing the config twice.
    with open(cfg_path) as fp:
        cfg_dict = json.load(fp)  # type: dict

    with open(simulation_path) as fp:
        simulation_dict = json.load(fp)  # type: dict

    def add_to_group(group_name, data):
        group = parser.add_argument_group(group_name)
        for k, v in data.items():
            if isinstance(v, list) or isinstance(v["value"], list) or isinstance(v["value"], dict):
                continue # TODO: deal with them later, e.g. prop_a.prop_b for dicts?
            value = v["value"]
            description = v["description"]
            value_type = type(value)


            group.add_argument(f"--{k.replace('_', '-')}", type=value_type, help=description, default=value)

    add_to_group("Organisation parameters", simulation_dict)
    add_to_group("Runtime parameters", cfg_dict)

    return parser


if __name__ == "__main__":
    # NOTE: missing_ok argument of unlink is only supported from Python 3.8
    # TODO: decide if that's ok or whether to revamp the file handling
    if sys.version_info < (3, 8, 0):
        sys.stderr.write("You need python 3.8 or later to run this script\n")
        sys.exit(1)

    # set up and read the command line arguments
    parser = argparse.ArgumentParser(description="Sea lice simulation")
    parser.add_argument("simulation_path",
                        type=str,
                        help="Output directory path. The base directory will be used for logging")
    parser.add_argument("param_dir",
                        type=str,
                        help="Directory of simulation parameters files.")
    parser.add_argument("--quiet",
                        help="Don't log to console or file.",
                        default=False,
                        action="store_true")
    parser.add_argument("--save-rate",
                        help="Interval to dump the simulation state. Useful for visualisation and debugging. Warning: saving a run is a slow operation.",
                        type=int,
                        required=False)
    parser.add_argument("--resume",
                        type=str,
                        help="(DEBUG) resume the simulator from a given timestep. All configuration variables will be ignored.")
    args, unknown = parser.parse_known_args()

    # set up the data folders
    output_path = Path(args.simulation_path)
    simulation_id = output_path.name
    output_basename = output_path.parent

    output_folder = Path.cwd() / output_basename
    output_folder.mkdir(parents=True, exist_ok=True)

    cfg_basename = Path(args.param_dir).parent
    cfg_path = str(cfg_basename / "config.json")

    # set up config class and logger (logging to file and screen.)
    logger = create_logger()

    # silence if needed
    if args.quiet:
        logger.addFilter(lambda record: False)

    config_parser = generate_argparse_from_config(cfg_path, args.param_dir + "/params.json")
    config_args = config_parser.parse_args(unknown)

    # create the config object
    cfg = Config(cfg_path, args.param_dir, logger, vars(config_args), args.save_rate)

    # run the simulation

    if args.resume:
        resume_time = to_dt(args.resume)
        sim = Simulator.reload(output_folder, simulation_id, resume_time)  # type: Simulator
    else:
        sim = Simulator(output_folder, simulation_id, cfg)
    sim.run_model()
