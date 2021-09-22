"""
Simulate effects of sea lice population on salmon population in
a given body of water.
See README.md for details.
"""
import argparse
import json
import logging
import sys

from pathlib import Path

from src import logger
from src.Config import Config, to_dt
from src.Simulator import Simulator


def create_logger():
    """
    Create a logger that logs to both file (in debug mode) and terminal (info).
    """
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

    # set up config class and logger (logging to file and screen.)
    create_logger()

    # set up the data folders
    output_path = Path(args.simulation_path)
    simulation_id = output_path.name
    output_basename = output_path.parent

    output_folder = Path.cwd() / output_basename
    output_folder.mkdir(parents=True, exist_ok=True)

    cfg_basename = Path(args.param_dir).parent
    cfg_path = str(cfg_basename / "config.json")

    # silence if needed
    if args.quiet:
        logger.addFilter(lambda record: False)

    config_parser = generate_argparse_from_config(cfg_path, args.param_dir + "/params.json")
    config_args = config_parser.parse_args(unknown)

    # create the config object
    cfg = Config(cfg_path, args.param_dir, vars(config_args), args.save_rate)

    # run the simulation
    if args.resume:
        resume_time = to_dt(args.resume)
        sim = Simulator.reload(output_folder, simulation_id, resume_time)  # type: Simulator
    else:
        sim = Simulator(output_folder, simulation_id, cfg)
    sim.run_model()
