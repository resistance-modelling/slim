"""
Simulate effects of sea lice population on salmon population in
a given body of water.
See README.md for details.
"""
import argparse
import datetime as dt
import json
import logging
import sys

from pathlib import Path

from src.Config import Config
from src.Organisation import Organisation
from src.JSONEncoders import CustomFarmEncoder


def setup(data_folder, sim_id):
    """
    Set up the folders and files to hold the output of the simulation
    :param data_folder: the folder where the output will bbe saved
    :return: None
    """
    lice_counts = data_folder / "lice_counts_{}.txt".format(sim_id)


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


def initialise(data_folder, sim_id, cfg):
    """
    Create the Organisation(s) data files that we will need.
    For now only one organisation can run at any time.
    :return: Organisation
    """

    # set up the data files, deleting them if they already exist
    lice_counts = data_folder / "lice_counts_{}.txt".format(sim_id)
    lice_counts.unlink(missing_ok=True)

    return Organisation(cfg)


def run_model(path: Path, sim_id: str, cfg: Config, org: Organisation):
    """Perform the simulation by running the model.

    :param path: Path to store the results in
    :param sim_id: Simulation name
    :param cfg: Configuration object holding parameter information.
    :param Organisation: the organisation to work on.
    """
    cfg.logger.info("running simulation, saving to %s", path)
    cur_date = cfg.start_date

    # create a file to store the population data from our simulation
    data_file = path / "simulation_data_{}.json".format(sim_id)
    data_file.unlink(missing_ok=True)
    data_file = (data_file).open(mode="a")

    while cur_date <= cfg.end_date:
        cfg.logger.debug("Current date = %s", cur_date)
        cur_date += dt.timedelta(days=1)
        org.step(cur_date)

        # Save the data
        # TODO: log as json, serialise as pickle
        data_file.write(org.to_json())
        data_file.write("\n")
        cfg.logger.info(repr(org))
    data_file.close()

def generate_argparse_from_config(cfg_path: str):
    parser = argparse.ArgumentParser(description="Sea lice simulation")

    # TODO: we are parsing the config twice.
    with open(cfg_path) as fp:
        cfg_struct = json.load(fp)  # type: dict

    group = parser.add_argument_group('Runtime parameters')

    for k, v in cfg_struct.items():
        value = v["value"]
        description = v["description"]
        value_type = type(value)

        if isinstance(value, list) or isinstance(value, dict):
            continue # TODO: deal with them later, e.g. prop_a.prop_b for dicts?

        group.add_argument(f"--{k.replace('_', '-')}", type=value_type, help=description, default=value)

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
    args, unknown = parser.parse_known_args()

    # set up the data folders

    output_path = Path(args.simulation_path)
    simulation_id = output_path.name
    output_basename = output_path.parent

    # TODO: do we want to keep an output directory like this?
    output_folder = Path.cwd() / "outputs" / output_basename
    output_folder.mkdir(parents=True, exist_ok=True)

    cfg_basename = Path(args.param_dir).parent
    cfg_path = str(cfg_basename / "config.json")

    # set up config class and logger (logging to file and screen.)
    logger = create_logger()

    # silence if needed
    if args.quiet:
        logger.addFilter(lambda record: False)

    config_parser = generate_argparse_from_config(cfg_path)
    config_parser.parse_args(unknown)

    # create the config object
    cfg = Config(cfg_path, args.param_dir, logger)

    # set the seed
    if "seed" in args:
        cfg.params.seed = args.seed

    # run the simulation
    org = initialise(output_folder, simulation_id, cfg)
    run_model(output_folder, simulation_id, cfg, org)
