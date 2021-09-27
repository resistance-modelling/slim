"""
Entry point to generate a strategy optimiser
"""

import argparse
from pathlib import Path
import sys

from src import logger, create_logger
from src.Simulator import Simulator
from src.Config import Config

def annealing(starting_cfg: Config, **kwargs):
    iterations = kwargs["iterations"]
    # TODO: logging + tqdm would be nice to have
    for i in range(iterations):
        pass


if __name__ == "__main__":
    if sys.version_info < (3, 8, 0):
        sys.stderr.write("You need python 3.8.0 or higher to run this script\n")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Sea lice strategy optimiser")

    # set up and read the command line arguments
    parser.add_argument("output_path",
                        type=str,
                        help="Output directory path. The base directory will be used for logging. All intermediate run files will be generated under subfolders")
    parser.add_argument("param_dir",
                        type=str,
                        help="Directory of simulation parameters files.")
    parser.add_argument("--quiet",
                        help="Don't log to console or file.",
                        default=False,
                        action="store_true")
    parser.add_argument("--iterations",
                        help="Number of iterations to run in the optimiser",
                        type=int,
                        default=10)
    parser.add_argument("--repeat-experiment",
                        help="How many times to repeat the same experiment with autoregressive seeds",
                        type=int,
                        default=3)
    parser.add_argument("--objective",
                        help="What is the objective function to optimise",
                        choices=["cumulative_payoff"],
                        default="cumulative_payoff")

    args, unknown = parser.parse_known_args()

    # set up config class and logger (logging to file and screen.)
    create_logger()

    # set up the data folders
    output_path = Path(args.output_path)
    simulation_id = output_path.name
    output_basename = output_path.parent

    output_folder = Path.cwd() / output_basename
    output_folder.mkdir(parents=True, exist_ok=True)

    cfg_basename = Path(args.param_dir).parent
    cfg_path = str(cfg_basename / "config.json")
    cfg_schema_path = str(cfg_basename / "config.schema.json")

    # silence if needed
    if args.quiet:
        logger.addFilter(lambda record: False)

    config_parser = Config.generate_argparse_from_config(cfg_schema_path, str(cfg_basename / "params.schema.json"))
    config_args = config_parser.parse_args(unknown)

    # create the config object
    cfg = Config(cfg_path, args.param_dir, vars(config_args), args.save_rate)

    annealing(cfg, **vars(args))