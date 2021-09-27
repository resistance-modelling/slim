"""
Entry point to generate a strategy optimiser
"""

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np

from src import logger, create_logger
from src.Simulator import Simulator
from src.Config import Config

def get_simulator_from_probas(starting_cfg, probas, out_path, sim_name):
    current_cfg = deepcopy(starting_cfg)
    for farm, defection_proba in zip(current_cfg.farms, probas):
        farm.defection_proba = defection_proba

    return Simulator(out_path, sim_name, current_cfg)

def get_neighbour(probas, rng):
    return np.clip(rng.normal(probas, 0.1), 0, 1)


def annealing(starting_cfg: Config, **kwargs):
    iterations = kwargs["iterations"]
    repeating_iterations = kwargs["repeat_experiment"]
    output_path = kwargs["output_path"]
    # TODO: logging + tqdm would be nice to have
    optimiser_rng =  np.random.default_rng(kwargs["optimiser_seed"])
    farm_no = len(starting_cfg.farms)
    current_best_sol = np.clip(optimiser_rng.normal(0.5, 0.5, farm_no), 0, 1)
    current_best = -np.inf
    for i in range(iterations):
        candidate_probas = get_neighbour(current_best_sol, optimiser_rng)
        payoff_sum = 0.0
        for t in range(repeating_iterations):
            sim_name = f"optimisation_{i}_{t}"
            sim = get_simulator_from_probas(starting_cfg, candidate_probas, output_path, sim_name)
            sim.run_model()
            payoff_sum += float(sim.payoff)

        candidate_payoff = payoff_sum / repeating_iterations

        if candidate_payoff / current_best > optimiser_rng.uniform(0, 1):
            current_best = candidate_payoff
            current_best_sol = candidate_probas

    return current_best_sol


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
    parser.add_argument("--optimiser-seed",
                        help="Seed used by the optimiser",
                        type=int,
                        default=42)

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

    print(annealing(cfg, **vars(args)))
