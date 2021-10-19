"""
Entry point for our optimisation framework
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np

from src import logger, create_logger
from src.Simulator import Simulator
from src.Config import Config

def get_simulator_from_probas(starting_cfg, probas, out_path: Path, sim_name: str, rng):
    current_cfg = deepcopy(starting_cfg)
    for farm, defection_proba in zip(current_cfg.farms, probas):
        farm.defection_proba = defection_proba

    current_cfg.seed = rng.integers(1<<32)
    current_cfg.rng = np.random.default_rng(current_cfg.seed)

    return Simulator(out_path, sim_name, current_cfg)

def get_neighbour(probas, rng):
    return np.clip(rng.normal(probas, 0.1), 0, 1)

def save_settings(method, output_path: Path, **kwargs):
    param_path = output_path / "params.json"

    if method == "annealing":
        payload = {
            "average_iterations": kwargs["repeat_experiment"],
            "walk_iterations": kwargs["iterations"]
        }
    else:
        # other methods
        raise NotImplementedError("Only annealing is supported")

    with param_path.open("w") as f:
        json.dump({"method": method, **payload}, indent=4)

def annealing(starting_cfg: Config,
              iterations,
              repeat_experiment,
              output_path,
              optimiser_seed,
              **kwargs):
   # TODO: logging + tqdm would be nice to have
    farm_no = len(starting_cfg.farms)
    optimiser_rng =  np.random.default_rng(optimiser_seed)
    current_state = np.clip(optimiser_rng.normal(0.5, 0.5, farm_no), 0, 1)
    current_state_sol = 1e-6
    best_sol = current_state.copy()
    best_payoff = -np.inf

    save_settings("annealing", output_path,
                  iterations=iterations, repeat_experiment=repeat_experiment)

    for i in range(iterations):
        logger.info(f"Started iteration {i}")
        candidate_state = get_neighbour(current_state, optimiser_rng)
        logger.info(f"Chosen candidate state {candidate_state}")
        payoff_sum = 0.0
        for t in range(repeat_experiment):
            sim_name = f"optimisation_{i}_{t}"
            sim = get_simulator_from_probas(
                starting_cfg,
                candidate_state,
                Path(output_path),
                sim_name,
                optimiser_rng
            )
            sim.run_model()
            payoff_sum += float(sim.payoff)


        candidate_payoff = payoff_sum / repeat_experiment
        candidate_payoff = max(candidate_payoff, 0)

        if candidate_payoff / current_state_sol > optimiser_rng.uniform(0, 1):
            current_state = candidate_state
            current_state_sol = candidate_payoff

        if best_payoff < candidate_payoff:
            best_payoff = candidate_payoff
            best_sol = candidate_state

    return best_sol


if __name__ == "__main__":
    if sys.version_info < (3, 8, 0):
        sys.stderr.write("You need python 3.8.0 or higher to run this script\n")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Sea lice strategy optimiser")

    # set up and read the command line arguments
    parser.add_argument("output_path",
                        type=str,
                        help="Output directory path. The base directory will be used for logging. " + \
                             "All intermediate run files will be generated under subfolders")
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

    # create the basic config object
    cfg = Config(cfg_path, args.param_dir, vars(config_args))

    print(annealing(cfg, **vars(args)))
