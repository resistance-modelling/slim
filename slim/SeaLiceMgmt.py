#!/bin/env python

"""
Simulate effects of sea lice population on salmon population in
a given body of water.
See README.md for details.
"""
import argparse
import cProfile
import sys
from pathlib import Path

import ray

import slim
from slim import logger, create_logger
from slim.simulation.simulator import Simulator, reload
from slim.simulation.config import Config, to_dt


if __name__ == "__main__":
    # set up and read the command line arguments
    parser = slim.common_cli_options("SLIM runner")
    parser.add_argument(
        "--profile",
        help="(DEBUG) Dump cProfile stats. The output path is your_simulation_path/profile.bin. Recommended to run together with --local-mode",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--save-rate",
        help="Interval (in days) to log the simulation output. If 0, do not save run artifacts except for the last day"
        "and other benchmarks",
        type=int,
        default=1,
    )

    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        type=str,
        help="(DEBUG) resume the simulator from a given timestep. All configuration variables will be ignored.",
    )
    resume_group.add_argument(
        "--resume-after",
        type=int,
        help="(DEBUG) resume the simulator from a given number of days since the beginning of the simulation. All configuration variables will be ignored.",
    )
    resume_group.add_argument(
        "--checkpoint-rate",
        help="(DEBUG) Interval to dump the simulation state. Allowed in single-process mode only.",
        type=int,
        required=False,
    )

    cfg, args, output_folder = slim.get_config(parser)

    # set up config class and logger (logging to file and screen.)
    create_logger()
    simulation_id = cfg.name

    # run the simulation
    resume = True
    if args.resume is not None:
        resume_time = to_dt(args.resume)
        sim = reload(output_folder, simulation_id, timestamp=resume_time)
    elif args.resume_after is not None:
        sim = reload(output_folder, simulation_id, resume_after=args.resume_after)
    else:
        sim = Simulator(output_folder, simulation_id, cfg)
        resume = False

    if not args.profile:
        sim.run_model(resume, quiet=args.quiet)
    else:
        profile_output_path = output_folder / f"profile_{simulation_id}.bin"
        # atexit.register(lambda prof=prof: prof.print_stats(output_unit=1e-3))
        cProfile.run("sim.run_model()", str(profile_output_path))
