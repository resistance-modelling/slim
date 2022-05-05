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

from slim import logger, create_logger
from slim.simulation.simulator import Simulator, reload
from slim.simulation.config import Config, to_dt


if __name__ == "__main__":
    # NOTE: missing_ok argument of unlink is only supported from Python 3.8
    if sys.version_info < (3, 8, 0):
        sys.stderr.write("You need python 3.8 or later to run this script\n")
        sys.exit(1)

    # set up and read the command line arguments
    parser = argparse.ArgumentParser(description="Sea lice simulation")
    parser.add_argument(
        "simulation_path",
        type=str,
        help="Output directory path. The base directory will be used for logging and serialisation "
        + "of the simulator state.",
    )
    parser.add_argument(
        "param_dir", type=str, help="Directory of simulation parameters files."
    )
    parser.add_argument(
        "--quiet",
        help="Don't log to console or file.",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--save-rate",
        help="Interval (in days) to log the simulation output. Ignored",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--profile",
        help="(DEBUG) Dump cProfile stats. The output path is your_simulation_path/profile.bin. Recommended to run together with --local-mode",
        default=False,
        action="store_true",
    )

    ray_group = parser.add_argument_group()
    ray_group.add_argument(
        "--ray-address",
        help="Address of a ray cluster address",
    )
    ray_group.add_argument(
        "--ray--redis_password",
        help="Password for the ray cluster",
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
    cfg_schema_path = str(cfg_basename / "config.schema.json")

    # silence if needed
    if args.quiet:
        logger.addFilter(lambda record: False)

    config_parser = Config.generate_argparse_from_config(
        cfg_schema_path, str(cfg_basename / "params.schema.json")
    )
    config_args = config_parser.parse_args(unknown)

    # create the config object
    cfg = Config(cfg_path, args.param_dir, **vars(args), **vars(config_args))

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
        sim.run_model(resume)
    else:
        profile_output_path = output_folder / f"profile_{simulation_id}.bin"
        # atexit.register(lambda prof=prof: prof.print_stats(output_unit=1e-3))
        cProfile.run("sim.run_model()", str(profile_output_path))
