"""
Simulate effects of sea lice population on salmon population in
a given body of water.
See README.md for details.
"""
import argparse
import cProfile
import os
import sys
from pathlib import Path

if sys.gettrace() is None or sys.gettrace():
    os.environ["NUMBA_DISABLE_JIT"] = "1"

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
        "--profile",
        help="(DEBUG) Dump cProfile stats. The output path is your_simulation_path/profile.bin.",
        default=False,
        action="store_true",
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
        "--save-rate",
        help="Interval to dump the simulation state. Useful for visualisation and debugging."
        + "Warning: saving a run is a slow operation. Saving and resuming at the same time"
        + "is forbidden. If this is not provided, only the last timestep is serialised",
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
    cfg = Config(cfg_path, args.param_dir, vars(config_args), args.save_rate)

    # run the simulation
    resume = True
    if args.resume:
        resume_time = to_dt(args.resume)
        sim: Simulator = reload(output_folder, simulation_id, timestamp=resume_time)
    elif args.resume_after:
        sim: Simulator = reload(
            output_folder, simulation_id, resume_after=args.resume_after
        )
    else:
        sim = Simulator(output_folder, simulation_id, cfg)
        resume = False

    if not args.profile:
        sim.run_model(resume)
    else:
        profile_output_path = output_folder / f"profile_{simulation_id}.bin"
        # atexit.register(lambda prof=prof: prof.print_stats(output_unit=1e-3))
        cProfile.run("sim.run_model()", str(profile_output_path))
