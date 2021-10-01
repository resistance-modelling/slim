"""
Simulate effects of sea lice population on salmon population in
a given body of water.
See README.md for details.
"""
import argparse
import sys
from pathlib import Path

from src import logger, create_logger
from src.Config import Config, to_dt
from src.Simulator import Simulator


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
                        help="Output directory path. The base directory will be used for logging and serialisation " + \
                             "of the simulator state.")
    parser.add_argument("param_dir",
                        type=str,
                        help="Directory of simulation parameters files.")
    parser.add_argument("--quiet",
                        help="Don't log to console or file.",
                        default=False,
                        action="store_true")

    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume",
                              type=str,
                              help="(DEBUG) resume the simulator from a given timestep. All configuration variables will be ignored.")
    resume_group.add_argument("--resume-after",
                              type=int,
                              help="(DEBUG) resume the simulator from a given number of days since the beginning of the simulation. All configuration variables will be ignored.")
    resume_group.add_argument("--save-rate",
                              help="Interval to dump the simulation state. Useful for visualisation and debugging." + \
                                   "Warning: saving a run is a slow operation. Saving and resuming at the same time" + \
                                   "is forbidden. If this is not provided, only the last timestep is serialised",
                              type=int,
                              required=False)

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

    config_parser = Config.generate_argparse_from_config(cfg_schema_path, str(cfg_basename / "params.schema.json"))
    config_args = config_parser.parse_args(unknown)

    # create the config object
    cfg = Config(cfg_path, args.param_dir, vars(config_args), args.save_rate)

    # run the simulation
    if args.resume:
        resume_time = to_dt(args.resume)
        sim: Simulator = Simulator.reload(output_folder, simulation_id, timestamp=resume_time) 
    elif args.resume_after:
        sim: Simulator = Simulator.reload(output_folder, simulation_id, resume_after=args.resume_after) 
    else:
        sim = Simulator(output_folder, simulation_id, cfg)
    sim.run_model()
