"""
Simulate effects of sea lice population on salmon population in
a given body of water.
See README.md for details.
"""
import argparse
import cProfile
import logging
from pathlib import Path
import sys

# This needs to be done before import src.
# Maybe a cleaner approach would be to use an env var?
import ray

from src import create_logger
from src.Config import Config, to_dt
from src.Simulator import Simulator

def debugger_is_active() -> bool:
    """Return if the debugger is currently active
    Taken from: https://stackoverflow.com/a/67065084
    """
    gettrace = getattr(sys, 'gettrace', lambda : None) 
    return gettrace() is not None

if __name__ == "__main__":
    # NOTE: missing_ok argument of unlink is only supported from Python 3.8
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
    parser.add_argument("--profile",
                        help="(DEBUG) Dump cProfile stats. The output path is your_simulation_path/profile.bin.",
                        default=False,
                        action="store_true"
                        )
    parser.add_argument('--verbose', '-v',
                        help="Verbosity level (the more 'v', the more verbose)",
                        action='count', default=1)

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
                              type=int)
    resume_group.add_argument("--buffer-rate",
                              help="Number of dumped farm \"days\" to keep in memory before flusing. " + \
                                   "Ignored if --save-rate is not set",
                              default=100,
                              type=int)

    args, unknown = parser.parse_known_args()


    # set up the data folders
    output_path = Path(args.simulation_path)
    simulation_id = output_path.name
    output_basename = output_path.parent

    output_folder = Path.cwd() / output_basename
    output_folder.mkdir(parents=True, exist_ok=True)

    cfg_basename = Path(args.param_dir).parent
    cfg_path = str(cfg_basename / "config.json")
    cfg_schema_path = str(cfg_basename / "config.schema.json")


    config_parser = Config.generate_argparse_from_config(cfg_schema_path, str(cfg_basename / "params.schema.json"))
    config_args = config_parser.parse_args(unknown)

    # Set up config class and logger for the main driver (i.e. spawning ray process)
    # use https://gist.github.com/ms5/9f6df9c42a5f5435be0e for the -v flag counting trick
    log_level = logging.ERROR - (10*args.verbose) if args.verbose > 0 else 0
    create_logger(log_level)

    cfg = Config(
        cfg_path,
        args.param_dir,
        **{
            **vars(config_args),
            "save_rate": args.save_rate,
            "buffer_rate": args.buffer_rate,
            "log_level": log_level
        }
    )

    # Debug mode: make ray spawn only one process
    ray.init(local_mode=debugger_is_active())

    # run the simulation
    resume = True
    if args.resume:
        resume_time = to_dt(args.resume)
        sim: Simulator = Simulator.reload(output_folder, simulation_id, timestamp=resume_time) 
    elif args.resume_after:
        sim: Simulator = Simulator.reload(output_folder, simulation_id, resume_after=args.resume_after) 
    else:
        sim = Simulator(output_folder, simulation_id, cfg)
        resume = False

    if not args.profile:
        sim.run_model(resume)
    else:
        profile_output_path = output_folder / f"profile_{simulation_id}.bin"
        # atexit.register(lambda prof=prof: prof.print_stats(output_unit=1e-3))
        cProfile.run("sim.run_model()", str(profile_output_path))
