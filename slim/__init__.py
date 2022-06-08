import logging

# one-stop entry point for logging
from typing import Any, Dict
import sys

import argparse
import os.path
from os import execv
from pathlib import Path

from slim.simulation.config import Config


def launch():
    # NOTE: missing_ok argument of unlink is only supported from Python 3.8
    if sys.version_info < (3, 8, 0):
        sys.stderr.write("You need python 3.8 or later to run this script\n")
        exit(1)

    parser = argparse.ArgumentParser(prog="SLIM", description="Sea lice simulation")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("run", help="Run the main simulator.")
    subparsers.add_parser("gui", help="Run the main GUI")
    subparsers.add_parser("fit", help="Run the fitter on reports")
    subparsers.add_parser(
        "optimise", help="(DEPRECATED) run the optimiser on the bernoullian policy"
    )

    x, extra = parser.parse_known_args()
    self_path = Path(__file__).parent

    if x.command == "run":
        path = self_path / "SeaLiceMgmt.py"
    elif x.command == "gui":
        path = self_path / "SeaLiceMgmtGUI.py"
    elif x.command == "fit":
        path = self_path / "Fitter.py"
    elif x.command == "optimise":
        path = self_path / "Optimiser.py"
    else:
        parser.print_help()
        exit(1)
    execv(str(path), ["command"] + extra)


def common_cli_options(description) -> argparse.ArgumentParser:
    """
    Generate common CLI parameters across all the sub-tools

    :param description: The description of the subtool to provide in the help
    """

    parser = argparse.ArgumentParser(description=description)
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

    ray_group = parser.add_argument_group()
    ray_group.add_argument(
        "--ray-address",
        help="Address of a ray cluster address",
    )
    ray_group.add_argument(
        "--ray--redis_password",
        help="Password for the ray cluster",
    )

    return parser


def get_config(parser: argparse.ArgumentParser):
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

    config_parser = Config.generate_argparse_from_config(
        cfg_schema_path, str(cfg_basename / "params.schema.json")
    )
    config_args = config_parser.parse_args(unknown)
    config_args.name = simulation_id

    return (
        Config(cfg_path, args.param_dir, **vars(args), **vars(config_args)),
        args,
        output_folder,
    )
