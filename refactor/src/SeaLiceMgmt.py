"""
Simulate effects of sea lice population on salmon population in
a given body of water.
See README.md for details.
"""
import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

from src.Config import Config
from src.Farm import Farm


def setup(data_folder, sim_id):
    """
    Set up the folders and files to hold the output of the simulation
    :param data_folder: the folder where the output will bbe saved
    :return: None
    """
    lice_counts = data_folder / "lice_counts_{}.txt".format(sim_id)
    resistance_bv = data_folder / "resistanceBVs_{}.txt".format(sim_id)

    # delete the files if they already exist
    # lice_counts.unlink(missing_ok=True)
    # resistance_bv.unlink(missing_ok=False)

    resistance_bv.write_text("cur_date, muEMB, sigEMB, prop_ext")


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
    Create the farms, cages and data files that we will need.
    :return:
    """

    # set up the data files, deleting them if they already exist
    lice_counts = data_folder / "lice_counts_{}.txt".format(sim_id)
    resistance_bv = data_folder / "resistanceBVs_{}.txt".format(sim_id)
    lice_counts.unlink(missing_ok=True)
    resistance_bv.unlink(missing_ok=True)
    resistance_bv.write_text("cur_date, muEMB, sigEMB, prop_ext")

    farms = [Farm(i, cfg) for i in range(cfg.nfarms)]

    # print(cfg.prop_arrive)
    # print(cfg.hrs_travel)
    return farms


def run_model(path, sim_id, cfg, farms):
    """Perform the simulation by running the model.

    :param path: Path to store the results in
    :type path: pathlib.PosixPath
    :param sim_id: Simulation name
    :type sim_id: str
    :param cfg: Configuration object holding parameter information.
    :type cfg: src.Config
    :param farms: List of Farm objects.
    :type farms: list
    """
    cfg.logger.info("running simulation, saving to %s", path)
    cur_date = cfg.start_date

    # create a file to store the population data from our simulation
    data_file = path / "simulation_data_{}.txt".format(sim_id)
    data_file.unlink(missing_ok=True)
    data_file = (data_file).open(mode="a")

    while cur_date <= cfg.end_date:
        cfg.logger.debug("Current date = %s", cur_date)

        cur_date += dt.timedelta(days=cfg.tau)
        days = (cur_date - cfg.start_date).days

        # update the farms and get the offspring
        offspring_dict = {}
        for farm in farms:

            # if days == 1:
            #    resistance_bv.write_text(cur_date, prev_muEMB[farm], prev_sigEMB[farm], prop_ext)

            offspring = farm.update(cur_date, cfg.tau)
            offspring_dict[farm.name] = offspring

        # once all of the offspring is collected
        # it can be dispersed (interfarm and intercage movement)
        # note: this is done on driver level because it requires access to
        # other farms - and will allow muliprocessing of the main update
        for farm_ix, offspring in offspring_dict.items():
            farms[farm_ix].disperse_offspring(offspring, farms, cur_date)

        # Save the data
        data_str = str(cur_date) + ", " + str(days)
        for farm in farms:
            data_str = data_str + ", "
            data_str = data_str + farm.to_csv()
        cfg.logger.info(data_str)
        data_file.write(data_str + "\n")
    data_file.close()


if __name__ == "__main__":
    # NOTE: missing_ok argument of unlink is only supported from Python 3.8
    # TODO: decide if that's ok or whether to revamp the file handling
    if sys.version_info < (3, 8, 0):
        sys.stderr.write("You need python 3.8 or later to run this script\n")
        sys.exit(1)

    # set up and read the command line arguments
    parser = argparse.ArgumentParser(description="Sea lice simulation")
    parser.add_argument("path",
                        type=str, help="Output directory path")
    parser.add_argument("id",
                        type=str,
                        help="Experiment name")
    parser.add_argument("cfg_path",
                        type=str,
                        help="Path to simulation config JSON file")
    parser.add_argument("param_dir",
                        type=str,
                        help="Directory of simulation parameters files.")
    parser.add_argument("--quiet",
                        help="Don't log to console or file.",
                        default=False,
                        action="store_true")
    parser.add_argument("--seed",
                        type=int,
                        help="Provide a seed for random generation.",
                        required=False)
    args = parser.parse_args()

    # set up the data folders
    output_folder = Path.cwd() / "outputs" / args.path
    output_folder.mkdir(parents=True, exist_ok=True)

    # set up config class and logger (logging to file and screen.)
    logger = create_logger()

    # silence if needed
    if args.quiet:
        logger.addFilter(lambda record: False)

    # create the config object
    cfg = Config(args.cfg_path, args.param_dir, logger)

    # set the seed
    if "seed" in args:
        cfg.params.seed = args.seed

    # run the simulation
    FARMS = initialise(output_folder, args.id, cfg)
    run_model(output_folder, args.id, cfg, FARMS)
