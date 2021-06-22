"""
TODO: Describe this simulation.
"""
import argparse
import copy
import datetime as dt
import logging
import sys
from pathlib import Path

from src.Config import Config
from src.Farm import Farm
from src.Reservoir import Reservoir


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

    resistance_bv.write_text('cur_date, muEMB, sigEMB, prop_ext')


def create_logger():
    """
    Create a logger that logs to both file (in debug mode) and terminal (info).
    """
    logger = logging.getLogger('SeaLiceManagementGame')
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler('SeaLiceManagementGame.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
    term_handler = logging.StreamHandler()
    term_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    term_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(term_handler)
    logger.addHandler(file_handler)

    return logger


def initialise(data_folder, sim_id, cfg):
    """
    Create the farms, cages, reservoirs and data files that we will need.
    :return:
    """

    # set up the data files, deleting them if they already exist
    lice_counts = data_folder / "lice_counts_{}.txt".format(sim_id)
    resistance_bv = data_folder / "resistanceBVs_{}.txt".format(sim_id)
    lice_counts.unlink(missing_ok=True)
    resistance_bv.unlink(missing_ok=True)
    resistance_bv.write_text('cur_date, muEMB, sigEMB, prop_ext')

    # Create the reservoir, farms and cages.
    wildlife_reservoir = Reservoir(cfg)
    #farms[0].cages[0].stage = np.random.choice(range(2, 7), cfg.ext_pressure)

    farms = [Farm(i, cfg) for i in range(cfg.nfarms)]
    
    #print(cfg.prop_arrive)
    #print(cfg.hrs_travel)
    return farms, wildlife_reservoir


def run_model(path, sim_id, cfg, farms, reservoir):
    """
    TODO
    :param path: TODO
    :return:
    """
    cfg.logger.info('running simulation, saving to %s', path)
    cur_date = cfg.start_date

    # create a file to store the population data from our simulation
    data_file = path / "simulation_data_{}.txt".format(sim_id)
    data_file.unlink(missing_ok=True)
    data_file = (data_file).open(mode='a')

    while cur_date <= cfg.end_date:
        cfg.logger.debug('Current date = %s', cur_date)

        cur_date += dt.timedelta(days=cfg.tau)
        days = (cur_date - cfg.start_date).days

        # Since we are using a tau-leap like algorithm, we want to update the cages using
        # the current time step. To make things consistent, we copy the current state of
        # farms and this copy as the current state.
        farms_at_date = copy.deepcopy(farms)

        # TODO: can this be done in paralled?
        #from multiprocessing import Pool
        #pool = Pool()
        #result1 = pool.apply_async(solve1, [A])    # evaluate "solve1(A)" asynchronously
        #result2 = pool.apply_async(solve2, [B])    # evaluate "solve2(B)" asynchronously
        #answer1 = result1.get(timeout=10)
        #answer2 = result2.get(timeout=10)

        for farm in farms:
            # update each farm who will need to know about their neighbours. To get the list of neighbbours
            # just remove this farm from the list of farms (the remove method actually removes the element so
            # I need to create another copy!).
            other_farms = copy.deepcopy(farms_at_date)
            other_farms.remove(farm)

            #if days == 1:
            #    resistance_bv.write_text(cur_date, prev_muEMB[farm], prev_sigEMB[farm], prop_ext)

            farm.update(cur_date, cfg.tau, other_farms, reservoir)

        # Now update the reservoir. The farms have already been updated so we don't need
        # to use the copy of the farms
        reservoir.update(cfg.tau, farms)

        # Save the data
        data_str = str(cur_date) + ", " + str(days) + ", " + reservoir.to_csv()
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
    parser.add_argument("path", type=str, help="Output directory path")
    parser.add_argument("id", type=str, help="Experiment name")
    parser.add_argument("cfg_path", type=str, help="Path to config JSON file")
    args = parser.parse_args()

    # set up the data folders
    output_folder = Path.cwd() / "outputs" / args.path
    output_folder.mkdir(parents=True, exist_ok=True)

    # set up config class and logger (logging to file and screen.)
    logger = create_logger()
    cfg = Config(args.cfg_path, logger)

    # run the simulation
    FARMS, RESERVOIR = initialise(output_folder, args.id, cfg)
    run_model(output_folder, args.id, cfg, FARMS, RESERVOIR)
