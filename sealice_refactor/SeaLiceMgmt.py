"""
TODO: Describe this simulation.
"""
import sys
import getopt
import logging
from pathlib import Path
import math
import datetime as dt
from scipy.spatial import distance
import numpy as np
import config as cfg
import Farm as frm


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

def initialise():
    """
    TODO
    :return:
    """
    # create cfg.nfarms farms
    farms = [frm.Farm(x, cfg.ncages[x], cfg.ext_pressure) for x in range(cfg.nfarms)]
    farms[0].cages[0].stage = np.random.choice(range(2, 7), cfg.ext_pressure)

    wildlife_reservoir = frm.Reservoir(cfg.ext_pressure)


    #print(cfg.prop_arrive)
    #print(cfg.hrs_travel)
    return farms, wildlife_reservoir

def read_command_line_args(args):
    """
    Read the command line arguments (the output data folder and the simulation id)
    :param args: the command line arguments
    :return: the output data folder and the simulation id
    """
    usage = "usage: SeaLiceMgmt -p <path>"
    data_folder = Path.cwd() / "outputs"
    sim_id = 0

    try:
        opts, args = getopt.getopt(args, "hi:p:", ["id=", "path="])
        for opt, arg in opts:
            if opt == '-h':
                print(usage)
                sys.exit()
            elif opt in ("-i", "--id"):
                sim_id = int(arg)
            elif opt in ("-p", "--path"):
                # create the path and create the data files.
                data_folder = data_folder / arg
    except getopt.GetoptError:
        print(usage)
        sys.exit("")
    return data_folder, sim_id

def run_model(path, farms, reservoir):
    """
    TODO
    :param path: TODO
    :return:
    """
    cfg.logger.info('running simulation, saving to %s', path)
    cur_date = cfg.start_date

    
    while cur_date <= cfg.end_date:
        cfg.logger.debug('Current date = %s', cur_date)

        cur_date += dt.timedelta(days=cfg.tau)
        days = (cur_date - cfg.start_date).days

        # Since we are using a tau-leap like algorithm, we want to update the cages using
        # the current time step. To make things consistent, we copy the current state of 
        # farms and this copy as the current state.
        farms_at_date = farms.copy()

        for farm in farms:
            # update each farm who will need to know about their neighbours. To get the list of neighbbours
            # just remove this farm from the list of farms (the remove method actually removes the element so
            # I need to create another copy!).
            other_farms = farms_at_date.copy()
            other_farms.remove(farm)
            
            #if days == 1:
            #    resistance_bv.write_text(cur_date, prev_muEMB[farm], prev_sigEMB[farm], prop_ext)

            farm.update(other_farms, reservoir)
        
        # Now update the reservoir. The farms have already been updated so we don't need to use the copy of the farms
        reservoir.update(farms)

if __name__ == "__main__":
    # We need at least python 3.8, check it here and abort if necessary
    # NOTE: I can't remember why I needed 3.8 (I think it was to do with path handling) but I've worked around it
    # and only need 3.7
    if sys.version_info < (3, 7, 0):
        sys.stderr.write("You need python 3.7 or later to run this script\n")
        sys.exit(1)

    # Set up the logger, logging to file and screen.
    cfg.logger = logging.getLogger('SeaLiceManagementGame')
    cfg.logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('SeaLiceManagementGame.log', mode='w')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    cfg.logger.addHandler(ch)
    cfg.logger.addHandler(fh)


    # set up the data folders and read the command line arguments.
    data_folder, sim_id = read_command_line_args(sys.argv[1:])
    data_folder.mkdir(parents=True, exist_ok=True)
    setup(data_folder, sim_id)

    # run the simulation
    farms, reservoir = initialise()
    run_model(data_folder, farms, reservoir)
