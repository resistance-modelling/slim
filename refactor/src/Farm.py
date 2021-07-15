"""
Defines a Farm class that encapsulates a salmon farm containing several cages.
"""
import datetime as dt
import json
from collections import Counter

import numpy as np

from src.Cage import Cage
from src.Config import Config
from src.JSONEncoders import CustomFarmEncoder


class Farm:
    """
    Define a salmon farm containing salmon cages. Over time the salmon in the cages grow and are
    subjected to external infestation pressure from sea lice.
    """

    def __init__(self, name: int, cfg: Config):
        """
        Create a farm.
        :param name: the id of the farm.
        :param cfg: the farm configuration
        """

        self.logger = cfg.logger
        self.cfg = cfg

        farm_cfg = cfg.farms[name]
        self.farm_cfg = farm_cfg
        self.name = name
        self.loc_x = farm_cfg.farm_location[0]
        self.loc_y = farm_cfg.farm_location[1]
        self.start_date = farm_cfg.farm_start
        self.treatment_dates = farm_cfg.treatment_dates
        self.cages = [Cage(i, cfg, self) for i in range(farm_cfg.n_cages)]

        self.year_temperatures = self.initialize_temperatures(cfg.farm_data)

    def __str__(self):
        """
        Get a human readable string representation of the farm.
        :return: a description of the cage
        """
        cages = ", ".join(str(a) for a in self.cages)
        return f"id: {self.name}, Cages: {cages}"

    def __repr__(self):
        return json.dumps(self, cls=CustomFarmEncoder, indent=4)

    def __eq__(self, other):
        if not isinstance(other, Farm):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.name == other.name

    def initialize_temperatures(self, temperatures):
        """
        Calculate the mean sea temperature at the northing coordinate of the farm at
        month c_month interpolating data taken from
        www.seatemperature.org
        """

        ardrishaig_data = temperatures["ardrishaig"]
        ardrishaig_temps, ardrishaig_northing = np.array(ardrishaig_data["temperatures"]), ardrishaig_data["northing"]
        tarbert_data = temperatures["tarbert"]
        tarbert_temps, tarbert_northing = np.array(tarbert_data["temperatures"]), tarbert_data["northing"]

        degs = (tarbert_temps - ardrishaig_temps) / abs(tarbert_northing - ardrishaig_northing)

        Ndiff = self.loc_y - tarbert_northing
        return np.round(tarbert_temps - Ndiff * degs, 1)

    def update(self, cur_date: dt.datetime, step_size: int) -> dict:
        """Update the status of the farm given the growth of fish and change
        in population of parasites.

        :param cur_date: Current date
        :param step_size: Step size for tau-leaping
        :return: Dictionary of genotype distributions based on hatch date
        """

        self.logger.debug("Updating farm {}".format(self.name))

        # get number of lice from reservoir to be put in each cage
        pressures_per_cage = self.get_cage_pressures()

        # collate egg batches by hatch time
        eggs_by_hatch_date = {}
        for cage in self.cages:

            # update the cage and collect the offspring info
            egg_distrib, hatch_date = cage.update(cur_date,
                                                  step_size,
                                                  pressures_per_cage[cage.id])

            # update the total offspring info
            if hatch_date in eggs_by_hatch_date:
                eggs_by_hatch_date[hatch_date] += Counter(egg_distrib)
            else:
                eggs_by_hatch_date[hatch_date] = Counter(egg_distrib)

        return eggs_by_hatch_date

    def get_cage_pressures(self) -> list:
        """Get external pressure divided into cages

        :return: List of values of external pressure for each cage
        """

        if len(self.cages) < 1:
            raise Exception("Farm must have at least one cage.")
        if self.cfg.ext_pressure < 0:
            raise Exception("External pressure cannot be negative.")


        # assume equal chances for each cage
        probs_per_cage = np.full(len(self.cages), 1/len(self.cages))

        return list(self.cfg.rng.multinomial(self.cfg.ext_pressure,
                                             probs_per_cage,
                                             size=1)[0])

    def get_egg_allocation(self, nbins: int, eggs_by_hatch_date: dict) -> list:
        """Return allocation of eggs for given number of bins.

        :param nbins: Number of bins to allocate to
        :param eggs_by_hatch_date: Dictionary of genotype distributions based on hatch date
        :return: List of dictionaries of genotype distributions based on hatch date per bin
        """

        if nbins < 1:
            raise Exception("Number of bins must be positive.")

        # dummy implmentation - assumes equal probabilities
        # for both intercage and interfarm travel
        # TODO: complete with actual probabilities
        # probs_per_farm = self.cfg.interfarm_probs[self.name]
        probs_per_bin = np.full(nbins, 1 / nbins)

        # preconstruct the data structure
        hatch_list = [{hatch_date: {} for hatch_date in eggs_by_hatch_date} for n in range(nbins)]
        for hatch_date, geno_dict in eggs_by_hatch_date.items():
            for genotype in geno_dict:
                # generate the bin distribution of this genotype with 
                # this hatch date
                genotype_per_bin = self.cfg.rng.multinomial(geno_dict[genotype],
                                                            probs_per_bin,
                                                            size=1)[0]
                # update the info
                for bin_ix, n in enumerate(genotype_per_bin):
                    hatch_list[bin_ix][hatch_date][genotype] = n

        return hatch_list

    def disperse_offspring(self, eggs_by_hatch_date: dict, farms: list, cur_date: dt.datetime):
        """Allocate new offspring between the farms and cages.

        Assumes the lice can float freely across a given farm so that
        they are not bound to a single cage while not attached to a fish.

        NOTE: This method is not multiprocessing safe.

        :param eggs_by_hatch_date: Dictionary of genotype distributions based
        on hatch date
        :param farms: List of Farm objects
        :param cur_date: Current date of the simulation
        """

        self.logger.debug("\tDispersing total offspring")

        # allocate eggs (in batches based on hatch date) between the farms
        arrivals_per_farm = self.get_egg_allocation(len(farms), eggs_by_hatch_date)

        for farm in farms:

            if farm.name == self.name:
                self.logger.debug("\tFarm {}:".format(farm.name))
            else:
                self.logger.debug("\tFarm {} (current):".format(farm.name))

            # allocate eggs to cages
            farm_arrivals = arrivals_per_farm[farm.name]
            arrivals_per_cage = self.get_egg_allocation(len(farm.cages), farm_arrivals)

            # TODO: add logging
            # self.logger.debug("\t\tTotal egg travels = {}".format(farm_arrivals))
            # self.logger.debug("\t\tCage lice arrivals distribution = {}".format(arrivals_per_cage))

            # get the arrival time of the egg batch at the allocated
            # destination
            travel_time = self.cfg.rng.poisson(self.cfg.interfarm_times[self.name][farm.name])
            arrival_date = cur_date + dt.timedelta(days=travel_time)

            # update the cages
            for cage in farm.cages:
                cage.update_arrivals(arrivals_per_cage[cage.id], arrival_date)

    def to_csv(self):
        """
        Save the contents of this cage as a CSV string for writing to a file later.
        """
        farm_data = "farm, " + str(self.name) + ", " + str(self.loc_x) + ", " + str(self.loc_y)
        cages_data = ""
        for i in range(len(self.cages)):
            # I want to keep a consistent order, hence the loop in this way
            cages_data = cages_data + ", " + self.cages[i].to_csv()
        return farm_data + cages_data


#def d_hatching(c_temp):
#    """
#    TODO: ???
#    """
#    return 3*(3.3 - 0.93*np.log(c_temp/3) -0.16*np.log(c_temp/3)**2) #for 3 broods
#
#def ave_dev_days(del_p, del_m10, del_s, temp_c):
#    """
#    Average dev days using dev_time method, not used in model but handy to have
#        # 5deg: 5.2,-,67.5,2
#        # 10deg: 3.9,-,24,5.3
#        # 15deg: 3.3,-,13.1,9.4
#    :param del_p:
#    :param del_m10:
#    :param del_s:
#    :param temp_c:
#    :return:
#    """
#    return 100 * dev_time(del_p, del_m10, del_s, temp_c, 100) \
#                   - 0.001 * sum([dev_time(del_p, del_m10, del_s, temp_c, i)
#                          for i in np.arange(0, 100.001, 0.001)])
#
#def eudist(point_a, point_b):
#    """
#    Obtain the [Euclidean] distance between two points.
#    :param point_a: the first point (location of farm 1)
#    :param point_b: the second point (location of farm 2)
#    :return: the Euclidean distance between point_a and point_b
#    """
#    return distance.euclidean(point_a, point_b)
#
#def egg_gen(farm, sig, eggs_plus, data):
#    """
#    TODO ???
#    :param farm:
#    :param sig:
#    :param eggs_plus:
#    :param data:
#    :return:
#    """
#    if farm == 0:
#        if np.random.uniform(0, 1, 1) > cfg.prop_influx:
#            bvs = 0.5 * data['resistanceT1'].values + 0.5 * data['mate_resistanceT1'].values + \
#                          np.random.normal(0, sig, eggs_plus) / np.sqrt(2)
#        else:
#            bvs = np.random.normal(cfg.f_muEMB, cfg.f_sigEMB, eggs_plus)
#    else:
#        bvs = 0.5 * data['resistanceT1'].values + 0.5 * data['mate_resistanceT1'].values + \
#              np.random.normal(0, sig, eggs_plus) / np.sqrt(2)
#    return bvs
#
#
