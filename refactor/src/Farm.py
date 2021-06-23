"""
Defines a Farm class that encapsulates a salmon farm containing several cages.
"""
from src.Cage import Cage
from src.JSONEncoders import CustomFarmEncoder
from src.Config import Config

import json

class Farm:
    """
    Define a salmon farm containing salmon cages. Over time the salmon in the cages grow and are
    subjected to external infestation pressure from sea lice.
    """

    def __init__(self, name, cfg: Config):
        """
        Create a farm.
        :param name: the id of the farm.
        :param cfg: the farm configuration
        """

        self.logger = cfg.logger

        farm_cfg = cfg.farms[name]
        self.name = name
        self.loc_x = farm_cfg.farm_location[0]
        self.loc_y = farm_cfg.farm_location[1]
        self.start_date = farm_cfg.farm_start
        self.treatment_dates = farm_cfg.treatment_dates
        self.cages = [Cage(name, i, cfg) for i in range(farm_cfg.n_cages)]

    def __str__(self):
        """
        Get a human readable string representation of the farm.
        :return: a description of the cage
        """
        cages = ', '.join(str(a) for a in self.cages)
        return f'id: {self.name}, Cages: {cages}'

    def __repr__(self):
        return json.dumps(self, cls=CustomFarmEncoder, indent=4)

    def __eq__(self, other): 
        if not isinstance(other, Farm):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.name == other.name

    def update(self, cur_date, step_size, other_farms, reservoir):
        """
        Update the status of the farm given the growth of fish and change in population of
        parasites.
        :return: none
        """
        self.logger.debug("Updating farm {}".format(self.name))

        # TODO: add new offspring to cages

        # update cages
        for cage in self.cages:
            cage.update(cur_date, step_size, other_farms, reservoir)

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