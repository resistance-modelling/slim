"""
Defines a Farm class that encapsulates a salmon farm containing several cages.
"""
import json
import datetime as dt
import numpy as np
from scipy import stats
import config as cfg

#def temp_f(c_month,farm_N):
#    """
#    TODO: Calculate the mean sea temperature of farm_N at month c_month interpolating data taken from
#    www.seatemperature.org
#    """
#    ardrishaig_avetemp = np.array([8.2, 7.55, 7.45, 8.25, 9.65, 11.35, 13.15, 13.75, 13.65, 12.85, 11.75, 9.85])
#    tarbert_avetemp = np.array([8.4, 7.8, 7.7, 8.6, 9.8, 11.65, 13.4, 13.9, 13.9, 13.2, 12.15, 10.2])
#    degs = (tarbert_avetemp[c_month-1]-ardrishaig_avetemp[c_month-1])/(685715-665300)
#    Ndiff = farm_N - 665300 #farm Northing - tarbert Northing
#    return round(tarbert_avetemp[c_month-1] - Ndiff*degs, 1)
#    
#def d_hatching(c_temp):
#    """
#    TODO: ???
#    """
#    return 3*(3.3 - 0.93*np.log(c_temp/3) -0.16*np.log(c_temp/3)**2) #for 3 broods
#
#def fb_mort(days):
#    """
#    Fish background mortality rate, decreasing as in Soares et al 2011
#    :param days: TODO ???
#    :return: TODO ???
#    """
#    return 0.00057 #(1000 + (days - 700)**2)/490000000
#
#def dev_time(del_p, del_m10, del_s, temp_c, n_days):
#    """
#    Probability of developing after n_days days in a stage as given by Aldrin et al 2017
#    :param n_days:
#    :param del_p:
#    :param del_m10:
#    :param del_s:
#    :param temp_c:
#    :return:
#    """
#    unbounded = math.log(2)*del_s*n_days**(del_s-1)*(del_m10*10**del_p/temp_c**del_p)**(-del_s)
#    unbounded[unbounded == 0] = 10**(-30)
#    unbounded[unbounded > 1] = 1
#    return unbounded.astype('float64')
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

class CustomEncoder(json.JSONEncoder):
    """
    Bespoke encoder to encode objects to json. Specifically, numpy arrays and datetime objects
    are not automatically converted to json.
    """
    def default(self, o): # pylint: disable=E0202
        """
        Provide a json string of an object.
        :param o: The object to be encoded as a json string.
        :return: the json representation of o.
        """
        return_str = ''
        if isinstance(o, np.ndarray):
            return_str = str(o)
        elif isinstance(o, dt.datetime):
            return_str = o.strftime("%Y-%m-%d %H:%M:%S")
        else:
            return_str = {'__{}__'.format(o.__class__.__name__): o.__dict__}

        return return_str


    
class Reservoir:
    """
    The reservoir for sea lice, essentially modelled as a sea cage.
    """
    def __init__(self, nplankt):
        """
        Create a cage on a farm
        :param farm: the farm this cage is attached to
        :param label: the label (id) of the cage within the farm
        :param nplankt: TODO ???
        """
        self.date = cfg.start_date
        self.nFish = 4000

        lice_mortality_rates = {'L1': 0.17, 'L2': 0.22, 'L3': 0.008, 'L4': 0.05, 'L5f': 0.02, 'L5m': 0.06}
        lice_population = {'L1': 25, 'L2': 25, 'L3': 25, 'L4': 25, 'L5f': 25, 'L5m': 25}

    def update(self, farms):
        """
        Update the reservoir at the current time step.
        :return:
        """



class Farm:
    """
    Define a salmon farm containing salmon cages. Over time the salmon in the cages grow and are
    subjected to external infestation pressure from sea lice.
    """

    def __init__(self, name, n_cages, nplankt):
        """
        Create a farm
        :param name: the id of the farm
        :param n_cages: the number of cages on the farm
        :param nplankt: TODO ???
        """
        self.name = name
        self.cages = [Cage(self, i, nplankt) for i in range(n_cages)]

    def __str__(self):
        """
        Get a human readable string representation of the farm.
        :return: a description of the cage
        """
        cages = ', '.join(str(a) for a in self.cages)
        return f'id: {self.name}, Cages: {cages}'

    def __repr__(self):
        return json.dumps(self, cls=CustomEncoder, indent=4)

    def update(self, other_farms, reservoir):
        """
        Update the status of the farm given the growth of fish and change in population of
        parasites.
        :return: none
        """
        # TODO: add new offspring to cages

        # update cages
        for cage in self.cages:
            cage.update(other_farms, reservoir)

class Cage:# pylint: disable=R0902
    """
    Fish cages contain the fish.
    """

    def __init__(self, farm, label, nplankt):
        """
        Create a cage on a farm
        :param farm: the farm this cage is attached to
        :param label: the label (id) of the cage within the farm
        :param nplankt: TODO ???
        """
        self.label = label
        # TODO: Note - having the farm here causes a circular refence when saving/printing the farms.
        self.farm = farm
        self.date = cfg.start_date
        self.nFish = 4000


        lice_mortality_rates = {'L1': 0.17, 'L2': 0.22, 'L3': 0.008, 'L4': 0.05, 'L5f': 0.02, 'L5m': 0.06}
        lice_population = {'L1': 25, 'L2': 25, 'L3': 25, 'L4': 25, 'L5f': 25, 'L5m': 25}
        lice_gender = {} #np.random.choice(['F','M'],nplankt)
        lice_age = {} # np.random.choice(range(15),nplankt,p=p)
        

        #self.nplankt = nplankt
        #self.m_f = np.random.choice(['F', 'M'], nplankt)
        #self.stage = 1
        #prob = stats.poisson.pmf(range(15), 3)
        #prob = prob/np.sum(prob) #probs need to add up to one
        #self.stage_age = np.random.choice(range(15), nplankt, p=prob)
        #self.avail = 0
        #self.resistance_t1 = np.random.normal(cfg.f_muEMB, cfg.f_sigEMB, nplankt)
        #self.avail = 0
        #self.arrival = 0
        #self.nmates = 0

    def __str__(self):
        """
        Get a human readable string representation of the cage.
        :return: a description of the cage
        """
        return json.dumps(self, cls=CustomEncoder, indent=4)

    def update(self, other_farms, reservoir):
        """
        Update the cage at the current time step.
        :return:
        """
        cfg.logger.debug("Updating cage {} on farm {}".format(self.label, self.farm.name))


        # TODO: Background lice mortality events

        # TODO: Treatment mortality events

        # TODO: Development events

        # TODO: Fish growth and death

        # TODO: Infection events

        # TODO: Mating events

        # TODO: create offspring

        # TODO: remove dead individuals

        return
