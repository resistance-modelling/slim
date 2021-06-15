"""
Defines a Farm class that encapsulates a salmon farm containing several cages.
"""
import json
import datetime as dt
import math
import numpy as np
from scipy import stats
from scipy.spatial import distance
import config as cfg

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

def update_background_lice_mortality(lice_population, days):
    """
    Background death in a stage (remove entry) -> rate = number of individuals in
    stage*stage rate (nauplii 0.17/d, copepods 0.22/d, pre-adult female 0.05,
    pre-adult male ... Stien et al 2005)
    """
    lice_mortality_rates = {'L1': 0.17, 'L2': 0.22, 'L3': 0.008, 'L4': 0.05, 'L5f': 0.02,
                            'L5m': 0.06}

    dead_lice_dist = {}
    for stage in lice_population:
        mortality_rate = lice_population[stage] * lice_mortality_rates[stage] * days
        mortality = min(np.random.poisson(mortality_rate), lice_population[stage])
        dead_lice_dist[stage] = mortality

    cfg.logger.debug('    background mortality distribn of dead lice = {}'.format(dead_lice_dist))
    return dead_lice_dist

def fish_growth_rate(days):
    return 10000/(1 + math.exp(-0.01*(days-475)))

class CustomFarmEncoder(json.JSONEncoder):
    """
    Bespoke encoder to encode Farm objects to json. Specifically, numpy arrays and datetime objects
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
        elif isinstance(o, np.int64):
            return_str = str(o)
        elif isinstance(o, dt.datetime):
            return_str = o.strftime("%Y-%m-%d %H:%M:%S")
        else:
            return_str = {'__{}__'.format(o.__class__.__name__): o.__dict__}

        return return_str

class CustomCageEncoder(json.JSONEncoder):
    """
    Bespoke encoder to encode Cage objects to json. Specifically, numpy arrays and datetime objects
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
        elif isinstance(o, Farm):
            return_str = ""
        elif isinstance(o, np.int64):
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

        # this is by no means optimal: I want to distribute nplankt lice across
        # each stage at random. (the method I use here is to create a random list
        # of stages and add up the number for each.
        lice_stage = np.random.choice(range(2, 7), nplankt)
        num_lice_in_stage = np.array([sum(lice_stage == i) for i in range(1, 7)])
        self.lice_population = {'L1': num_lice_in_stage[0], 'L2': num_lice_in_stage[1],
                                'L3': num_lice_in_stage[2], 'L4': num_lice_in_stage[3],
                                'L5f': num_lice_in_stage[4],
                                'L5m': num_lice_in_stage[5]}

    def to_csv(self):
        """
        Save the contents of this cage as a CSV string for writing to a file later.
        """
        return f"reservoir, {self.num_fish}, {self.lice_population['L1']}, \
                {self.lice_population['L2']}, {self.lice_population['L3']}, \
                {self.lice_population['L4']}, {self.lice_population['L5f']}, \
                {self.lice_population['L5m']}, {sum(self.lice_population.values())}"


    def update(self, step_size, farms):
        """
        Update the reservoir at the current time step.
        :return:
        """
        cfg.logger.debug("Updating reservoir")
        cfg.logger.debug("  initial lice population = {}".format(self.lice_population))

        self.lice_population = update_background_lice_mortality(self.lice_population, step_size)

        # TODO - infection events in the reservoir
        #eta_aldrin = -2.576 + log(inpt.Enfish_res[cur_date.month-1]) + 0.082*(log(inpt.Ewt_res)-0.55)
        #Einf = (exp(eta_aldrin)/(1+exp(eta_aldrin)))*tau*cop_cage
        #inf_ents = np.random.poisson(Einf)
        #inf_ents = min(inf_ents,cop_cage)
        #inf_inds = np.random.choice(df_list[fc].loc[(df_list[fc].stage==2) & (df_list[fc].arrival<=df_list[fc].stage_age)].index, inf_ents, replace=False)

        cfg.logger.debug("  final lice population = {}".format(self.lice_population))


class Farm:
    """
    Define a salmon farm containing salmon cages. Over time the salmon in the cages grow and are
    subjected to external infestation pressure from sea lice.
    """

    def __init__(self, name, loc, start_date, treatment_dates, n_cages, cages_start_date, nplankt):
        """
        Create a farm.
        :param name: the id of the farm.
        :param loc: the [x,y] location of the farm.
        :param start_date: the date the farm commences.
        :param treatment_dates: the dates the farm is allowed to treat their cages.
        :param n_cages: the number of cages on the farm.
        :param cages_start_date: a list of the start dates for each cage.
        :param nplankt: initial number of planktonic lice.
        """
        self.name = name
        self.loc_x = loc[0]
        self.loc_y = loc[1]
        self.start_date = start_date
        self.treatment_dates = treatment_dates
        self.cages = [Cage(self, i, cages_start_date[i], nplankt) for i in range(n_cages)]

    def __str__(self):
        """
        Get a human readable string representation of the farm.
        :return: a description of the cage
        """
        cages = ', '.join(str(a) for a in self.cages)
        return f'id: {self.name}, Cages: {cages}'

    def __repr__(self):
        return json.dumps(self, cls=CustomEncoder, indent=4)

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
        cfg.logger.debug("Updating farm {}".format(self.name))

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
        # TODO: having the farm here causes a circular refence when saving/printing the farms.
        # I need to figure out a way of either ignoring this or saving it as an id.
        self.farm = farm
        self.start_date = start_date
        self.date = cfg.start_date
        self.num_fish = 4000
        self.num_infected_fish = 0

        self.lice_population = {'L1': nplankt, 'L2': 0, 'L3': 30, 'L4': 30, 'L5f': 10, 'L5m': 0}
 
        # The original code was a IBM, here we act on populations so the age in each stage must
        # be a distribution.

        #lice_gender = {} #np.random.choice(['F','M'],nplankt)
        #lice_age = {} # np.random.choice(range(15),nplankt,p=p)
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
        Get a human readable string representation of the cage in json form.
        Note: having a reference to the farm here causeѕ a circular reference, we need to
        ignore the farm attribute of this class in this method.
        :return: a description of the cage
        """
        return json.dumps(self, cls=CustomCageEncoder, indent=4)

    def to_csv(self):
        """
        Save the contents of this cage as a CSV string for writing to a file later.
        """
        return f"{self.label}, {self.num_fish}, {self.lice_population['L1']}, \
                {self.lice_population['L2']}, {self.lice_population['L3']}, \
                {self.lice_population['L4']}, {self.lice_population['L5f']}, \
                {self.lice_population['L5m']}, {sum(self.lice_population.values())}"



    def update(self, cur_date, step_size, other_farms, reservoir):
        """
        Update the cage at the current time step.
        :return:
        """
        cfg.logger.debug(f"  Updating farm {self.farm.name} / cage {self.label}")
        cfg.logger.debug(f"    initial lice population = {self.lice_population}")
        cfg.logger.debug(f"    initial fish population = {self.num_fish}")

        days_since_start = (cur_date - self.date).days

        # Background lice mortality events
        dead_lice_dist = update_background_lice_mortality(self.lice_population, step_size)

        # Treatment mortality events
        treatment_mortality = self.update_lice_treatment_mortality(cur_date)

        # Development events
        new_L2, new_L4, new_females, new_males = self.update_lice_lifestage(cur_date.month)

        # Fish growth and death
        fish_deaths_natural, fish_deaths_from_lice = self.update_fish_growth(days_since_start, step_size)

        # Infection events
        num_infected_fish = self.do_infection_events(days_since_start)

        # TODO: Mating events
        self.do_mating_events()

        # TODO: create offspring
        self.create_offspring()

        # TODO
        self.update_deltas(dead_lice_dist, treatment_mortality, fish_deaths_natural,
                fish_deaths_from_lice, new_L2, new_L4, new_females, new_males, num_infected_fish)

        cfg.logger.debug("    final lice population = {}".format(self.lice_population))
        cfg.logger.debug("    final fish population = {}".format(self.num_fish))

    def update_lice_treatment_mortality(self, cur_date):
        """
        Calculate the number of lice in each stage killed by treatment.
        """
        if cur_date - dt.timedelta(days=14) in self.farm.treatment_dates:
            cfg.logger.debug('    treating farm {}/cage {} on date {}'.format(self.farm.name,
                                                                              self.label, cur_date))

            # number of lice in those stages that are susceptible to Emamectin Benzoate (i.e.
            # those L3 or above)
            susceptible_stages = list(self.lice_population)[2:]
            num_susc = sum(self.lice_population[x] for x in susceptible_stages)

            # model the resistence of each lice in the susceptible stages (phenoEMB) and estimate
            # the mortality rate due to treatment (ETmort). We will pick the number of lice that
            # die due to treatment from a Poisson distribution
            pheno_emb = np.random.normal(cfg.f_meanEMB, cfg.f_sigEMB, num_susc) \
                    + np.random.normal(cfg.env_meanEMB, cfg.env_sigEMB, num_susc)
            pheno_emb = 1/(1 + np.exp(pheno_emb))
            mortality_rate = sum(pheno_emb)*cfg.EMBmort

            if mortality_rate > 0:
                num_dead_lice = np.random.poisson(mortality_rate)
                num_dead_lice = min(num_dead_lice, num_susc)

                # Now we need to decide how many lice from each stage die,
                #   the algorithm is to label each louse  1...num_susc
                #   assign each of these a probability of dying as (phenoEMB)/np.sum(phenoEMB)
                #   randomly pick lice according to this probability distribution
                #        now we need to find out which stages these lice are in by calculating the
                #        cumulative sum of the lice in each stage and finding out how many of our
                #        dead lice falls into this bin.
                p = (pheno_emb)/np.sum(pheno_emb)
                dead_lice = np.random.choice(range(num_susc), num_dead_lice, p=p,
                                             replace=False).tolist()
                total_so_far = 0
                dead_lice_dist = {}
                for stage in susceptible_stages:
                    num_dead = len([x for x in dead_lice if total_so_far <= x <
                                    (total_so_far + self.lice_population[stage])])
                    if num_dead > 0:
                        self.lice_population[stage] -= num_dead
                        total_so_far += self.lice_population[stage]
                        dead_lice_dist[stage] = num_dead

                cfg.logger.debug('      distribution of dead lice on farm {}/cage {} = {}'
                                 .format(self.farm.name, self.label, dead_lice_dist))
        return dead_lice_dist

    def update_lice_lifestage(self, cur_month):
        """
        Move lice between lifecycle stages.
        """
        cfg.logger.debug('    updating lice lifecycle stages')

        def get_stage_ages(size):
            """
            Create an age distribution (in days) for the sea lice within a lifecycle stage.
            """
            p = stats.poisson.pmf(range(15), 3)
            p = p / np.sum(p) # probs need to add up to one
            return np.random.choice(range(15), size, p=p)

        def dev_time(del_p, del_m10, del_s, temp_c, n_days):
            """
            Probability of developing after n_days days in a stage as given by Aldrin et al 2017
            :param n_days:
            :param del_p:
            :param del_m10:
            :param del_s:
            :param temp_c:
            :return:
            """
            unbounded = math.log(2)*del_s*n_days**(del_s-1)* \
                    (del_m10*10**del_p/temp_c**del_p)**(-del_s)
            unbounded[unbounded == 0] = 10**(-30)
            unbounded[unbounded > 1] = 1
            return unbounded.astype('float64')

        def ave_temperature_at(c_month, farm_northing):
            """
            TODO: Calculate the mean sea temperature at the northing coordinate of the farm at
            month c_month interpolating data taken from
            www.seatemperature.org
            """
            ardrishaig_avetemp = np.array([8.2, 7.55, 7.45, 8.25, 9.65, 11.35, 13.15, 13.75, 13.65,
                                           12.85, 11.75, 9.85])
            tarbert_avetemp = np.array([8.4, 7.8, 7.7, 8.6, 9.8, 11.65, 13.4, 13.9, 13.9, 13.2,
                                        12.15, 10.2])
            degs = (tarbert_avetemp[c_month-1]-ardrishaig_avetemp[c_month-1])/(685715-665300)
            Ndiff = farm_northing - 665300              #farm Northing - tarbert Northing
            return round(tarbert_avetemp[c_month-1] - Ndiff*degs, 1)

        lice_dist = {}

        # L4 -> L5
        num_lice = self.lice_population['L4']
        l4_to_l5 = dev_time(0.866, 10.742, 1.643, ave_temperature_at(cur_month, self.farm.loc_x),
                            get_stage_ages(num_lice))
        num_to_move = min(np.random.poisson(sum(l4_to_l5)), num_lice)
        new_females = np.random.choice([math.floor(num_to_move/2.0), math.ceil(num_to_move/2.0)])
        new_males = (num_to_move - new_females)

        lice_dist['L5f'] = new_females
        lice_dist['L5m'] = (num_to_move - new_females)

        # L3 -> L4
        num_lice = self.lice_population['L3']
        l3_to_l4 = dev_time(1.305, 18.934, 7.945, ave_temperature_at(cur_month, self.farm.loc_x),
                            get_stage_ages(num_lice))
        num_to_move = min(np.random.poisson(sum(l3_to_l4)), num_lice)
        new_L4 = num_to_move

        lice_dist['L4'] = num_to_move

        # L2 -> L3 
        # This is done in do_infection_events()

        # L1 -> L2
        num_lice = self.lice_population['L2']
        l1_to_l2 = dev_time(0.401, 8.814, 18.869, ave_temperature_at(cur_month, self.farm.loc_x),
                            get_stage_ages(num_lice))
        num_to_move = min(np.random.poisson(sum(l1_to_l2)), num_lice)
        new_L2 = num_to_move

        lice_dist['L2'] = num_to_move

        cfg.logger.debug('      distribution of new lice lifecycle stages on farm {}/cage {} = {}'
                         .format(self.farm.name, self.label, lice_dist))

        return new_L2, new_L4, new_females, new_males



    def update_fish_growth(self, days, step_size):
        """
        Fish growth rate -> 10000/(1+exp(-0.01*(t-475))) fitted logistic curve to data from
        http://www.fao.org/fishery/affris/species-profiles/atlantic-salmon/growth/en/

        Fish death rate: constant background daily rate 0.00057 based on
        www.gov.scot/Resource/0052/00524803.pdf
        multiplied by lice coefficient see surv.py (compare to threshold of 0.75 lice/g fish)
        """

        cfg.logger.debug('    updating fish population')

        def fb_mort(days):
            """
            Fish background mortality rate, decreasing as in Soares et al 2011
            :param days: TODO ???
            :return: TODO ???
            """
            return 0.00057 #(1000 + (days - 700)**2)/490000000

        # detemine the number of fish with lice and the number of attached lice on each.
        # for now, assume there is only one TODO fix this when I understand the infestation
        # (next stage)
        num_fish_with_lice = 3
        adlicepg = np.array([1] * num_fish_with_lice)/fish_growth_rate(days)
        prob_lice_death = 1/(1+np.exp(-19*(adlicepg-0.63)))

        ebf_death = fb_mort(days)*step_size*(self.num_fish)
        elf_death = np.sum(prob_lice_death)*step_size
        fish_deaths_natural = np.random.poisson(ebf_death)
        fish_deaths_from_lice = np.random.poisson(elf_death)

        cfg.logger.debug('      number of background fish death {}, from lice {}'
                         .format(fish_deaths_natural, fish_deaths_from_lice))

        #self.num_fish -= fish_deaths_natural
        #self.num_fish -= fish_deaths_from_lice
        return fish_deaths_natural, fish_deaths_from_lice

    def do_infection_events(self, days):
        """
        Infect fish in this cage if the sea lice are in stage L2 and TODO - the 'arrival time' <= 'stage_age' ?????
        """
        # TODO - need to fix this - we have a dictionary of lice lifestage but no age within this stage.
        # Perhaps we can have a distribution which can change per day (the mean/median increaseѕ?
        # but at what point does the distribution mean decrease).
        num_avail_lice = self.lice_population['L2']
        if num_avail_lice > 0:
            num_fish_in_farm = sum([c.num_fish for c in self.farm.cages]) 

            eta_aldrin = -2.576 + math.log(num_fish_in_farm) + 0.082*(math.log(fish_growth_rate(days))-0.55)
            Einf = (math.exp(eta_aldrin)/(1 + math.exp(eta_aldrin)))*tau*num_avail_lice
            num_infected_fish = np.random.poisson(Einf)
            inf_ents = np.random.poisson(Einf)

            # inf_ents now go on to infect fish so get removed from the general lice population.
            return  min(inf_ents, num_avail_lice)
        else
            return 0

    def do_mating_events(self):
        """
        TODO
        """

        """
        TODO - convert this (it is an IBM so we will also need to think about the numbers of events rather than who is mating).
                        #Mating events---------------------------------------------------------------------
                #who is mating               
                females = df_list[fc].loc[(df_list[fc].stage==5) & (df_list[fc].avail==0)].index
                males = df_list[fc].loc[(df_list[fc].stage==6) & (df_list[fc].avail==0)].index
                nmating = min(sum(df_list[fc].index.isin(females)),\
                          sum(df_list[fc].index.isin(males)))
                if nmating>0:
                    sires = np.random.choice(males, nmating, replace=False)
                    p_dams = 1 - (df_list[fc].loc[df_list[fc].index.isin(females),'stage_age']/
                                np.sum(df_list[fc].loc[df_list[fc].index.isin(females),'stage_age'])+1)
                    dams = np.random.choice(females, nmating, p=np.array(p_dams/np.sum(p_dams)).tolist(), replace=False)
                else:
                    sires = []
                    dams = []
                df_list[fc].loc[df_list[fc].index.isin(dams),'avail'] = 1
                df_list[fc].loc[df_list[fc].index.isin(sires),'avail'] = 1
                df_list[fc].loc[df_list[fc].index.isin(dams),'nmates'] = df_list[fc].loc[df_list[fc].index.isin(dams),'nmates'].values + 1
                df_list[fc].loc[df_list[fc].index.isin(sires),'nmates'] = df_list[fc].loc[df_list[fc].index.isin(sires),'nmates'].values + 1
                #Add genotype of sire to dam info
                df_list[fc].loc[df_list[fc].index.isin(dams),'mate_resistanceT1'] = \
                df_list[fc].loc[df_list[fc].index.isin(sires),'resistanceT1'].values
        """
        pass

    def create_offspring(self):
        """
        TODO
        """

        """
        TODO - convert this ().
                #create offspring
                bv_lst = []
                eggs_now = int(round(eggs*tau/d_hatching(temp_now)))
                for i in dams:
                    if farm==0:
                        r = np.random.uniform(0,1,1)
                        if r>inpt.prop_influx:
                            underlying = 0.5*df_list[fc].loc[df_list[fc].index==i,'resistanceT1'].values\
                               + 0.5*df_list[fc].loc[df_list[fc].index==i,'mate_resistanceT1'].values + \
                               np.random.normal(0, farms_sigEMB[farm], eggs_now+250)/np.sqrt(2)
                        else:
                            underlying = np.random.normal(inpt.f_muEMB,inpt.f_sigEMB,eggs_now+250)
                    else:
                        underlying = 0.5*df_list[fc].loc[df_list[fc].index==i,'resistanceT1'].values\
                               + 0.5*df_list[fc].loc[df_list[fc].index==i,'mate_resistanceT1'].values + \
                               np.random.normal(0, farms_sigEMB[farm], eggs_now+250)/np.sqrt(2)
                    bv_lst.extend(underlying.tolist())  
                new_offs = len(dams)*eggs_now

                #print("       farms =  {}".format(inpt.nfarms))
                num = 0
                for f in range(inpt.nfarms):
                    #print("       farm {}".format(f))
                    arrivals = np.random.poisson(prop_arrive[farm][f]*new_offs)
                    if arrivals>0:
                        num = num + 1
                        offs = pd.DataFrame(columns=df_list[fc].columns)
                        offs['MF'] = np.random.choice(['F','M'], arrivals)
                        offs['Farm'] = f
                        offs['Cage'] = np.random.choice(range(1,inpt.ncages[farm]+1), arrivals)
                        offs['stage'] = np.repeat(1, arrivals)
                        offs['stage_age'] = np.repeat(0, arrivals)
                        if len(bv_lst)<arrivals:
                            randams = np.random.choice(dams,arrivals-len(bv_lst))
                            for i in randams:
                                underlying = 0.5*df_list[fc].resistanceT1[df_list[fc].index==i]\
                                   + 0.5*df_list[fc].mate_resistanceT1[df_list[fc].index==i]+ \
                                   np.random.normal(0, farms_sigEMB[farm], 1)/np.sqrt(2)
                                bv_lst.extend(underlying)  
                        ran_bvs = np.random.choice(len(bv_lst),arrivals,replace=False)
                        offs['resistanceT1'] = [bv_lst[i] for i in ran_bvs]  
                        for i in sorted(ran_bvs, reverse=True):
                            del bv_lst[i]     
                        Earrival = [hrs_travel[farm][i] for i in offs.Farm]
                        offs['arrival'] = np.random.poisson(Earrival)
                        offs['avail'] = 0
                        offs['date'] = cur_date
                        offs['nmates'] = 0
                        offspring = offspring.append(offs, ignore_index=True)
                        del offs
        """
        pass


    def update_deltas(dead_lice_dist, treatment_mortality, fish_deaths_natural, fish_deaths_from_lice, new_L2, new_L4, new_females, new_males, num_infected_fish):
        """
        Update the number of fish and the lice in each life stage given the number that move between stages in this time period.
        """

        for stage in lice_population:
            # update background mortality
            self.lice_population[stage] -= dead_lice_dist[stage]

            # update population due to treatment
            num_dead = treatment_mortality[stage]
            if num_dead > 0:
                self.lice_population[stage] -= num_dead

        self.lice_population['L5m'] += new_males
        self.lice_population['L5f'] += new_females
        self.lice_population['L4'] = self.lice_population['L4'] - (new_names + new_females) + new_L4
        self.lice_population['L3'] = self.lice_population['L3'] - new_L4 + num_infected_fish
        self.lice_population['L2'] = self.lice_population['L2'] + new_L2 - num_infected_fish
        self.lice_population['L1'] -= new_L2

        self.num_fish -= fish_deaths_natural
        self.num_fish -= fish_deaths_from_lice

        return self.lice_population
