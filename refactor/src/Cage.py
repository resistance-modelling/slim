import datetime as dt
import math
import json

import numpy as np
from scipy import stats

from src.CageTemplate import CageTemplate
from src.JSONEncoders import CustomCageEncoder


class Cage(CageTemplate):
    """
    Fish cages contain the fish.
    """

    def __init__(self, farm_id, cage_id, cfg):
        """
        Create a cage on a farm
        :param farm: the farm this cage is attached to
        :param label: the label (id) of the cage within the farm
        :param nplankt: TODO ???
        """

        # sets access to cfg and logger
        super().__init__(cfg)

        self.id = cage_id
        self.farm_id = farm_id
        self.start_date = cfg.farms[farm_id].cages_start[cage_id]
        self.date = cfg.start_date
        self.num_fish = cfg.farms[farm_id].num_fish
        self.num_infected_fish = 0

        # TODO: update with calculations
        self.lice_population = {'L1': cfg.ext_pressure, 'L2': 0, 'L3': 30, 'L4': 30, 'L5f': 10, 'L5m': 0}

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
        return f"{self.id}, {self.num_fish}, {self.lice_population['L1']}, \
                {self.lice_population['L2']}, {self.lice_population['L3']}, \
                {self.lice_population['L4']}, {self.lice_population['L5f']}, \
                {self.lice_population['L5m']}, {sum(self.lice_population.values())}"

    def update(self, cur_date, step_size, other_farms, reservoir):
        """
        Update the cage at the current time step.
        :return:
        """
        self.logger.debug(f"  Updating farm {self.farm_id} / cage {self.id}")
        self.logger.debug(f"    initial lice population = {self.lice_population}")
        self.logger.debug(f"    initial fish population = {self.num_fish}")

        days_since_start = (cur_date - self.date).days

        # Background lice mortality events
        dead_lice_dist = self.update_background_lice_mortality(self.lice_population, step_size)

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

        self.logger.debug("    final lice population = {}".format(self.lice_population))
        self.logger.debug("    final fish population = {}".format(self.num_fish))

    def update_lice_treatment_mortality(self, cur_date):
        """
        Calculate the number of lice in each stage killed by treatment.
        """

        dead_lice_dist = {"L1": 0, "L2": 0, "L3": 0, "L4": 0, "L5m": 0, "L5f": 0}

        # TODO: take temperatures into account? See #22
        if cur_date - dt.timedelta(days=self.cfg.delay_EMB) in self.cfg.farms[self.farm_id].treatment_dates:
            self.logger.debug('    treating farm {}/cage {} on date {}'.format(self.farm_id,
                                                                              self.id, cur_date))

            # number of lice in those stages that are susceptible to Emamectin Benzoate (i.e.
            # those L3 or above)
            susceptible_stages = list(self.lice_population)[2:]
            num_susc = sum(self.lice_population[x] for x in susceptible_stages)

            # model the resistence of each lice in the susceptible stages (phenoEMB) and estimate
            # the mortality rate due to treatment (ETmort). We will pick the number of lice that
            # die due to treatment from a Poisson distribution
            pheno_emb = np.random.normal(self.cfg.f_meanEMB, self.cfg.f_sigEMB, num_susc) \
                    + np.random.normal(self.cfg.env_meanEMB, self.cfg.env_sigEMB, num_susc)
            pheno_emb = 1/(1 + np.exp(pheno_emb))
            mortality_rate = sum(pheno_emb)*self.cfg.EMBmort

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
                for stage in susceptible_stages:
                    num_dead = len([x for x in dead_lice if total_so_far <= x <
                                    (total_so_far + self.lice_population[stage])])
                    total_so_far += self.lice_population[stage]
                    if num_dead > 0:
                        self.lice_population[stage] -= num_dead
                        dead_lice_dist[stage] = num_dead

                self.logger.debug('      distribution of dead lice on farm {}/cage {} = {}'
                                 .format(self.farm_id, self.id, dead_lice_dist))

                assert num_dead_lice == sum(list(dead_lice_dist.values()))
        return dead_lice_dist

    @staticmethod
    def get_stage_ages(size: int, min: int, mean: int, development_days=25):
        """
        Create an age distribution (in days) for the sea lice within a lifecycle stage.
        :param size the number of lice to consider
        :param mean the mean of the distribution. mean must be bigger than min.
        :param development_days the maximum age to consider
        :return a size-long array of ages (in days)
        """

        # Create a shifted poisson distribution,
        k = mean - min
        max_quantile = development_days - min

        assert min > 0, "min must be positive"
        assert k > 0, "mean must be greater than min."

        p = stats.poisson.pmf(range(max_quantile), k)
        p = p / np.sum(p)  # probs need to add up to one
        return np.random.choice(range(max_quantile), size, p=p) + min


    @staticmethod
    def ave_temperature_at(c_month, farm_northing):
        """
        Calculate the mean sea temperature at the northing coordinate of the farm at
        month c_month interpolating data taken from
        TODO if the farm never moves there is no point in interpolating every time but only during initialization
        www.seatemperature.org
        """
        ardrishaig_avetemp = np.array([8.2, 7.55, 7.45, 8.25, 9.65, 11.35, 13.15, 13.75, 13.65,
                                       12.85, 11.75, 9.85])
        tarbert_avetemp = np.array([8.4, 7.8, 7.7, 8.6, 9.8, 11.65, 13.4, 13.9, 13.9, 13.2,
                                    12.15, 10.2])
        degs = (tarbert_avetemp[c_month - 1] - ardrishaig_avetemp[c_month - 1]) / (685715 - 665300)
        Ndiff = farm_northing - 665300  # farm Northing - tarbert Northing
        return np.round(tarbert_avetemp[c_month - 1] - Ndiff * degs, 1)

    def update_lice_lifestage(self, cur_month):
        """
        Move lice between lifecycle stages.
        """
        self.logger.debug('    updating lice lifecycle stages')

        def dev_time(del_p, del_m10, del_s, temp_c, n_days):
            """
            Probability of developing after n_days days in a stage as given by Aldrin et al 2017
            See section 2.2.4 of Aldrin et al. (2017)
            :param n_days: stage-age
            :param del_p: power transformation constant on temp_c
            :param del_m10: 10 °C median reference value
            :param del_s: fitted Weibull shape parameter
            :param temp_c: average temperature in °C
            :return: expected development rate
            """
            epsilon = 1e-30
            del_m = del_m10*(10/temp_c)**del_p

            unbounded = math.log(2)*del_s*n_days**(del_s-1)*del_m**(-del_s)
            unbounded = np.clip(unbounded, epsilon, 1)
            return unbounded.astype('float64')

        lice_dist = {}
        farm_loc_x = self.cfg.farms[self.farm_id].farm_location[0]

        ave_temp = self.ave_temperature_at(cur_month, farm_loc_x)

        # L4 -> L5
        # TODO move these magic numbers somewhere else...
        # TODO these blocks look like the same?
        num_lice = self.lice_population['L4']
        stage_ages = self.get_stage_ages(num_lice, min=10, mean=15)

        l4_to_l5 = dev_time(0.866, 10.742, 1.643, ave_temp, stage_ages)
        num_to_move = min(np.random.poisson(np.sum(l4_to_l5)), num_lice)
        new_females = np.random.choice([math.floor(num_to_move/2.0), math.ceil(num_to_move/2.0)])
        new_males = (num_to_move - new_females)

        lice_dist['L5f'] = new_females
        lice_dist['L5m'] = (num_to_move - new_females)

        # L3 -> L4
        num_lice = self.lice_population['L3']
        stage_ages = self.get_stage_ages(num_lice, min=15, mean=18)
        l3_to_l4 = dev_time(1.305, 18.934, 7.945, ave_temp, stage_ages)
        num_to_move = min(np.random.poisson(np.sum(l3_to_l4)), num_lice)
        new_L4 = num_to_move

        lice_dist['L4'] = num_to_move

        # L2 -> L3
        # This is done in do_infection_events()

        # L1 -> L2
        num_lice = self.lice_population['L2']
        stage_ages = self.get_stage_ages(num_lice, min=3, mean=4)
        l1_to_l2 = dev_time(0.401, 8.814, 18.869, ave_temp, stage_ages)
        num_to_move = min(np.random.poisson(np.sum(l1_to_l2)), num_lice)
        new_L2 = num_to_move

        lice_dist['L2'] = num_to_move

        self.logger.debug('      distribution of new lice lifecycle stages on farm {}/cage {} = {}'
                         .format(self.farm_id, self.id, lice_dist))

        return new_L2, new_L4, new_females, new_males

    def update_fish_growth(self, days, step_size):
        """
        Fish growth rate -> 10000/(1+exp(-0.01*(t-475))) fitted logistic curve to data from
        http://www.fao.org/fishery/affris/species-profiles/atlantic-salmon/growth/en/

        Fish death rate: constant background daily rate 0.00057 based on
        www.gov.scot/Resource/0052/00524803.pdf
        multiplied by lice coefficient see surv.py (compare to threshold of 0.75 lice/g fish)
        """

        self.logger.debug('    updating fish population')

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
        adlicepg = np.array([1] * num_fish_with_lice)/self.fish_growth_rate(days)
        prob_lice_death = 1/(1+np.exp(-19*(adlicepg-0.63)))

        ebf_death = fb_mort(days)*step_size*(self.num_fish)
        elf_death = np.sum(prob_lice_death)*step_size
        fish_deaths_natural = np.random.poisson(ebf_death)
        fish_deaths_from_lice = np.random.poisson(elf_death)

        self.logger.debug('      number of background fish death {}, from lice {}'
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

            eta_aldrin = -2.576 + math.log(num_fish_in_farm) + 0.082*(math.log(self.fish_growth_rate(days))-0.55)
            Einf = (math.exp(eta_aldrin)/(1 + math.exp(eta_aldrin)))*tau*num_avail_lice
            num_infected_fish = np.random.poisson(Einf)
            inf_ents = np.random.poisson(Einf)

            # inf_ents now go on to infect fish so get removed from the general lice population.
            return min(inf_ents, num_avail_lice)
        else:
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


    def update_deltas(self, dead_lice_dist, treatment_mortality, fish_deaths_natural, fish_deaths_from_lice, new_L2, new_L4, new_females, new_males, num_infected_fish):
        """
        Update the number of fish and the lice in each life stage given the number that move between stages in this time period.
        """

        for stage in self.lice_population:
            # update background mortality
            self.lice_population[stage] -= dead_lice_dist[stage]

            # update population due to treatment
            num_dead = treatment_mortality.get(stage, 0)
            if num_dead > 0:
                self.lice_population[stage] -= num_dead

        self.lice_population['L5m'] += new_males
        self.lice_population['L5f'] += new_females
        self.lice_population['L4'] = self.lice_population['L4'] - (new_males + new_females) + new_L4
        self.lice_population['L3'] = self.lice_population['L3'] - new_L4 + num_infected_fish
        self.lice_population['L2'] = self.lice_population['L2'] + new_L2 - num_infected_fish
        self.lice_population['L1'] -= new_L2

        self.num_fish -= fish_deaths_natural
        self.num_fish -= fish_deaths_from_lice

        return self.lice_population
