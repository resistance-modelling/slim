import datetime as dt
import math
import json
import random
import copy

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
 
        self.eggs = {}
        # TODO/Question: what's the best way to deal with having multiple possible genetic schemes?
        # TODO/Question: I suppose some of this initial genotype information ought to come from the config file
        # TODO/Question: the genetic mechanism will be the same for all lice in a simulation, so should it live in the driver?
        # for now I've hard-coded in one mechanism in this setup, and a particular genotype starting point. Should probably be from a config file?
        self.genetic_mechanism = 'discrete'
        
        generic_discrete_props = {('A'):0.25, ('a'): 0.25, ('A', 'a'): 0.5}
        
        self.geno_by_lifestage = {}
        for stage in self.lice_population:
            self.geno_by_lifestage[stage] = {}
            for geno in generic_discrete_props:
                self.geno_by_lifestage[stage][geno] = np.round(generic_discrete_props[geno]*self.lice_population[stage], 0)


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
        delta_avail_dams, delta_eggs = self.do_mating_events()

        # TODO: create offspring
        self.create_offspring()

        # TODO
        self.update_deltas(dead_lice_dist, treatment_mortality, fish_deaths_natural,
                fish_deaths_from_lice, new_L2, new_L4, new_females, new_males, num_infected_fish, delta_avail_dams, delta_eggs)

        self.logger.debug("    final lice population = {}".format(self.lice_population))
        self.logger.debug("    final fish population = {}".format(self.num_fish))

    def update_lice_treatment_mortality(self, cur_date):
        """
        Calculate the number of lice in each stage killed by treatment.
        """
        if cur_date - dt.timedelta(days=14) in self.cfg.farms[self.farm_id].treatment_dates:
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
                dead_lice_dist = {}
                for stage in susceptible_stages:
                    num_dead = len([x for x in dead_lice if total_so_far <= x <
                                    (total_so_far + self.lice_population[stage])])
                    if num_dead > 0:
                        self.lice_population[stage] -= num_dead
                        total_so_far += self.lice_population[stage]
                        dead_lice_dist[stage] = num_dead

                self.logger.debug('      distribution of dead lice on farm {}/cage {} = {}'
                                 .format(self.farm_id, self.id, dead_lice_dist))
        return dead_lice_dist

    def update_lice_lifestage(self, cur_month):
        """
        Move lice between lifecycle stages.
        """
        self.logger.debug('    updating lice lifecycle stages')

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
        farm_loc_x = self.cfg.farms[self.farm_id].farm_location[0]

        # L4 -> L5
        num_lice = self.lice_population['L4']
        l4_to_l5 = dev_time(0.866, 10.742, 1.643, ave_temperature_at(cur_month, farm_loc_x),
                            get_stage_ages(num_lice))
        num_to_move = min(np.random.poisson(sum(l4_to_l5)), num_lice)
        new_females = np.random.choice([math.floor(num_to_move/2.0), math.ceil(num_to_move/2.0)])
        new_males = (num_to_move - new_females)

        lice_dist['L5f'] = new_females
        lice_dist['L5m'] = (num_to_move - new_females)

        # L3 -> L4
        num_lice = self.lice_population['L3']
        l3_to_l4 = dev_time(1.305, 18.934, 7.945, ave_temperature_at(cur_month, farm_loc_x),
                            get_stage_ages(num_lice))
        num_to_move = min(np.random.poisson(sum(l3_to_l4)), num_lice)
        new_L4 = num_to_move

        lice_dist['L4'] = num_to_move

        # L2 -> L3
        # This is done in do_infection_events()

        # L1 -> L2
        num_lice = self.lice_population['L2']
        l1_to_l2 = dev_time(0.401, 8.814, 18.869, ave_temperature_at(cur_month, farm_loc_x),
                            get_stage_ages(num_lice))
        num_to_move = min(np.random.poisson(sum(l1_to_l2)), num_lice)
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
    
    def get_num_mating_events(self):
        """
        TODO
        """
        return 10
    
    def get_num_eggs_distrib(self):
        """
        TODO
        
        probably depends on water temperature, maybe other things?
        """
        return {400:5, 500 : 10, 600:5}
    
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
        num_mating_events = self.get_num_mating_events()
        
        distrib_sire_available = self.geno_by_lifestage['L5m']
        distrib_dam_available = self.geno_by_lifestage['L5f']
        distrib_eggs_per_mating = self.get_num_eggs_distrib()
        
        delta_avail_dams, delta_eggs = self.generate_matings_discrete(distrib_sire_available, distrib_dam_available, distrib_eggs_per_mating, num_mating_events)
        return delta_avail_dams, delta_eggs

        
    # if we're in the discrete 2-gene setting, assume for now that genotypes are tuples - so in a A/a genetic system, genotypes
    # could be ('A'), ('a'), or ('A', 'a')     
    #  right now number of offspring with each genotype are deterministic, and might be missing one (we should update to add jitter in future,
    # but this is a reasonable approx)
    # note another todo: doesn't do anything sensible re: integer/real numbers of offspring     
    def generate_eggs(self, sire, dam, breeding_method, number_eggs):
      if breeding_method == 'discrete':
        eggs_generated = {}
        if len(sire) == 1 and len(dam) == 1:
          eggs_generated[tuple(set((sire[0], dam[0])))] = number_eggs
        elif len(sire) == 2 and len(dam) == 1:
          eggs_generated[tuple(set((sire[0], dam[0])))] = number_eggs/2
          eggs_generated[tuple(set((sire[1], dam[0])))] = number_eggs/2
        elif len(sire) == 1 and len(dam) == 2:
          eggs_generated[tuple(set((sire[0], dam[0])))] = number_eggs/2
          eggs_generated[tuple(set((sire[0], dam[1])))] = number_eggs/2
        else: #
          # drawing both from the sire in the first case ensures heterozygotes
          # but is a bit hacky.
          eggs_generated[tuple(set((sire[0], sire[1])))] = number_eggs/2
          # and the below gets us our two types of homozygotes
          eggs_generated[tuple(set((sire[0], dam[0])))] = number_eggs/4
          eggs_generated[tuple(set((sire[1], dam[1])))] = number_eggs/4
        return eggs_generated
      else:
        # additive genes, assume genetic state for an individual looks like a number between 0 and 1.
        # because we're only dealing with the heritable part here don't need to do any of the comparison
        # to population mean or impact of heritability, etc - that would appear in the code dealing with treatment
        # so we could use just the mid-parent value for this genetic recording for children
        # as with the discrete genetic model, this will be deterministic for now
        eggs_generated = {}
        mid_parent = np.round((sire + dam)/2, 1)
        eggs_generated[mid_parent] = number_eggs
        
        return eggs_generated
    
    
    # I've assumed that both distrib and delta are dictionaries
    # and are *not* normalised (that is, they are effectively counts)
    # I've also assumed that we never want a count below zero
    # Code is naive - interpret as algorithm spec
    # combine these two functions with a negation of a dict?
    def update_distrib_discrete_add(self, distrib_delta, distrib):
      # print('adding distribs')
      for geno in distrib_delta:
        if geno not in distrib:
          distrib[geno] = 0
        distrib[geno] += distrib_delta[geno]
        if distrib[geno] < 0:
          distrib[geno] = 0
    
    def update_distrib_discrete_subtract(self, distrib_delta, distrib):
      for geno in distrib:
        distrib[geno] -= distrib_delta[geno]
        if distrib[geno] < 0:
          distrib[geno] = 0
    
    
    # Assumes the usual dictionary representation of number
    # of dams - genotype:number_available
    # function separated from other breeding processses to make it easier to come back and optimise
    # related TODO: there must be a faster way to do this. 
    # returns a dictionary in the same format giving genotpye:numer_selected
    # if the num_dams exceeds number available, gives back all the dams
    def select_dams(self, distrib_dams_available, num_dams):
        delta_dams_selected = {}
        copy_dams_avail = copy.deepcopy(distrib_dams_available)
        if sum(distrib_dams_available.values()) <= num_dams:
            return copy_dams_avail
        
        for _ in range(num_dams):
            this_dam = choose_from_distrib(copy_dams_avail)
            copy_dams_avail[this_dam] -= 1
            if this_dam not in delta_dams_selected:
                delta_dams_selected[this_dam] = 0
            delta_dams_selected[this_dam] += 1
            
        return delta_dams_selected
            
        
        
    
    # Assuming that these distribs are counts, and that they 
    # contain only individuals available for mating
    # will generate two deltas:  one to add to unavailable dams and subtract from available dams, one to add to eggs 
    # assume males don't become unavailable? in this case we don't need a delta for sires
    # right now only does one mating - need to deal with dams becoming unavailable if mating events is similar to number of dams
    def generate_matings_discrete(self, distrib_sire_available, distrib_dam_available, distrib_eggs_per_mating, num_matings):
      delta_eggs = {}
      delta_dams = self.select_dams(distrib_dam_available, num_matings)
      if sum(distrib_sire_available.values()) == 0 or sum(distrib_dam_available.values()) == 0:
        return {}, {}
    
#       TODO - need to add dealing with fewer males/females than the number of matings
      
      for dam_geno in delta_dams:
        for _ in range(int(delta_dams[dam_geno])):
          sire_geno = self.choose_from_distrib(distrib_sire_available)
          num_eggs = self.choose_from_distrib(distrib_eggs_per_mating)
      
          new_eggs = self.generate_eggs(sire_geno, dam_geno, 'discrete', num_eggs)
          
          self.update_distrib_discrete_add(new_eggs, delta_eggs)
    
      return delta_dams, delta_eggs
    
    
    
    def choose_from_distrib(self, distrib):
      max_val = sum(distrib.values())
      this_draw = random.randrange(max_val)
      sum_so_far = 0
      for val in distrib:
        sum_so_far += distrib[val]
        if this_draw <= sum_so_far:
          return val
        

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


    def update_deltas(self, dead_lice_dist, treatment_mortality, fish_deaths_natural, fish_deaths_from_lice, new_L2, new_L4, new_females, new_males, num_infected_fish, delta_avail_dams, delta_eggs):
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
        
        #  TODO: update genotypes as well
        
        
        #  TODO: update available dams and eggs
        self.geno_by_lifestage

        self.num_fish -= fish_deaths_natural
        self.num_fish -= fish_deaths_from_lice

        return self.lice_population
