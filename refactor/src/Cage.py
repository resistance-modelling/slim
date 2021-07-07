import datetime as dt
import math
import json

import numpy as np
from scipy import stats

from src.JSONEncoders import CustomCageEncoder

class Cage:
    """
    Fish cages contain the fish.
    """

    lice_stages = ['L1', 'L2', 'L3', 'L4', 'L5f', 'L5m']
    susceptible_stages = list(lice_stages)[2:]

    def __init__(self, cage_id, cfg, farm):
        """
        Create a cage on a farm
        :param farm_id: the farm this cage is attached to
        :param cage_id: the label (id) of the cage within the farm
        :param cfg the farm configuration
        :param farm a Farm object
        """

        self.cfg = cfg
        self.logger = cfg.logger
        self.id = cage_id

        self.farm_id = farm.name
        self.start_date = cfg.farms[self.farm_id].cages_start[cage_id]
        self.date = cfg.start_date
        self.num_fish = cfg.farms[self.farm_id].num_fish
        self.num_infected_fish = 0
        self.farm = farm

        # TODO: update with calculations
        self.lice_population = {'L1': cfg.ext_pressure, 'L2': 0, 'L3': 30, 'L4': 30, 'L5f': 10, 'L5m': 0}

        # The original code was a IBM, here we act on populations so the age in each stage must
        # be a distribution.

        # lice_gender = {} #np.random.choice(['F','M'],nplankt)
        # lice_age = {} # np.random.choice(range(15),nplankt,p=p)
        # self.nplankt = nplankt
        # self.m_f = np.random.choice(['F', 'M'], nplankt)
        # self.stage = 1
        # prob = stats.poisson.pmf(range(15), 3)
        # prob = prob/np.sum(prob) #probs need to add up to one
        # self.stage_age = np.random.choice(range(15), nplankt, p=prob)
        # self.avail = 0
        # self.resistance_t1 = np.random.normal(cfg.f_muEMB, cfg.f_sigEMB, nplankt)
        # self.avail = 0
        # self.arrival = 0
        # self.nmates = 0

    def __str__(self):
        """
        Get a human readable string representation of the cage in json form.
        :return: a description of the cage
        """

        filtered_vars = vars(self).copy()
        del filtered_vars["farm"]
        del filtered_vars["logger"]
        del filtered_vars["cfg"]
        for k in filtered_vars:
            if isinstance(filtered_vars[k], dt.datetime):
                filtered_vars[k] = filtered_vars[k].strftime("%Y-%m-%d %H:%M:%S")

        return json.dumps(filtered_vars, indent=4)
        # return json.dumps(self, cls=CustomCageEncoder, indent=4)

    def update(self, cur_date, step_size, other_farms, pressure):
        """
        Update the cage at the current time step.
        :return:
        """
        self.logger.debug(f"  Updating farm {self.farm_id} / cage {self.id}")
        self.logger.debug(f"    initial lice population = {self.lice_population}")
        self.logger.debug(f"    initial fish population = {self.num_fish}")

        days_since_start = (cur_date - self.date).days

        # Background lice mortality events
        dead_lice_dist = self.get_background_lice_mortality(self.lice_population)

        # Treatment mortality events
        treatment_mortality = self.update_lice_treatment_mortality(cur_date)

        # Development events
        new_L2, new_L4, new_females, new_males = self.get_lice_lifestage(cur_date.month)

        # Fish growth and death
        fish_deaths_natural, fish_deaths_from_lice = self.get_fish_growth(days_since_start, step_size)

        # Infection events
        num_infection_events = self.do_infection_events(days_since_start)

        # TODO: Mating events
        self.do_mating_events()

        # TODO: create offspring
        self.create_offspring()

        # lice coming from reservoir
        lice_from_reservoir = self.get_reservoir_lice(pressure)

        # TODO
        self.update_deltas(dead_lice_dist,
                           treatment_mortality,
                           fish_deaths_natural,
                           fish_deaths_from_lice,
                           new_L2,
                           new_L4,
                           new_females,
                           new_males,
                           num_infection_events,
                           lice_from_reservoir)

        self.logger.debug("    final lice population = {}".format(self.lice_population))
        self.logger.debug("    final fish population = {}".format(self.num_fish))

    def get_lice_treatment_mortality_rate(self, cur_date):
        """
        Compute the mortality rate due to chemotherapeutic treatment (See Aldrin et al, 2017, §2.2.3)
        """
        num_susc = sum(self.lice_population[x] for x in self.susceptible_stages)

        # TODO: take temperatures into account? See #22
        if cur_date - dt.timedelta(days=self.cfg.delay_EMB) in self.cfg.farms[self.farm_id].treatment_dates:
            self.logger.debug('    treating farm {}/cage {} on date {}'.format(self.farm_id,
                                                                               self.id, cur_date))

            # number of lice in those stages that are susceptible to Emamectin Benzoate (i.e.
            # those L3 or above)
            # we assume the mortality rate is the same across all stages and ages, but this may change in the future
            # (or with different chemicals)

            # model the resistence of each lice in the susceptible stages (phenoEMB) and estimate
            # the mortality rate due to treatment (ETmort).
            pheno_emb = np.random.normal(self.cfg.f_meanEMB, self.cfg.f_sigEMB, num_susc) \
                        + np.random.normal(self.cfg.env_meanEMB, self.cfg.env_sigEMB, num_susc)
            pheno_emb = 1 / (1 + np.exp(pheno_emb))
            mortality_rate = sum(pheno_emb) * self.cfg.EMBmort
            return mortality_rate, pheno_emb, num_susc

        else:
            return 0, 0, num_susc

    def update_lice_treatment_mortality(self, cur_date):
        """
        Calculate the number of lice in each stage killed by treatment.
        """

        dead_lice_dist = {"L1": 0, "L2": 0, "L3": 0, "L4": 0, "L5m": 0, "L5f": 0}

        mortality_rate, pheno_emb, num_susc = self.get_lice_treatment_mortality_rate(cur_date)

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
            p = (pheno_emb) / np.sum(pheno_emb)
            dead_lice = np.random.choice(range(num_susc), num_dead_lice, p=p,
                                         replace=False).tolist()
            total_so_far = 0
            for stage in self.susceptible_stages:
                num_dead = len([x for x in dead_lice if total_so_far <= x <
                                (total_so_far + self.lice_population[stage])])
                total_so_far += self.lice_population[stage]
                if num_dead > 0:
                    dead_lice_dist[stage] = num_dead

            self.logger.debug('      distribution of dead lice on farm {}/cage {} = {}'
                              .format(self.farm_id, self.id, dead_lice_dist))

            assert num_dead_lice == sum(list(dead_lice_dist.values()))
        return dead_lice_dist

    def get_stage_ages_distrib(self, stage: str, size=15):
        """
        Create an age distribution (in days) for the sea lice within a lyfecycle stage.
        In absence of further data or constraints, we simply assume it's a uniform distribution
        """

        stage_age_max_days = self.cfg.stage_age_evolutions[stage]
        p = np.ones((size,))
        p[size - stage_age_max_days:] = 0
        return p / np.sum(p)

    @staticmethod
    def get_evolution_ages(size: int, minimum_age: int, mean: int, development_days=25):
        """
        Create an age distribution (in days) for the sea lice within a lifecycle stage.
        TODO: This actually computes the evolution ages.
        :param size the number of lice to consider
        :param minimum_age the minimum age before an evolution is allowed
        :param mean the mean of the distribution. mean must be bigger than min.
        :param development_days the maximum age to consider
        :return a size-long array of ages (in days)
        """

        # Create a shifted poisson distribution,
        k = mean - minimum_age
        max_quantile = development_days - minimum_age

        assert minimum_age > 0, "min must be positive"
        assert k > 0, "mean must be greater than min."

        p = stats.poisson.pmf(range(max_quantile), k)
        p = p / np.sum(p)  # probs need to add up to one
        return np.random.choice(range(max_quantile), size, p=p) + minimum_age

    def get_lice_lifestage(self, cur_month):
        """
        Move lice between lifecycle stages.
        See Section 2.1 of Aldrin et al. (2017)
        """
        self.logger.debug('    updating lice lifecycle stages')

        def dev_times(del_p, del_m10, del_s, temp_c, ages):
            """
            Probability of developing after n_days days in a stage as given by Aldrin et al 2017
            See section 2.2.4 of Aldrin et al. (2017)
            :param ages: stage-age
            :param del_p: power transformation constant on temp_c
            :param del_m10: 10 °C median reference value
            :param del_s: fitted Weibull shape parameter
            :param temp_c: average temperature in °C
            :return: expected development rate
            """
            epsilon = 1e-30
            del_m = del_m10 * (10 / temp_c) ** del_p

            unbounded = math.log(2) * del_s * ages ** (del_s - 1) * del_m ** (-del_s)
            unbounded = np.clip(unbounded, epsilon, 1)
            return unbounded.astype('float64')

        lice_dist = {}
        ave_temp = self.farm.year_temperatures[cur_month - 1]

        # L4 -> L5
        # TODO these blocks look like the same?

        num_lice = self.lice_population['L4']
        stage_ages = self.get_evolution_ages(num_lice, minimum_age=10, mean=15)

        l4_to_l5 = dev_times(self.cfg.delta_p["L4"], self.cfg.delta_m10["L4"], self.cfg.delta_s["L4"],
                             ave_temp, stage_ages)
        num_to_move = min(np.random.poisson(np.sum(l4_to_l5)), num_lice)
        new_females = np.random.choice([math.floor(num_to_move / 2.0), math.ceil(num_to_move / 2.0)])
        new_males = (num_to_move - new_females)

        lice_dist['L5f'] = new_females
        lice_dist['L5m'] = (num_to_move - new_females)

        # L3 -> L4
        num_lice = self.lice_population['L3']
        stage_ages = self.get_evolution_ages(num_lice, minimum_age=15, mean=18)
        l3_to_l4 = dev_times(self.cfg.delta_p["L3"], self.cfg.delta_m10["L3"], self.cfg.delta_s["L3"],
                             ave_temp, stage_ages)
        num_to_move = min(np.random.poisson(np.sum(l3_to_l4)), num_lice)
        new_L4 = num_to_move

        lice_dist['L4'] = num_to_move

        # L2 -> L3
        # This is done in do_infection_events()

        # L1 -> L2
        num_lice = self.lice_population['L2']
        stage_ages = self.get_evolution_ages(num_lice, minimum_age=3, mean=4)
        l1_to_l2 = dev_times(self.cfg.delta_p["L1"], self.cfg.delta_m10["L1"], self.cfg.delta_s["L1"],
                             ave_temp, stage_ages)
        num_to_move = min(np.random.poisson(np.sum(l1_to_l2)), num_lice)
        new_L2 = num_to_move

        lice_dist['L2'] = num_to_move

        self.logger.debug('      distribution of new lice lifecycle stages on farm {}/cage {} = {}'
                          .format(self.farm_id, self.id, lice_dist))

        return new_L2, new_L4, new_females, new_males

    def get_fish_growth(self, days, step_size):
        """
        Get the new number of fish after a step size.
        """

        self.logger.debug('    updating fish population')

        def fb_mort(days):
            """
            Fish background mortality rate, decreasing as in Soares et al 2011

            Fish death rate: constant background daily rate 0.00057 based on
            www.gov.scot/Resource/0052/00524803.pdf
            multiplied by lice coefficient see surv.py (compare to threshold of 0.75 lice/g fish)

            TODO: what should we do with the formulae?

            :param days: number of days elapsed
            :return: fish background mortality rate
            """
            return 0.00057  # (1000 + (days - 700)**2)/490000000

        # determine the number of fish with lice and the number of attached lice on each.
        # for now, assume there is only one TODO fix this when I understand the infestation
        # (next stage)
        adlicepg = 1 / self.fish_growth_rate(days)
        prob_lice_death = 1 / (1 + math.exp(-self.cfg.fish_mortality_k * (adlicepg - self.cfg.fish_mortality_center)))

        ebf_death = fb_mort(days) * step_size * self.num_fish
        elf_death = self.num_infected_fish * step_size * prob_lice_death
        fish_deaths_natural = np.random.poisson(ebf_death)
        fish_deaths_from_lice = np.random.poisson(elf_death)

        self.logger.debug('      number of background fish death {}, from lice {}'
                          .format(fish_deaths_natural, fish_deaths_from_lice))

        return fish_deaths_natural, fish_deaths_from_lice

    def compute_eta_aldrin(self, num_fish_in_farm, days):
        return self.cfg.infection_main_delta + math.log(num_fish_in_farm) + self.cfg.infection_weight_delta * \
               (math.log(self.fish_growth_rate(days)) - self.cfg.delta_expectation_weight_log)

    def get_infection_rates(self, days) -> (float, int):
        """
        Compute the number of lice that can infect and what their infection rate (number per fish) is

        :param days the amount of time it takes
        :returns a pair (Einf, num_avail_lice)
        """
        # Perhaps we can have a distribution which can change per day (the mean/median increaseѕ?
        # but at what point does the distribution mean decrease).
        age_distrib = self.get_stage_ages_distrib('L2')
        num_avail_lice = round(self.lice_population['L2'] * np.sum(age_distrib[1:]))
        if num_avail_lice > 0:
            num_fish_in_farm = sum([c.num_fish for c in self.farm.cages])

            # TODO: this has O(c^2) complexity
            etas = np.array([c.compute_eta_aldrin(num_fish_in_farm, days) for c in self.farm.cages])
            Einf = math.exp(etas[self.id]) / (1 + np.sum(np.exp(etas)))

            return Einf, num_avail_lice

        return 0, num_avail_lice

    def do_infection_events(self, days) -> int:
        """
        Infect fish in this cage if the sea lice are in stage L2 and at least 1 day old

        :param days the number of days elapsed
        :returns number of evolving lice, or equivalently the new number of infections
        """
        Einf, num_avail_lice = self.get_infection_rates(days)

        if Einf == 0:
            return 0

        inf_events = np.random.poisson(Einf * num_avail_lice)
        return min(inf_events, num_avail_lice)

    def get_infected_fish(self):
        # the number of infections is equal to the population size from stage 3 onward
        infective_stages = ["L3", "L4", "L5m", "L5f"]
        attached_lice = sum(self.lice_population[stage] for stage in infective_stages)

        # see: https://stats.stackexchange.com/a/296053
        num_infected_fish = int(self.num_fish * (1 - ((self.num_fish - 1) / self.num_fish) ** attached_lice))
        return num_infected_fish

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

    def update_deltas(self, dead_lice_dist, treatment_mortality, fish_deaths_natural, fish_deaths_from_lice, new_L2,
                      new_L4, new_females, new_males, new_infections, lice_from_reservoir):
        """
        Update the number of fish and the lice in each life stage given the number that move between stages in this time
        period.
        """

        for stage in self.lice_population:
            # update background mortality
            bg_delta = self.lice_population[stage] - dead_lice_dist[stage]
            self.lice_population[stage] = max(0, bg_delta)

            # update population due to treatment
            num_dead = treatment_mortality.get(stage, 0)
            treatment_delta = self.lice_population[stage] - num_dead
            self.lice_population[stage] = max(0, treatment_delta)

        self.lice_population['L5m'] += new_males
        self.lice_population['L5f'] += new_females

        update_L4 = self.lice_population['L4'] - (new_males + new_females) + new_L4
        self.lice_population['L4'] = max(0, update_L4)

        update_L3 = self.lice_population['L3'] - new_L4 + new_infections
        self.lice_population['L3'] = max(0, update_L3)

        update_L2 = self.lice_population['L2'] + new_L2 - new_infections + lice_from_reservoir["L2"]
        self.lice_population['L2'] = max(0, update_L2)

        update_L1 = self.lice_population['L1'] - new_L2 + lice_from_reservoir["L1"]
        self.lice_population['L1'] = max(0, update_L1)

        self.num_fish -= fish_deaths_natural
        self.num_fish -= fish_deaths_from_lice

        # treatment may kill some lice attached to the fish, thus update at the very end
        self.num_infected_fish = self.get_infected_fish()

        return self.lice_population

    def get_background_lice_mortality(self, lice_population):
        """
        Background death in a stage (remove entry) -> rate = number of
        individuals in stage*stage rate (nauplii 0.17/d, copepods 0.22/d,
        pre-adult female 0.05, pre-adult male ... Stien et al 2005)
        """
        lice_mortality_rates = {'L1': 0.17,
                                'L2': 0.22,
                                'L3': 0.008,
                                'L4': 0.05,
                                'L5f': 0.02,
                                'L5m': 0.06}

        dead_lice_dist = {}
        for stage in lice_population:
            mortality_rate = lice_population[stage] * lice_mortality_rates[stage] * self.cfg.tau
            mortality = min(np.random.poisson(mortality_rate), lice_population[stage])
            dead_lice_dist[stage] = mortality

        self.logger.debug('    background mortality distribn of dead lice = {}'.format(dead_lice_dist))
        return dead_lice_dist

    def fish_growth_rate(self, days):
        return 10000/(1 + math.exp(-0.01*(days-475)))

    def to_csv(self):
        """
        Save the contents of this cage as a CSV string
        for writing to a file later.
        """

        data = [str(self.id), str(self.num_fish)]
        data.extend([str(val) for val in self.lice_population.values()])
        data.append(str(sum(self.lice_population.values())))

        return ", ".join(data)

    def get_reservoir_lice(self, pressure):
        """Get distribution of lice coming from the reservoir

        :param pressure: External pressure
        :type pressure: int
        :return: Distribution of lice in L1 and L2
        :rtype: dict
        """

        num_L1 = self.cfg.rng.integers(low=0, high=pressure, size=1)
        new_lice_dist = {"L1": num_L1, "L2": pressure - num_L1}
        self.logger.debug('    distribn of new lice from reservoir = {}'.format(new_lice_dist))
        return new_lice_dist