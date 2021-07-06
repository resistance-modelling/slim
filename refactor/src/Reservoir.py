import math

import numpy as np

from src.CageTemplate import CageTemplate


class Reservoir(CageTemplate):
    """
    The reservoir for sea lice, essentially modelled as a sea cage.
    """

    def __init__(self, cfg):
        """Create a reservoir

        :param cfg: Simulation configuration and parameters
        :type cfg: src.Config.Config
        """

        super().__init__(cfg, "reservoir")

        self.date = cfg.start_date
        self.num_fish = cfg.reservoir_num_fish

        # define life stage labels
        life_stages = ["L1", "L2", "L3", "L4", "L5f", "L5m"]
        n = len(life_stages)

        # set probability of each life stage - in this case,
        # the same of all lifestages
        life_stages_prob = np.full(n, 1/n)

        # construct rng generator
        rng = np.random.default_rng(seed=self.cfg.seed)

        # generate distribution based on life stage probs that sums
        # up to total number of initial lice in reservoir
        dist = rng.multinomial(cfg.reservoir_num_lice,
                               life_stages_prob,
                               size=1)[0]

        # construct the population dict by assigning the numbers to labels
        self.lice_population = {life_stages[ix]: dist[ix] for ix in range(n)}

    def update(self, cur_date, farms):
        """Update the reservoir at the current time step.
        * update lice population
        * perform infection
        * TODO: mating?

        :param cur_date: Current date of the simulation
        :type cur_date: datetime.datetime
        :param farms: List of Farm objects
        :type farms: list
        """

        self.logger.debug("Updating reservoir")
        self.logger.debug("  initial lice population = {}".format(self.lice_population))

        # get new lice population after background deaths
        dead_lice_dist = self.get_background_lice_mortality(self.lice_population)
        self.lice_population = {stage: max(0, self.lice_population[stage] - dead) for stage, dead in dead_lice_dist.items()}

        # retrieve deltas
        main_delta = self.cfg.infection_main_delta
        weight_delta = self.cfg.infection_weight_delta

        # term involving month-based number of fish in millions
        # in the reservoir
        num_fish_term = math.log(self.cfg.reservoir_enfish_res[cur_date.month - 1])

        # term involving average fish weight
        # 0.55 is constant from (Aldrin, 2017)
        weight_term = math.log(self.cfg.reservoir_ewt_res) - 0.55

        # calculate the eta
        eta_aldrin = main_delta + num_fish_term + weight_delta * weight_term

        # TODO: this variable depends on arrival and stage age - update once
        # age dependency handling is agreed upon
        # `cop_cage = sum((df_list[fc].stage==2) & (df_list[fc].arrival<=df_list[fc].stage_age))``
        num_avail_lice = self.lice_population["L3"]

        # calculate infection rate
        infection_rate = (math.exp(eta_aldrin) / (1 + math.exp(eta_aldrin))) * self.cfg.tau * num_avail_lice

        # get the number of infections
        infections = np.random.poisson(infection_rate)

        # get lower out of infections and lice - can't infect
        # more fish then there are lice
        infections = min(infections, num_avail_lice)

        self.logger.debug("  final lice population = {}".format(self.lice_population))
        self.logger.debug("  number of infections = {}".format(infections))
