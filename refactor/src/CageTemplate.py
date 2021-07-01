import math

import numpy as np


class CageTemplate:
    """Class for methods shared between the sea cages and reservoir."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = cfg.logger

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
