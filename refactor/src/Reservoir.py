import numpy as np
import src.config as cfg

from src.CageTemplate import CageTemplate


class Reservoir(CageTemplate):
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
        self.num_fish = 4000

        # this is by no means optimal: I want to distribute nplankt lice across
        # each stage at random. (the method I use here is to create a random list
        # of stages and add up the number for each.
        lice_stage = np.random.choice(range(2, 7), nplankt)
        num_lice_in_stage = np.array([sum(lice_stage == i) for i in range(1, 7)])
        self.lice_population = {'L1': num_lice_in_stage[0], 
                                'L2': num_lice_in_stage[1],
                                'L3': num_lice_in_stage[2], 
                                'L4': num_lice_in_stage[3],
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

        self.lice_population = self.update_background_lice_mortality(self.lice_population, step_size)

        # TODO - infection events in the reservoir
        #eta_aldrin = -2.576 + log(inpt.Enfish_res[cur_date.month-1]) + 0.082*(log(inpt.Ewt_res)-0.55)
        #Einf = (exp(eta_aldrin)/(1+exp(eta_aldrin)))*tau*cop_cage
        #inf_ents = np.random.poisson(Einf)
        #inf_ents = min(inf_ents,cop_cage)
        #inf_inds = np.random.choice(df_list[fc].loc[(df_list[fc].stage==2) & (df_list[fc].arrival<=df_list[fc].stage_age)].index, inf_ents, replace=False)

        cfg.logger.debug("  final lice population = {}".format(self.lice_population))