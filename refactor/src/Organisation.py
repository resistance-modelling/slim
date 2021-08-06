"""
An organisation controls multiple farms.
For now, we assume an organisation is also an agent that can control all the
farms at the same time.
TODO: treatments should be decided rather than preemptively scheduled
"""

from src.Config import Config
from src.Farm import Farm
from src.JSONEncoders import CustomFarmEncoder
import json

class Organisation:
    def __init__(self, cfg: Config):
        self.name = cfg.name  # type: str
        self.current_capital = cfg.start_capital
        self.cfg = cfg
        self.farms = [Farm(i, cfg) for i in range(cfg.nfarms)]

    def step(self, cur_date):
        days = (cur_date - self.cfg.start_date).days

        # update the farms and get the offspring
        offspring_dict = {}
        for farm in self.farms:

            # if days == 1:
            #    resistance_bv.write_text(cur_date, prev_muEMB[farm], prev_sigEMB[farm], prop_ext)

            offspring = farm.update(cur_date)
            offspring_dict[farm.name] = offspring

        # once all of the offspring is collected
        # it can be dispersed (interfarm and intercage movement)
        # note: this is done on organisation level because it requires access to
        # other farms - and will allow muliprocessing of the main update
        for farm_ix, offspring in offspring_dict.items():
            self.farms[farm_ix].disperse_offspring(offspring, self.farms, cur_date)

    def to_json(self, **kwargs):
        json_dict = kwargs.copy()
        json_dict.update(self.to_json_dict())
        return json.dumps(json_dict, cls=CustomFarmEncoder, indent=4)

    def to_json_dict(self):
        return {
            "name": self.name,
            "capital": self.current_capital,
            "farms": [farm.to_json_dict() for farm in self.farms]
        }

    def __repr__(self):
        return json.dumps(self.to_json_dict(), cls=CustomFarmEncoder, indent=4)
