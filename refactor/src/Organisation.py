"""
An organisation controls multiple farms.
For now, we assume an organisation is also an agent that can control all the
farms at the same time.
TODO: treatments should be decided rather than preemptively scheduled
"""

import json
from src.Config import Config
from src.Farm import Farm
from src.JSONEncoders import CustomFarmEncoder
from src.TreatmentTypes import Money


class Organisation:
    def __init__(self, cfg: Config):
        self.name = cfg.name  # type: str
        self.cfg = cfg
        self.farms = [Farm(i, cfg) for i in range(cfg.nfarms)]

    @property
    def capital(self):
        return sum((farm.current_capital for farm in self.farms), Money())

    def step(self, cur_date) -> Money:
        # days = (cur_date - self.cfg.start_date).days

        # update the farms and get the offspring
        offspring_dict = {}
        payoff = Money()
        for farm in self.farms:
            offspring, cost = farm.update(cur_date)
            offspring_dict[farm.name] = offspring
            # TODO: take into account other types of disadvantages, not just the mere treatment cost
            # e.g. how are environmental factors measured? Are we supposed to add some coefficients here?
            # TODO: if we take current fish population into account then what happens to the infrastructure cost?
            payoff += farm.get_profit(cur_date) - cost

        # once all of the offspring is collected
        # it can be dispersed (interfarm and intercage movement)
        # note: this is done on organisation level because it requires access to
        # other farms - and will allow multiprocessing of the main update
        for farm_ix, offspring in offspring_dict.items():
            self.farms[farm_ix].disperse_offspring(offspring, self.farms, cur_date)

        return payoff

    def to_json(self, **kwargs):
        json_dict = kwargs.copy()
        json_dict.update(self.to_json_dict())
        return json.dumps(json_dict, cls=CustomFarmEncoder, indent=4)

    def to_json_dict(self):
        return {
            "name": self.name,
            "farms": [farm.to_json_dict() for farm in self.farms]
        }

    def __repr__(self):
        return json.dumps(self.to_json_dict(), cls=CustomFarmEncoder, indent=4)
