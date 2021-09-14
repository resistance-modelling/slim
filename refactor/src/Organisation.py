"""
An organisation controls multiple farms.
For now, we assume an organisation is also an agent that can control all the
farms at the same time.
TODO: treatments should be decided rather than preemptively scheduled
"""

import datetime as dt
import json
from src.Config import Config
from src.Farm import Farm
from src.JSONEncoders import CustomFarmEncoder
from src.QueueTypes import pop_from_queue, FarmResponse, SamplingResponse
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
        for farm in self.farms:
            self.handle_farm_messages(cur_date, farm)

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
            "farms": self.farms
        }

    def __repr__(self):
        return json.dumps(self.to_json_dict(), cls=CustomFarmEncoder, indent=4)

    def handle_farm_messages(self, cur_date: dt.datetime, farm: Farm):
        def cts(farm_response: FarmResponse):
            if isinstance(farm_response, SamplingResponse) and \
                    farm_response.detected_rate >= self.cfg.aggregation_rate_threshold:
                # send a treatment command to everyone
                for other_farm in self.farms:
                    other_farm.ask_for_treatment(cur_date, can_defect=other_farm != farm)

        pop_from_queue(farm.farm_to_org, cur_date, cts)
