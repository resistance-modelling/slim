"""
A collection of policies.
"""

import datetime as dt
import abc

from .config import Config
from slim import logger
from slim.types.policies import ObservationSpace, TREATMENT_NO, FALLOW, NO_ACTION
from slim.types.treatments import Treatment

from gym import spaces
import numpy as np
import pandas as pd


class BernoullianPolicy:
    """
    Perhaps the simplest policy here.
    Never apply treatment except for when asked by the organisation, and in which case whether to apply
    a treatment with a probability :math:`p` or not (:math:`1-p`). In the first case each treatment
    will be chosen with likelihood :math:`q`.
    Also culls if the lice count goes beyond a threshold repeatedly (e.g. after 4 weeks).
    See :class:`Config` for details.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.proba = [farm_cfg.defection_proba for farm_cfg in cfg.farms]
        n = len(Treatment)
        self.treatment_probas = np.ones(n) / n
        self.treatment_threshold = cfg.aggregation_rate_threshold
        self.seed = cfg.seed
        self.reset()

    def _predict(self, asked_to_treat, agg_rate: np.ndarray, agent: int):
        if not asked_to_treat and np.any(agg_rate < self.treatment_threshold):
            return NO_ACTION

        p = [self.proba[agent], 1 - self.proba[agent]]

        want_to_treat = self.rng.choice([False, True], p=p)
        logger.debug(f"Outcome of the vote: {want_to_treat}")

        if not want_to_treat:
            logger.debug("\tFarm {} refuses to treat".format(agent))
            return NO_ACTION

        picked_treatment = self.rng.choice(
            np.arange(TREATMENT_NO), p=self.treatment_probas
        )
        return picked_treatment

    def predict(self, observation: ObservationSpace, agent: str):
        if isinstance(observation, spaces.Box):
            raise NotImplementedError("Only dict spaces are supported for now")

        agent_id = int(agent[len("farm_") :])
        asked_to_treat = bool(observation["asked_to_treat"])
        return self._predict(asked_to_treat, observation["aggregation"], agent_id)

    def reset(self):
        self.rng = np.random.default_rng(self.seed)


class UntreatedPolicy:
    """
    A treatment stub that performs no action.
    """

    def predict(self, *args, **_kwargs):
        return NO_ACTION


class MosaicPolicy:
    """
    A simple treatment: as soon as farms receive treatment the farmers apply treatment.
    Treatments are selected in rotation
    """

    def __init__(self, cfg: Config):
        self.last_action = [0] * len(cfg.farms)

    def predict(self, observation: ObservationSpace, agent: str):
        if not observation["asked_to_treat"] or observation["aggregation"].mean() <= 2.0:
            return NO_ACTION

        agent_id = int(agent[len("farm_") :])
        action = self.last_action[agent_id]
        self.last_action[agent_id] = (action + 1) % TREATMENT_NO
        return action


class PilotedPolicy:
    def __init__(
        self,
        cfg: Config,
        weekly_data: pd.DataFrame,
        monthly_data_collated: pd.DataFrame,
    ):
        # Now, we need to translate each of these actions into treatment objects
        conversion_map = {
            "Physical": [Treatment.THERMOLICER.value],
            "Bath": [Treatment.EMB.value],
            "Harvesting": [FALLOW],
            "EoC": [FALLOW],
            "Bio reduction": [FALLOW],  # TODO: implement thinning (?)
            "Bath & Physical": [Treatment.EMB.value, Treatment.THERMOLICER.value],
            "Physical & Harvesting": [Treatment.THERMOLICER.value, FALLOW],
            "Environmental": [Treatment.EMB.value],
            "Treatment / Handling": [Treatment.EMB.value],
            "Handling": [Treatment.EMB.value],
            "Gill Health/Treatment losses": [Treatment.EMB.value],
            "Gill Health/Handling": [Treatment.EMB.value],
            "Sea lice management / Viral disease": [Treatment.EMB.value],
            "Handling / Sea lice management": [Treatment.EMB.value],
            "Sea lice management" "Gill health related": [Treatment.EMB.value],
        }

        self.map_to_action = {}
        self.cur_day_per_farm = {}
        self.expected_lice_aggregation = {}
        self.expected_fish_population = {}

        for farm_idx, farm in enumerate(cfg.farms):
            weekly_data_farm = weekly_data[weekly_data["site_name"] == farm.name]
            monthly_data_farm = monthly_data_collated[
                monthly_data_collated["site_name"] == farm.name
            ]
            if len(weekly_data_farm) == 0:
                data = monthly_data_farm
                mitigations = data["lice_note"]
                time = data["date"]
            else:
                data = weekly_data_farm
                mitigations = data["mitigation"]
                time = data["time"]
            mitigations = mitigations.apply(
                lambda x: conversion_map.get(x, [NO_ACTION])
            )
            ops = (
                pd.DataFrame({"time": time.to_list(), "actions": mitigations.to_list()})
                .set_index("time")
                .to_dict()
            )
            self.map_to_action[farm_idx] = ops["actions"]
            self.cur_day_per_farm[farm_idx] = cfg.start_date

    def predict(self, _, agent):
        farm = int(agent[len("farm_") :])
        date = self.cur_day_per_farm[farm]
        self.cur_day_per_farm[farm] += dt.timedelta(days=1)
        # take the last monday
        date_to_take = date - dt.timedelta(days=date.weekday())
        actions = self.map_to_action[farm].get(pd.Timestamp(date_to_take), [NO_ACTION])
        if NO_ACTION not in actions:
            print(actions)
        if len(actions) > date.weekday():
            actions = actions[date.weekday()]
        else:
            actions = NO_ACTION
        return actions
