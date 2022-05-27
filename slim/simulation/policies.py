"""
A collection of policies.
"""

import datetime as dt

from .config import Config
from slim import logger
from slim.types.policies import ObservationSpace, TREATMENT_NO, FALLOW, NO_ACTION
from slim.types.treatments import Treatment

from gym import spaces
import numpy as np
import pandas as pd

# TODO: figure out if I can use StableBaselines' interfaces here


class BernoullianPolicy:
    """
    This policy is arguably one of the simplest baselines in SLIM. It implements a bernoullian mechanism
    to choose when to apply a treatment depending on the current lice aggregation and the organisation's
    recommendation.

    * if a farm has LC < suggested threshold then no treatment can be applied at any time
    * if a farm has enforcement threshold >= LC >= suggested threshold AND there is a request to treat then treatment
      will be applied with probability :math:`p` < 1.
    * if a farm has LC >= enforcement threshold then treatment is performed with p=1

    The value of :math:`p` is farm-specific.

    Therefore: a farm can either act egoistically (treat only when the situation is bordering illegality)
    or altruistically (treat at much lower counts).
    Keep in mind that treatments need to satisfy cage requirements (see :meth:`slim.simulation.farm.Farm.add_treatment`)

    As for the action, each action can be chosen with likelihood :math:`1/n` with :math:`n` being the number of available treatments.
    See :class:`slim.simulation.config.Config` for details.
    """

    def __init__(self, cfg: Config):
        self.proba = [farm_cfg.defection_proba for farm_cfg in cfg.farms]
        n = len(Treatment)
        self.treatment_probas = np.ones(n) / n
        self.enforced_threshold = cfg.agg_rate_enforcement_threshold
        self.seed = cfg.seed
        self.reset()

    def _predict(self, asked_to_treat, agg_rate: np.ndarray, agent: int):
        must_treat = np.any(agg_rate >= self.enforced_threshold)
        p = [self.proba[agent], 1 - self.proba[agent]]
        want_to_treat = self.rng.choice([False, True], p=p) if asked_to_treat else False
        logger.debug("Outcome of the vote: %s (was forced: %s)", want_to_treat, must_treat)

        if not (must_treat or want_to_treat):
            logger.debug("\tFarm %s refuses to treat", agent)
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
        if (
            not observation["asked_to_treat"]
            or observation["aggregation"].mean() <= 2.0
        ):
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
            "Sea lice management / Gill health related": [Treatment.EMB.value],
            "Stocking": [Treatment.CLEANERFISH.value],
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
