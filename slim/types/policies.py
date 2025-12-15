"""
Useful type definitions for Gym/PettingZoo

Note: Gym uses spaces as an abstraction to perform run-time typechecking.
Spaces are _not_ containers but rather container descriptions.

"""
import functools
from typing import List, Union, TypedDict, Dict

import numpy as np
from gymnasium.spaces import Discrete, MultiBinary, Box, Space, Dict as GymDict

from .treatments import TREATMENT_NO

__all__ = [
    "ACTION_SPACE",
    "SAMPLED_ACTIONS",
    "ACTION_SPACE",
    "FALLOW",
    "NO_ACTION",
    "CURRENT_TREATMENTS",
    "TREATMENT_NO",
    "ObservationSpace",
    "ObservationSpace",
    "SimulatorSpace",
    "agent_to_id",
    "get_observation_space_schema",
    "no_observation",
]

# 0 - (treatment_no - 1): treatment code
# treatment_no: fallowing
# treatment_no+1: no action
ACTION_SPACE = Discrete(TREATMENT_NO + 2)
# For efficiency reasons, these need to be integer rather than enums
FALLOW = TREATMENT_NO
NO_ACTION = TREATMENT_NO + 1


ActionType = int
SAMPLED_ACTIONS = Union[List[ActionType], np.ndarray]
# 0th (MSB) - (treatment_no - 1)th bit: treatment code
# treatment_no: fallowing
# Each of those bits is 1 if that treatment is performed.
CURRENT_TREATMENTS = MultiBinary(TREATMENT_NO + 1)


class ObservationSpace(Space):
    aggregation: np.ndarray
    fish_population: np.ndarray
    cleaner_fish: np.ndarray
    reported_aggregation: np.ndarray
    current_treatments: np.ndarray
    allowed_treatments: int
    asked_to_treat: np.ndarray  # gym doesn't like bools


def get_observation_space_schema(agents: List[str], num_applications: int):
    single_obs_space = GymDict({
        "aggregation": Box(low=0, high=20, shape=(1,), dtype=np.float32),
        "reported_aggregation": Box(low=0, high=20, shape=(1,), dtype=np.float32),
        "fish_population": Box(low=0, high=1e8, shape=(1,), dtype=np.int64),
        "cleaner_fish": Box(low=0, high=1e8, shape=(1,), dtype=np.int64),
        "current_treatments": CURRENT_TREATMENTS,
        "allowed_treatments": Discrete(num_applications),
        "asked_to_treat": MultiBinary(1),  # Yes or no
    })

    observation = GymDict({"observation": single_obs_space})

    # Return the same observation space for each agent
    return {agent: observation for agent in agents}


    # return {
    #     agent: GymDict(
    #         {
    #             "observation": GymDict({
    #             "aggregation": Box(low=0, high=20, shape=(1,), dtype=np.float32),
    #             "reported_aggregation": Box(
    #                 low=0, high=20, shape=(1,), dtype=np.float32
    #             ),
    #             "fish_population": Box(low=0, high=1e8, shape=(1,), dtype=np.int64),
    #             "cleaner_fish": Box(low=0, high=1e8, shape=(1,), dtype=np.int64),
    #             "current_treatments": CURRENT_TREATMENTS,
    #             "allowed_treatments": Discrete(num_applications),
    #             "asked_to_treat": MultiBinary(1),  # Yes or no
    #             })
    #         }
    #     )
    #     for agent in agents
    # }


@functools.lru_cache(maxsize=None)
def no_observation() -> ObservationSpace:
    """
    Generate an empty observation.

    :returns an empty observation that matches the schema
    """
    return {
        "aggregation": np.zeros((1,), dtype=np.float32),
        "reported_aggregation": np.zeros((1,), dtype=np.float32),
        "fish_population": np.zeros((1,), dtype=np.int64),
        "cleaner_fish": np.zeros((1,), dtype=np.int64),
        "current_treatments": np.zeros((TREATMENT_NO + 1,), dtype=np.int8),
        "allowed_treatments": 0,
        "asked_to_treat": np.zeros((1,), dtype=np.int8),
    }


def agent_to_id(agent_str: str) -> int:
    return int(agent_str[len("farm_") :])


SimulatorSpace = Dict[str, ObservationSpace]
