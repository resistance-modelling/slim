"""
Useful type definitions for Gym/PettingZoo
"""

from typing import List, Union

import numpy as np

from .treatments import TREATMENT_NO
from gym.spaces import Discrete, MultiBinary, Dict, Box

__all__ = [
    "ACTION_SPACE",
    "SAMPLED_ACTIONS",
    "ACTION_SPACE",
    "FALLOW",
    "NO_ACTION",
    "CURRENT_TREATMENTS",
    "TREATMENT_NO",
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
# Problem: the observation type needs to be flattened...
ObservationType = Union[Dict, Box]
# 0th (MSB) - (treatment_no - 1)th bit: treatment code
# treatment_no: fallowing
# Each of those bits is 1 if that treatment is performed.
CURRENT_TREATMENTS = MultiBinary(TREATMENT_NO + 1)
