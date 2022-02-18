from typing import List, Union

import numpy as np

from .TreatmentTypes import Treatment
from gym.spaces import Discrete, MultiBinary

__all__ = ["ACTION_SPACE", "CURRENT_TREATMENTS"]

treatment_no = len(Treatment)
# 0 - (treatment_no - 1): treatment code
# treatment_no: fallowing
# treatment_no+1: no action
ACTION_SPACE = Discrete(treatment_no + 2)
ACTION_TYPE = int
SAMPLED_ACTIONS = Union[List[ACTION_TYPE], np.ndarray[np.int32]]
# 0th (MSB) - (treatment_no - 1)th bit: treatment code
# treatment_no: fallowing
# Each of those bits is 1 if that treatment is performed.
CURRENT_TREATMENTS = MultiBinary(treatment_no + 1)
