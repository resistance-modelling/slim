from __future__ import annotations

import dataclasses
import datetime as dt
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from src.Simulator import Simulator
    import pandas as pd

@dataclasses.dataclass
class SimulatorSingleRunState:
    """State of the single simulator"""
    # TODO: should I rather subclass from AbstractModel
    states: List[Simulator]
    times: List[dt.datetime]
    states_as_df: pd.DataFrame

@dataclasses.dataclass
class SimulatorOptimiserState:
    """Optimiser state"""
    states: List[Simulator]
    states_as_df: pd.DataFrame
