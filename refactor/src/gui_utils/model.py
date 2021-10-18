from __future__ import annotations

import dataclasses
import datetime as dt
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from src.Simulator import Simulator
    import pandas as pd

@dataclasses.dataclass
class SimulatorSingleRunState:
    # TODO: should I rather subclass from AbstractModel
    # TODO: should I move this to another file?
    states: List[Simulator]
    times: List[dt.datetime]
    states_as_df: pd.DataFrame
