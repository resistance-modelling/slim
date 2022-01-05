from __future__ import annotations

from dataclasses import dataclass, field
import datetime as dt
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from slim.Simulator import Simulator
    import pandas as pd


@dataclass
class SimulatorSingleRunState:
    """State of the single simulator"""
    # TODO: should I rather subclass from AbstractModel
    states: List[Simulator]
    times: List[dt.datetime]
    states_as_df: pd.DataFrame


@dataclass
class SimulatorOptimiserState:
    """Optimiser state"""
    states: List[List[Simulator]]
    states_as_df: pd.DataFrame


@dataclass
class CurveListState:
    L1: bool = field(default=True)
    L2: bool = field(default=True)
    L3: bool = field(default=True)
    L4: bool = field(default=True)
    L5f: bool = field(default=True)
    L5m: bool = field(default=True)
    Eggs: bool = field(default=True)
    ExtP: bool = field(default=True)
