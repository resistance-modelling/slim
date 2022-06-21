"""
Models for communication across widgets in accordance to MVC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime as dt
from typing import List, TYPE_CHECKING, Tuple, Optional

import numpy as np
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QColor
from PyQt5.QtPositioning import QGeoCoordinate

if TYPE_CHECKING:
    from slim.simulation.simulator import Simulator
    from slim.simulation.config import Config
    import pandas as pd


@dataclass
class SimulatorSingleRunState:
    """State of the single simulator"""

    times: List[dt.datetime]
    states_as_df: pd.DataFrame
    report_df: Optional[pd.DataFrame]
    cfg: Config
    sim_name: str


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
    sum_: bool = field(default=False)


@dataclass
class PositionMarker:
    """Position marker on the GUI"""

    name: str
    position: QGeoCoordinate
    source: QUrl


@dataclass
class TransitionEndpoint:
    """Lice transition on the map"""

    endpoint: Tuple[QGeoCoordinate, QGeoCoordinate]
    color: QColor
    width: int
    value: float
