from collections import Counter
from dataclasses import is_dataclass, asdict
from enum import Enum
import datetime as dt
import json

import numpy as np

from src.TreatmentTypes import Money
from src.QueueTypes import PriorityQueue


class CustomFarmEncoder(json.JSONEncoder):
    """
    Bespoke encoder to encode Farm objects to json. Specifically, numpy arrays
    and datetime objects are not automatically converted to json.
    """

    def default(self, o):
        """
        Provide a json string of an object.
        :param o: The object to be encoded as a json string.
        :return: the json representation of o.
        """
        if isinstance(o, np.ndarray):
            return o.tolist()

        elif isinstance(o, dt.datetime):
            return o.strftime("%Y-%m-%d %H:%M:%S")

        elif isinstance(o, Enum):
            return str(o)

        elif isinstance(o, PriorityQueue):
            return sorted(list(o.queue))

        elif hasattr(o, "to_json_dict"):
            return o.to_json_dict()

        elif isinstance(o, np.number):
            return o.item()

        elif isinstance(o, Money):
            return str(o)

        return o