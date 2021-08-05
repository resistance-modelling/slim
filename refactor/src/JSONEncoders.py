from dataclasses import is_dataclass, asdict
from enum import Enum
import datetime as dt
import json
from queue import PriorityQueue

import numpy as np

from src.QueueBatches import TreatmentEvent, Treatment


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

        elif is_dataclass(o):
            return asdict(o)

        elif isinstance(o, PriorityQueue):
            return sorted(list(o.queue))

        # Python's circular dependencies are annoying
        elif type(o).__name__ == "Cage":
            return o.to_json_dict()

        elif isinstance(o, np.number):
            return o.item()

        return o