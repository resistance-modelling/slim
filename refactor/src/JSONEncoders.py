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
        return_str = ""
        if isinstance(o, np.ndarray):
            return_str = str(o)
        elif isinstance(o, int):
            return_str = str(o)
        elif isinstance(o, dt.datetime):
            return_str = o.strftime("%Y-%m-%d %H:%M:%S")
        else:
            return_str = {"__{}__".format(o.__class__.__name__): o.__dict__}

        return return_str

class CustomCageEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, dt.datetime):
            return o.strftime("%Y-%m-%d %H:%M:%S")

        elif isinstance(o, Treatment):
            return str(o)

        elif isinstance(o, TreatmentEvent):
            return vars(o)

        elif isinstance(o, PriorityQueue):
            return sorted(list(o.queue))

        return o
