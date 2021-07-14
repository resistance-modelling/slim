import datetime as dt
import json

import numpy as np


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
        elif isinstance(o, np.int64):
            return_str = str(o)
        elif isinstance(o, dt.datetime):
            return_str = o.strftime("%Y-%m-%d %H:%M:%S")
        else:
            return_str = {"__{}__".format(o.__class__.__name__): o.__dict__}

        return return_str
