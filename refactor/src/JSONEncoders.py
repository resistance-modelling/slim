import datetime as dt
import json

import numpy as np

import src.Farm as frm
from src.Config import Config, FarmConfig


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
        return_str = ''
        if isinstance(o, np.ndarray):
            return_str = str(o)
        elif isinstance(o, np.int64):
            return_str = str(o)
        elif isinstance(o, dt.datetime):
            return_str = o.strftime("%Y-%m-%d %H:%M:%S")
        else:
            return_str = {'__{}__'.format(o.__class__.__name__): o.__dict__}

        return return_str


class CustomCageEncoder(json.JSONEncoder):
    """
    Bespoke encoder to encode Cage objects to json. Specifically, numpy arrays
    and datetime objects are not automatically converted to json.
    TODO: this is unused as filtering out keys is impossible via a JSONEncoder, see https://stackoverflow.com/a/56138540
    """
    def default(self, o):
        """
        Provide a json string of an object.
        :param o: The object to be encoded as a json string.
        :return: the json representation of o.
        """
        return_str = ''
        if isinstance(o, np.ndarray):
            return_str = str(o)
        elif isinstance(o, frm.Farm):
            return_str = ""
        elif isinstance(o, np.int64):
            return_str = str(o)
        elif isinstance(o, dt.datetime):
            return_str = o.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(o, frm.Farm) or isinstance(o, Config) or isinstance(o, FarmConfig):
            return_str = ""
        else:
            #return_str = json.dumps(o)
            #return_str = {'__{}__'.format(o.__class__.__name__): o.__dict__}
            return_str = ""

        return return_str
