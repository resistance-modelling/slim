"""Some global actors to be available across all modules"""
from typing import Tuple

import ray
from ray.util import ActorPool
from src import create_logger
from src.Config import Config


class LoggingActor:
    """A shim of an actor that sets up a logging instance.
    Note: subclasses must still use the @ray.remote decorator!"""

    def __init__(self, log_level):
        create_logger(log_level)

def setup_workers(farm_type, cfg: Config) -> ActorPool:
    """
    Create workers.
    :return Farm actor pool
    """
    cage_pool = ActorPool([farm_type.remote(cfg.log_level) for _ in range(cfg.num_workers)])
    farm_pool = ActorPool([farm_type.remote(cfg.log_level, cage_pool) for _ in range(cfg.num_workers)])

    return farm_pool
