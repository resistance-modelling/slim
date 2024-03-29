"""
Cages, Farms and Organisations communicate via a mix of traditional API and message passing via channels.
Here are some types and tools used to handle such queues.
"""
from __future__ import annotations

import abc
import datetime as dt
from dataclasses import dataclass, field, asdict
from functools import singledispatch
from heapq import heappush, heappop
from typing import Callable, TypeVar, TYPE_CHECKING, Generic


# from queue import PriorityQueue

if TYPE_CHECKING:
    from slim.simulation.lice_population import GenoDistrib
    from slim.types.policies import ObservationSpace
    from slim.types.treatments import Treatment


class Event:
    """An empty parent class for all the events with a default serialiser."""

    def to_json_dict(self):
        dct = asdict(self)

        for k in dct:
            v = dct[k]
            if hasattr(v, "to_json_dict"):
                dct[k] = v.to_json_dict()
        return dct


class CageEvent(abc.ABC, Event):
    @property
    @abc.abstractmethod
    def event_time(self):
        pass


@dataclass(order=True)
class EggBatch(CageEvent):
    hatch_date: dt.datetime
    geno_distrib: GenoDistrib = field(compare=False)

    @property
    def event_time(self):
        # TODO: in theory everything in here is an Event. The problem is that "event_time" is a poor name here.
        # All classes in here are a total mess, but I am too lazy to rewrite that.
        return self.hatch_date


@dataclass(order=True)
class TravellingEggBatch(CageEvent):
    arrival_date: dt.datetime
    hatch_date: dt.datetime = field(compare=False)
    geno_distrib: GenoDistrib = field(compare=False)

    @property
    def event_time(self):
        return self.hatch_date


@dataclass(order=True)
class DamAvailabilityBatch(CageEvent):
    availability_date: dt.datetime  # expected return time
    geno_distrib: GenoDistrib = field(compare=False)

    @property
    def event_time(self):
        return self.availability_date


@dataclass
class TreatmentEvent(CageEvent):
    # date at which the effects are noticeable = application date + delay
    affecting_date: dt.datetime
    # type of treatment
    treatment_type: Treatment
    # Effectiveness duration
    effectiveness_duration_days: int
    # date at which the treatment is applied for the first time. Note that first_application_date < affecting_date
    first_application_date: dt.datetime
    # date at which the treatment is no longer applied
    end_application_date: dt.datetime

    def __lt__(self, other: TreatmentEvent):
        # in the rare case two treatments are applied in a row it would be better to prefer longer treatments.
        # Apparently there is no way to force a reverse lexicographical order for some fields with a @dataclass
        return self.first_application_date < other.first_application_date or (
            self.first_application_date == other.first_application_date
            and self.effectiveness_duration_days > other.effectiveness_duration_days
        )

    @property
    def event_time(self):
        return self.first_application_date

    @property
    def treatment_window(self):
        return self.affecting_date + dt.timedelta(days=self.effectiveness_duration_days)


@dataclass(order=True)
class SamplingEvent(Event):
    """Internal sampling event used inside farm"""

    # TODO: move inside Farm?
    sampling_date: dt.datetime


# TODO: As we start deprecating PriorityQueues we should also deprecate the datetime parameter


@dataclass
class StepResponse:
    total_offspring: GenoDistrib
    profit: float
    total_cost: float
    observation_space: ObservationSpace
    loggable: dict  # any extra to log


T = TypeVar("T")


class PriorityQueue(Generic[T]):
    """Mutex-free priority queue. Only use where thread-safety is not required!"""

    def __init__(self):
        self.queue = []

    def qsize(self):
        return len(self.queue)

    def empty(self):
        return self.qsize() == 0

    def put(self, item):
        heappush(self.queue, item)

    def get(self):
        return heappop(self.queue)


EventT = TypeVar("EventT", CageEvent, SamplingEvent)


def pop_from_queue(
    queue: PriorityQueue[EventT],
    cur_time: dt.datetime,
    continuation: Callable[[EventT], None],
):
    """
    Pops an event from a queue and call a continuation function

    :param queue: the queue to process
    :param cur_time: the current time to compare the events with. Only events past cur_time will be popped
    :param continuation: the function to call for each event in the queue
    """

    @singledispatch
    def access_time_lt(_peek_element, _cur_time: dt.datetime):
        pass  # pragma: no cover

    @access_time_lt.register
    def _(arg: CageEvent, _cur_time: dt.datetime):
        return arg.event_time <= _cur_time

    @access_time_lt.register
    def _(arg: TravellingEggBatch, _cur_time: dt.datetime):
        return arg.arrival_date <= _cur_time

    # have mypy ignore redefinitions of '_'
    # see https://github.com/python/mypy/issues/2904 for details
    @access_time_lt.register  # type: ignore[no-redef]
    def _(arg: SamplingEvent, _cur_time: dt.datetime):
        return arg.sampling_date <= _cur_time  # pragma: no cover

    while not queue.empty() and access_time_lt(queue.queue[0], cur_time):
        event = queue.get()
        continuation(event)
