import datetime as dt
from dataclasses import dataclass, field


@dataclass(order=True)
class EggBatch:
    hatch_date: dt.datetime
    geno_distrib: dict = field(compare=False)


@dataclass(order=True)
class TravellingEggBatch:
    arrival_date: dt.datetime
    hatch_date: dt.datetime = field(compare=False)
    geno_distrib: dict = field(compare=False)


@dataclass(order=True)
class DamAvailabilityBatch:
    availability_date: dt.datetime # expected return time
    geno_distrib: dict = field(compare=False)
