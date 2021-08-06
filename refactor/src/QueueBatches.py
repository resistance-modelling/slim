"""
Cages and Farms communicate via a mix of traditional API and message passing via channels. This allows
for greater flexibility when there is a need to communicate back and forth. Furthermore it allows for greater
multithreading capabilities.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from src.LicePopulation import GenoDistrib
from src.TreatmentTypes import Treatment

@dataclass(order=True)
class EggBatch:
    hatch_date: dt.datetime
    geno_distrib: GenoDistrib = field(compare=False)


@dataclass(order=True)
class TravellingEggBatch:
    arrival_date: dt.datetime
    hatch_date: dt.datetime = field(compare=False)
    geno_distrib: GenoDistrib = field(compare=False)


@dataclass(order=True)
class DamAvailabilityBatch:
    availability_date: dt.datetime  # expected return time
    geno_distrib: GenoDistrib = field(compare=False)


@dataclass(repr=True)
class TreatmentEvent:
    # date at which the effects are noticeable = application date + delay
    affecting_date: dt.datetime
    treatment_type: Treatment
    effectiveness_duration_days: int

    def __lt__(self, other: TreatmentEvent):
        # in the rare case two treatments are applied in a row it would be better to prefer longer treatments.
        # Apparently there is no way to force a reverse lexicographical order for some fields with a @dataclass
        return self.affecting_date < other.affecting_date or \
               (self.affecting_date == other.affecting_date and
                self.effectiveness_duration_days > other.effectiveness_duration_days)
