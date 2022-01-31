# #!/usr/bin/python
#
import json

import numpy as np
import datetime as dt


nfarms = 10
ncages = [20, 24, 12, 8, 10, 14, 14, 9, 9, 10]
fishf = [50000, 50000, 50000, 90000, 60000, 60000, 60000, 60000, 60000, 60000]
ext_pressure = (
    500  # planktonic lice per day per cage/farm arriving from wildlife -> seasonal?
)

oban_avetemp = [
    8.2,
    7.5,
    7.4,
    8.2,
    9.6,
    11.3,
    13.1,
    13.7,
    13.6,
    12.8,
    11.7,
    9.8,
]  # www.seatemperature.org
fw_avetemp = [8, 7.4, 7.1, 8, 9.3, 11.1, 12.9, 13.5, 13.3, 12.5, 11.3, 9.6]

xy_array = [
    [206100, 770600],
    [201300, 764700],
    [207900, 759500],
    [185200, 752500],
    [192500, 749700],
    [186500, 745300],
    [193400, 741700],
    [184700, 740000],
    [186700, 734100],
    [183500, 730700],
]  # Loch Linnhe

start_date = dt.datetime(2015, 10, 1)
end_date = dt.datetime(2016, 7, 3)  # dt.datetime(2017, 9, 1)
max_num_treatments = [10] * nfarms
sampling_spacing = [7] * nfarms

opening_date = [start_date for _ in range(nfarms)]
defection_proba = [0.2] * nfarms

to_generate = []

for i in range(nfarms):
    farm_ardrishaig = oban_avetemp[i]
    farm_tarbert = fw_avetemp[i]
    farm_location = xy_array[i]
    farm_opening_date = opening_date[i]
    treatment_type = "emb"
    treatment_dates = []
    num_fish = 40000
    farm_ncages = ncages[i]
    farm_max_num_treatments = max_num_treatments[i]
    farm_sampling = sampling_spacing[i]
    farm_defection_proba = defection_proba[i]

    to_generate.append(
        {
            "ncages": farm_ncages,
            "location": farm_location,
            "num_fish": num_fish,
            "start_date": str(start_date),
            "cages_start_dates": [str(start_date)] * farm_ncages,
            "treatment_type": treatment_type,
            "treatment_dates": treatment_dates,
            "max_num_treatments": farm_max_num_treatments,
            "sampling_spacing": farm_sampling,
            "defection_proba": farm_defection_proba,
        }
    )

to_dump = {
    "name": "Loch Linnhe",
    "start_date": "2017-10-01 00:00:00",
    "end_date": "2019-10-01 00:00:00",
    "ext_pressure": 150,
    "monthly_cost": "5000.00",
    "genetic_ratios": {"A": 0.001, "a": 0.099, "A,a": 0.001},
    "farms": to_generate,
}
print(json.dumps(to_dump, indent=4))
print("")
