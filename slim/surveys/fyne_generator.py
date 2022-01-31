# #!/usr/bin/python
#
import json

import numpy as np
import datetime as dt


nfarms = 9
ncages = [14, 10, 14, 12, 10, 10, 14, 9, 9]
fishf = [40000, 40000, 40000, 40000, 60000, 60000, 60000, 60000, 60000]
ext_pressure = (
    500  # planktonic lice per day per cage/farm arriving from wildlife -> seasonal?
)

ardrishaig_avetemp = [
    8.2,
    7.55,
    7.45,
    8.25,
    9.65,
    11.35,
    13.15,
    13.75,
    13.65,
    12.85,
    11.75,
    9.85,
]  # www.seatemperature.org
tarbert_avetemp = [8.4, 7.8, 7.7, 8.6, 9.8, 11.65, 13.4, 13.9, 13.9, 13.2, 12.15, 10.2]

xy_array = [
    [190300, 665300],
    [192500, 668200],
    [191800, 669500],
    [186500, 674500],
    [190400, 676800],
    [186300, 679600],
    [190800, 681000],
    [195300, 692200],
    [199800, 698000],
]

# 1. Tarbert, 2. Rubha Stillaig, 3. Glenan Bay, 4. Meall Mhor, 5. Gob a Bharra, 6. Strondoir Bay, 7. Ardgadden, 8. Ardcastle Bay, 9. Quarry Point

start_date = dt.datetime(2015, 10, 1)
end_date = dt.datetime(2016, 7, 3)  # dt.datetime(2017, 9, 1)
max_num_treatments = [10, 8] + [10] * 7
sampling_spacing = [7] * nfarms

opening_date = [start_date for _ in range(nfarms)]
defection_proba = [0.2] * nfarms

to_generate = []

for i in range(nfarms):
    farm_ardrishaig = ardrishaig_avetemp[i]
    farm_tarbert = tarbert_avetemp[i]
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

print(json.dumps(to_generate, indent=4))
print("")
