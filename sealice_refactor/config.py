"""
Defines the configurable items for the simulation.
"""
import datetime as dt

import dateutil.relativedelta as dtr
import numpy as np
import pandas as pd



# The logger will be set up later
logger = None


# Simulation start and end dates, time interval
start_date: dt.datetime = dt.datetime(2017, 10, 1)
end_date: dt.datetime = dt.datetime(2017, 10, 2)
#start_date: dt.datetime = dt.datetime(2017, 8, 1)
#end_date: dt.datetime = dt.datetime(2019, 8, 1)
tau: float = 1


# Farm and cage configuration,
# For testing and development only
nfarms: int = 1
ncages: np.array = np.array([6])
fishf: np.array = [40000]
farm_locations: np.array = np.array([[190300, 665300]])
farm_start = [dt.datetime(2017, 10, 1)]
cages_start =  [[dt.datetime(2017, 10, 1, 0, 0), dt.datetime(2017, 10, 8, 0, 0), dt.datetime(2017, 10, 15, 0, 0), dt.datetime(2017, 10, 22, 0, 0), dt.datetime(2017, 10, 29, 0, 0), dt.datetime(2017, 11, 5, 0, 0)]]
treatment_dates =  [[] for i in range(nfarms)]
treatment_dates[0].extend(pd.date_range(dt.datetime(2017, 9, 15), dt.datetime(2017, 9, 22)).tolist())
treatment_dates[0].extend(pd.date_range(dt.datetime(2017, 12, 1), dt.datetime(2018, 1, 1)).tolist())
treatment_dates[0].extend(pd.date_range(dt.datetime(2018, 2, 1), dt.datetime(2018, 3, 1)).tolist())
treatment_dates[0].extend(pd.date_range(dt.datetime(2018, 5, 1), dt.datetime(2018, 6, 1)).tolist())




"""
nfarms: int = 9
ncages: np.array = np.array([6, 4, 8, 12, 9, 9, 8, 9, 9])
fishf: np.array = [40000, 40000, 40000, 40000, 40000, 40000, 40000, 40000, 40000]
farm_locations: np.array = np.array([[190300, 665300], [192500, 668200], [191800, 669500],
                               [186500, 674500], [190400, 676800], [186300, 679600],
                               [190800, 681000], [195300, 692200], [199800, 698000]])
farm_start = [dt.datetime(2017, 10, 1), dt.datetime(2017, 9, 1), dt.datetime(2017, 10, 1), dt.datetime(2017, 10, 1), dt.datetime(2017, 10, 1), dt.datetime(2017, 8, 1), dt.datetime(2017, 9, 1), dt.datetime(2017, 10, 1), dt.datetime(2017, 9, 1)]
cages_start =  [[dt.datetime(2017, 10, 1, 0, 0), dt.datetime(2017, 10, 8, 0, 0), dt.datetime(2017, 10, 15, 0, 0), dt.datetime(2017, 10, 22, 0, 0), dt.datetime(2017, 10, 29, 0, 0), dt.datetime(2017, 11, 5, 0, 0)], [dt.datetime(2017, 9, 1, 0, 0), dt.datetime(2017, 9, 8, 0, 0), dt.datetime(2017, 9, 15, 0, 0), dt.datetime(2017, 9, 22, 0, 0)], [dt.datetime(2017, 10, 1, 0, 0), dt.datetime(2017, 10, 8, 0, 0), dt.datetime(2017, 10, 15, 0, 0), dt.datetime(2017, 10, 22, 0, 0), dt.datetime(2017, 10, 1, 0, 0), dt.datetime(2017, 10, 8, 0, 0), dt.datetime(2017, 10, 15, 0, 0), dt.datetime(2017, 10, 22, 0, 0)], [dt.datetime(2017, 10, 1, 0, 0), dt.datetime(2017, 10, 8, 0, 0), dt.datetime(2017, 10, 15, 0, 0), dt.datetime(2017, 10, 22, 0, 0), dt.datetime(2017, 10, 1, 0, 0), dt.datetime(2017, 10, 8, 0, 0), dt.datetime(2017, 10, 15, 0, 0), dt.datetime(2017, 10, 22, 0, 0), dt.datetime(2017, 10, 1, 0, 0), dt.datetime(2017, 10, 8, 0, 0), dt.datetime(2017, 10, 15, 0, 0), dt.datetime(2017, 10, 22, 0, 0)], [dt.datetime(2017, 10, 1, 0, 0), dt.datetime(2017, 10, 8, 0, 0), dt.datetime(2017, 10, 15, 0, 0), dt.datetime(2017, 10, 22, 0, 0), dt.datetime(2017, 10, 29, 0, 0), dt.datetime(2017, 11, 5, 0, 0), dt.datetime(2017, 11, 12, 0, 0), dt.datetime(2017, 11, 19, 0, 0), dt.datetime(2017, 11, 26, 0, 0)], [dt.datetime(2017, 8, 1, 0, 0), dt.datetime(2017, 8, 8, 0, 0), dt.datetime(2017, 8, 15, 0, 0), dt.datetime(2017, 8, 22, 0, 0), dt.datetime(2017, 8, 29, 0, 0), dt.datetime(2017, 9, 5, 0, 0), dt.datetime(2017, 9, 12, 0, 0), dt.datetime(2017, 9, 19, 0, 0), dt.datetime(2017, 9, 26, 0, 0)], [dt.datetime(2017, 9, 1, 0, 0), dt.datetime(2017, 9, 8, 0, 0), dt.datetime(2017, 9, 15, 0, 0), dt.datetime(2017, 9, 22, 0, 0), dt.datetime(2017, 9, 1, 0, 0), dt.datetime(2017, 9, 8, 0, 0), dt.datetime(2017, 9, 15, 0, 0), dt.datetime(2017, 9, 22, 0, 0)], [dt.datetime(2017, 10, 1, 0, 0), dt.datetime(2017, 10, 8, 0, 0), dt.datetime(2017, 10, 15, 0, 0), dt.datetime(2017, 10, 22, 0, 0), dt.datetime(2017, 10, 29, 0, 0), dt.datetime(2017, 11, 5, 0, 0), dt.datetime(2017, 11, 12, 0, 0), dt.datetime(2017, 11, 19, 0, 0), dt.datetime(2017, 11, 26, 0, 0)], [dt.datetime(2017, 9, 1, 0, 0), dt.datetime(2017, 9, 8, 0, 0), dt.datetime(2017, 9, 15, 0, 0), dt.datetime(2017, 9, 22, 0, 0), dt.datetime(2017, 9, 29, 0, 0), dt.datetime(2017, 10, 6, 0, 0), dt.datetime(2017, 10, 13, 0, 0), dt.datetime(2017, 10, 20, 0, 0), dt.datetime(2017, 10, 27, 0, 0)]]

treatment_dates = [[] for i in range(nfarms)]
treatment_dates[0].extend(pd.date_range(dt.datetime(2017, 12, 1), dt.datetime(2018, 1, 1)).tolist())
treatment_dates[0].extend(pd.date_range(dt.datetime(2018, 2, 1), dt.datetime(2018, 3, 1)).tolist())
treatment_dates[0].extend(pd.date_range(dt.datetime(2018, 5, 1), dt.datetime(2018, 6, 1)).tolist())
treatment_dates[1].extend(pd.date_range(dt.datetime(2017, 10, 1), dt.datetime(2018, 1, 1)).tolist())
treatment_dates[1].extend(pd.date_range(dt.datetime(2018, 2, 1), dt.datetime(2018, 3, 1)).tolist())
treatment_dates[1].extend(pd.date_range(dt.datetime(2018, 5, 1), dt.datetime(2018, 6, 1)).tolist())
treatment_dates[2].extend(pd.date_range(dt.datetime(2017, 12, 1), dt.datetime(2018, 1, 1)).tolist())
treatment_dates[2].extend(pd.date_range(dt.datetime(2018, 2, 1), dt.datetime(2018, 3, 1)).tolist())
treatment_dates[2].extend(pd.date_range(dt.datetime(2018, 5, 1), dt.datetime(2018, 6, 1)).tolist())
treatment_dates[3].extend(pd.date_range(dt.datetime(2017, 10, 1), dt.datetime(2017, 11, 1)).tolist())
treatment_dates[3].extend(pd.date_range(dt.datetime(2017, 12, 1), dt.datetime(2018, 1, 1)).tolist())
treatment_dates[3].extend(pd.date_range(dt.datetime(2018, 2, 1), dt.datetime(2018, 3, 1)).tolist())
treatment_dates[3].extend(pd.date_range(dt.datetime(2018, 5, 1), dt.datetime(2018, 6, 1)).tolist())
treatment_dates[4].extend(pd.date_range(dt.datetime(2017, 10, 1), dt.datetime(2018, 1, 1)).tolist())
treatment_dates[4].extend(pd.date_range(dt.datetime(2018, 2, 1), dt.datetime(2018, 3, 1)).tolist())
treatment_dates[4].extend(pd.date_range(dt.datetime(2018, 5, 1), dt.datetime(2018, 6, 1)).tolist())
treatment_dates[5].extend(pd.date_range(dt.datetime(2017, 10, 1), dt.datetime(2018, 1, 1)).tolist())
treatment_dates[5].extend(pd.date_range(dt.datetime(2018, 2, 1), dt.datetime(2018, 3, 1)).tolist())
treatment_dates[5].extend(pd.date_range(dt.datetime(2018, 5, 1), dt.datetime(2018, 6, 1)).tolist())
treatment_dates[6].extend(pd.date_range(dt.datetime(2017, 11, 1), dt.datetime(2018, 1, 1)).tolist())
treatment_dates[6].extend(pd.date_range(dt.datetime(2018, 2, 1), dt.datetime(2018, 3, 1)).tolist())
treatment_dates[6].extend(pd.date_range(dt.datetime(2018, 5, 1), dt.datetime(2018, 6, 1)).tolist())
treatment_dates[7].extend(pd.date_range(dt.datetime(2017, 10, 1), dt.datetime(2017, 12, 1)).tolist())
treatment_dates[7].extend(pd.date_range(dt.datetime(2018, 2, 1), dt.datetime(2018, 3, 1)).tolist())
treatment_dates[7].extend(pd.date_range(dt.datetime(2018, 5, 1), dt.datetime(2018, 6, 1)).tolist())
treatment_dates[8].extend(pd.date_range(dt.datetime(2017, 10, 1), dt.datetime(2017, 12, 1)).tolist())
treatment_dates[8].extend(pd.date_range(dt.datetime(2018, 2, 1), dt.datetime(2018, 3, 1)).tolist())
treatment_dates[8].extend(pd.date_range(dt.datetime(2018, 5, 1), dt.datetime(2018, 6, 1)).tolist())

"""

ext_pressure = 150 #planktonic lice per day per cage/farm arriving from wildlife -> seasonal?
start_pop = 150*np.sum(ncages)


# Treatment variables
f_meanEMB: float = 3.5               # mean of Emamectin Benzoate 
f_sigEMB: float = 0.7                # standard deviation of Emamectin Benzoate 
EMBmort = 0.9                        # probability of mortality due to Emamectin Benzoate
env_meanEMB = 0.0                    # mean Emamectin Benzoate in the environment
env_sigEMB = 1.0                     # standard deviation of Emamectin Benzoate in the environment

prop_influx: float = 0.33







# ???
eggs: int = 1200  # 3 broods #[50,50,40,40,50,60,80,80,80,80,70,50]
d_hatching: np.array = [9, 10, 11, 9, 8, 6, 4, 4, 4, 4, 5, 7]


# www.seatemperature.org
Ardrishaig_AveTemp: np.array = np.array([8.2, 7.55, 7.45, 8.25, 9.65, 11.35, 13.15, 13.75, 13.65,
                                         12.85, 11.75, 9.85])
Tarbert_AveTemp: np.array = np.array([8.4, 7.8, 7.7, 8.6, 9.8, 11.65, 13.4, 13.9, 13.9, 13.2,
                                      12.15, 10.2])


# "roughly" based on marine scotland fixed engine catch data for west of scotland from 2000-2018
Enfish_res: np.array = [3] * 12  # [4,5,6,6,8,15,37,22,10,2,2,3]
Ewt_res: int = 2500


cpw: np.array = [1, 1, 1, 2, 3, 1, 1, 2, 1, 1]
numwk: np.array = [1, 6, 4, 4, 4, 9, 9, 4, 9, 9]

#prop_arrive: pd.DataFrame = pd.read_csv('./Data/Fyne_props.csv', header=None)
#hrs_travel: pd.DataFrame = pd.read_csv('./Data/Fyne_Edays.csv', header=None)


"""
bool_treat = '(cur_date - dt.timedelta(days=14)) in inpt.dates_list[farm-1]'

NSbool_str = 'cur_date>=inpt.cage_start[farm][cage-1]'

"""
