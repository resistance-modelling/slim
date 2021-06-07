"""
Defines the configurable items for the simulation.
"""
import datetime as dt

import dateutil.relativedelta as dtr
import numpy as np
import pandas as pd



# The logger will be set up later
logger = None


# Farm and cage configuration
nfarms: int = 10
ncages: np.array = np.array([1, 6, 4, 8, 12, 9, 9, 8, 9, 9])
fishf: np.array = [40000, 40000, 40000, 40000, 40000, 40000, 40000, 40000, 40000]
ext_pressure: int = 150              # planktonic lice per day per cage/farm arriving from wildlife -> seasonal?
prop_influx: float = 0.33
f_muEMB: float = 3.5                 # mean of ???
f_sigEMB: float = 0.7                # standard deviation of ???

# Simulation start and end dates, time interval
start_date: dt.datetime = dt.datetime(2017, 8, 1)
end_date: dt.datetime = dt.datetime(2019, 8, 1)
tau: float = 1







# ???
eggs: int = 1200  # 3 broods #[50,50,40,40,50,60,80,80,80,80,70,50]
mort_rates: np.array = np.array([0.17, 0.22, 0.008, 0.05, 0.02, 0.06]) # L1,L2,L3,L4,L5f,L5m
d_hatching: np.array = [9, 10, 11, 9, 8, 6, 4, 4, 4, 4, 5, 7]

EMBmort: int = 0.9


# www.seatemperature.org
Ardrishaig_AveTemp: np.array = np.array([8.2, 7.55, 7.45, 8.25, 9.65, 11.35, 13.15, 13.75, 13.65,
                                         12.85, 11.75, 9.85])
Tarbert_AveTemp: np.array = np.array([8.4, 7.8, 7.7, 8.6, 9.8, 11.65, 13.4, 13.9, 13.9, 13.2,
                                      12.15, 10.2])

xy_array: np.array = np.array([[190300, 665300], [192500, 668200], [191800, 669500],
                               [186500, 674500], [190400, 676800], [186300, 679600],
                               [190800, 681000], [195300, 692200], [199800, 698000]])


# "roughly" based on marine scotland fixed engine catch data for west of scotland from 2000-2018
Enfish_res: np.array = [3] * 12  # [4,5,6,6,8,15,37,22,10,2,2,3]
Ewt_res: int = 2500


cpw: np.array = [1, 1, 1, 2, 3, 1, 1, 2, 1, 1]
numwk: np.array = [1, 6, 4, 4, 4, 9, 9, 4, 9, 9]
farm_start: np.array = [dt.datetime(2017, 8, 1), dt.datetime(2017, 10, 1),
                        dt.datetime(2017, 9, 1), dt.datetime(2017, 10, 1),
                        dt.datetime(2017, 10, 1), dt.datetime(2017, 10, 1),
                        dt.datetime(2017, 8, 1), dt.datetime(2017, 9, 1),
                        dt.datetime(2017, 10, 1), dt.datetime(2017, 9, 1)]
cage_start: np.array = [[farm_start[i] + dtr.relativedelta(weeks=j) for j in range(numwk[i])]\
                        * cpw[i] for i in range(nfarms)]



#prop_arrive: pd.DataFrame = pd.read_csv('./Data/Fyne_props.csv', header=None)
#hrs_travel: pd.DataFrame = pd.read_csv('./Data/Fyne_Edays.csv', header=None)


"""
bool_treat = '(cur_date - dt.timedelta(days=14)) in inpt.dates_list[farm-1]'

NSbool_str = 'cur_date>=inpt.cage_start[farm][cage-1]'

dates_list = [[] for i in range(nfarms-1)]
dates_list[0].extend(pd.date_range(dt.datetime(2017, 12, 1), dt.datetime(2018, 1, 1)).tolist())
dates_list[0].extend(pd.date_range(dt.datetime(2018, 2, 1), dt.datetime(2018, 3, 1)).tolist())
dates_list[0].extend(pd.date_range(dt.datetime(2018, 5, 1), dt.datetime(2018, 6, 1)).tolist())
dates_list[1].extend(pd.date_range(dt.datetime(2017, 10, 1), dt.datetime(2018, 1, 1)).tolist())
dates_list[1].extend(pd.date_range(dt.datetime(2018, 2, 1), dt.datetime(2018, 3, 1)).tolist())
dates_list[1].extend(pd.date_range(dt.datetime(2018, 5, 1), dt.datetime(2018, 6, 1)).tolist())
dates_list[2].extend(pd.date_range(dt.datetime(2017, 12, 1), dt.datetime(2018, 1, 1)).tolist())
dates_list[2].extend(pd.date_range(dt.datetime(2018, 2, 1), dt.datetime(2018, 3, 1)).tolist())
dates_list[2].extend(pd.date_range(dt.datetime(2018, 5, 1), dt.datetime(2018, 6, 1)).tolist())
dates_list[3].extend(pd.date_range(dt.datetime(2017, 10, 1), dt.datetime(2017, 11, 1)).tolist())
dates_list[3].extend(pd.date_range(dt.datetime(2017, 12, 1), dt.datetime(2018, 1, 1)).tolist())
dates_list[3].extend(pd.date_range(dt.datetime(2018, 2, 1), dt.datetime(2018, 3, 1)).tolist())
dates_list[3].extend(pd.date_range(dt.datetime(2018, 5, 1), dt.datetime(2018, 6, 1)).tolist())
dates_list[4].extend(pd.date_range(dt.datetime(2017, 10, 1), dt.datetime(2018, 1, 1)).tolist())
dates_list[4].extend(pd.date_range(dt.datetime(2018, 2, 1), dt.datetime(2018, 3, 1)).tolist())
dates_list[4].extend(pd.date_range(dt.datetime(2018, 5, 1), dt.datetime(2018, 6, 1)).tolist())
dates_list[5].extend(pd.date_range(dt.datetime(2017, 10, 1), dt.datetime(2018, 1, 1)).tolist())
dates_list[5].extend(pd.date_range(dt.datetime(2018, 2, 1), dt.datetime(2018, 3, 1)).tolist())
dates_list[5].extend(pd.date_range(dt.datetime(2018, 5, 1), dt.datetime(2018, 6, 1)).tolist())
dates_list[6].extend(pd.date_range(dt.datetime(2017, 11, 1), dt.datetime(2018, 1, 1)).tolist())
dates_list[6].extend(pd.date_range(dt.datetime(2018, 2, 1), dt.datetime(2018, 3, 1)).tolist())
dates_list[6].extend(pd.date_range(dt.datetime(2018, 5, 1), dt.datetime(2018, 6, 1)).tolist())
dates_list[7].extend(pd.date_range(dt.datetime(2017, 10, 1), dt.datetime(2017, 12, 1)).tolist())
dates_list[7].extend(pd.date_range(dt.datetime(2018, 2, 1), dt.datetime(2018, 3, 1)).tolist())
dates_list[7].extend(pd.date_range(dt.datetime(2018, 5, 1), dt.datetime(2018, 6, 1)).tolist())
dates_list[8].extend(pd.date_range(dt.datetime(2017, 10, 1), dt.datetime(2017, 12, 1)).tolist())
dates_list[8].extend(pd.date_range(dt.datetime(2018, 2, 1), dt.datetime(2018, 3, 1)).tolist())
dates_list[8].extend(pd.date_range(dt.datetime(2018, 5, 1), dt.datetime(2018, 6, 1)).tolist())

"""
