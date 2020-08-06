# #!/usr/bin/python
#

import numpy as np
import pandas as pd
import datetime as dt
import dateutil.relativedelta as dtr

def temp_f(c_month,farm_N):
    degs = (tarbert_avetemp[c_month-1]-ardrishaig_avetemp[c_month-1])/(685715-665300)
    Ndiff = farm_N - 665300 #farm Northing - tarbert Northing
    return round(tarbert_avetemp[c_month-1] - Ndiff*degs, 1)
    
def d_hatching(c_temp):
    return 3*(3.3 - 0.93*np.log(c_temp/3) -0.16*np.log(c_temp/3)**2) #for 3 broods

nfarms = 9
ncages = np.array([6,4,8,12,9,9,8,9,9])
fishf = [40000,40000,40000,40000,40000,40000,40000,40000,40000]
prop_feedback = 0.001
ext_pressure = '150 + round(inpt.prop_feedback*prevOffs_len)' #planktonic lice per day per cage/farm arriving from wildlife
start_pop = 6000

prop_influx = 0.33

eggs = 1200#3 broods #[50,50,40,40,50,60,80,80,80,80,70,50]
#d_hatching = [9,10,11,9,8,6,4,4,4,4,5,7]

ardrishaig_avetemp = np.array([8.2,7.55,7.45,8.25,9.65,11.35,13.15,13.75,13.65,12.85,11.75,9.85]) #www.seatemperature.org
tarbert_avetemp = np.array([8.4,7.8,7.7,8.6,9.8,11.65,13.4,13.9,13.9,13.2,12.15,10.2])

xy_array = np.array([[190300,665300],[192500,668200],[191800,669500],[186500,674500],
    [190400,676800],[186300,679600],[190800,681000],[195300,692200],[199800,698000]])
    
#Tarbert South, Rubha Stillaig, Glenan Bay, Meall Mhor, Gob a Bharra, Strondoir Bay, Ardgaddan, Ardcastle, Quarry Point

f_muEMB = 3.5
f_sigEMB = 0.7

dates_list = [[] for i in range(nfarms)]
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

bool_treat = '(cur_date - dt.timedelta(days=14)) in inpt.dates_list[farm-1]'
		
prob_arrive = pd.read_csv('./Data/Fyne_props.csv',header=None)
E_days = pd.read_csv('./Data/Fyne_Edays.csv',header=None)

Enfish_res = [3]*12 #[4,5,6,6,8,15,37,22,10,2,2,3] #"roughly" based on marine scotland fixed engine catch data for west of scotland from 2000-2018
Ewt_res = 2500

start_date = dt.datetime(2017, 8, 1)
end_date = dt.datetime(2019, 8, 1)

cpw = [1, 1, 2, 3, 1, 1, 2, 1, 1]
numwk = [6, 4, 4, 4, 9, 9, 4, 9, 9]
farm_start = [dt.datetime(2017, 10, 1), dt.datetime(2017, 9, 1), dt.datetime(2017, 10, 1), dt.datetime(2017, 10, 1), dt.datetime(2017, 10, 1), dt.datetime(2017, 8, 1), dt.datetime(2017, 9, 1), dt.datetime(2017, 10, 1), dt.datetime(2017, 9, 1)]
cage_start = [[farm_start[i] + dtr.relativedelta(weeks=j) for j in range(numwk[i])]*cpw[i] for i in range(nfarms)]

NSbool_str = 'cur_date>=inpt.cage_start[farm-1][cage-1]'