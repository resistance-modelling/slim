import numpy as np
import pandas as pd
import datetime as dt
from math import exp, log, sqrt, ceil
import matplotlib.pyplot as plt

lice = pd.read_csv('M:/slm/outputs/Fyne/lice_df.csv')
lice = lice.drop(lice[lice.Farm=='Farm'].index)
lice[['Farm','Cage','Fish','stage','stage_age','avail','mate_resistanceT1','resistanceT1','arrival']] \
   = lice[['Farm','Cage','Fish','stage','stage_age','avail','mate_resistanceT1','resistanceT1','arrival']].apply(pd.to_numeric)
lice = lice.drop(['Unnamed: 0'], axis=1)
lice.date = pd.to_datetime(lice.date)
lice.MF = lice.MF.astype('category')

min(lice.date)
#Out[17]: Timestamp('2015-10-02 00:00:00')

max(lice.date)
#Out[19]: Timestamp('2015-11-23 00:00:00')

nfish = [40000,60000]
ncages = np.array([4,3])

cur_date = min(lice.date)
while cur_date<=dt.datetime(2015,11,23):
    for farm in range(1,3):
        print('date=', cur_date, \
              ' nlice=', sum((lice.Farm==farm) & (lice.date==cur_date)), \
              ' femaleAL=', sum((lice.Farm==farm) & (lice.date==cur_date) & (lice.MF=='F') & (lice.stage==5)), \
              ' nfish=', nfish[farm-1], \
              ' prop=', sum((lice.Farm==farm) & (lice.date==cur_date) & (lice.MF=='F') & (lice.stage==5))/(nfish[farm-1]*ncages[farm-1]))
    cur_date = cur_date + dt.timedelta(days=1)