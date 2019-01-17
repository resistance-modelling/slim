# #!/usr/bin/python
#

import numpy as np
import pandas as pd
import datetime as dt

def temp_f(c_month,farm_N):
    degs = (tarbert_avetemp[c_month-1]-ardrishaig_avetemp[c_month-1])/(685715-665300)
    Ndiff = farm_N - 665300 #farm Northing - tarbert Northing
    return round(tarbert_avetemp[c_month-1] - Ndiff*degs, 1)

nfarms = 2
ncages = np.array([4,3])
fishf = [40000,60000]
ext_pressure = 500 #planktonic lice per day per cage/farm arriving from wildlife -> seasonal?

ardrishaig_avetemp = np.array([8.2,7.55,7.45,8.25,9.65,11.35,13.15,13.75,13.65,12.85,11.75,9.85]) #www.seatemperature.org
tarbert_avetemp = np.array([8.4,7.8,7.7,8.6,9.8,11.65,13.4,13.9,13.9,13.2,12.15,10.2])

xy_array = np.array([[190300,665300],[192500,668200]])
#1. Tarbert, 2. Rubha Stillaig, 3. Glenan Bay, 4. Meall Mhor, 5. Gob a Bharra, 6. Strondoir Bay, 7. Ardgadden, 8. Ardcastle Bay, 9. Quarry Point   
					 
prob_arrive = pd.read_csv('M:/slm/Data/Fynenetwork.csv',header=None)
prob_arrive = prob_arrive.iloc[:2,:2]

start_date = dt.datetime(2015, 10, 1)
end_date = dt.datetime(2016, 7, 3)#dt.datetime(2017, 9, 1)

file_path = "M:/slm/outputs/Fyne/outputs" + dt.datetime.today().strftime('%Y%m%d') + '/tst/' ####################

NSbool_str = 'True'