# #!/usr/bin/python
#

import numpy as np
import pandas as pd
import datetime as dt

def temp_f(c_month,farm_N):
    degs = (tarbert_avetemp[c_month-1]-ardrishaig_avetemp[c_month-1])/(685715-665300)
    Ndiff = farm_N - 665300 #farm Northing - tarbert Northing
    return round(tarbert_avetemp[c_month-1] - Ndiff*degs, 1)

nfarms = 9
ncages = np.array([14,10,14,12,10,10,14,9,9])
fishf = [60000,60000,60000,60000,90000,90000,90000,90000,90000]
ext_pressure = 100 #planktonic lice per day per cage/farm arriving from wildlife -> seasonal?

ardrishaig_avetemp = np.array([8.2,7.55,7.45,8.25,9.65,11.35,13.15,13.75,13.65,12.85,11.75,9.85]) #www.seatemperature.org
tarbert_avetemp = np.array([8.4,7.8,7.7,8.6,9.8,11.65,13.4,13.9,13.9,13.2,12.15,10.2])

xy_array = np.array([[190300,665300],[192500,668200],[191800,669500],[186500,674500],
    [190400,676800],[186300,679600],[190800,681000],[195300,692200],[199800,698000]])
					 
prob_arrive = pd.read_csv('./Data/Fynenetwork.csv',header=None)

start_date = dt.datetime(2015, 10, 1)
end_date = dt.datetime(2016, 4, 3)#dt.datetime(2017, 9, 1)

file_path = "./outputs/Fyne/outputs" + dt.datetime.today().strftime('%Y%m%d') + '/ext100_2/' ####################

NSbool_str = 'True'