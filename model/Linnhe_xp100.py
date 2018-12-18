# #!/usr/bin/python
#

import numpy as np
import pandas as pd
import datetime as dt

def temp_f(c_month,farm_N):
    degs = (oban_avetemp[c_month-1]-fw_avetemp[c_month-1])/(774181-729693)
    Ndiff = farm_N - 729693 #farm Northing - oban Northing
    return round(oban_avetemp[c_month-1] - Ndiff*degs, 1)

nfarms = 10
ncages = np.array([20,24,12,8,10,14,14,9,9,10])
fishf = [50000,50000,50000,90000,60000,60000,60000,60000,60000,60000]
ext_pressure = 100 #planktonic lice per day per cage/farm arriving from wildlife -> seasonal?

oban_avetemp = np.array([8.2,7.5,7.4,8.2,9.6,11.3,13.1,13.7,13.6,12.8,11.7,9.8]) #www.seatemperature.org
fw_avetemp = np.array([8,7.4,7.1,8,9.3,11.1,12.9,13.5,13.3,12.5,11.3,9.6])

xy_array = np.array([[206100,770600],[201300,764700],[207900,759500],[185200,752500],
                     [192500,749700],[186500,745300],[193400,741700],[184700,740000],
                     [186700,734100],[183500,730700]]) #Loch Linnhe
					 
prob_arrive = pd.read_csv('./Data/Linnhenetwork.csv',header=None)

start_date = dt.datetime(2015, 11, 1)
end_date = dt.datetime(2016, 7, 3)#dt.datetime(2017, 9, 1)

file_path = "./outputs/Linnhe/outputs" + dt.datetime.today().strftime('%Y%m%d') + '/ext100/' ####################

NSbool_str = '(farm>3) | (cur_date>dt.datetime(2016, 1, 1))'