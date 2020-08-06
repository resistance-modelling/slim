import numpy as np
import pandas as pd
from scipy import stats
import datetime as dt
from math import exp, log, sqrt, ceil
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

locations = ['res_func4/extFuni2_2','res_func4/extFuni3','res_func4/extFuni4']
end_date = dt.datetime(2019,8,2)
figtitle = 'group3rf4'

titles = ['date', 'tm', 'count_mu', 'count_sd']
dtypes = {'date':'str' , 'tm':'str', 'count_mu':'float', 'count_sd':'float'}
dtypesbv = {'cur_date':'str' ,'muEMB':'float', 'sigEMB':'float', 'prop_ext':'float'}
parse_dates = ['date']
parse_datesbv = ['cur_date']

one = []
two = []
three = []

Lice_counts_one = []
Lice_counts_two = []
Lice_counts_three = []


one.extend([pd.read_csv('/exports/cmvm/eddie/eb/groups/EnrightNetworksEpi/slm/outputs/'+locations[0]+'/resistanceBVs1.csv', header=0,dtype = dtypesbv, parse_dates = parse_datesbv)])
Lice_counts_one.extend([pd.read_csv('/exports/cmvm/eddie/eb/groups/EnrightNetworksEpi/slm/outputs/'+locations[0]+'/lice_counts1.txt', header=None, sep=' ', names=titles,dtype = dtypes, parse_dates = parse_dates)])

two.extend([pd.read_csv('/exports/cmvm/eddie/eb/groups/EnrightNetworksEpi/slm/outputs/'+locations[1]+'/resistanceBVs1.csv', header=0,dtype = dtypesbv, parse_dates = parse_datesbv)])
Lice_counts_two.extend([pd.read_csv('/exports/cmvm/eddie/eb/groups/EnrightNetworksEpi/slm/outputs/'+locations[1]+'/lice_counts1.txt', header=None, sep=' ', names=titles,dtype = dtypes, parse_dates = parse_dates)])

three.extend([pd.read_csv('/exports/cmvm/eddie/eb/groups/EnrightNetworksEpi/slm/outputs/'+locations[2]+'/resistanceBVs1.csv', header=0,dtype = dtypesbv, parse_dates = parse_datesbv)])
Lice_counts_three.extend([pd.read_csv('/exports/cmvm/eddie/eb/groups/EnrightNetworksEpi/slm/outputs/'+locations[2]+'/lice_counts1.txt', header=None, sep=' ', names=titles,dtype = dtypes, parse_dates = parse_dates)])

one_sum = one[0].muEMB
two_sum = two[0].muEMB
three_sum = three[0].muEMB

Lice_counts_one_sum = Lice_counts_one[0].count_mu
Lice_counts_two_sum = Lice_counts_two[0].count_mu
Lice_counts_three_sum = Lice_counts_three[0].count_mu

for i in range(1, 5):
    one.extend([pd.read_csv('/exports/cmvm/eddie/eb/groups/EnrightNetworksEpi/slm/outputs/'+locations[0]+'/resistanceBVs' + str(i+1) + '.csv', header=0,dtype = dtypesbv, parse_dates = parse_datesbv)])
    Lice_counts_one.extend([pd.read_csv('/exports/cmvm/eddie/eb/groups/EnrightNetworksEpi/slm/outputs/'+locations[0]+'/lice_counts' + str(i+1) + '.txt', header=None, sep=' ', names=titles,dtype = dtypes, parse_dates = parse_dates)])
    
    two.extend([pd.read_csv('/exports/cmvm/eddie/eb/groups/EnrightNetworksEpi/slm/outputs/'+locations[1]+'/resistanceBVs' + str(i+1) + '.csv', header=0,dtype = dtypesbv, parse_dates = parse_datesbv)])
    Lice_counts_two.extend([pd.read_csv('/exports/cmvm/eddie/eb/groups/EnrightNetworksEpi/slm/outputs/'+locations[1]+'/lice_counts' + str(i+1) + '.txt', header=None, sep=' ', names=titles,dtype = dtypes, parse_dates = parse_dates)])
    
    three.extend([pd.read_csv('/exports/cmvm/eddie/eb/groups/EnrightNetworksEpi/slm/outputs/'+locations[2]+'/resistanceBVs' + str(i+1) + '.csv', header=0,dtype = dtypesbv, parse_dates = parse_datesbv)])
    Lice_counts_three.extend([pd.read_csv('/exports/cmvm/eddie/eb/groups/EnrightNetworksEpi/slm/outputs/'+locations[2]+'/lice_counts' + str(i+1) + '.txt', header=None, sep=' ', names=titles,dtype = dtypes, parse_dates = parse_dates)])
    
    one_sum = one_sum + one[i].muEMB
    two_sum = two_sum + two[i].muEMB
    three_sum = three_sum + three[i].muEMB
    
    Lice_counts_one_sum = Lice_counts_one_sum + Lice_counts_one[i].count_mu
    Lice_counts_two_sum = Lice_counts_two_sum + Lice_counts_two[i].count_mu
    Lice_counts_three_sum = Lice_counts_three_sum + Lice_counts_three[i].count_mu
    
bv_df = pd.DataFrame(columns=['Date','one', 'two', 'three']) 
counts_df = pd.DataFrame(columns=['Date','one', 'two', 'three']) 

bv_df['Date'] = one[0].cur_date
bv_df['one'] = one_sum/5
bv_df['two'] = two_sum/5
bv_df['three'] = three_sum/5


bv_df = bv_df[bv_df.Date < end_date]

fig_bv, ax = plt.subplots()
date_form = DateFormatter("%m-%Y")
ax.xaxis.set_major_formatter(date_form)
ax.set_xlim(dt.date(2017,9,1),end_date.date())
ax.set_ylim(3.4,5.0)
ax.plot(bv_df.groupby('Date').one.mean(),'cx-')
ax.plot(bv_df.Date,bv_df.one,'c.')
ax.plot(bv_df.groupby('Date').two.mean(),'rx-')
ax.plot(bv_df.Date,bv_df.two,'r.')
ax.plot(bv_df.groupby('Date').three.mean(),'mx-')
ax.plot(bv_df.Date,bv_df.three,'m.')
plt.xlabel('Date')
plt.ylabel('Mean underlying resistance')
plt.savefig('/exports/cmvm/eddie/eb/groups/EnrightNetworksEpi/slm/Figures/bv_'+figtitle+'.png')

counts_df['Date'] = one[0].cur_date
counts_df['one'] = Lice_counts_one_sum/5
counts_df['two'] = Lice_counts_two_sum/5
counts_df['three'] = Lice_counts_three_sum/5

fig_counts, ax = plt.subplots()
date_form = DateFormatter("%m-%Y")
ax.xaxis.set_major_formatter(date_form)
ax.set_xlim(dt.date(2017,9,1),end_date.date())
ax.plot(counts_df.groupby('Date').one.mean(),'cx-')
ax.plot(counts_df.Date,counts_df.one,'c.')
ax.plot(counts_df.groupby('Date').two.mean(),'rx-')
ax.plot(counts_df.Date,counts_df.two,'r.')
ax.plot(counts_df.groupby('Date').three.mean(),'mx-')
ax.plot(counts_df.Date,counts_df.three,'m.')
plt.xlabel('Date')
plt.ylabel('Female adults per fish')
plt.savefig('/exports/cmvm/eddie/eb/groups/EnrightNetworksEpi/slm/Figures/counts_'+figtitle+'.png')

