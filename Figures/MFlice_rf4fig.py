import numpy as np
import pandas as pd
from scipy import stats
import datetime as dt
from math import exp, log, sqrt, ceil
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

dtypesl = {'Company':'str', 'Farm':'str', 'Adult_female_lice_Average_per_fish':'str', 'date':'str' ,'farm':'int'}

titles = ['date', 'tm', 'count_mu', 'count_sd']
dtypes = {'date':'str' , 'tm':'str', 'count_mu':'float', 'count_sd':'float'}
parse_dates = ['date']



counts_df = pd.read_csv('./Data/MFlice_fyne.csv', header=0,dtype = dtypesl) 
counts_df.Adult_female_lice_Average_per_fish = pd.to_numeric(counts_df.Adult_female_lice_Average_per_fish, errors = 'coerce')
counts_df.date = [dt.datetime.strptime(i, '%d/%m/%Y') for i in counts_df.date]

Lice_counts_one = pd.read_csv('/exports/cmvm/eddie/eb/groups/EnrightNetworksEpi/slm/outputs/res_func4/extFuni2_2/lice_counts1.txt', header=None, sep=' ', names=titles,dtype = dtypes, parse_dates = parse_dates)


fig_counts, ax = plt.subplots()
date_form = DateFormatter("%m-%Y")
ax.xaxis.set_major_formatter(date_form)
ax.set_xlim(dt.date(2017,9,1),dt.date(2019,5,2))
#ax.set_ylim(0,0.03)
ax.plot(counts_df.groupby('date').Adult_female_lice_Average_per_fish.mean(),'cx-')
ax.plot(counts_df.date,counts_df.Adult_female_lice_Average_per_fish,'c.')
ax.plot(Lice_counts_one.groupby('date').count_mu.mean(),'mx-')
ax.plot(Lice_counts_one.date,Lice_counts_one.count_mu,'m.')
plt.xlabel('Date')
plt.ylabel('Female adults per fish')
plt.savefig('/exports/cmvm/eddie/eb/groups/EnrightNetworksEpi/slm/Figures/counts_MFlicerf4.png')


