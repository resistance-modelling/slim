import numpy as np
import pandas as pd
import datetime as dt
import dateutil.relativedelta as dtr
from math import exp, log, sqrt, ceil
import matplotlib.pyplot as plt

Fyne_cm = []
for i in range(1,15):
    Fyne_cm.extend([np.loadtxt('M:/slm/Data/Fyne_TA/c_d'+str(i)+'.dat')])
    
E_days = np.zeros([9, 9])
for i in range(14):
    E_days = E_days+Fyne_cm[i]*(i+1)/sum(Fyne_cm)
    
Fyne_props = sum(Fyne_cm)

np.savetxt('M:/slm/Data/Fyne_Edays.csv', E_days, delimiter=',')
np.savetxt('M:/slm/Data/Fyne_props.csv', Fyne_props, delimiter=',')

# pdftotext -layout Data/Lice-averages-Oct-2018.pdf
#sed -i 's/^ *//g' Data/Lice-averages-Oct-2018.txt 

sep18 = pd.read_csv('M:/slm/Data/Lice-averages-Sept-2018.txt',header=1)
jan18 = pd.read_csv('M:/slm/Data/Lice-averages-Jan-2018.txt',header=0)
feb18 = pd.read_csv('M:/slm/Data/Lice-averages-Feb-2018.txt',header=1)
mar18 = pd.read_csv('M:/slm/Data/Lice-averages-Mar-2018.txt',header=1)
apr18 = pd.read_csv('M:/slm/Data/Lice-averages-Apr-2018.txt',header=1)
may18 = pd.read_csv('M:/slm/Data/Lice-averages-May-2018.txt',header=1)
jun18 = pd.read_csv('M:/slm/Data/Lice-averages-Jun-2018.txt',header=1)
jul18 = pd.read_csv('M:/slm/Data/Lice-averages-Jul-2018.txt',header=1)
aug18 = pd.read_csv('M:/slm/Data/Lice-averages-Aug-2018.txt',header=1)

oct18 = pd.read_csv('M:/slm/Data/Lice-averages-Oct-2018.txt',header=1)
nov18 = pd.read_csv('M:/slm/Data/Lice-averages-Nov-2018.txt',header=1)
dec18 = pd.read_csv('M:/slm/Data/Lice-averages-Dec-2018.txt',header=1)
jan19 = pd.read_csv('M:/slm/Data/Lice-averages-Jan-2019.txt',header=1)
feb19 = pd.read_csv('M:/slm/Data/Lice-averages-Feb-2019.txt',header=1)
mar19 = pd.read_csv('M:/slm/Data/Lice-averages-Mar-2019.txt',header=1)
apr19 = pd.read_csv('M:/slm/Data/Lice-averages-Apr-2019.txt',header=1)

Fyne_farms = ['Tarbert South', 'Rubha Stillaig', 'Glenan Bay', 'Meall Mhor', 'Gob a Bharra', 'Strondoir Bay', 'Ardgadden', 'Ardcastle Bay', 'Quarry Point']

#jan18[jan18.Farm.str.contains('[Aa]rd.*')]

jan18.columns = feb18.columns
MFlice = jan18[jan18.Farm.isin(Fyne_farms)]
MFlice['date'] = [dt.datetime(2018, 1, 1)]*9

tlst = [feb18,mar18,apr18,may18,jun18,jul18,aug18,sep18,oct18,nov18,dec18,jan19,feb19,mar19,apr19]
c = 0
for i in tlst:
    c = c + 1
    tdf = i[i.Farm.isin(Fyne_farms)]
    tdf['date'] = [dt.datetime(2018, 1, 1) + dtr.relativedelta(months=c)]*len(tdf.index)
    MFlice  = MFlice.append(tdf, ignore_index=True)

MFlice.to_csv('M:/slm/Data/MFlice_fyne.csv')

#MFlice2 = oct18[oct18.Farm.isin(Fyne_farms)]
#MFlice2['date'] = [dt.datetime(2018, 10, 1)]*9
#
#tlst = [nov18,dec18,jan19,feb19]
#c = 0
#for i in tlst:
#    c = c + 1
#    tdf = i[i.Farm.isin(Fyne_farms)]
#    tdf['date'] = [dt.datetime(2018, 10, 1) + dtr.relativedelta(months=c)]*len(tdf.index)
#    MFlice2  = MFlice2.append(tdf, ignore_index=True)
#
#MFlice2.to_csv('M:/slm/Data/MFlice2.csv')


sites_sub = [['Sgeir Dughall','Kenmore','Aird','Torridon'],['Geasgill', 'Gometra', 'Inch Kenneth','Loch Tuath'],
             ['Bring Head', 'Chalmers Hope', 'Lyrawa Bay', 'Pegal Bay','Westerbister','Fara West','South Cava','Toyness'],
             ['Drumbeg','Badcall','Calbha Bay','Loch a Chairn Bhain','Outer Bay']]

torridon_lice = jan18[jan18.Farm.isin(sites_sub[0])]
torridon_lice['date'] = [dt.datetime(2018, 1, 1)]*4

tlst = [feb18,mar18,apr18,may18,jun18,jul18,aug18,sep18,oct18,nov18,dec18,jan19,feb19]
c = 0
for i in tlst:
    c = c + 1
    tdf = i[i.Farm.isin(sites_sub[0])]
    tdf['date'] = [dt.datetime(2018, 1, 1) + dtr.relativedelta(months=c)]*len(tdf.index)
    torridon_lice  = torridon_lice.append(tdf, ignore_index=True)

torridon_lice.to_csv('M:/slm/Data/torridon_lice.csv')

tuath_lice = jan18[jan18.Farm.isin(sites_sub[1])]
tuath_lice['date'] = [dt.datetime(2018, 1, 1)]*4

tlst = [feb18,mar18,apr18,may18,jun18,jul18,aug18,sep18,oct18,nov18,dec18,jan19,feb19]
c = 0
for i in tlst:
    c = c + 1
    tdf = i[i.Farm.isin(sites_sub[1])]
    tdf['date'] = [dt.datetime(2018, 1, 1) + dtr.relativedelta(months=c)]*len(tdf.index)
    tuath_lice  = tuath_lice.append(tdf, ignore_index=True)

tuath_lice.to_csv('M:/slm/Data/tuath_lice.csv')

scapa_lice = jan18[jan18.Farm.isin(sites_sub[2])]
scapa_lice['date'] = [dt.datetime(2018, 1, 1)]*8

tlst = [feb18,mar18,apr18,may18,jun18,jul18,aug18,sep18,oct18,nov18,dec18,jan19,feb19]
c = 0
for i in tlst:
    c = c + 1
    tdf = i[i.Farm.isin(sites_sub[2])]
    tdf['date'] = [dt.datetime(2018, 1, 1) + dtr.relativedelta(months=c)]*len(tdf.index)
    scapa_lice  = scapa_lice.append(tdf, ignore_index=True)

scapa_lice.to_csv('M:/slm/Data/scapa_lice.csv')

badcall_lice = jan18[jan18.Farm.isin(sites_sub[3])]
badcall_lice['date'] = [dt.datetime(2018, 1, 1)]*5

tlst = [feb18,mar18,apr18,may18,jun18,jul18,aug18,sep18,oct18,nov18,dec18,jan19,feb19]
c = 0
for i in tlst:
    c = c + 1
    tdf = i[i.Farm.isin(sites_sub[3])]
    tdf['date'] = [dt.datetime(2018, 1, 1) + dtr.relativedelta(months=c)]*len(tdf.index)
    badcall_lice  = badcall_lice.append(tdf, ignore_index=True)

badcall_lice.to_csv('M:/slm/Data/badcall_lice.csv')


farm_mu=pd.read_csv('M:/slm/outputs/farm_mu.csv')
farm_mu=farm_mu[farm_mu.index<126]
farm_mu.columns=['blip','date','Lcount']
farm_mu['farm']=[i for n in range(int(len(farm_mu.index)/9)) for i in range(1,10)]
farm_mu['date']=pd.to_datetime(farm_mu.date)

MFlice.columns=MFlice.columns.str.replace(' ','_')
MFlice.columns=MFlice.columns.str.replace('[()]','')
MFlice['farm']=0
for i in range(1,10):
    MFlice.loc[MFlice.Farm==Fyne_farms[i-1],'farm'] = i
    print(i,Fyne_farms[i-1])
    
farm_mu = farm_mu.merge(MFlice,how='left',on=['date','farm'])
farm_mu.loc[(farm_mu.date<dt.date(2018,1,1))&(farm_mu.Lcount.notnull()),'Adult_female_lice_Average_per_fish']=0

farm_mu['Adult_female_lice_Average_per_fish'] = pd.to_numeric(farm_mu['Adult_female_lice_Average_per_fish'],errors='coerce')
farm_mu['delta_est'] = farm_mu.Lcount - farm_mu.Adult_female_lice_Average_per_fish

fig,ax=plt.subplots()
farm_mu.set_index('date', inplace=True)
farm_mu.groupby('farm')['delta_est'].plot(legend=True)