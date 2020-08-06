import numpy as np
import pandas as pd
from scipy import stats
import datetime as dt
from math import exp, log, sqrt, ceil
import matplotlib.pyplot as plt
import re
import geopandas as gpd
from shapely.geometry import Point, Polygon
import mplleaflet

FMA_shapefile = gpd.read_file('M:/slm/Data/FMA_shapefile/FMA_OS.shp')
FMA_shapefile.Locality[['Linnhe' in str(FMA_shapefile.Locality[i]) for i in range(86)]]
#Out[20]: 
#49    Loch Linnhe south, Kerrera, Loch Etive, Crera
#50          Loch Linnhe north, Loch Leven, Loch Eil
#Name: Locality, dtype: object

FMA_shapefile.Id[['Linnhe' in str(FMA_shapefile.Locality[i]) for i in range(86)]]
#Out[21]: 
#49    336
#50    333
#Name: Id, dtype: int64

FMA_shapefile.Locality[['Fyne' in str(FMA_shapefile.Locality[i]) for i in range(86)]]
#Out[22]: 
#57    Loch Fyne
#Name: Locality, dtype: object

FMA_shapefile.Id[['Fyne' in str(FMA_shapefile.Locality[i]) for i in range(86)]]
#Out[23]: 
#57    342
#Name: Id, dtype: int64

se_mr = pd.read_csv('M:/slm/Data/se_monthly_reports.csv')

linnheN_mr = se_mr[[Point(se_mr.Easting[i],se_mr.Northing[i]).within(FMA_shapefile.loc[50,'geometry']) for i in range(len(se_mr.Easting))]]
linnheS_mr = se_mr[[Point(se_mr.Easting[i],se_mr.Northing[i]).within(FMA_shapefile.loc[49,'geometry']) for i in range(len(se_mr.Easting))]]
fyne_mr = se_mr[[Point(se_mr.Easting[i],se_mr.Northing[i]).within(FMA_shapefile.loc[57,'geometry']) for i in range(len(se_mr.Easting))]]


linnheN_mr.columns = [linnheN_mr.columns[i].replace(' ','_') for i in range(len(linnheN_mr.columns))]
linnheN_mr.columns = [re.sub('[()]', '', linnheN_mr.columns[i]) for i in range(len(linnheN_mr.columns))]

linnheS_mr.columns = [linnheS_mr.columns[i].replace(' ','_') for i in range(len(linnheS_mr.columns))]
linnheS_mr.columns = [re.sub('[()]', '', linnheS_mr.columns[i]) for i in range(len(linnheS_mr.columns))]

fyne_mr.columns = [fyne_mr.columns[i].replace(' ','_') for i in range(len(fyne_mr.columns))]
fyne_mr.columns = [re.sub('[()]', '', fyne_mr.columns[i]) for i in range(len(fyne_mr.columns))]

linnheN_mr['prop_mort'] = linnheN_mr.Mortalities_Kilograms/(1000*linnheN_mr.Actual_Biomass_on_Site_tonnes*30)
linnheN_mr['Year'] = pd.to_datetime(linnheN_mr['Year'])
linnheN_pmort_mean = linnheN_mr.groupby('Year').prop_mort.mean()

lNpm15 = linnheN_pmort_mean[linnheN_pmort_mean.index>dt.datetime(2015, 9, 1)]

plt.plot(lNpm15)
 

linnheS_mr['prop_mort'] = linnheS_mr.Mortalities_Kilograms/(1000*linnheS_mr.Actual_Biomass_on_Site_tonnes*30)
linnheS_mr['Year'] = pd.to_datetime(linnheS_mr['Year'])
linnheS_pmort_mean = linnheS_mr.groupby('Year').prop_mort.mean()

lSpm15 = linnheS_pmort_mean[linnheS_pmort_mean.index>dt.datetime(2015, 9, 1)]

plt.plot(lSpm15)

fyne_mr['prop_mort'] = fyne_mr.Mortalities_Kilograms/(1000*fyne_mr.Actual_Biomass_on_Site_tonnes*30)
fyne_mr['Year'] = pd.to_datetime(fyne_mr['Year'])
fyne_pmort_mean = fyne_mr.groupby('Year').prop_mort.mean()

fpm15 = fyne_pmort_mean[fyne_pmort_mean.index>dt.datetime(2015, 9, 1)]

plt.plot(fpm15)

ncages = np.array([14,10,14,12,10,10,14,9,9])
fishf = [60000,60000,60000,60000,90000,90000,90000,90000,90000]
farm_names = ['Tarbert South', 'Rubha Stillaig', 'Glenan Bay', 'Meall Mhor', 'Gob a Bharra', 'Strondoir Bay', 'Ardgaddan', 'Ardcastle', 'Quarry Point']
weight0 = [(1000*fyne_mr.Actual_Biomass_on_Site_tonnes[(fyne_mr.Site_Name==farm_names[i]) & (fyne_mr.Year==dt.datetime(2015, 10, 1))])/(ncages[i]*fishf[i]) for i in range(9)]
#0.038095    0.046667    0.008333    0.031944    0.026667    0.057778    0.01746    0.04321    0.034568


fish0 = [(1000*fyne_mr.Actual_Biomass_on_Site_tonnes[(fyne_mr.Site_Name==farm_names[i]) & (fyne_mr.Year==dt.datetime(2015, 10, 1))])/(ncages[i]*0.065) for i in range(9)]
#35164.835165    43076.923077    7692.307692    29487.179487    36923.076923    80000.0    24175.824176    59829.059829    47863.247863

ncages = np.array([14,10,14,12,10,10,14,9,9])
fishf = [40000,40000,40000,40000,60000,60000,60000,60000,60000]
farm_names = ['Tarbert South', 'Rubha Stillaig', 'Glenan Bay', 'Meall Mhor', 'Gob a Bharra', 'Strondoir Bay', 'Ardgaddan', 'Ardcastle', 'Quarry Point']
weight0 = [(1000*fyne_mr.Actual_Biomass_on_Site_tonnes[(fyne_mr.Site_Name==farm_names[i]) & (fyne_mr.Year==dt.datetime(2015, 10, 1))])/(ncages[i]*fishf[i]) for i in range(9)]

fyne_ebq1 = fyne_mr.loc[(fyne_mr.Year>=dt.datetime(2015, 9, 1))&(fyne_mr.Year<=dt.datetime(2016, 1, 31))&(fyne_mr.Emamectin_Benzoate_grams>0),['Year','Mortalities_Kilograms',
       'Actual_Biomass_on_Site_tonnes', 'Emamectin_Benzoate_grams',
       'Site_Name','Receiving_Water', 'Easting', 'Northing', 'Water_Type',
       'Local_Authority', 'prop_mort']]
       

fyne_ebq1['eb_prop'] = fyne_ebq1.Emamectin_Benzoate_grams/fyne_ebq1.Actual_Biomass_on_Site_tonnes
fyne_ebq1['eb_prop'] = fyne_ebq1.Emamectin_Benzoate_grams/fyne_ebq1.Feed_Kilograms

linnheS_ebq1 = linnheS_mr.loc[(linnheS_mr.Year>=dt.datetime(2015, 9, 1))&(linnheS_mr.Year<=dt.datetime(2016, 1, 31))&
        ((linnheS_mr.Emamectin_Benzoate_grams>0)|(linnheS_mr.Azamethiphos_grams>0)),['Year','Mortalities_Kilograms',
       'Actual_Biomass_on_Site_tonnes', 'Azamethiphos_grams', 'Emamectin_Benzoate_grams',
       'Site_Name','Receiving_Water', 'Easting', 'Northing', 'Water_Type',
       'Local_Authority', 'prop_mort']]
       

linnheS_ebq1['eb_prop'] = linnheS_ebq1.Emamectin_Benzoate_grams/linnheS_ebq1.Actual_Biomass_on_Site_tonnes

def fb_mort(jours):
    return (1000 + (jours - 700)**2)/490000000
nf = 30000

bm = []
for td in range(1,491):
    wt = 8000/(1+exp(-0.01*(td-475)))
    nf = nf - nf*0.00057
    tot_wt = nf*wt/10**6
    bm.extend([tot_wt])

plt.plot(bm)

dates_fyne = pd.unique(fyne_mr.Year[(fyne_mr.Year>=dt.datetime(2015, 9, 1))&(fyne_mr.Year<=dt.datetime(2017, 1, 31))])

ndays = [(dt.datetime.utcfromtimestamp((i.astype('M8[s]')).astype(int)) - dt.datetime(2015, 8, 31)).days for i in dates_fyne]

bm_farm1 = [((fyne_mr.Actual_Biomass_on_Site_tonnes[(fyne_mr.Site_Name==farm_names[0]) & (fyne_mr.Year==i)])/(ncages[0])).values for i in dates_fyne]
plt.plot(np.asarray(ndays),np.asarray(bm_farm1))

bm_farm2 = [((fyne_mr.Actual_Biomass_on_Site_tonnes[(fyne_mr.Site_Name==farm_names[1]) & (fyne_mr.Year==i)])/(ncages[1])).values for i in dates_fyne]
plt.plot(np.asarray(ndays),np.asarray(bm_farm2))

bm_farm3 = [((fyne_mr.Actual_Biomass_on_Site_tonnes[(fyne_mr.Site_Name==farm_names[2]) & (fyne_mr.Year==i)])/(ncages[2])).values for i in dates_fyne]
plt.plot(np.asarray(ndays),np.asarray(bm_farm3))

bm_farm4 = [((fyne_mr.Actual_Biomass_on_Site_tonnes[(fyne_mr.Site_Name==farm_names[3]) & (fyne_mr.Year==i)])/(ncages[3])).values for i in dates_fyne]
plt.plot(np.asarray(ndays),np.asarray(bm_farm4))


nf = 400000

bm = []
for td in range(1,491):
    wt = 4000/(1+exp(-0.015*(td-300)))
    nf = nf - nf*0.00057
    tot_wt = nf*wt/10**6
    bm.extend([tot_wt])

plt.plot(bm)

nf = 400000
bm = []
for td in range(1,491):
    wt = 4000/(1+exp(-0.015*(td-300)))
    nf = nf - nf*fb_mort(td)
    tot_wt = nf*wt/10**6
    bm.extend([tot_wt])

plt.plot(bm)

dates_fyne = pd.unique(fyne_mr.Year[(fyne_mr.Year>=dt.datetime(2015, 9, 1))&(fyne_mr.Year<=dt.datetime(2017, 1, 31))])

ndays = [(dt.datetime.utcfromtimestamp((i.astype('M8[s]')).astype(int)) - dt.datetime(2015, 8, 31)).days for i in dates_fyne]

# =============================================================================
# bm_farm5 = [((fyne_mr.Actual_Biomass_on_Site_tonnes[(fyne_mr.Site_Name==farm_names[4]) & (fyne_mr.Year==i)])/(ncages[4])).values for i in dates_fyne]
# plt.plot(np.asarray(ndays),np.asarray(bm_farm5))
# 
# bm_farm6 = [((fyne_mr.Actual_Biomass_on_Site_tonnes[(fyne_mr.Site_Name==farm_names[5]) & (fyne_mr.Year==i)])/(ncages[5])).values for i in dates_fyne]
# plt.plot(np.asarray(ndays),np.asarray(bm_farm6))
# 
# bm_farm7 = [((fyne_mr.Actual_Biomass_on_Site_tonnes[(fyne_mr.Site_Name==farm_names[6]) & (fyne_mr.Year==i)])/(ncages[6])).values for i in dates_fyne]
# plt.plot(np.asarray(ndays),np.asarray(bm_farm7))
# 
# bm_farm8 = [((fyne_mr.Actual_Biomass_on_Site_tonnes[(fyne_mr.Site_Name==farm_names[7]) & (fyne_mr.Year==i)])/(ncages[7])).values for i in dates_fyne]
# plt.plot(np.asarray(ndays),np.asarray(bm_farm8))
# 
# bm_farm9 = [((fyne_mr.Actual_Biomass_on_Site_tonnes[(fyne_mr.Site_Name==farm_names[8]) & (fyne_mr.Year==i)])/(ncages[8])).values for i in dates_fyne]
# plt.plot(np.asarray(ndays),np.asarray(bm_farm9))
# 
# bm_farm10 = [((fyne_mr.Actual_Biomass_on_Site_tonnes[(fyne_mr.Site_Name==farm_names[9]) & (fyne_mr.Year==i)])/(ncages[9])).values for i in dates_fyne]
# plt.plot(np.asarray(ndays),np.asarray(bm_farm10))
# =============================================================================

Floch_bm = fyne_mr.groupby('Year')['Actual_Biomass_on_Site_tonnes'].sum()
Floch_bm[Floch_bm<100]
fyne_mr.loc[(fyne_mr.Year>'2015-08-01')&(fyne_mr.Year<'2015-12-01'), ['Year','Site_Name','Actual_Biomass_on_Site_tonnes']]

LSloch_bm = linnheS_mr[linnheS_mr.Receiving_Water!='Loch Etive'].groupby('Year')['Actual_Biomass_on_Site_tonnes'].sum()
LSloch_bm[LSloch_bm<100]
linnheS_mr.loc[(linnheS_mr.Year>'2014-02-01')&(linnheS_mr.Year<'2014-05-01')&(linnheS_mr.Receiving_Water!='Loch Etive'), ['Year','Site_Name','Actual_Biomass_on_Site_tonnes']]
#Linnhe is just a mess!!!!

for farm in range(len(farm_names)):
    plt.plot(np.asarray(ndays),np.asarray(fyne_mr.Actual_Biomass_on_Site_tonnes[(fyne_mr.Site_Name==farm_names[farm]) & (fyne_mr.Year>=dt.datetime(2015, 9, 1)) & (fyne_mr.Year<=dt.datetime(2017, 1, 31))]))

fyne_df = fyne_mr.loc[(fyne_mr.Year>=dt.datetime(2015, 9, 1))&(fyne_mr.Year<=dt.datetime(2017, 6, 30))]
fyne_df.to_csv('M:/slm/Data/fyne_df.csv')

fyne18_df = fyne_mr.loc[(fyne_mr.Year>=dt.datetime(2017, 8, 1))&(fyne_mr.Year<=dt.datetime(2018, 9, 30))]
fyne18_df.to_csv('M:/slm/Data/fyne18_df.csv')

#Farms using EMB only
se_mr.columns = [se_mr.columns[i].replace(' ','_') for i in range(len(se_mr.columns))]
se_mr.columns = [re.sub('[()]', '', se_mr.columns[i]) for i in range(len(se_mr.columns))]
se_mr['Year'] = pd.to_datetime(se_mr['Year'])

se_mr['yr'] = [se_mr.Year[i].year for i in se_mr.index]

names = sorted(set(se_mr.Site_Name.unique()))
yrs = sorted(set(se_mr.yr.unique()))

EMBfarms = []
for y in yrs:
    df = se_mr[se_mr.yr==y]
    sublist = []
    for n in names:
        if df.loc[df.Site_Name==n,['Deltamethrin_grams', 'Cypermethrin_grams', 'Azamethiphos_grams', 'Teflubenzuron_grams']].values.sum()==0:
            sublist.extend([n])
    EMBfarms.extend([sublist])
    
EMBfarms = []
for n in names:
    if se_mr.loc[se_mr.Site_Name==n,['Deltamethrin_grams', 'Cypermethrin_grams', 'Azamethiphos_grams', 'Teflubenzuron_grams']].values.sum()==0:
        EMBfarms.extend([n])
        
lochs = sorted(set(se_mr.Receiving_Water.unique()))

EMBlochs = []
for y in yrs:
    df = se_mr[se_mr.yr==y]
    sublist = []
    for l in lochs:
        bool1 = sum(df.loc[df.Receiving_Water==l,['Deltamethrin_grams', 'Cypermethrin_grams', 'Azamethiphos_grams', 'Teflubenzuron_grams']].sum(skipna=True))==0
        bool2 = sum(df.loc[df.Receiving_Water==l,['Emamectin_Benzoate_grams']].sum(skipna=True))>0
        if (bool1==True)&(bool2==True):
            sublist.extend([l])
    EMBlochs.extend([sublist])

uniqEL = [elem for sub in EMBlochs for elem in sub]
uniqEL = sorted(set(uniqEL))

EMBlochs2 = []
for y in yrs:
    df = se_mr[se_mr.yr==y]
    sublist = []
    for l in lochs:
        bool1 = sum(df.loc[df.Receiving_Water==l,['Deltamethrin_grams', 'Cypermethrin_grams', 'Azamethiphos_grams', 'Teflubenzuron_grams']].sum(skipna=True))==0
#        bool2 = sum(df.loc[df.Receiving_Water==l,['Emamectin_Benzoate_grams']].sum(skipna=True))>0
        if bool1==True:
            sublist.extend([l])
    EMBlochs2.extend([sublist])
    
uniqEL2 = [elem for sub in EMBlochs[14:17] for elem in sub]
uniqEL2 = sorted(set(uniqEL2))

lochsubs = [[u for u in uniqEL2 if u in sub] for sub in EMBlochs2]
lochsubs = [u for u in lochsubs[15] if (u in lochsubs[14])&(u in lochsubs[16])] 
lochsubs
Out[40]: 
['Aith Voe',
# 'Bluemull Sound',
 'Eday Sound',
 'Firth of Lorn',
 'Gairsay Sound',
 'Gruting Voe',
# 'Hamna Voe, Yell', #yell sound shetland
 'Lamlash Bay',
 'Lax Firth',
 'Loch Carron',
 'Loch Tuath',
 'Loch na Keal',
 'Lynn of Lorn',
 'Scapa Flow',
 'Snarraness Voe']

s_info = pd.read_csv('M:/slm/Data/ms_site_details.csv')
s_info.columns = [s_info.columns[i].replace(' ','_') for i in range(len(s_info.columns))]
s_info.columns = [re.sub('[()]', '', s_info.columns[i]) for i in range(len(s_info.columns))]

sites = se_mr.National_Grid_Reference.unique()

se_mr['NGR'] = [str(i)[:5]+str(i)[6:9] for i in se_mr.National_Grid_Reference]

se_mr['name'] = se_mr.Site_Name.str.lower()
s_info['name'] = s_info.Site_Name.str.lower()

se_mr['name'] = se_mr.name.str.replace('[-(,)].*','')
s_info['name'] = s_info.name.str.replace('[-(,)].*','')

FMA = nu_df.MS_Management_Area.unique()
FMA=FMA[1:]

EMBfma = []
for y in yrs:
    df = nu_df[nu_df.yr==y]
    sublist = []
    for l in FMA:
        bool1 = sum(df.loc[df.MS_Management_Area==l,['Deltamethrin_grams', 'Cypermethrin_grams', 'Azamethiphos_grams', 'Teflubenzuron_grams']].sum(skipna=True))==0
        bool2 = sum(df.loc[df.MS_Management_Area==l,['Emamectin_Benzoate_grams']].sum(skipna=True))>0
        if (bool1==True)&(bool2==True):
            sublist.extend([l])
    EMBfma.extend([sublist])

uniqEL = [elem for sub in EMBfma for elem in sub]
uniqEL = sorted(set(uniqEL))

EMBfma2 = []
for y in yrs:
    df = nu_df[nu_df.yr==y]
    sublist = []
    for l in FMA:
        bool1 = sum(df.loc[df.MS_Management_Area==l,['Deltamethrin_grams', 'Cypermethrin_grams', 'Azamethiphos_grams', 'Teflubenzuron_grams']].sum(skipna=True))==0
#        bool2 = sum(df.loc[df.MS_Management_Area==l,['Emamectin_Benzoate_grams']].sum(skipna=True))>0
        if bool1==True:
            sublist.extend([l])
    EMBfma2.extend([sublist])
    
uniqEL2 = [elem for sub in EMBfma[14:17] for elem in sub]
uniqEL2 = sorted(set(uniqEL2))

fmaubs = [[u for u in uniqEL2 if u in sub] for sub in EMBfma2]
fmaubs = [u for u in fmaubs[15] if (u in fmaubs[14])&(u in fmaubs[16])] 
fmaubs
# =============================================================================
# Out[103]: 
# ['16a - Tuath, na Keal',
#  '8c - Scapa Flow',
#  '9c - Badcall, Chairn Bhain, Eddrachillis']
# =============================================================================

fma_sub = ['11a - Torridon',
 '16a - Tuath, na Keal',
 '19a - Fyne',
 '8c - Scapa Flow',
 '9c - Badcall, Chairn Bhain, Eddrachillis']

[nu_df.loc[nu_df.MS_Management_Area==i,'Site_Name_x'].unique() for i in fma_sub]

# =============================================================================
# [nu_df.loc[nu_df.MS_Management_Area==i,'Site_Name_x'].unique() for i in fma_sub]
# Out[122]: 
# [array(['Sgeir Dughall'], dtype=object),
#  array(['Geasgill', 'Gometra', 'Inch Kenneth'], dtype=object),
#  array(['Glenan Bay', 'Quarry Point', 'Rubha Stillaig', 'Strondoir Bay',
#         'Tarbert South'], dtype=object),
#  array(['Bring Head, Hoy', 'Chalmers Hope', 'Lyrawa Bay', 'Pegal Bay',
#         'Westerbister'], dtype=object),
#  array(['Drumbeg (Loch Dhrombaig)'], dtype=object)]
# =============================================================================
 
 sites_sub = [['Sgeir Dughall','Kenmore','Aird','Torridon'],['Geasgill', 'Gometra', 'Inch Kenneth','Loch Tuath'],
  ['Tarbert South', 'Rubha Stillaig', 'Glenan Bay', 'Meall Mhor', 'Gob a Bharra', 'Strondoir Bay', 'Ardgadden', 'Ardcastle Bay', 'Quarry Point'],
  ['Bring Head, Hoy', 'Chalmers Hope', 'Lyrawa Bay', 'Pegal Bay','Westerbister','Fara West','South Cava','Toyness','Lober Rock'],
  ['Drumbeg (Loch Dhrombaig)','Badcall Bay','Calva Bay (Calbha Beag)','Loch A Chain Bhain','Outer Bay (Loch Droighniche)','Nedd','Clashnessie Bay']]
 
 sites_sub = [['Sgeir Dughall','Kenmore','Aird','Torridon'],['Geasgill', 'Gometra', 'Inch Kenneth','Loch Tuath'],
              ['Tarbert South', 'Rubha Stillaig', 'Glenan Bay', 'Meall Mhor', 'Gob a Bharra', 'Strondoir Bay', 'Ardgadden', 'Ardcastle Bay', 'Quarry Point'],
             ['Bring Head', 'Chalmers Hope', 'Lyrawa Bay', 'Pegal Bay','Westerbister','Fara West','South Cava','Toyness'],
             ['Drumbeg','Badcall','Calbha Bay','Loch a Chairn Bhain','Outer Bay']]
 
bools = [nm in sites_sub[0] for nm in se_mr.Site_Name]
torridonEMB = se_mr[(bools)|(se_mr.Receiving_Water=='Loch Torridon')]
torridonEMB['name'] = torridonEMB.Site_Name

fig, ax = plt.subplots()
torridon_lice.groupby('Farm').plot('date','Adult female lice (Average per fish)',ax=ax)

torridonEMB['EMBbin'] = 0
torridonEMB.EMBbin[torridonEMB.Emamectin_Benzoate_grams>0] = 1

torridonEMB['other_bin'] = 0
torridonEMB.other_bin[(torridonEMB.Deltamethrin_grams>0)|(torridonEMB.Cypermethrin_grams>0)|(torridonEMB.Azamethiphos_grams>0)|(torridonEMB.Teflubenzuron_grams>0)] = 1

torridonEMB[torridonEMB.Year>=min(torridon_lice.date)].groupby('Site_Name').plot('Year','EMBbin',ax=ax,kind='bar')