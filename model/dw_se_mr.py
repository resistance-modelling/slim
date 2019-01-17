import numpy as np
import pandas as pd
from scipy import stats
import datetime as dt
from math import exp, log, sqrt, ceil
import matplotlib.pyplot as plt
import re
import geopandas as gpd
from shapely.geometry import Point, Polygon

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

