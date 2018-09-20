#!/usr/bin/python
#

#Input data: location of cages/farms, (Average) monthly(?) water T at those locations

#Sea lice lifecycle
#Lice values held in data frame with entries for: farm, cage, fish, MF, 
##stage, stage-age, resistanceT1, resitanceT2,...
#Initialise by setting initial lice population for each cage and sample 
##resistance values from distribution for natural occurance of resistance 
##***[INSERT APPROPRIATE DISTRIBUTION]*** and male/female=1.22 (cox et al 2017) maybe 1 
##as females live longer


#For each cage-subset simultaneously:
#Calculate rate of events
#For each event sample number of times it happened during a fixed period tau=? 
##***(to be estimated)*** from Pois(E(events during tau))
#Sample who they happen to and update value
##If event is infection sample fish 
##If event is mating sample both and create new entries with status 0 and 
###create resistance values based on parents ***[INSERT MODE OF INHERITANCE]***
### 6-11 broods of 500 ova on average (see Costello 2006) extrude a pair every 3 days 
#Update fish growth and death (remove all lice entries associated w dead fish) 
#Migrate lice in L1/2 to other cages/farms within range 11-45km mean 27km Siegal et al 2003
##***INSERT FUNCTION***

#Update all values, add seasonal inputs e.g. treatment, wildlife reservoir (input from  
##reservoir can be changed to model use of skirt/closed cage) and run all cages again

#Lice events:
#infecting a fish (L2->L3) -> rate from Aldrin et al 2017
#L1->L2,L3->L4,L4->L5 -> rate=sum(prob of dev after A days in a stage as given 
##by Aldrin et al 2017)
#Background death in a stage (remove entry) -> rate= number of individuals in 
##stage*stage rate (nauplii 0.17/d, copepods 0.22/d, pre-adult female 0.05, 
##pre-adult male ... Stien et al 2005)
#Treatment death in a stage (remove entry) -> rate= (1 - individual resistance 
##for individuals in appropriate stage)*[TREATMENT EFFECT]*[DOSE PRESENCE/DECAY]****
#Mating -> rate fish_i= adult males on fish_i*adult females on fish_i*(1/4)*
##(2.5/ave adult female lifetime) 
##Set lice as unavailable for 3 days after mating and assume that a female 
##will mate as soon as available if there's an available male

#Fish model:
#Fish growth rate -> 10000/(1+exp(-0.01*(t-475))) fitted logistic curve to data from 
##http://www.fao.org/fishery/affris/species-profiles/atlantic-salmon/growth/en/
#Fish death rate: constant background daily rate 0.00057 based on 
##www.gov.scot/Resource/0052/00524803.pdf 
##multiplied by lice coefficient see surv.py (compare to threshold of 0.75 lice/g fish)

#----------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from scipy import stats
import datetime as dt
from math import exp, log

np.random.seed(545836870)

#Functions-------------------------------------------------------------------------------------

#random truncated normal variates with more easily readable inputs
def ranTruncNorm(mean=0, sd=1, low=0, upp=10, length=1):    
    return stats.truncnorm.rvs((low - mean)/sd, (upp - mean)/sd, 
                               loc=mean, scale=sd, size=length)

#prob of developping after D days in a stage as given by Aldrin et al 2017
def devTimeAldrin(del_p, del_m10, del_s, temp_c, D):   
    unbounded = log(2)*log(del_s)*D**(del_s-1)*(del_m10*10**del_p/temp_c**del_p)**(-del_s)
    return min(unbounded,1)

def aveDevDays(del_p, del_m10, del_s, temp_c):
    return 100*devTimeAldrin(del_p, del_m10, del_s, temp_c, 100)\
    -0.001*sum([devTimeAldrin(del_p, del_m10, del_s, temp_c, i) 
    for i in np.arange(0,100.001,0.001)])
    
def chooseDinds(df,nents_list):
    stg = df.stage.unique()
    if len(stg) != 1: 
        return print('Error - grouping malfunctioned, stages are',stg)
    else:
        n_ents = nents_list[stg - 1]
        p = df.stage_age/np.sum(df.stage_age)
        return np.random.choice(df.index, n_ents, p=p, replace=False)

def chooseFish(df):
    cg = df.Cage.unique()
    fm = df.Farm.unique()
    if [len(cg),len(fm)] != [1,1]:
        return print('Error - grouping malfunctioned, farms & cages', fm, cg)
    else:
        a = [None]*len(df.index)
        for i in range(len(df.index)+1):
            if df.stage[i] > 2:###########################################
                a[i] = np.random.choice(range(1,fish0[*fm-1][*cg-1]+1))#######
        return a 

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------


#Input Data------------------------------------------------------------------------------------
oban_avetemp = np.array([8.2,7.5,7.4,8.2,9.6,11.3,13.1,13.7,13.6,12.8,11.7,9.8]) #www.seatemperature.org
ncages = 2
fish0 = np.array([[3,3]])
mort_rates = np.array([0.17, 0.22, 0.008, 0.05, 0.02, 0.06]) # L1,L2,L3,L4,L5f,L5m
fb_mort = 0.00057
lice_coef = -2.4334
start_date = dt.datetime(2016, 3, 1)
tau = 1 ###############################################

#Initial Values--------------------------------------------------------------------------------
t = 0        #Day 0
cur_date = start_date
licepfish0 = 8
cur_fish = fish0.copy() 

lice_tot = np.sum(licepfish0*fish0)
#Initial lice population
lice = pd.DataFrame(columns=['Farm','Cage','Fish','MF','stage','stage_age','avail','resistanceT1'])
lice['Farm'] = np.repeat(1,lice_tot)
lice['Cage'] = np.array([np.repeat(i,licepfish0*fish0[j-1][i-1]) 
                          for i in range(1,ncages+1) for j in lice.Farm.unique()]).flatten()
lice['Fish'] = np.array(lice.groupby(['Farm','Cage']).apply(chooseFish)).flatten()                     
lice['MF'] = np.random.choice(['F','M'],lice_tot)
#Based on median development time at 10 degrees from Aldrin et al 2017
lice['stage'] = np.random.choice([1,2],lice_tot,p=[0.55,0.45]) 

p=stats.poisson.pmf(range(15),3)
p = p/sum(p) #probs need to add up to one 
lice['stage_age'] = np.random.choice(range(15),lice_tot,p=p)
lice['avail'] = np.repeat(1,lice_tot)

rands = ranTruncNorm(mean=-6, sd=2, low=-100000000, upp=0, length=lice_tot)
lice['resistanceT1'] = np.exp(rands)


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#--------------------------Simulation----------------------------------------------------------
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

cur_date = cur_date + dt.timedelta(days=tau)
t = cur_date - start_date
lice.stage_age = lice.stage_age + tau


#----------------------------------------------------------------------------------------------
#Events during tau in cage---------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
farm = 1 ###############################
cage = 1 ###############################
cur_cage = lice.loc[(lice['Farm']==farm) & (lice['Cage']==cage)]

#Background mortality events-------------------------------------------------------------------
inds_stage = [sum(cur_cage['stage']==i) for i in range(1,7)]
Emort = np.multiply(mort_rates, inds_stage)
mort_ents = np.random.poisson(Emort)
mort_inds = cur_cage.groupby('stage').apply(chooseDinds,nents_list=mort_ents) 
             
#Development events----------------------------------------------------------------------------
temp_now = oban_avetemp[cur_date.month-1] #change when farm location is available
L1toL2 = [devTimeAldrin(0.401,8.814,18.869,temp_now, i) 
             for i in cur_cage.loc[cur_cage['stage']==1].stage_age]
L1toL2_ents = np.random.poisson(sum(L1toL2))
L1toL2_inds = np.random.choice(cur_cage.loc[cur_cage['stage']==1].index, L1toL2_ents,\
                               p=L1toL2, replace=False)
    
L3toL4 = [devTimeAldrin(1.305,18.934,7.945,temp_now,i) 
             for i in cur_cage.loc[cur_cage['stage']==3].stage_age]
L3toL4_ents = np.random.poisson(sum(L3toL4))
L3toL4_inds = np.random.choice(cur_cage.loc[cur_cage['stage']==3].index, L3toL4_ents,\
                               p=L3toL4, replace=False)

L4toL5 = sum([devTimeAldrin(0.866,10.742,1.643,temp_now,i) 
             for i in cur_cage.loc[cur_cage['stage']==4].stage_age])
L4toL5_ents = np.random.poisson(sum(L4toL5))
L4toL5_inds = np.random.choice(cur_cage.loc[cur_cage['stage']==4].index, L4toL5_ents,\
                               p=L4toL5, replace=False)

 
#Fish growth and death-------------------------------------------------------------------------
wt = 10000/(1+exp(-0.01*(t.days-475)))
fish_fc = cur_cage.Fish.unique()
adlicepg = np.array([(sum(cur_cage.loc[cur_cage['Fish']==i].stage>3))/wt 
           for i in fish_fc])
Plideath = 1/(1+np.exp(-19*(adlicepg-0.63)))
Efish_death = (fb_mort - np.log(1-Plideath))*tau*cur_fish[farm-1][cage-1]
fd_ents = np.random.poisson(Efish_death)
fd_inds = np.random.choice(fish_fc, fd_ents, p=Plideath, replace=False)

#Infection events------------------------------------------------------------------------------
eta_aldrin = -2.576 + log(cur_fish[farm-1][cage-1]) + 0.082*(log(wt)-0.55)
Einf = (exp(eta_aldrin)/(1+exp(eta_aldrin)))*tau*inds_stage[1]
inf_ents = np.random.poisson(Einf)
inf_inds = ######################################################

#Mating events---------------------------------------------------------------------------------
nmating = min(sum((cur_cage.stage==5) & (cur_cage.avail==1)),\
              sum((cur_cage.stage==6) & (cur_cage.avail==1)))