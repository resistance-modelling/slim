#!/usr/bin/python
#

#Input data: location of cages/farms, (Average) monthly(?) water T at those locations

#Sea lice lifecycle
#Lice values held in data frame (FILE?) with entries for: farm, cage, fish, MF, 
##stage, stage-age, resistanceT1, resitanceT2,...
#Initialise by setting initial lice population for each cage and sample 
##resistance values from distribution for natural occurance of resistance 
##***[INSERT APPROPRIATE DISTRIBUTION]*** and male/female=1.22 (cox et al 2017)


#For each cage-subset simultaneously:
#Calculate rate of events
#For each event sample number of times it happened during a fixed period tau=? 
##***(to be estimated)*** from Pois(rate*tau)
#Sample who they happen to and update value
##If event is infection sample fish ***density dependent???****
##If event is mating sample both and create new entries with status 0 and 
###create resistance values based on parents ***[INSERT MODE OF INHERITANCE]***
### 6-11 broods of 500 ova on average (see Costello 2006) extrude a pair every 3 days 
###***Average number of eggs for all or distribution???****
#Update fish growth and death (remove all lice entries associated w dead fish) 
#Migrate lice in L1/2 to other cages/farms within range 11-45km mean 27km Siegal et al 2003
##***INSERT FUNCTION***

#Update all values, add seasonal inputs e.g. treatment, wildlife reservoir and 
##run all cages again

#Lice events:
#infecting a fish (L2->L3) -> rate= ***[INSERT FUNCTION FISH WEIGHT BY AGE]**** 
##to be used with function from Aldrin et al 2017
#L1->L2,L3->L4,L4->L5 -> rate=sum(prob of dev after A days in a stage as given 
##by Aldrin et al 2017)
#Background death in a stage (remove entry) -> rate= number of individuals in 
##stage*stage rate (nauplii 0.17/d, copepods 0.22/d, pre-adult female 0.05, 
##pre-adult male ... Stien et al 2005)
#Treatment death in a stage (remove entry) -> rate= (1 - individual resistance 
##for individuals in appropriate stage)*[TREATMENT EFFECT]*[DOSE PRESENCE/DECAY]****
#Mating -> rate fish_i= adult males on fish_i*adult females on fish_i*(1/4)*
##(2.5/ave adult female lifetime) 
##Or set lice as unavailable for x days after mating and assume that a female 
##will mate as soon as available if there's an available male

#Fish model:
#Fish growth rate ****[INSERT FUNCTION]**** -> 
#Fish death rate: sum of constant *****background rate=??**** and individual 
##lice dependent term *****[INSERT APPROPRIATE DISTRIBUTION]**** ->
##sample with prob=rate_i/rate_tot
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
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------


#Input Data------------------------------------------------------------------------------------
oban_avetemp = [8.2,7.5,7.4,8.2,9.6,11.3,13.1,13.7,13.6,12.8,11.7,9.8] #www.seatemperature.org
mort_rates = [0.17, 0.22, 0.008, 0.05, 0.02, 0.06] # L1,L2,L3,L4,L5f,L5m
start_date = dt.datetime(2016, 3, 1)
tau = 1 ###############################################

#Initial Values--------------------------------------------------------------------------------
t = 0        #Day 0
cur_date = start_date
licepcage0 = 12

#Initial lice population
lice = pd.DataFrame(columns=['Farm','Cage','Fish','MF','stage','stage_age','resistanceT1'])
lice['Farm'] = np.repeat(1,licepcage0)
lice['Cage'] = np.repeat(1,licepcage0)
lice['Fish'] = np.repeat(range(1,4),4)
lice['MF'] = np.random.choice(['F','M'],licepcage0,p=[0.45,0.55])
#Based on median development time at 10 degrees from Aldrin et al 2017
lice['stage'] = np.random.choice([1,2],licepcage0,p=[0.55,0.45]) 


p=stats.poisson.pmf(range(15),3)
p[2] = p[2] + 1-sum(p) #probs need to add up to one 
lice['stage_age'] = np.random.choice(range(15),licepcage0,p=p)

rands = ranTruncNorm(mean=-6, sd=2, low=-100000000, upp=0, length=licepcage0)
lice['resistanceT1'] = [exp(i) for i in rands]

#pre-calc values-------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#--------------------------Simulation----------------------------------------------------------
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

cur_date = cur_date + dt.timedelta(days=tau)
lice.stage_age = lice.stage_age + tau

#----------------------------------------------------------------------------------------------
#Events during tau in cage---------------------------------------------------------------------
#----------------------------------------------------------------------------------------------


#Expected number of dev events-----------------------------------------------------------------
temp_now = oban_avetemp[cur_date.month-1] #change when farm location is available
L1toL2 = sum([devTimeAldrin(0.401,8.814,18.869,temp_now, i) 
             for i in lice.loc[lice['stage']=='L1'].stage_age])
#lice['Farm']==farm & lice['Cage']==cage when they have more than 1 value
    
L3toL4 = sum([devTimeAldrin(1.305,18.934,7.945,temp_now,i) 
             for i in lice.loc[lice['stage']=='L3'].stage_age])
L4toL5 = sum([devTimeAldrin(0.866,10.742,1.643,temp_now,i) 
             for i in lice.loc[lice['stage']=='L4'].stage_age])


#Mortality rates-------------------------------------------------------------------------------

    
    


