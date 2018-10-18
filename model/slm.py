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
### Or constant production of 70 nauplii/day for 20 days post-mating
#Update fish growth and death (remove all lice entries associated w dead fish) 
#Migrate lice in L1/2 to other cages/farms within range 11-45km mean 27km Siegal et al 2003
##biased random walk with different with constant speed and different prob for direction dep on 
##prevalent wind direction, current and location of farms

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
from math import exp, log, sqrt
import geopandas as gpd
from shapely.geometry import Point, Polygon

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

#average dev days using devTimeAldrin, not used in model but handy to have
#5deg: 12.2,-,67.5,2
#10deg: 9.1,-,24,5.3
#15deg: 7.6,-,13.1,9.4
def aveDevDays(del_p, del_m10, del_s, temp_c):
    return 100*devTimeAldrin(del_p, del_m10, del_s, temp_c, 100)\
    -0.001*sum([devTimeAldrin(del_p, del_m10, del_s, temp_c, i) 
    for i in np.arange(0,100.001,0.001)])
    
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------


#Input Data------------------------------------------------------------------------------------
nfarms = 1
ncages = np.array([2])
oban_avetemp = np.array([8.2,7.5,7.4,8.2,9.6,11.3,13.1,13.7,13.6,12.8,11.7,9.8]) #www.seatemperature.org
portree_avetemp = np.array([8.2,7.6,7.4,8,9.4,10.9,12.7,13.3,13,12.3,11.2,9.6])
fcID = ['f'+str(i)+'c'+str(j) for i in range(1,nfarms+1) for j in range(1,ncages[i-1]+1)]
xy_array = np.array([[56.52166,-5.36803],[56.52120,-5.36774]]) #from google maps
cages_xy = dict(zip(fcID,xy_array))

fish0 = np.array([[3,4]])
mort_rates = np.array([0.17, 0.22, 0.008, 0.05, 0.02, 0.06]) # L1,L2,L3,L4,L5f,L5m
fb_mort = 0.00057
lice_coef = -2.4334
eggspday = 10#100
ext_pressure = 1 #planktonic lice per day per cage/farm arriving from wildlife -> seasonal?

start_date = dt.datetime(2016, 3, 1)
end_date = dt.datetime(2016, 3, 8) #dt.datetime(2017, 9, 1)
tau = 1 ###############################################

#Initial Values--------------------------------------------------------------------------------
cur_date = start_date
cur_fish = fish0.copy() 

#Initial Fish population
salmon = [np.arange(1,fish0[i][j]+1) for i in range(nfarms) for j in range(ncages[i])]
all_fish = dict(zip(fcID,salmon))
    
#Lice population in farms
lice = pd.DataFrame(columns=['Farm','Cage','Fish','MF','stage','stage_age','avail',\
                             'mate_resistanceT1','resistanceT1','date'])

#Freefloating plantonic lice reservoir
offspring = pd.DataFrame(columns=['lat','lon','MF','stage','stage_age','resistanceT1','date'])


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#--------------------------Simulation----------------------------------------------------------
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

while cur_date <= end_date: 
    
    cur_date = cur_date + dt.timedelta(days=tau)
    if not lice.empty:
        lice.date = cur_date
        lice.stage_age = lice.stage_age + tau
        lice.loc[lice.avail>0, 'avail'] = lice.avail + tau
        lice.loc[(lice.MF=='M') & (lice.avail>4), 'avail'] = 0
        lice.loc[(lice.MF=='F') & (lice.avail>12), 'avail'] = 0
        lice.loc[(lice.MF=='F') & (lice.avail>12), 'mate_resistanceT1'] = None
         
    offspring.stage_age = offspring.stage_age + tau
    offspring.date = cur_date
    #Biased random walk of offspring-----------------------------------------------------------
    ###########################################
    
    #------------------------------------------------------------------------------------------
    #Events during tau in cage-----------------------------------------------------------------
    #------------------------------------------------------------------------------------------
    for farm in range(1, nfarms+1):
        for cage in range(1, ncages[farm-1]+1): 
            cur_cage = lice.loc[(lice['Farm']==farm) & (lice['Cage']==cage)].copy()
                        
            #new planktonic stages arriving from wildlife reservoir
            nplankt = ext_pressure*tau
            plankt_cage = pd.DataFrame(columns=offspring.columns)
            plankt_cage['MF'] = np.random.choice(['F','M'],nplankt)
            #Based on median development time at 10 degrees from Aldrin et al 2017
            plankt_cage['stage'] = np.random.choice([1,2],nplankt,p=[0.55,0.45]) 
            plankt_cage['lat'] = cages_xy['f'+str(farm)+'c'+str(cage)][0]
            plankt_cage['lon'] = cages_xy['f'+str(farm)+'c'+str(cage)][1]
            p=stats.poisson.pmf(range(15),3)
            p = p/sum(p) #probs need to add up to one 
            plankt_cage['stage_age'] = np.random.choice(range(15),nplankt,p=p)
            rands = ranTruncNorm(mean=-6, sd=2, low=-100000000, upp=0, length=nplankt)
            plankt_cage['resistanceT1'] = np.exp(rands)
            plankt_cage['date'] = cur_date
            
            #planktonic stages arriving from other cages (or still around)
            cage_shp = Point(cages_xy['f'+str(farm)+'c'+str(cage)]).buffer(0.0005)
            
            if offspring.lat.size>0:
                plankt_pts = [Point([offspring.lat[i],offspring.lon[i]]) for i in offspring.index]
                bool_pts = [cage_shp.contains(plankt_pts[i]) for i in range(len(plankt_pts))]
                other_cages = offspring[bool_pts].copy()
                plankt_cage = plankt_cage.append(other_cages, ignore_index=True)
                offspring = offspring.drop(offspring[bool_pts].index)
            
            #Background mortality events-------------------------------------------------------
            inds_stage = [sum(plankt_cage['stage']==i) for i in range(1,3)]
            inds_stage.extend([sum(cur_cage['stage']==i) for i in range(3,7)])
            Emort = np.multiply(mort_rates, inds_stage)
            mort_ents = np.random.poisson(Emort)
            planktm_inds = []
            for i in range(1,3):
                df = plankt_cage.loc[plankt_cage.stage==i].copy()
                if not df.empty:
                    p = df.stage_age/np.sum(df.stage_age)
                    p=pd.to_numeric(p)
                    values = np.random.choice(df.index, mort_ents[i-1], p=p, replace=False).tolist()
                    planktm_inds.extend(values)
            mort_inds = []
            for i in range(3,7):
                df = cur_cage.loc[cur_cage.stage==i].copy()
                if not df.empty:
                    p = df.stage_age/np.sum(df.stage_age)
                    p=pd.to_numeric(p)
                    values = np.random.choice(df.index, mort_ents[i-1], p=p, replace=False).tolist()
                    mort_inds.extend(values)
                         
            #Development events----------------------------------------------------------------
            temp_now = oban_avetemp[cur_date.month-1] #change when farm location is available
            if inds_stage[0]>0:
                L1toL2 = [devTimeAldrin(0.401,8.814,18.869,temp_now, i) 
                         for i in plankt_cage.loc[plankt_cage['stage']==1].stage_age]
                L1toL2_ents = np.random.poisson(sum(L1toL2))
                L1toL2_inds = np.random.choice(plankt_cage.loc[plankt_cage['stage']==1].index, \
                                               L1toL2_ents, p=L1toL2/sum(L1toL2), replace=False)
            else:
                L1toL2_inds = []
                
            if inds_stage[2]>0:
                L3toL4 = [devTimeAldrin(1.305,18.934,7.945,temp_now,i) 
                         for i in cur_cage.loc[cur_cage['stage']==3].stage_age]
                L3toL4_ents = np.random.poisson(sum(L3toL4))
                L3toL4_inds = np.random.choice(cur_cage.loc[cur_cage['stage']==3].index, \
                                               L3toL4_ents, p=L3toL4/sum(L3toL4), replace=False)
            else:
                L3toL4_inds = []
            
            if inds_stage[3]>0:
                L4toL5 = sum([devTimeAldrin(0.866,10.742,1.643,temp_now,i) 
                         for i in cur_cage.loc[cur_cage['stage']==4].stage_age])
                L4toL5_ents = np.random.poisson(sum(L4toL5))
                L4toL5_inds = np.random.choice(cur_cage.loc[cur_cage['stage']==4].index, \
                                               L4toL5_ents, p=L4toL5/sum(L4toL5), replace=False)
            else:
                L4toL5_inds = []
             
            #Fish growth and death-------------------------------------------------------------
            t = cur_date - start_date
            wt = 10000/(1+exp(-0.01*(t.days-475)))
            fish_fc = cur_cage.Fish.unique() #fish with lice
            adlicepg = np.array([(sum(cur_cage.loc[cur_cage['Fish']==i].stage>3))/wt 
                       for i in fish_fc])
            Plideath = 1/(1+np.exp(-19*(adlicepg-0.63)))
            Ebf_death = fb_mort*tau*cur_fish[farm-1][cage-1]
            Elf_death = np.sum(Plideath)*tau
            bfd_ents = np.random.poisson(Ebf_death)
            lfd_ents = np.random.poisson(Elf_death)
            fd_inds = []
            if fish_fc.size>0:
                fd_inds.extend(np.random.choice(fish_fc, lfd_ents, p=Plideath/sum(Plideath), 
                                                          replace=False).tolist())
            fd_inds.extend(np.random.choice(all_fish['f'+str(farm)+'c'+str(cage)], \
                                                bfd_ents, replace=False).tolist())
            
            #Infection events------------------------------------------------------------------
            if sum(plankt_cage['stage']==2)>0:
                eta_aldrin = -2.576 + log(cur_fish[farm-1][cage-1]) + 0.082*(log(wt)-0.55)
                Einf = (exp(eta_aldrin)/(1+exp(eta_aldrin)))*tau*inds_stage[1]
                inf_ents = np.random.poisson(Einf)
                inf_inds = np.random.choice(plankt_cage.loc[plankt_cage['stage']==2].index, inf_ents, replace=False)
            else:
                inf_inds = []
            
            #Mating events---------------------------------------------------------------------
     
            #who is mating          
            for fish in fish_fc:
                females = cur_cage.loc[(cur_cage.stage==5) & (cur_cage.Fish==fish) & (cur_cage.avail==0)].index
                males = cur_cage.loc[(cur_cage.stage==6) & (cur_cage.Fish==fish) & (cur_cage.avail==0)].index
                nmating = min(sum(cur_cage.index.isin(females)),\
                          sum(cur_cage.index.isin(males)))
                if nmating>0:
                    sires = np.random.choice(males, nmating, replace=False)
                    p_dams = 1 - (cur_cage.loc[cur_cage.index.isin(females),'stage_age']/
                                sum(cur_cage.loc[cur_cage.index.isin(females),'stage_age']))
                    dams = np.random.choice(females, nmating, p=p_dams, replace=False)
                else:
                    sires = []
                    dams = []
            if fish_fc.size>0:
                cur_cage.loc[cur_cage.index.isin(dams),'avail'] = 1
                cur_cage.loc[cur_cage.index.isin(sires),'avail'] = 1
                #Add phenotype of sire to dam info
                cur_cage.loc[cur_cage.index.isin(dams),'mate_resistanceT1'] = \
                cur_cage.loc[cur_cage.index.isin(sires),'resistanceT1']
            
            #create offspring
            offs = pd.DataFrame(columns=offspring.columns)
            gravid_females = cur_cage.loc[cur_cage.avail>0].index
            new_offs = eggspday*tau*len(gravid_females)
            offs['MF'] = np.random.choice(['F','M'], new_offs)
            offs['lat'] = cages_xy['f'+str(farm)+'c'+str(cage)][0]
            offs['lon'] = cages_xy['f'+str(farm)+'c'+str(cage)][1]
            offs['stage'] = np.repeat(1, new_offs)
            offs['stage_age'] = np.repeat(0, new_offs)
            offs['date'] = cur_date
            k = 0
            for i in gravid_females:
                for j in range(eggspday*tau):
                    underlying = 6 + 0.5*np.log(cur_cage.resistanceT1[cur_cage.index==i])\
                               + 0.5*np.log(cur_cage.mate_resistanceT1[cur_cage.index==i]) + \
                               ranTruncNorm(mean=0, sd=2, low=-100000000, upp=0, length=1)/sqrt(2)
                    offs.loc[offs.index[k], 'resistanceT1'] = np.exp(underlying)
                    k = k + 1                
            offspring = offspring.append(offs, ignore_index=True)
            
            #Update cage info------------------------------------------------------------------
            #----------------------------------------------------------------------------------
            plankt_cage.loc[plankt_cage.index.isin(L1toL2_inds),'stage'] = 2
            plankt_cage.loc[plankt_cage.index.isin(L1toL2_inds),'stage_age'] = 0
            cur_cage.loc[cur_cage.index.isin(L3toL4_inds),'stage'] = 4
            cur_cage.loc[cur_cage.index.isin(L3toL4_inds),'stage_age'] = 0
            cur_cage.loc[cur_cage.index.isin(L4toL5_inds) & (cur_cage.MF=='F'),'stage'] = 5
            cur_cage.loc[cur_cage.index.isin(L4toL5_inds) & (cur_cage.MF=='M'),'stage'] = 6
            cur_cage.loc[cur_cage.index.isin(L4toL5_inds),'stage_age'] = 0
            
            plankt_cage.loc[plankt_cage.index.isin(inf_inds), 'stage'] = 3
            plankt_cage.loc[plankt_cage.index.isin(inf_inds),'stage_age'] = 0
            offspring = offspring.append(plankt_cage[plankt_cage.stage<3], ignore_index=True)
                                 
            chalimi = pd.DataFrame(columns=lice.columns)
            fsh = all_fish['f'+str(farm)+'c'+str(cage)]
            chalimi['Fish'] = np.random.choice(fsh, len(inf_inds))
            chalimi['Farm'] = farm
            chalimi['Cage'] = cage
            chalimi['stage'] = 3
            chalimi['stage_age'] = 0
            chalimi['avail'] = 0
            chalimi['MF'] = plankt_cage.loc[plankt_cage.stage==3, 'MF'].values
            chalimi['resistanceT1'] = plankt_cage.loc[plankt_cage.stage==3, 'resistanceT1'].values
            chalimi.date = cur_date
                      
            file_path = 'M:/slm/model/cage'+str(cage)+'.txt'
            cur_cage.to_csv(file_path, sep='\t', mode='a')
            
            file_path = 'M:/slm/model/plankt_cage'+str(cage)+'.txt'
            plankt_cage.to_csv(file_path, sep='\t', mode='a')
            
            lice.loc[(lice['Farm']==farm) & (lice['Cage']==cage)] = cur_cage.copy()
                                   
            #remove dead individuals
            lice = lice.drop(mort_inds)
            lice = lice.drop(lice.loc[lice.Fish.isin(fd_inds)].index)
            if len(fd_inds)>0:
                fish = [i for i in fsh]
                fish.remove(fd_inds)
                all_fish['f'+str(farm)+'c'+str(cage)] = fish
            cur_fish[farm-1,cage-1] = len(all_fish['f'+str(farm)+'c'+str(cage)])
            
            #Add new chalimi
            lice = lice.append(chalimi, ignore_index=True)
            
            lice.to_csv('M:/slm/model/lice.txt', sep='\t', mode='a')
            offspring.to_csv('M:/slm/model/offspring.txt', sep='\t', mode='a')
        