#!/usr/bin/python
#

#Input data: location of cages/farms, (Average) monthly(?) water T at those locations

#Sea lice lifecycle
#Lice values held in data frame with entries for: farm, cage, fish, MF, 
##stage, stage-age, resistanceT1, resitanceT2,...
#Initialise by setting initial lice population for each cage and sample 
##resistance values from distribution for natural occurance of resistance 
##***[INSERT APPROPRIATE DISTRIBUTION]*** and male/female=1.22 (cox et al 2017) 


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
#Indicate farm that will be reached using probability matrix from Salama et al 2013 and sample
##cage randomly

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
from math import exp, log, sqrt, ceil
import time
import os

np.random.seed(545836870)

#Functions-------------------------------------------------------------------------------------

#random truncated normal variates with more easily readable inputs
def ranTruncNorm(mean=0, sd=1, low=0, upp=10, length=1):    
    return stats.truncnorm.rvs((low - mean)/sd, (upp - mean)/sd, 
                               loc=mean, scale=sd, size=length)

#prob of developping after D days in a stage as given by Aldrin et al 2017
def devTimeAldrin(del_p, del_m10, del_s, temp_c, D):   
    unbounded = log(2)*del_s*D**(del_s-1)*(del_m10*10**del_p/temp_c**del_p)**(-del_s)
    return min(unbounded,1)

#average dev days using devTimeAldrin, not used in model but handy to have
#5deg: 5.2,-,67.5,2
#10deg: 3.9,-,24,5.3
#15deg: 3.3,-,13.1,9.4
def aveDevDays(del_p, del_m10, del_s, temp_c):
    return 100*devTimeAldrin(del_p, del_m10, del_s, temp_c, 100)\
    -0.001*sum([devTimeAldrin(del_p, del_m10, del_s, temp_c, i) 
    for i in np.arange(0,100.001,0.001)])
    
def eudist(pointA,pointB):
    return sqrt((pointA[0]-pointB[0])**2 + (pointA[1]-pointB[1])**2)

def Linnhe_temp(c_month,farm_N):
    degs = (oban_avetemp[c_month-1]-fw_avetemp[c_month-1])/(774181-729693)
    Ndiff = farm_N - 729693 #farm Northing - oban Northing
    return round(oban_avetemp[c_month-1] - Ndiff*degs, 1)


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------


#Input Data------------------------------------------------------------------------------------
nfarms = 10
ncages = np.array([20,24,12,8,10,14,14,9,9,10])
oban_avetemp = np.array([8.2,7.5,7.4,8.2,9.6,11.3,13.1,13.7,13.6,12.8,11.7,9.8]) #www.seatemperature.org
fw_avetemp = np.array([8,7.4,7.1,8,9.3,11.1,12.9,13.5,13.3,12.5,11.3,9.6])
#portree_avetemp = np.array([8.2,7.6,7.4,8,9.4,10.9,12.7,13.3,13,12.3,11.2,9.6])
xy_array = np.array([[206100,770600],[201300,764700],[207900,759500],[185200,752500],
                     [192500,749700],[186500,745300],[193400,741700],[184700,740000],
                     [186700,734100],[183500,730700]]) #Loch Linnhe
#xy_array = np.array([[190300,665300],[192500,668200],[191800,669500],[186500,674500],
#    [190400,676800],[186300,679600],[190800,681000],[195300,692200],[199800,698000]])
farm_dist = np.array([eudist(i,j) for i in xy_array for j in xy_array]).reshape(nfarms,nfarms)
hrs_travel = farm_dist/(0.05*60*60)
prob_arrive = pd.read_csv('M:/slm/data/Linnhenetwork.csv',header=None)
prop_arrive = np.array([prob_arrive[i][j]/exp(-0.01*hrs_travel[i][j]) for i in range(nfarms) 
                                             for j in range(nfarms)]).reshape(nfarms,nfarms)

fishf = [50000,50000,50000,90000,60000,60000,60000,60000,60000,60000]
c_weight = np.array([85, 85, 85, 85, 85, 85, 85, 85, 85, 85])
mort_rates = np.array([0.17, 0.22, 0.008, 0.05, 0.02, 0.06]) # L1,L2,L3,L4,L5f,L5m
fb_mort = 0.00057
lice_coef = -2.4334
eggspday = [50,50,40,40,50,60,80,80,80,80,70,50]
d_hatching = [9,10,11,9,8,6,4,4,4,4,5,7]            
ext_pressure = 100 #planktonic lice per day per cage/farm arriving from wildlife -> seasonal?

start_date = dt.datetime(2015, 11, 1)
end_date = dt.datetime(2016, 7, 3)#dt.datetime(2017, 9, 1)
tau = 1 ###############################################

#Initial Values--------------------------------------------------------------------------------
cur_date = start_date

fcID = ['f'+str(i)+'c'+str(j) for i in range(1,nfarms+1) for j in range(1,ncages[i-1]+1)]
fsh = [list(range(1,fishf[i]+1)) for i in range(nfarms) for j in range(ncages[i])]
all_fish = dict(zip(fcID,fsh))
    
#Lice population in farms
lice = pd.DataFrame(columns=['Farm','Cage','Fish','MF','stage','stage_age','avail',\
                             'mate_resistanceT1','resistanceT1','date'])

#Freefloating plantonic lice reservoir
offspring = pd.DataFrame(columns=['Farm','Cage','MF','stage','stage_age','resistanceT1',\
                                  'arrival','date'])


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#--------------------------Simulation----------------------------------------------------------
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
file_path = "M:/slm/outputs/outputs" + dt.datetime.today().strftime('%Y%m%d') + '/'
directory = os.path.dirname(file_path)
try:
    if not os.path.exists(directory):
        os.makedirs(directory)
except OSError:
    print('Error creating directory')
file = open(file_path + 'timings.txt','a+') 
prev_time = time.time() 
prev_time2 = time.time()
counter = 0 
while cur_date <= end_date: 
    
    print(cur_date, time.time()-prev_time2)
    prev_time2 = time.time()
    counter = counter + 1
    if (counter%7)==0:
        file.write(str((time.time()-prev_time)/60) + ' date ' + str(cur_date) + '\n')
        prev_time = time.time()
#    lice.to_csv(file_path + 'lice.txt', sep='\t', mode='a')
#    offspring.to_csv(file_path + 'offspring.txt', sep='\t', mode='a')
  
    cur_date = cur_date + dt.timedelta(days=tau)
    if cur_date.day==1:
        femaleAL = [sum((lice.MF=='F') & (lice.Farm==i) & (lice.stage==5) & (lice.Cage==j))/fishf[i-1] \
                    for i in range(1,nfarms+1)
                    for j in range(1,ncages[i-1]+1)]
        farms = [i for i in range(1,nfarms+1) for j in range(1,ncages[i-1]+1)]
        femaleAL_df = pd.DataFrame({'femaleAL':femaleAL,'farms':farms})
        file.write(str(cur_date) + '\n')
        file.write('All, mean, ' + str(femaleAL_df.femaleAL.mean()) + ', std, ' + str(femaleAL_df.femaleAL.std()) + '\n')
        file.write('North, mean, ' + str(femaleAL_df[femaleAL_df.farms<4].femaleAL.mean()) + ', std, ' + str(femaleAL_df[femaleAL_df.farms<4].femaleAL.std()) + '\n')
        file.write('South, mean, ' + str(femaleAL_df[femaleAL_df.farms>3].femaleAL.mean()) + ', std, ' + str(femaleAL_df[femaleAL_df.farms>3].femaleAL.std()) + '\n')
        file.write(str(femaleAL_df.groupby(farms).femaleAL.mean()))
        file.write(str(femaleAL_df.groupby(farms).femaleAL.std()))
 
            
    if not lice.empty:
        lice.date = cur_date
        lice.stage_age = lice.stage_age + tau
        lice.loc[lice.avail>0, 'avail'] = lice.avail + tau
        lice.loc[(lice.MF=='M') & (lice.avail>4), 'avail'] = 0
        lice.loc[(lice.MF=='F') & (lice.avail>d_hatching[cur_date.month-1]), 'avail'] = 0
        lice.loc[(lice.MF=='F') & (lice.avail>d_hatching[cur_date.month-1]), 'mate_resistanceT1'] = None
         
   
    offspring['stage_age'] = offspring['stage_age'] + tau
    offspring['date'] = cur_date
    nu_cppds = offspring.loc[offspring.stage_age>=offspring.arrival, offspring.columns!='arrival'].copy()
    nu_cppds['avail'] = 0
        
    if not nu_cppds.empty:
        lice = lice.append(nu_cppds, ignore_index=True, sort=False)
        offspring = offspring.drop(nu_cppds.index)
    
    #------------------------------------------------------------------------------------------
    #Events during tau in cage-----------------------------------------------------------------
    #------------------------------------------------------------------------------------------
    for farm in range(1, nfarms+1):
        NSbool = (farm>3) | (cur_date>dt.datetime(2016, 1, 1))
        if NSbool==True:
            for cage in range(1, ncages[farm-1]+1): 
                
                #new planktonic stages arriving from wildlife reservoir
                nplankt = ext_pressure*tau
                plankt_cage = pd.DataFrame(columns=offspring.columns)
                plankt_cage['MF'] = np.random.choice(['F','M'],nplankt)
                plankt_cage['stage'] = 2 
                plankt_cage['Farm'] = farm
                plankt_cage['Cage'] = cage
                p=stats.poisson.pmf(range(15),3)
                p = p/sum(p) #probs need to add up to one 
                plankt_cage['stage_age'] = np.random.choice(range(15),nplankt,p=p)
                plankt_cage['avail'] = 0
                rands = ranTruncNorm(mean=-6, sd=2, low=-100000000, upp=0, length=nplankt) ########################################
                plankt_cage['resistanceT1'] = np.exp(rands)
                plankt_cage['date'] = cur_date
                
                lice = lice.append(plankt_cage, ignore_index=True, sort=False)
                
                cur_cage = lice.loc[(lice['Farm']==farm) & (lice['Cage']==cage)].copy()
                dead_fish = set([])
                
                #Background mortality events-------------------------------------------------------
                inds_stage = np.array([sum(cur_cage['stage']==i) for i in range(1,7)])
                Emort = np.multiply(mort_rates, inds_stage)
                mort_ents = np.random.poisson(Emort)
                mort_ents=[min(mort_ents[i],inds_stage[i]) for i in range(len(mort_ents))]
                mort_inds = []
                for i in range(1,7):
                    df = cur_cage.loc[cur_cage.stage==i].copy()
                    if not df.empty:
                        if np.sum(df.stage_age)>0:
                            p = (df.stage_age+1)/np.sum(df.stage_age+1)
                            p = pd.to_numeric(p)
                            values = np.random.choice(df.index, mort_ents[i-1], p=p, replace=False).tolist()
                        else:
                            values = np.random.choice(df.index, mort_ents[i-1], replace=False).tolist()                    
                        mort_inds.extend(values)
                             

                #Development events----------------------------------------------------------------
                temp_now = Linnhe_temp(cur_date.month, xy_array[farm-1][1])
                    
                if inds_stage[2]>0:
                    L3toL4 = [devTimeAldrin(1.305,18.934,7.945,temp_now,i) 
                             for i in cur_cage.loc[cur_cage['stage']==3].stage_age]
                    L3toL4_ents = np.random.poisson(sum(L3toL4))
                    L3toL4_ents = min(L3toL4_ents, inds_stage[2])
                    L3toL4_inds = np.random.choice(cur_cage.loc[cur_cage['stage']==3].index, \
                                                   L3toL4_ents, p=L3toL4/sum(L3toL4), replace=False)
                else:
                    L3toL4_inds = []
                
                if inds_stage[3]>0:
                    L4toL5 = [devTimeAldrin(0.866,10.742,1.643,temp_now,i) 
                             for i in cur_cage.loc[cur_cage['stage']==4].stage_age]
                    L4toL5_ents = np.random.poisson(sum(L4toL5))
                    L4toL5_ents = min(L4toL5_ents, inds_stage[3])
                    L4toL5_inds = np.random.choice(cur_cage.loc[cur_cage['stage']==4].index, \
                                                   L4toL5_ents, p=L4toL5/sum(L4toL5), replace=False)
                else:
                    L4toL5_inds = []
                
                #Fish growth and death-------------------------------------------------------------
                t = cur_date - start_date
                wt = 10000/(1+exp(-0.01*(t.days-475)))
                fish_fc = np.array(cur_cage[cur_cage.stage>3].Fish.unique().tolist()) #fish with lice
                fish_fc = fish_fc[~np.isnan(fish_fc)]
                adlicepg = np.array(cur_cage[cur_cage.stage>3].groupby('Fish').stage.count())/wt
                Plideath = 1/(1+np.exp(-19*(adlicepg-0.63)))
                nfish = len(all_fish['f'+str(farm)+'c'+str(cage)])
                Ebf_death = fb_mort*tau*(nfish)
                Elf_death = np.sum(Plideath)*tau
                bfd_ents = np.random.poisson(Ebf_death)
                lfd_ents = np.random.poisson(Elf_death) 
                           
                if fish_fc.size>0:
                    dead_fish.update(np.random.choice(fish_fc, 
                              lfd_ents, p=Plideath/sum(Plideath), replace=False).tolist())
                if len(dead_fish)>0:
                    all_fish['f'+str(farm)+'c'+str(cage)] = [i for i in all_fish['f'+str(farm)+'c'+str(cage)] if i not in dead_fish]
                dead_fish.update(np.random.choice(all_fish['f'+str(farm)+'c'+str(cage)], bfd_ents, replace=False).tolist())
                if len(dead_fish)>0:
                    all_fish['f'+str(farm)+'c'+str(cage)] = [i for i in all_fish['f'+str(farm)+'c'+str(cage)] if i not in dead_fish]
                
                #Infection events------------------------------------------------------------------
                if sum(cur_cage['stage']==2)>0:
                    eta_aldrin = -2.576 + log(nfish) + 0.082*(log(wt)-0.55)
                    Einf = (exp(eta_aldrin)/(1+exp(eta_aldrin)))*tau*inds_stage[1]
                    inf_ents = np.random.poisson(Einf)
                    inf_ents = min(inf_ents,sum(cur_cage['stage']==2))
                    inf_inds = np.random.choice(cur_cage.loc[cur_cage['stage']==2].index, inf_ents, replace=False)
                else:
                    inf_inds = []
                
                #Mating events---------------------------------------------------------------------
                
                #who is mating               
                females = cur_cage.loc[(cur_cage.stage==5) & (cur_cage.avail==0)].index
                males = cur_cage.loc[(cur_cage.stage==6) & (cur_cage.avail==0)].index
                nmating = min(sum(cur_cage.index.isin(females)),\
                          sum(cur_cage.index.isin(males)))
                if nmating>0:
                    sires = np.random.choice(males, nmating, replace=False)
                    p_dams = 1 - (cur_cage.loc[cur_cage.index.isin(females),'stage_age']/
                                sum(cur_cage.loc[cur_cage.index.isin(females),'stage_age']+1))
                    dams = np.random.choice(females, nmating, p=np.array(p_dams/sum(p_dams)).tolist(), replace=False)
                else:
                    sires = []
                    dams = []
                cur_cage.loc[cur_cage.index.isin(dams),'avail'] = 1
                cur_cage.loc[cur_cage.index.isin(sires),'avail'] = 1
                #Add phenotype of sire to dam info
                cur_cage.loc[cur_cage.index.isin(dams),'mate_resistanceT1'] = \
                cur_cage.loc[cur_cage.index.isin(sires),'resistanceT1'].values
                

                #create offspring
                bv_lst = []
                for i in dams:
                    for j in range(eggspday[cur_date.month-1]*tau):
                        underlying = 6 + 0.5*np.log(cur_cage.resistanceT1[cur_cage.index==i])\
                                   + 0.5*np.log(cur_cage.mate_resistanceT1[cur_cage.index==i].tolist()) + \
                                   ranTruncNorm(mean=0, sd=2, low=-100000000, upp=0, length=1)/sqrt(2)
                        bv_lst.extend(np.exp(underlying))  
                new_offs = len(bv_lst)
                num = 0
                for f in range(1,nfarms+1):
                    arrivals = np.random.poisson(prop_arrive[farm-1][f-1]*new_offs)
                    if arrivals>0:
                        num = num + 1
                        offs = pd.DataFrame(columns=offspring.columns)
                        offs['MF'] = np.random.choice(['F','M'], arrivals)
                        offs['Farm'] = f
                        offs['Cage'] = np.random.choice(range(1,ncages[farm-1]+1), arrivals)
                        offs['stage'] = np.repeat(2, arrivals)
                        offs['stage_age'] = np.repeat(-4, arrivals)
                        if len(bv_lst)>0:
                            ran_bvs = np.random.choice(len(bv_lst),arrivals,replace=False)
                            offs['resistanceT1'] = [bv_lst[i] for i in ran_bvs]
                            for i in sorted(ran_bvs, reverse=True):
                                del bv_lst[i]
                        else:
                            randams = np.random.choice(dams,arrivals)
                            for i in randams:
                                underlying = 6 + 0.5*np.log(cur_cage.resistanceT1[cur_cage.index==i])\
                                   + 0.5*np.log(cur_cage.mate_resistanceT1[cur_cage.index==i].tolist()) + \
                                   ranTruncNorm(mean=0, sd=2, low=-100000000, upp=0, length=1)/sqrt(2)
                                bv_lst.extend(np.exp(underlying))
                            offs['resistanceT1'] = [bv_lst[i] for i in range(len(arrivals))]                        
                        Earrival = [hrs_travel[farm-1][i-1]/24 for i in offs.Farm]
                        offs['arrival'] = np.random.poisson(Earrival)
                        offs['date'] = cur_date   
                        offspring = offspring.append(offs, ignore_index=True, sort=False)                                             
            
                
                #Update cage info------------------------------------------------------------------
                #----------------------------------------------------------------------------------
                cur_cage.loc[cur_cage.index.isin(L3toL4_inds),'stage'] = 4
                cur_cage.loc[cur_cage.index.isin(L3toL4_inds),'stage_age'] = 0
                cur_cage.loc[cur_cage.index.isin(L4toL5_inds) & (cur_cage.MF=='F'),'stage'] = 5
                cur_cage.loc[cur_cage.index.isin(L4toL5_inds) & (cur_cage.MF=='M'),'stage'] = 6
                cur_cage.loc[cur_cage.index.isin(L4toL5_inds),'stage_age'] = 0
                
                cur_cage.loc[cur_cage.index.isin(inf_inds), 'stage'] = 3
                cur_cage.loc[cur_cage.index.isin(inf_inds),'stage_age'] = 0
                cur_cage.loc[cur_cage.index.isin(inf_inds),'Fish'] = \
                          np.random.choice(all_fish['f'+str(farm)+'c'+str(cage)], len(inf_inds))                     
                
                                     
                lice.loc[(lice['Farm']==farm) & (lice['Cage']==cage)] = cur_cage.copy()
                                       
                #remove dead individuals
                lice = lice.drop(mort_inds)
                lice = lice.drop(lice.loc[lice.Fish.isin(dead_fish)].index)
 
file.close()       