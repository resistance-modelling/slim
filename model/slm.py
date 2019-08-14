# #!/usr/bin/python
# #

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
import numpy as np
import pandas as pd
from scipy import stats
import datetime as dt
from math import exp, log, sqrt, ceil
import time
import os
import sys

v_path = sys.argv[2]
file_path = "./outputs" + v_path

sys.path.append(file_path)

file_in = sys.argv[1]
inpt=__import__(file_in, globals(), locals(), ['*'])

v_file = str(sys.argv[3])

#np.random.seed(545836870)

#Functions-------------------------------------------------------------------------------------

#Update resistance distribution and sample from it 
def resistEMB(prp_ext, frms_muEMB, frms_sigEMB, length=1): 
    EMB_out = []
    for i in range(length):
        r = np.random.uniform(0,1,1)
        if r<prp_ext:
            EMB_out.extend([np.random.normal(inpt.f_muEMB, inpt.f_sigEMB)])        
        else:
            EMB_out.extend([np.random.normal(frms_muEMB, frms_sigEMB)])
    return EMB_out
                               
#Fish background mortality rate, decreasing as in Soares et al 2011                           
def fb_mort(jours):
    return 0.00057#(1000 + (jours - 700)**2)/490000000 

#prob of developping after D days in a stage as given by Aldrin et al 2017
def devTimeAldrin(del_p, del_m10, del_s, temp_c, D):   
    unbounded = log(2)*del_s*D**(del_s-1)*(del_m10*10**del_p/temp_c**del_p)**(-del_s)
    if unbounded==0:
        unbounded = 10**(-30)
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


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------


#Input Data------------------------------------------------------------------------------------
EMBmort = 0.9

hrs_travel = inpt.E_days*24
prop_arrive = inpt.prob_arrive


mort_rates = np.array([0.17, 0.22, 0.008, 0.05, 0.02, 0.06]) # L1,L2,L3,L4,L5f,L5m
eggspday = inpt.eggspday #50#[50,50,40,40,50,60,80,80,80,80,70,50]
d_hatching = inpt.d_hatching #8#[9,10,11,9,8,6,4,4,4,4,5,7]            

tau = 1 ###############################################

#Initial Values--------------------------------------------------------------------------------
cur_date = inpt.start_date

fcID = ['f'+str(i)+'c'+str(j) for i in range(1,inpt.nfarms+1) for j in range(1,inpt.ncages[i-1]+1)]
fsh = [list(range(1,inpt.fishf[i]+1)) for i in range(inpt.nfarms) for j in range(inpt.ncages[i])]
all_fish = dict(zip(fcID,fsh))
    
#Lice population in farms
df_list = [pd.DataFrame(columns=['Farm','Cage','Fish','MF','stage','stage_age','avail',\
                             'mate_resistanceT1','resistanceT1','date','arrival']) \
                              for i in range(inpt.nfarms) for j in range(inpt.ncages[i])]
offspring = pd.DataFrame(columns=df_list[0].columns)

offs_len = 0
env_sigEMB = 1.0
farms_muEMB = [inpt.f_muEMB]*inpt.nfarms
farms_sigEMB = [inpt.f_sigEMB]*inpt.nfarms
prev_muEMB = farms_muEMB.copy()
prev_sigEMB = farms_sigEMB.copy()
prop_ext = 1
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#--------------------------Simulation----------------------------------------------------------
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
directory = os.path.dirname(file_path)
try:
    if not os.path.exists(directory):
        os.makedirs(directory)
except OSError:
    print('Error creating directory')
file1 = open(file_path + 'lice_counts' + v_file + '.txt','a+')
file2 = open(file_path + 'resistanceBVs' + v_file + '.csv','a+') 
print('cur_date', 'muEMB', 'sigEMB', 'prop_ext', file=file2, sep=',', flush=True)
#prev_time = time.time() 
while cur_date <= inpt.end_date: 
    
    cur_date = cur_date + dt.timedelta(days=tau)
    t = (cur_date - inpt.start_date).days
    
    # if (t%7)==0:
        # print((time.time()-prev_time)/60, ' date ' + str(cur_date), file=file, flush=True)
        # prev_time = time.time()
        
          
    #add new offspring to cages
    k = -1
    for f in range(1,inpt.nfarms+1):
        for c in range(1,inpt.ncages[f-1]+1):
            k = k + 1
            df_list[k] = df_list[k].append(offspring[(offspring.Farm==f) & (offspring.Cage==c)].copy(), ignore_index=True)
            offs_len = offs_len + len(offspring.index)
    offspring = pd.DataFrame(columns=df_list[0].columns) 

    if (t%35)==0:
        prop_ext = (sum(inpt.ncages)*inpt.ext_pressure)/(sum(inpt.ncages)*inpt.ext_pressure + offs_len/35)
        offs_len = 0        
            
    #------------------------------------------------------------------------------------------
    #Events during tau in cage-----------------------------------------------------------------
    #------------------------------------------------------------------------------------------
    fc = -1
    for farm in range(1, inpt.nfarms+1):
        if cur_date.day==1:
            femaleAL = np.array([],dtype=float)
        
        #Estimate distribution params for external pressure that originated from farms
        if (t%35)==0:
            farms_muEMB[farm-1] = prev_muEMB[farm-1]
            farms_sigEMB[farm-1] = prev_sigEMB[farm-1]
            resistanceT1 = []
            if farm==1:
                for i in range(inpt.ncages[farm-1]):
                    resistanceT1.extend(df_list[i].resistanceT1)
            else:
                for i in range(sum(inpt.ncages[0:farm-1])-1,sum(inpt.ncages[0:farm])):
                    resistanceT1.extend(df_list[i].resistanceT1)
            prev_muEMB[farm-1] = np.array(resistanceT1).mean()
            prev_sigEMB[farm-1] = np.array(resistanceT1).std()
            print(cur_date, prev_muEMB[farm-1], prev_sigEMB[farm-1], prop_ext, file=file2, sep=',', flush=True)
        
        for cage in range(1, inpt.ncages[farm-1]+1):
            NSbool = eval(inpt.NSbool_str)
            if NSbool==True: 
            
                fc = fc + 1
                                
                if cur_date.day==1:
                    femaleAL = np.append(femaleAL, sum((df_list[fc].MF=='F') & (df_list[fc].stage==5))/len(all_fish['f'+str(farm)+'c'+str(cage)]))
                                                              
                if not df_list[fc].empty:
                    df_list[fc].date = cur_date
                    df_list[fc].stage_age = df_list[fc].stage_age + tau
                    df_list[fc].arrival = df_list[fc].arrival - tau
                    df_list[fc].loc[df_list[fc].avail>0, 'avail'] = df_list[fc].avail + tau
                    df_list[fc].loc[(df_list[fc].MF=='M') & (df_list[fc].avail>4), 'avail'] = 0
                    df_list[fc].loc[(df_list[fc].MF=='F') & (df_list[fc].avail>d_hatching), 'avail'] = 0
                    df_list[fc].loc[(df_list[fc].MF=='F') & (df_list[fc].avail>d_hatching), 'mate_resistanceT1'] = None
        
                #new planktonic stages arriving from wildlife reservoir
                nplankt = inpt.ext_pressure*tau
                plankt_cage = pd.DataFrame(columns=df_list[fc].columns)
                plankt_cage['MF'] = np.random.choice(['F','M'],nplankt)
                plankt_cage['stage'] = 2 
                plankt_cage['Farm'] = farm
                plankt_cage['Cage'] = cage
                p=stats.poisson.pmf(range(15),3)
                p = p/np.sum(p) #probs need to add up to one 
                plankt_cage['stage_age'] = np.random.choice(range(15),nplankt,p=p)
                plankt_cage['avail'] = 0
                plankt_cage['resistanceT1'] = resistEMB(prop_ext, farms_muEMB[farm-1], farms_sigEMB[farm-1], nplankt)
                plankt_cage['date'] = cur_date
                plankt_cage['avail'] = 0
                plankt_cage['arrival'] = 0
                           
                df_list[fc] = df_list[fc].append(plankt_cage, ignore_index=True)
                dead_fish = set([])
                
                #Background mortality events-------------------------------------------------------
                inds_stage = np.array([sum(df_list[fc]['stage']==i) for i in range(1,7)])
                Emort = np.multiply(mort_rates, inds_stage)
                mort_ents = np.random.poisson(Emort)
                mort_ents=[min(mort_ents[i],inds_stage[i]) for i in range(len(mort_ents))]
                mort_inds = []
                for i in range(1,7):
                    df = df_list[fc].loc[df_list[fc].stage==i].copy()
                    if not df.empty:
                        if np.sum(df.stage_age)>0:
                            p = (df.stage_age+1)/np.sum(df.stage_age+1)
                            p = pd.to_numeric(p)
                            values = np.random.choice(df.index, mort_ents[i-1], p=p, replace=False).tolist()
                        else:
                            values = np.random.choice(df.index, mort_ents[i-1], replace=False).tolist()                    
                        mort_inds.extend(values)
                        
                #Treatment mortality events------------------------------------------------------
                EMBsus = [1 if df_list[fc].stage[i]>2 else 0 for i in range(len(df_list[fc].index))]
                start_treat = cur_date - dt.timedelta(days=14)
                if start_treat in inpt.dates_list[farm-1]:
                    phenoEMB = df_list[fc].resistanceT1 + np.random.normal(0,env_sigEMB,len(df_list[fc].resistanceT1)) #add environmental deviation
                    phenoEMB = 1/(1 + np.exp(phenoEMB))  #1-resistance
                    phenoEMB =  phenoEMB*EMBsus #remove stages that aren't susceptible to EMB
                    ETmort = sum(phenoEMB)*EMBmort 
                    Tmort_ents = np.random.poisson(ETmort)
                    Tmort_ents = min(Tmort_ents,len(df_list[fc].resistanceT1))
                    p = (phenoEMB)/np.sum(phenoEMB)
                    mort_inds.extend(np.random.choice(df_list[fc].index, Tmort_ents, p=p, replace=False).tolist())
                    mort_inds = list(set(mort_inds))

                #Development events----------------------------------------------------------------
                temp_now = inpt.temp_f(cur_date.month, inpt.xy_array[farm-1][1])
                
                if inds_stage[0]>0:
                    L1toL2 = [devTimeAldrin(0.401,8.814,18.869,temp_now, i)
                             for i in df_list[fc].loc[df_list[fc].stage==1,'stage_age']]
                    L1toL2_ents = np.random.poisson(sum(L1toL2))
                    L1toL2_ents = min(L1toL2_ents, inds_stage[0])
                    L1toL2_inds = np.random.choice(df_list[fc].loc[df_list[fc].stage==1].index, \
                                                   L1toL2_ents, p=L1toL2/np.sum(L1toL2), replace=False)
                else:
                    L1toL2_inds = []
                                        
                if inds_stage[2]>0:
                    L3toL4 = [devTimeAldrin(1.305,18.934,7.945,temp_now,i) 
                             for i in df_list[fc].loc[df_list[fc]['stage']==3].stage_age]
                    L3toL4_ents = np.random.poisson(sum(L3toL4))
                    L3toL4_ents = min(L3toL4_ents, inds_stage[2])
                    L3toL4_inds = np.random.choice(df_list[fc].loc[df_list[fc]['stage']==3].index, \
                                                   L3toL4_ents, p=L3toL4/np.sum(L3toL4), replace=False)
                else:
                    L3toL4_inds = []
                
                if inds_stage[3]>0:
                    L4toL5 = [devTimeAldrin(0.866,10.742,1.643,temp_now,i) 
                             for i in df_list[fc].loc[df_list[fc]['stage']==4].stage_age]
                    L4toL5_ents = np.random.poisson(sum(L4toL5))
                    L4toL5_ents = min(L4toL5_ents, inds_stage[3])
                    L4toL5_inds = np.random.choice(df_list[fc].loc[df_list[fc]['stage']==4].index, \
                                                   L4toL5_ents, p=L4toL5/np.sum(L4toL5), replace=False)
                else:
                    L4toL5_inds = []
                
                #Fish growth and death-------------------------------------------------------------
                wt = 10000/(1+exp(-0.01*(t-475)))
                fish_fc = np.array(df_list[fc][df_list[fc].stage>3].Fish.unique().tolist()) #fish with lice
                fish_fc = fish_fc[~np.isnan(fish_fc)]
                adlicepg = np.array(df_list[fc][df_list[fc].stage>3].groupby('Fish').stage.count())/wt
                Plideath = 1/(1+np.exp(-19*(adlicepg-0.63)))
                nfish = len(all_fish['f'+str(farm)+'c'+str(cage)])
                Ebf_death = fb_mort(t)*tau*(nfish)
                Elf_death = np.sum(Plideath)*tau
                bfd_ents = np.random.poisson(Ebf_death)
                lfd_ents = np.random.poisson(Elf_death) 
                           
                if fish_fc.size>0:
                    dead_fish.update(np.random.choice(fish_fc, 
                              lfd_ents, p=Plideath/np.sum(Plideath), replace=False).tolist())
                if len(dead_fish)>0:
                    all_fish['f'+str(farm)+'c'+str(cage)] = [i for i in all_fish['f'+str(farm)+'c'+str(cage)] if i not in dead_fish]
                dead_fish.update(np.random.choice(all_fish['f'+str(farm)+'c'+str(cage)], bfd_ents, replace=False).tolist())
                if len(dead_fish)>0:
                    all_fish['f'+str(farm)+'c'+str(cage)] = [i for i in all_fish['f'+str(farm)+'c'+str(cage)] if i not in dead_fish]
                
                #Infection events------------------------------------------------------------------
                cop_cage = sum((df_list[fc].stage==2) & (df_list[fc].arrival<=df_list[fc].stage_age))
                if cop_cage>0:
                    eta_aldrin = -2.576 + log(nfish) + 0.082*(log(wt)-0.55)
                    Einf = (exp(eta_aldrin)/(1+exp(eta_aldrin)))*tau*cop_cage
                    inf_ents = np.random.poisson(Einf)
                    inf_ents = min(inf_ents,cop_cage)
                    inf_inds = np.random.choice(df_list[fc].loc[(df_list[fc].stage==2) & (df_list[fc].arrival<=df_list[fc].stage_age)].index, inf_ents, replace=False)
                else:
                    inf_inds = []
                
                #Mating events---------------------------------------------------------------------
                
                #who is mating               
                females = df_list[fc].loc[(df_list[fc].stage==5) & (df_list[fc].avail==0)].index
                males = df_list[fc].loc[(df_list[fc].stage==6) & (df_list[fc].avail==0)].index
                nmating = min(sum(df_list[fc].index.isin(females)),\
                          sum(df_list[fc].index.isin(males)))
                if nmating>0:
                    sires = np.random.choice(males, nmating, replace=False)
                    p_dams = 1 - (df_list[fc].loc[df_list[fc].index.isin(females),'stage_age']/
                                np.sum(df_list[fc].loc[df_list[fc].index.isin(females),'stage_age']+1))
                    dams = np.random.choice(females, nmating, p=np.array(p_dams/np.sum(p_dams)).tolist(), replace=False)
                else:
                    sires = []
                    dams = []
                df_list[fc].loc[df_list[fc].index.isin(dams),'avail'] = 1
                df_list[fc].loc[df_list[fc].index.isin(sires),'avail'] = 1
                #Add genotype of sire to dam info
                df_list[fc].loc[df_list[fc].index.isin(dams),'mate_resistanceT1'] = \
                df_list[fc].loc[df_list[fc].index.isin(sires),'resistanceT1'].values
                

                #create offspring
                bv_lst = []
                for i in dams:
                    for j in range(eggspday*tau):
                        underlying = 0.5*df_list[fc].resistanceT1[df_list[fc].index==i]\
                                   + 0.5*df_list[fc].mate_resistanceT1[df_list[fc].index==i]+ \
                                   np.random.normal(0, farms_sigEMB[farm-1], 1)/np.sqrt(2)
                        bv_lst.extend(underlying)  
                new_offs = len(bv_lst)
                num = 0
                for f in range(1,inpt.nfarms+1):
                    arrivals = np.random.poisson(prop_arrive[farm-1][f-1]*new_offs)
                    if arrivals>0:
                        num = num + 1
                        offs = pd.DataFrame(columns=df_list[fc].columns)
                        offs['MF'] = np.random.choice(['F','M'], arrivals)
                        offs['Farm'] = f
                        offs['Cage'] = np.random.choice(range(1,inpt.ncages[farm-1]+1), arrivals)
                        offs['stage'] = np.repeat(1, arrivals)
                        offs['stage_age'] = np.repeat(0, arrivals)
                        if len(bv_lst)>0:
                            ran_bvs = np.random.choice(len(bv_lst),arrivals,replace=False)
                            offs['resistanceT1'] = [bv_lst[i] for i in ran_bvs]
                            for i in sorted(ran_bvs, reverse=True):
                                del bv_lst[i]
                        else:
                            randams = np.random.choice(dams,arrivals)
                            for i in randams:
                                underlying = 0.5*df_list[fc].resistanceT1[df_list[fc].index==i]\
                                   + 0.5*df_list[fc].mate_resistanceT1[df_list[fc].index==i]+ \
                                   np.random.normal(0, farms_sigEMB[farm-1], 1)/np.sqrt(2)
                                bv_lst.extend(underlying)  
                            offs['resistanceT1'] = [bv_lst[i] for i in range(len(arrivals))]                        
                        Earrival = [hrs_travel[farm-1][i-1]/24 for i in offs.Farm]
                        offs['arrival'] = np.random.poisson(Earrival)
                        offs['avail'] = 0
                        offs['date'] = cur_date
                        offspring = offspring.append(offs, ignore_index=True)
            
                
                #Update cage info------------------------------------------------------------------
                #----------------------------------------------------------------------------------
                df_list[fc].loc[df_list[fc].index.isin(L1toL2_inds),'stage'] = 2
                df_list[fc].loc[df_list[fc].index.isin(L1toL2_inds),'stage_age'] = 0 
                df_list[fc].loc[df_list[fc].index.isin(L3toL4_inds),'stage'] = 4
                df_list[fc].loc[df_list[fc].index.isin(L3toL4_inds),'stage_age'] = 0
                df_list[fc].loc[df_list[fc].index.isin(L4toL5_inds) & (df_list[fc].MF=='F'),'stage'] = 5
                df_list[fc].loc[df_list[fc].index.isin(L4toL5_inds) & (df_list[fc].MF=='M'),'stage'] = 6
                df_list[fc].loc[df_list[fc].index.isin(L4toL5_inds),'stage_age'] = 0
                
                df_list[fc].loc[df_list[fc].index.isin(inf_inds), 'stage'] = 3
                df_list[fc].loc[df_list[fc].index.isin(inf_inds),'stage_age'] = 0
                df_list[fc].loc[df_list[fc].index.isin(inf_inds),'Fish'] = \
                          np.random.choice(all_fish['f'+str(farm)+'c'+str(cage)], len(inf_inds))             

                          
                #remove dead individuals
                df_list[fc] = df_list[fc].drop(mort_inds)
                df_list[fc] = df_list[fc].drop(df_list[fc].loc[df_list[fc].Fish.isin(dead_fish)].index)
                
                               
                #df_list[fc].to_csv(file_path + 'lice_df.csv', mode='a')
                
        if cur_date.day==1:
            print(cur_date, femaleAL.mean(), femaleAL.std(), file=file1, flush=True)
            femaleAL = np.array([],dtype=float)
                
file1.close()  
file2.close() 