#!/usr/bin/python
#

#Input data: location of cages/farms 

#Sea lice lifecycle
#Lice values held in data frame (FILE?) with entries for: farm, cage, fish, MF, stage, stage-age, resistanceT1, resitanceT2,...
#Initialise by setting initial lice population for each cage and sample resistance values from distribution for natural occurance of resistance ***[INSERT APPROPRIATE DISTRIBUTION]*** and male/female=1.22 (cox et al 2017)


#For each cage-subset simultaneously:
#Calculate rate of events
#For each event sample number of times it happened during a fixed period tau=? ***(to be estimated)*** from Pois(rate*tau)
#Sample who they happen to and update value
##If event is infection sample fish ***density dependent???****
##If event is mating sample both and create new entries with status 0 and create resistance values based on parents ***[INSERT MODE OF INHERITANCE]***
### 6-11 broods of 500 ova on average (see Costello 2006) extrude a pair every 3 days ***Average number of eggs for all or distribution???****
#Update fish growth and death (remove all lice entries associated with dead fish) 
#Migrate lice in L1/2 to other cages/farms within range 11-45km mean 27km Siegal et al 2003 ***INSERT FUNCTION***

#Update all values, add seasonal inputs e.g. treatment, wildlife reservoir and run all cages again

#Lice events:
#infecting a fish (L2->L3) -> rate= ***[INSERT FUNCTION OF FISH WEIGHT BY AGE]**** to be used with function from Aldrin et al 2017
#L1->L2,L3->L4,L4->L5 -> rate=sum(prob of dev after A days in a stage as given by Aldrin et al 2017)
#Background death in a stage (remove entry) -> rate= number of individuals in stage*stage rate (nauplii 0.17/d, copepods 0.22/d, pre-adult female 0.05, pre-adult male ... Stien et al 2005)
#Treatment death in a stage (remove entry) -> rate= (1 - individual resistance for individuals in appropriate stage)*[TREATMENT EFFECT]*[DOSE PRESENCE/DECAY]****
#Mating -> rate fish_i= adult males on fish_i*adult females on fish_i*(1/4)*(2.5/ave adult female lifetime) 

#Fish model:
#Fish growth rate ****[INSERT FUNCTION]**** -> deterministic
#Fish death rate: sum of constant *****background rate=??**** and individual lice dependent term *****[INSERT APPROPRIATE DISTRIBUTION]**** -> sample with prob=rate_i/rate_tot
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np

np.random.seed(545836870)

def gender(rnum01):
	if rnum01 <= 0.45:
		return 'F'
	else:
		return 'M'
	;

#Initial lice population
lice = pd.DataFrame(columns=['Farm','Cage','Fish','MF','stage','stage_age','resistanceT1'])
lice['Farm'] = np.repeat(1,12)
lice['Cage'] = np.repeat(1,12)
lice['Fish'] = np.repeat(range(1,4),4)

rands = np.random.random(12)
lice['MF'] = [gender(i) for i in rands]

lice['stage'] = 

