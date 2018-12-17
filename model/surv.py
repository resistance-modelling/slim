#Survival analysis of data from Finstad et al 2000

from math import log
import pandas as pd
import numpy as np
import lifelines as ll
import scipy.stats as ss

np.random.seed(545836870)

def integer_norm(x,mu,st_dev,length):
    xU, xL = x + 0.5, x - 0.5 
    prob = ss.norm.cdf(xU, loc=mu, scale=st_dev) - ss.norm.cdf(xL, loc=mu, scale=st_dev)
    prob = prob / prob.sum() 
    return np.random.choice(x, size = length, p = prob)

fish = pd.DataFrame(columns=['time','lice','death'])

end_sample = np.random.choice(range(33,41),9)
fish['time'] = [18,25,28,28,28,30,31,*end_sample,*np.repeat(40,184)]
fish['death'] = [*np.repeat(1,16),*np.repeat(0,184)]
lice28 = integer_norm(np.arange(24,85),54,9.7,3)
lice30 = integer_norm(np.arange(6,67),36,9.9,2)
lice33 = integer_norm(np.arange(0,47),22,8.1,9)
lice40 = integer_norm(np.arange(0,86),38,3,184)
fish['lice'] = [117,94,*lice28,*lice30,*lice33,*lice40]

fish_ll = fish.copy()
fish_ll['lice'] = [log(i) for i in fish_ll['lice']]

cph = CoxPHFitter()
cph.fit(fish,duration_col='time',event_col='death')
cph.print_summary()
#n=200, number of events=16
#
#       coef  exp(coef)  se(coef)      z      p  lower 0.95  upper 0.95   
#lice 0.0504     1.0517    0.0253 1.9905 0.0465      0.0008      0.1001  *
#---
#Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1 
#
#Concordance = 0.372
#Likelihood ratio test = 2.398 on 1 df, p=0.12151

print(np.mean(k_fold_cross_validation(cph, fish, duration_col='time', event_col='death')))
#0.2965964912280702

cph.fit(fish_ll,duration_col='time',event_col='death')
cph.print_summary()
#n=200, number of events=16
#
#        coef  exp(coef)  se(coef)       z      p  lower 0.95  upper 0.95     
#lice -2.4334     0.0877    0.6418 -3.7919 0.0001     -3.6912     -1.1756  ***
#---
#Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1 
#
#Concordance = 0.628
#Likelihood ratio test = 8.976 on 1 df, p=0.00274

