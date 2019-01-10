import numpy as np
import pandas as pd
import time

tlist_big = np.array([],dtype=float)
big = pd.DataFrame({'Farm':np.random.randint(low=1, high=10, size=100), \
                    'Cage':np.random.randint(low=1, high=10, size=100), \
                    'age':np.random.randint(low=1, high=6, size=100), \
                    'cat':np.random.choice(['a','b'],100)})
df_list = [big[big.Farm==i].copy() for i in range(1,11)]
df_list2 = [big[(big.Farm==i) & (big.Cage==j)].copy() for i in range(1,11) for j in range(1,11)]


for reps in range(30):
    day=0
    t0=time.time()
    while day<100:
        day = day + 1
        big.age = big.age + 1
        big.loc[(big.cat=='a') & (big.age>6),'age'] = 0
        big.loc[(big.cat=='b') & (big.age>8),'age'] = 0
        
        for farm in range(1,11):
            for cage in range(1,11):
                cur_cage = big.loc[(big.Farm==farm) & (big.Cage==cage)].copy()
                big.loc[(big.Farm==farm) & (big.Cage==cage)] = cur_cage.copy()
    
    tlist_big = np.append(tlist_big,time.time()-t0)
print(tlist_big.mean(), tlist_big.std())
#59.028608139355974 1.4079436833950365


tlist_ll = np.array([],dtype=float)
for reps in range(30):
    day=0
    t0=time.time()
    while day<100:
        day = day + 1
                
        for farm in range(1,11):
            df_list[farm-1].age = df_list[farm-1].age + 1
            df_list[farm-1].loc[(df_list[farm-1].cat=='a') & (df_list[farm-1].age>6),'age'] = 0
            df_list[farm-1].loc[(df_list[farm-1].cat=='b') & (df_list[farm-1].age>8),'age'] = 0
            for cage in range(1,11):
                cur_cage = df_list[farm-1].loc[(df_list[farm-1].Cage==cage)].copy()
                df_list[farm-1].loc[(df_list[farm-1].Cage==cage)] = cur_cage.copy()
    
    tlist_ll = np.append(tlist_ll,time.time()-t0)
print(tlist_ll.mean(), tlist_ll.std())
#41.71184564431508 1.1861610438716288


tlist_ll2 = np.array([],dtype=float)
for reps in range(30):
    day=0
    t0=time.time()
    while day<100:
        day = day + 1
        
        c = 0        
        for farm in range(1,11):
            for cage in range(1,11):
                df_list2[c].age = df_list2[c].age + 1
                df_list2[c].loc[(df_list2[c].cat=='a') & (df_list2[c].age>6),'age'] = 0
                df_list2[c].loc[(df_list2[c].cat=='b') & (df_list2[c].age>8),'age'] = 0                
                c = c + 1
    
    tlist_ll2 = np.append(tlist_ll2,time.time()-t0)
print(tlist_ll2.mean(), tlist_ll2.std())
#34.30218040943146 0.5885443036185948