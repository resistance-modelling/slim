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


tlist_1 = np.array([],dtype=float)
for reps in range(30):
    day=0
    t0=time.time()
    while day<100:
        day = day + 1
                
    k = -1
    for f in range(1,11):
        for c in range(1,11):
            k = k + 1
            df_list2[k] = df_list2[k].append(big[(big.Farm==f) & (big.Cage==c)].copy(), ignore_index=True)
    
    tlist_1 = np.append(tlist_1,time.time()-t0)
print(tlist_1.mean(), tlist_1.std())
#0.14232006867726643 0.005889815131302248

big2 = big.copy()
tlist_2 = np.array([],dtype=float)
for reps in range(30):
    day=0
    t0=time.time()
    while day<100:
        day = day + 1
                
    k = -1
    for f in range(1,11):
        for c in range(1,11):
            k = k + 1
            df_list2[k] = df_list2[k].append(big[(big.Farm==f) & (big.Cage==c)].copy(), ignore_index=True)
            big = big[(big.Farm!=f) & (big.Cage!=c)]
    
    tlist_2 = np.append(tlist_2,time.time()-t0)
    big = big2.copy()
print(tlist_2.mean(), tlist_2.std())
#0.2296535015106201 0.01745010327987059


tlist = np.array([],dtype=float)
for reps in range(30):   
    bv_lst = np.random.normal(0, 1, 60)
    dams = [6,4,7,9,2,1,3,8,5]
    
    t0=time.time()
    
    bv_lst = pd.DataFrame({'bv':bv_lst})
    num = 0
    for f in range(3):
        arrivals = np.random.poisson(10)
        if arrivals>0:
            num = num + 1
            if len(bv_lst.bv)<arrivals:
                randams = np.random.choice(dams,arrivals-len(bv_lst.bv))
                tlst = []
                for i in randams:
                    underlying = np.random.normal(0, 1, 1) 
                    tlst.extend(underlying) 
                tlst = pd.DataFrame({'bv':tlst})
                bv_lst = pd.concat([bv_lst,tlst],ignore_index=True)
                del tlst
            ran_bvs = np.random.choice(bv_lst.index,arrivals,replace=False)           
            bv_lst = bv_lst.drop(ran_bvs) 
    tlist = np.append(tlist,time.time()-t0)
print(tlist.mean(), tlist.std())
#0.002891993522644043 0.001395053321291246

tlist = np.array([],dtype=float)
for reps in range(30):
    bv_lst = np.random.normal(0, 1, 60).tolist()
    dams = [6,4,7,9,2,1,3,8,5]
    
    t0=time.time()
    
    num = 0
    for f in range(3):
        arrivals = np.random.poisson(10)
        if arrivals>0:
            num = num + 1
            if len(bv_lst)>=arrivals:
                ran_bvs = np.random.choice(len(bv_lst),arrivals,replace=False)
            else:
                randams = np.random.choice(dams,arrivals-len(bv_lst))
                for i in randams:
                    underlying = np.random.normal(0, 1, 1) 
                    bv_lst.extend(underlying)  
                ran_bvs = np.random.choice(len(bv_lst),arrivals,replace=False)
            for i in sorted(ran_bvs, reverse=True):
                del bv_lst[i]
    tlist = np.append(tlist,time.time()-t0)
print(tlist.mean(), tlist.std())
#0.00013297398885091146 0.00033901873348128077


tlist = np.array([],dtype=float)
tlist2 = np.array([],dtype=float)
for reps in range(30):   
    bv_lst = np.random.normal(0, 1, 60)
    dams = [6,4,7,9,2,1,3,8,5]
    
    t0=time.time()
    
    bv_lst = pd.DataFrame({'bv':bv_lst})
    ran_bvs = np.random.choice(bv_lst.index,10,replace=False)
    t1 = time.time()     
    tlist2 = np.append(tlist2,t1-t0)      
    bv_lst = bv_lst.drop(ran_bvs) 
    tlist = np.append(tlist,time.time()-t1)
print(tlist2.mean(), tlist2.std())
#0.0008974472681681315 0.000743228438650053
print(tlist.mean(), tlist.std())
#0.0007311185201009115 0.0005121420017283775

tlist = np.array([],dtype=float)
tlist2 = np.array([],dtype=float)
for reps in range(30):
    bv_lst = np.random.normal(0, 1, 60).tolist()
    dams = [6,4,7,9,2,1,3,8,5]
    
    t0=time.time()
    
    ran_bvs = np.random.choice(len(bv_lst),10,replace=False)
    t1 = time.time()     
    tlist2 = np.append(tlist2,t1-t0)
    for i in sorted(ran_bvs, reverse=True):
        del bv_lst[i]
    tlist = np.append(tlist,time.time()-t1)
print(tlist2.mean(), tlist2.std())
#0.00019945303599039715 0.000398943929199771
print(tlist.mean(), tlist.std())
#0.0 0.0


tlist = np.array([],dtype=float)
for reps in range(1000):    
    t0=time.time()
    
    bv_lst = []
    for i in dams:
        for j in range(100):
            if reps==0:
                r = np.random.uniform(0,1,1)
                if r>0.3:
                    underlying = np.random.normal(0, 1, 1)/np.sqrt(2)
                else:
                    underlying = np.random.normal(0,1,1)
            else:
                underlying = np.random.normal(0, 1, 1)/np.sqrt(2)
            bv_lst.extend(underlying)  
    tlist = np.append(tlist,time.time()-t0)
print(tlist.mean(), tlist.std())
#0.006316382646560669 0.0017018039112977343


tlist = np.array([],dtype=float)
for reps in range(1000):    
    t0=time.time()
    
    bv_lst = []
    for i in dams:
        if reps==0:
            r = np.random.uniform(0,1,1)
            if r>0.3:
                underlying = np.random.normal(0, 1, 100)/np.sqrt(2)
            else:
                underlying = np.random.normal(0,1,100)
        else:
            underlying = np.random.normal(0, 1, 100)/np.sqrt(2)
        bv_lst.extend(underlying)  
    tlist = np.append(tlist,time.time()-t0)
print(tlist.mean(), tlist.std())
#0.0001770961284637451 0.0007805228315077014

tdf = pd.DataFrame({'bv':np.random.normal(0, 1, 100)})
tlist = np.array([],dtype=float)
for reps in range(100): 
    dams = [6,4,7,9,2,1,3,8,5]
    
    t0=time.time()
    
    bv_lst = []
    for i in dams:
        for j in range(100):
            if reps==0:
                r = np.random.uniform(0,1,1)
                if r>0.3:
                    underlying = 0.5*tdf.bv[tdf.index==i]\
                                   + 0.5*tdf.bv[tdf.index==i]+ \
                                   np.random.normal(0, 1, 1)/np.sqrt(2)
                else:
                    underlying = np.random.normal(0,1,1)
            else:
                underlying = 0.5*tdf.bv[tdf.index==i]\
                                   + 0.5*tdf.bv[tdf.index==i]+ \
                                   np.random.normal(0, 1, 1)/np.sqrt(2)
            bv_lst.extend(underlying)  
    tlist = np.append(tlist,time.time()-t0)
print(tlist.mean(), tlist.std())
#1.0146438813209533 0.09019463734741422

tdf = pd.DataFrame({'bv':np.random.normal(0, 1, 100)})
tlist = np.array([],dtype=float)
for reps in range(100): 
    dams = [6,4,7,9,2,1,3,8,5]
    
    t0=time.time()
    
    bv_lst = []
    for i in dams:
        if reps==0:
            r = np.random.uniform(0,1,1)
            if r>0.3:
                underlying = 0.5*tdf[tdf.index==i].bv.values\
                               + 0.5*tdf[tdf==i].bv.values + \
                               np.random.normal(0, 1, 100)/np.sqrt(2)
            else:
                underlying = np.random.normal(0,1,100)
        else:
            underlying = 0.5*tdf[tdf.index==i].bv.values\
                               + 0.5*tdf[tdf.index==i].bv.values + \
                               np.random.normal(0, 1, 100)/np.sqrt(2)
        bv_lst.extend(underlying.tolist())  
    tlist = np.append(tlist,time.time()-t0)
print(tlist.mean(), tlist.std())
#0.0057443976402282715 0.0022452927174215903

tdf = pd.DataFrame({'bv':np.random.normal(0, 1, 100)})
tlist = np.array([],dtype=float)
for reps in range(100): 
    dams = [6,4,7,9,2,1,3,8,5]
    
    t0=time.time()
    
    bv_lst = []
    for i in dams:
        if reps==0:
            r = np.random.uniform(0,1,1)
            if r>0.3:
                underlying = 0.5*tdf.loc[tdf.index==i,'bv'].values\
                               + 0.5*tdf.loc[tdf.index==i,'bv'].values + \
                               np.random.normal(0, 1, 100)/np.sqrt(2)
            else:
                underlying = np.random.normal(0,1,100)
        else:
            underlying = 0.5*tdf.loc[tdf.index==i,'bv'].values\
                               + 0.5*tdf.loc[tdf.index==i,'bv'].values + \
                               np.random.normal(0, 1, 100)/np.sqrt(2)
        bv_lst.extend(underlying.tolist())  
    tlist = np.append(tlist,time.time()-t0)
print(tlist.mean(), tlist.std())
#0.002562863826751709 0.0006188030162264759