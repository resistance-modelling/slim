def cage_func(cur_date,inpt,farm,cage,cage_df,tau,mort_rates,devTimeAldrin,all_fish,eggspday,femaleAL):    
    if cur_date.day==1:
        femaleAL = np.append(femaleAL, sum((cage_df.MF=='F') & (cage_df.stage==5))/inpt.fishf[farm-1])
                                                  
    if not cage_df.empty:
        cage_df.date = cur_date
        cage_df.stage_age = cage_df.stage_age + tau
        cage_df.arrival = cage_df.arrival - tau
        cage_df.loc[cage_df.avail>0, 'avail'] = cage_df.avail + tau
        cage_df.loc[(cage_df.MF=='M') & (cage_df.avail>4), 'avail'] = 0
        cage_df.loc[(cage_df.MF=='F') & (cage_df.avail>d_hatching[cur_date.month-1]), 'avail'] = 0
        cage_df.loc[(cage_df.MF=='F') & (cage_df.avail>d_hatching[cur_date.month-1]), 'mate_resistanceT1'] = None

    #new planktonic stages arriving from wildlife reservoir
    nplankt = inpt.ext_pressure*tau
    plankt_cage = pd.DataFrame(columns=cage_df.columns)
    plankt_cage['MF'] = np.random.choice(['F','M'],nplankt)
    plankt_cage['stage'] = 2 
    plankt_cage['Farm'] = farm
    plankt_cage['Cage'] = cage
    p=stats.poisson.pmf(range(15),3)
    p = p/np.sum(p) #probs need to add up to one 
    plankt_cage['stage_age'] = np.random.choice(range(15),nplankt,p=p)
    plankt_cage['avail'] = 0
    rands = ranTruncNorm(mean=-6, sd=2, low=-100000000, upp=0, length=nplankt) ########################################
    plankt_cage['resistanceT1'] = np.exp(rands)
    plankt_cage['date'] = cur_date
    plankt_cage['avail'] = 0
    plankt_cage['arrival'] = 0
               
    cage_df = cage_df.append(plankt_cage, ignore_index=True)
    dead_fish = set([])
    
    #Background mortality events-------------------------------------------------------
    inds_stage = np.array([sum(cage_df['stage']==i) for i in range(1,7)])
    Emort = np.multiply(mort_rates, inds_stage)
    mort_ents = np.random.poisson(Emort)
    mort_ents=[min(mort_ents[i],inds_stage[i]) for i in range(len(mort_ents))]
    mort_inds = []
    for i in range(1,7):
        df = cage_df.loc[cage_df.stage==i].copy()
        if not df.empty:
            if np.sum(df.stage_age)>0:
                p = (df.stage_age+1)/np.sum(df.stage_age+1)
                p = pd.to_numeric(p)
                values = np.random.choice(df.index, mort_ents[i-1], p=p, replace=False).tolist()
            else:
                values = np.random.choice(df.index, mort_ents[i-1], replace=False).tolist()                    
            mort_inds.extend(values)
                 

    #Development events----------------------------------------------------------------
    temp_now = inpt.temp_f(cur_date.month, inpt.xy_array[farm-1][1])
    
    if inds_stage[0]>0:
        L1toL2 = [devTimeAldrin(0.401,8.814,18.869,temp_now, i)
                 for i in cage_df.loc[cage_df.stage==1,'stage_age']]
        L1toL2_ents = np.random.poisson(sum(L1toL2))
        L1toL2_ents = min(L1toL2_ents, inds_stage[0])
        L1toL2_inds = np.random.choice(cage_df.loc[cage_df.stage==1].index, \
                                       L1toL2_ents, p=L1toL2/np.sum(L1toL2), replace=False)
    else:
        L1toL2_inds = []
                            
    if inds_stage[2]>0:
        L3toL4 = [devTimeAldrin(1.305,18.934,7.945,temp_now,i) 
                 for i in cage_df.loc[cage_df['stage']==3].stage_age]
        L3toL4_ents = np.random.poisson(sum(L3toL4))
        L3toL4_ents = min(L3toL4_ents, inds_stage[2])
        L3toL4_inds = np.random.choice(cage_df.loc[cage_df['stage']==3].index, \
                                       L3toL4_ents, p=L3toL4/np.sum(L3toL4), replace=False)
    else:
        L3toL4_inds = []
    
    if inds_stage[3]>0:
        L4toL5 = [devTimeAldrin(0.866,10.742,1.643,temp_now,i) 
                 for i in cage_df.loc[cage_df['stage']==4].stage_age]
        L4toL5_ents = np.random.poisson(sum(L4toL5))
        L4toL5_ents = min(L4toL5_ents, inds_stage[3])
        L4toL5_inds = np.random.choice(cage_df.loc[cage_df['stage']==4].index, \
                                       L4toL5_ents, p=L4toL5/np.sum(L4toL5), replace=False)
    else:
        L4toL5_inds = []
    
    #Fish growth and death-------------------------------------------------------------
    t = cur_date - inpt.start_date
    wt = 10000/(1+exp(-0.01*(t.days-475)))
    fish_fc = np.array(cage_df[cage_df.stage>3].Fish.unique().tolist()) #fish with lice
    fish_fc = fish_fc[~np.isnan(fish_fc)]
    adlicepg = np.array(cage_df[cage_df.stage>3].groupby('Fish').stage.count())/wt
    Plideath = 1/(1+np.exp(-19*(adlicepg-0.63)))
    nfish = len(all_fish['f'+str(farm)+'c'+str(cage)])
    Ebf_death = fb_mort(t.days)*tau*(nfish)
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
    cop_cage = sum((cage_df.stage==2) & (cage_df.arrival<=cage_df.stage_age))
    if cop_cage>0:
        eta_aldrin = -2.576 + log(nfish) + 0.082*(log(wt)-0.55)
        Einf = (exp(eta_aldrin)/(1+exp(eta_aldrin)))*tau*cop_cage
        inf_ents = np.random.poisson(Einf)
        inf_ents = min(inf_ents,cop_cage)
        inf_inds = np.random.choice(cage_df.loc[(cage_df.stage==2) & (cage_df.arrival<=cage_df.stage_age)].index, inf_ents, replace=False)
    else:
        inf_inds = []
    
    #Mating events---------------------------------------------------------------------
    
    #who is mating               
    females = cage_df.loc[(cage_df.stage==5) & (cage_df.avail==0)].index
    males = cage_df.loc[(cage_df.stage==6) & (cage_df.avail==0)].index
    nmating = min(sum(cage_df.index.isin(females)),\
              sum(cage_df.index.isin(males)))
    if nmating>0:
        sires = np.random.choice(males, nmating, replace=False)
        p_dams = 1 - (cage_df.loc[cage_df.index.isin(females),'stage_age']/
                    np.sum(cage_df.loc[cage_df.index.isin(females),'stage_age']+1))
        dams = np.random.choice(females, nmating, p=np.array(p_dams/np.sum(p_dams)).tolist(), replace=False)
    else:
        sires = []
        dams = []
    cage_df.loc[cage_df.index.isin(dams),'avail'] = 1
    cage_df.loc[cage_df.index.isin(sires),'avail'] = 1
    #Add phenotype of sire to dam info
    cage_df.loc[cage_df.index.isin(dams),'mate_resistanceT1'] = \
    cage_df.loc[cage_df.index.isin(sires),'resistanceT1'].values
    

    #create offspring
    bv_lst = []
    for i in dams:
        for j in range(eggspday[cur_date.month-1]*tau):
            underlying = 6 + 0.5*np.log(cage_df.resistanceT1[cage_df.index==i])\
                       + 0.5*np.log(cage_df.mate_resistanceT1[cage_df.index==i].tolist()) + \
                       ranTruncNorm(mean=0, sd=2, low=-100000000, upp=0, length=1)/sqrt(2)
            bv_lst.extend(np.exp(underlying))  
    new_offs = len(bv_lst)
    num = 0
    for f in range(1,inpt.nfarms+1):
        arrivals = np.random.poisson(prop_arrive[farm-1][f-1]*new_offs)
        if arrivals>0:
            num = num + 1
            offs = pd.DataFrame(columns=cage_df.columns)
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
                    underlying = 6 + 0.5*np.log(cage_df.resistanceT1[cage_df.index==i])\
                       + 0.5*np.log(cage_df.mate_resistanceT1[cage_df.index==i].tolist()) + \
                       ranTruncNorm(mean=0, sd=2, low=-100000000, upp=0, length=1)/sqrt(2)
                    bv_lst.extend(np.exp(underlying))
                offs['resistanceT1'] = [bv_lst[i] for i in range(len(arrivals))]                        
            Earrival = [hrs_travel[farm-1][i-1]/24 for i in offs.Farm]
            offs['arrival'] = np.random.poisson(Earrival)
            offs['avail'] = 0
            offs['date'] = cur_date
            offspring = offspring.append(offs, ignore_index=True)

    
    #Update cage info------------------------------------------------------------------
    #----------------------------------------------------------------------------------
    cage_df.loc[cage_df.index.isin(L1toL2_inds),'stage'] = 2
    cage_df.loc[cage_df.index.isin(L1toL2_inds),'stage_age'] = 0 
    cage_df.loc[cage_df.index.isin(L3toL4_inds),'stage'] = 4
    cage_df.loc[cage_df.index.isin(L3toL4_inds),'stage_age'] = 0
    cage_df.loc[cage_df.index.isin(L4toL5_inds) & (cage_df.MF=='F'),'stage'] = 5
    cage_df.loc[cage_df.index.isin(L4toL5_inds) & (cage_df.MF=='M'),'stage'] = 6
    cage_df.loc[cage_df.index.isin(L4toL5_inds),'stage_age'] = 0
    
    cage_df.loc[cage_df.index.isin(inf_inds), 'stage'] = 3
    cage_df.loc[cage_df.index.isin(inf_inds),'stage_age'] = 0
    cage_df.loc[cage_df.index.isin(inf_inds),'Fish'] = \
              np.random.choice(all_fish['f'+str(farm)+'c'+str(cage)], len(inf_inds))             

              
    #remove dead individuals
    cage_df = cage_df.drop(mort_inds)
    cage_df = cage_df.drop(cage_df.loc[cage_df.Fish.isin(dead_fish)].index)
    
                   
    cage_df.to_csv(file_path + 'lice_df.csv', mode='a')