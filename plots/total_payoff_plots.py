import os
import pyarrow.parquet as pq
import pickle
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

farm_file_names = ['bipath_regular_cleanerfish_60',
 'bipath_regular_cleanerfish_30',
 'bipath_bernoulli_0.8',
 'bipath_regular_emb_30',
 'bipath_regular_thermolicer_30',
 'bipath_bernoulli_1.0',
 'bipath_mosaic',
 'bipath_regular_emb_60',
 'bipath_bernoulli_0.2',
 'bipath_regular_thermolicer_60',
 'bipath_untreated', 
 'Fyne_complete_bernoulli_0.0',
 'Linnhe_complete_bernoulli_0.0'
 ]

def load_data(dir_name, num_of_trials):
    parquet_files = []
    for file in os.listdir("./" + dir_name):
        if file.endswith(".parquet"):
            parquet_files.append(os.path.join("./" + dir_name, file))
        else:
            if file.endswith(".pickle"):
                pickle_file = os.path.join("./" + dir_name, file)
    return parquet_files[:num_of_trials], pickle_file

for farm_file_name in farm_file_names:
    total_payoff = []
    pqt_list, pkl_name = load_data(farm_file_name, 1000)
    for pqt_name in pqt_list:
        try:
            table = pq.read_table("./" + pqt_name).to_pandas()
            if table.loc[table.farm_name == "farm_0"].shape[0] == 730:
                for farm_id_name in table.farm_name.unique():
                    total_payoff.append(sum(list(table.loc[table.farm_name == farm_id_name].payoff)))
        except:
            print(pqt_name)

    with open("./payoff_temp/" + farm_file_name + '.pkl', 'wb') as f:
        pickle.dump(total_payoff, f)

    # creating the plot
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)

    sns.violinplot(data = total_payoff)

    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)

    ax.set_ylabel(r'GBP', fontsize=13)

    plt.title('Cumulative payoff - ' + farm_file_name.replace("_", " "), fontsize=18)
    plt.savefig('./new_plots/total_cummulative_payoff_' + farm_file_name + '.pdf')
    plt.close()
