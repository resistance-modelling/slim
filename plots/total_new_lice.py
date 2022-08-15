import os
import pyarrow.parquet as pq
import pickle
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


linestyles = ["solid", "dashdot", "dashed", "dashdot", "solid", "dotted", "dashed", "dashdot"]
colors = ["tomato", "mediumseagreen", "orange", "darkmagenta", "olive", "orange"]
linewidth = 1.7
hatchs = ["//", "OO", "XX", "\\", "oo", "//", "OO", "XX", "\\"]

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

def getting_bounds_new_lice(lice_geno_counts):
    y = []
    y_u = []
    y_l= []
    for lice_geno in range(3):
        y.append([])
        y_u.append([])
        y_l.append([])
        for day in range(730):
            y[lice_geno].append(round(np.mean(lice_geno_counts[:, day::730, lice_geno].flatten()), 2))
            y_u[lice_geno].append(round(np.quantile(lice_geno_counts[:, day::730, lice_geno].flatten(), 0.95), 2))
            y_l[lice_geno].append(round(np.quantile(lice_geno_counts[:, day::730, lice_geno].flatten(), 0.05), 2))
    return y, y_u, y_l

def new_reservoir_lice(pqt_list):
    l = []
    for i, pqt_name in enumerate(pqt_list):
        try:
            table = pq.read_table("./" + pqt_name).to_pandas()
            if table.loc[table.farm_name == "farm_0"].shape[0] == 730:
                l.append(np.array(list(map(np.array, list(table.sort_values(['farm_name', 'timestamp']).apply(lambda row: (row.new_reservoir_lice_ratios["A"], row.new_reservoir_lice_ratios["Aa"], row.new_reservoir_lice_ratios["a"]), axis=1))))))
        except:
            print(pqt_name)
    x = table.timestamp.unique()
    return np.array(l), x

for farm_file_name in farm_file_names:
    print(farm_file_name)
    pqt_list, pkl_name = load_data(farm_file_name, 1000)
    lice_stages_counts, x = new_reservoir_lice(pqt_list)
    y, y_u, y_l = getting_bounds_new_lice(lice_stages_counts)

    with open("./new_lice_temp/" + farm_file_name + '.pkl', 'wb') as f:
        pickle.dump(lice_stages_counts, f)

    # creating the plot
    fig, ax = plt.subplots(figsize=(13, 4), constrained_layout=True)

    labels = ["A", "aA", "a"]
    for i in range(3):
        ax.plot(x, y[i], label = labels[i], linestyle=linestyles[i], color=colors[i], linewidth=linewidth)
        ax.fill_between(x, y_l[i], y_u[i], alpha=0.3, edgecolor=colors[i], facecolor=colors[i], label="CI "+labels[i], hatch=hatchs[i])


    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.set_xlim(x[0], x[-1])
    ax.grid(axis='y')
    ax.set_ylabel(r'New reservoir lice', fontsize=13)

    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    plt.legend()

    plt.title("New reservoir lice - " + farm_file_name.replace("_", " "), fontsize=18)
    plt.savefig('./new_plots/total_new_reservoir_lice' + farm_file_name + '.pdf')

    plt.close()
