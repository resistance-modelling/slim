import os
import pyarrow.parquet as pq
import pickle
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

lice_stages_to_names = {
    "L3": "Chalimuses",
    "L4":"Pre-adults",
    "L5m":"Adult males",
    "L5f":"Adult females",
    "all":"Sum"
}

linestyles = ["solid", "dotted", "dashed", "dashdot", "solid", "dotted", "dashed", "dashdot"]
colors = ["tomato", "darkmagenta", "tomato", "mediumseagreen", "skyblue", "orange", "orange"]
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

def getting_bounds(lice_stages_counts):
    y = []
    y_u = []
    y_l= []
    for lice_stage in range(6):
        y.append([])
        y_u.append([])
        y_l.append([])
        for day in range(730):
            y[lice_stage].append(round(np.quantile(lice_stages_counts[:, day::730, lice_stage].flatten(), 0.5), 2))
            y_u[lice_stage].append(round(np.quantile(lice_stages_counts[:, day::730, lice_stage].flatten(), 0.95), 2))
            y_l[lice_stage].append(round(np.quantile(lice_stages_counts[:, day::730, lice_stage].flatten(), 0.05), 2))
    return y, y_u, y_l

def all_farms_all_trials_prep(pqt_list):
    l = []
    for i, pqt_name in enumerate(pqt_list):
        try:
            table = pq.read_table("./" + pqt_name).to_pandas()
            if table.loc[table.farm_name == "farm_0"].shape[0] == 730:
                l.append(np.array(list(map(np.array, list(table.sort_values(['farm_name', 'timestamp']).apply(lambda row: (sum(row["L1"].values()), sum(row["L2"].values()), sum(row["L3"].values()), sum(row["L4"].values()), sum(row["L5m"].values()), sum(row["L5f"].values())), axis=1))))))
        except:
            print(pqt_name)
    x = table.timestamp.unique()
    return np.array(l), x

for farm_file_name in farm_file_names:
    print(farm_file_name)
    pqt_list, pkl_name = load_data(farm_file_name, 1000)


    lice_stages_counts, x = all_farms_all_trials_prep(pqt_list)
    
    with open("./lice_count_temp/" + farm_file_name + '.pkl', 'wb') as f:
        pickle.dump(lice_stages_counts, f)

    y, y_u, y_l = getting_bounds(lice_stages_counts)

    include = [3, 4, 5]
    y_all = [sum([y[i][j] for i in include]) for j in range(730)]
    y_all_u = [sum([y_u[i][j] for i in include]) for j in range(730)]
    y_all_l = [sum([y_l[i][j] for i in include]) for j in range(730)]
    
    fig, ax = plt.subplots(figsize=(13, 4), constrained_layout=True)

    ax.plot(x, y_all, label = "Sum", linestyle="solid", color="tomato", linewidth=linewidth)
    ax.fill_between(x, y_all_l, y_all_u, alpha=0.15, edgecolor="tomato", facecolor="r", label="CI Sum", hatch="//")

    labels = ["L1", "L2", "Chalimus", "Pre-adults", "Adult females", "Adult males", "Adults"]
    for i in [3, 6]:
        if i == 6:
            ax.plot(x, [sum(k) for k in zip(y[4], y[5])], label = labels[i], linestyle=linestyles[i], color=colors[i], linewidth=linewidth)
            ax.fill_between(x, [sum(k) for k in zip(y_l[4], y_l[5])], [sum(k) for k in zip(y_u[4], y_u[5])], alpha=0.3, edgecolor=colors[i], facecolor=colors[i], label="CI "+labels[i], hatch=hatchs[i])
        else:
            ax.plot(x, y[i], label = labels[i], linestyle=linestyles[i], color=colors[i], linewidth=linewidth)
            ax.fill_between(x, y_l[i], y_u[i], alpha=0.3, edgecolor=colors[i], facecolor=colors[i], label="CI "+labels[i], hatch=hatchs[i])

            
    # Major ticks every half year, minor ticks every month,
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.set_xlim(x[0], x[-1])
    ax.grid(axis='y')
    ax.set_ylabel(r'Lice count ', fontsize=13)

    # ax.set_title('ConciseFormatter', loc='left', y=0.85, x=0.02, fontsize='medium')
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    plt.legend()
    plt.title("Average lice population across 1000 runs " + farm_file_name.replace("_", " "), fontsize=18)
    plt.savefig('./new_plots/total_lice_count' + farm_file_name + '.pdf')
    plt.close()
