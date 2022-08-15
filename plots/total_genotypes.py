import os
import pyarrow.parquet as pq
import pickle
import scipy, numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

colors = ["mediumseagreen", "orange", "salmon", "mediumseagreen", "skyblue", "orange", "orange"]
linestyles = ["solid", "dashed", "dotted", "dashdot", "solid", "dotted", "dashed", "dashdot"]
linewidth = 1.7
N=7

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


def getting_bounds(lice_geno_counts):
    y = []
    y_u = []
    y_l= []
    for lice_geno in range(3):
        y.append([])
        y_u.append([])
        y_l.append([])
        for day in range(730):
            y[lice_geno].append(round(np.quantile(lice_geno_counts[:, day::730, lice_geno].flatten(), 0.5), 2))
            y_u[lice_geno].append(round(np.quantile(lice_geno_counts[:, day::730, lice_geno].flatten(), 0.95), 2))
            y_l[lice_geno].append(round(np.quantile(lice_geno_counts[:, day::730, lice_geno].flatten(), 0.05), 2))
    return y, y_u, y_l


def total_genotypes(pqt_list):
    l = []
    for pqt_name in pqt_list:
        try:
            table = pq.read_table("./" + pqt_name).to_pandas()
            if table.loc[table.farm_name == "farm_0"].shape[0] == 730:
                l.append(np.array(list(map(np.array, list(table.sort_values(['farm_name', 'timestamp']).apply(lambda row: (row.L1['a']+row.L2['a']+row.L3['a']+row.L4['a']+row.L5f['a']+row.L5m['a'], row.L1['Aa']+row.L2['Aa']+row.L3['Aa']+row.L4['Aa']+row.L5f['Aa']+row.L5m['Aa'], row.L1['A']+row.L2['A']+row.L3['A']+row.L4['A']+row.L5f['A']+row.L5m['A']), axis=1))))))
        except:
            print(pqt_name)
    x = table.timestamp.unique()
    return np.array(l), x

for farm_file_name in farm_file_names:
    print(farm_file_name)
    pqt_list, pkl_name = load_data(farm_file_name, 1000)
    lice_stages_counts, x = total_genotypes(pqt_list)
    y, y_u, y_l = getting_bounds(lice_stages_counts)

    with open("./genotypes_temp/" + farm_file_name + '.pkl', 'wb') as f:
        pickle.dump(lice_stages_counts, f)

    with open("./" + pkl_name, 'rb') as f:
        farms_data = pickle.load(f)

    # creating the plot
    fig, ax = plt.subplots(figsize=(13, 4), constrained_layout=True)

    labels = ["a", "aA", "A"]
    for i in range(3):
        ax.plot(x, scipy.ndimage.uniform_filter1d(y[i], size=N, mode='nearest'), label = labels[i], linestyle=linestyles[i], color=colors[i], linewidth=linewidth)
        ax.fill_between(x, scipy.ndimage.uniform_filter1d(y_l[i], size=N, mode='nearest'), scipy.ndimage.uniform_filter1d(y_u[i], size=N, mode='nearest'), alpha=0.3, edgecolor=colors[i], facecolor=colors[i])


    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.set_xlim(x[0], x[-1])
    ax.grid(axis='y')
    ax.set_ylabel(r'Lice count', fontsize=13)

    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    plt.legend()

    plt.title("Lice population by genotype - " + farm_file_name.replace("_", " "), fontsize=18)
    plt.savefig('./new_plots/total_genotype_' + farm_file_name + '.pdf')

    plt.close()

    trials={}
    trials["a"] = lice_stages_counts[:, :, 0]
    trials["Aa"] = lice_stages_counts[:, :, 1]
    trials["A"] = lice_stages_counts[:, :, 2]

    ratios = {}
    ratios["a"] =  np.reshape(trials["a"] / (trials["a"] + trials["A"] + trials["Aa"]), (995*9, 730))
    ratios["Aa"] =  np.reshape(trials["Aa"] / (trials["a"] + trials["A"] + trials["Aa"]), (995*9, 730))
    ratios["A"] =  np.reshape(trials["A"] / (trials["a"] + trials["A"] + trials["Aa"]), (995*9, 730))

    mean = {}
    mean["a"] = np.mean(ratios["a"], axis=0)
    mean["Aa"] = np.mean(ratios["Aa"], axis=0)
    mean["A"] = np.mean(ratios["A"], axis=0)

    sorted_trials = {}
    sorted_trials["A"] = np.sort(ratios["A"], axis=0)
    sorted_trials["a"] = np.sort(ratios["a"], axis=0)
    sorted_trials["Aa"] = np.sort(ratios["Aa"], axis=0)

    conf = 0.95 # this mean the interval us from 0.05 to 0.95 which is confidence interval 0.9
    T = lice_stages_counts.shape[0]*len(farms_data.farms)
    minimum = {}
    maximum = {}

    minimum["A"] = sorted_trials["A"][int(T * (1-conf))]
    maximum["A"] = sorted_trials["A"][int(T * conf)]

    minimum["a"] = sorted_trials["a"][int(T * (1-conf))]
    maximum["a"] = sorted_trials["a"][int(T * conf)]

    minimum["Aa"] = sorted_trials["Aa"][int(T * (1-conf))]
    maximum["Aa"] = sorted_trials["Aa"][int(T * conf)]

    fig, ax = plt.subplots(figsize=(13, 4), constrained_layout=True)

    labels = ["AA", "Aa", "aa"]

    for i, geno in enumerate(["A", "Aa", "a"]):
        ax.plot(x, scipy.ndimage.uniform_filter1d(mean[geno], size=N, mode='nearest'), label = labels[i], linestyle=linestyles[i], color=colors[i], linewidth=linewidth)
        ax.fill_between(x, scipy.ndimage.uniform_filter1d(minimum[geno], size=N, mode='nearest'), scipy.ndimage.uniform_filter1d(maximum[geno], size=N, mode='nearest'), alpha=0.2, edgecolor=colors[i], facecolor=colors[i])


    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.set_xlim(x[0], x[-1])
    ax.grid(axis='y')
    ax.set_ylabel(r'Ratio', fontsize=14)
    ax.tick_params(axis='x', labelsize=12)

    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    plt.legend()

    plt.title("Normalised lice population by genotype - " + farm_file_name.replace("_", " "), fontsize=18)
    plt.savefig('./new_plots/total_genotype_sumto1_' + farm_file_name + '.pdf')

    plt.close()