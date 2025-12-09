import os
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Plotting configuration
COLORS = ["mediumseagreen", "orange", "salmon", "mediumseagreen", "skyblue", "orange", "orange"]
LINESTYLES = ["solid", "dashed", "dotted", "dashdot", "solid", "dotted", "dashed", "dashdot"]
LINEWIDTH = 1.7
SMOOTH_N = 7


def extract_genotype_data(table_df, farm_name="farm_0"):
    """
    Extract genotype counts from a parquet table DataFrame.

    Args:
        table_df: pandas DataFrame from parquet file
        farm_name: name of the farm to filter (default: "farm_0")

    Returns:
        tuple: (genotype_array, timestamps)
               genotype_array shape: (3,) containing [a_count, Aa_count, A_count]
               timestamps: unique timestamps from the table
    """
    farm_data = table_df.loc[table_df.farm_name == farm_name]

    if farm_data.shape[0] != 730:
        return None, None

    sorted_data = farm_data.sort_values(['farm_name', 'timestamp'])

    genotypes = sorted_data.apply(
        lambda row: (
            row.L1['a'] + row.L2['a'] + row.L3['a'] + row.L4['a'] + row.L5f['a'] + row.L5m['a'],
            row.L1['Aa'] + row.L2['Aa'] + row.L3['Aa'] + row.L4['Aa'] + row.L5f['Aa'] + row.L5m['Aa'],
            row.L1['A'] + row.L2['A'] + row.L3['A'] + row.L4['A'] + row.L5f['A'] + row.L5m['A']
        ),
        axis=1
    )

    genotype_array = np.array(list(map(np.array, list(genotypes))))
    timestamps = table_df.timestamp.unique()

    return genotype_array, timestamps


def calculate_quantile_bounds(lice_geno_counts, days=730):
    """
    Calculate median and confidence bounds for genotype counts.

    Args:
        lice_geno_counts: array of shape (trials, days*farms, 3)
        days: number of days per farm cycle (default: 730)

    Returns:
        tuple: (medians, upper_bounds, lower_bounds) - each is a list of 3 arrays
    """
    medians = []
    upper_bounds = []
    lower_bounds = []

    for geno_idx in range(3):
        med = []
        upper = []
        lower = []

        for day in range(days):
            day_data = lice_geno_counts[:, day::days, geno_idx].flatten()
            med.append(round(np.quantile(day_data, 0.5), 2))
            upper.append(round(np.quantile(day_data, 0.95), 2))
            lower.append(round(np.quantile(day_data, 0.05), 2))

        medians.append(med)
        upper_bounds.append(upper)
        lower_bounds.append(lower)

    return medians, upper_bounds, lower_bounds


def plot_genotype_counts(lice_counts, timestamps, title, output_path, smooth_n=SMOOTH_N):
    """
    Plot absolute genotype counts with confidence intervals.

    Args:
        lice_counts: array of shape (trials, days*farms, 3)
        timestamps: array of datetime values
        title: plot title
        output_path: path to save the figure
        smooth_n: smoothing window size (default: 7)
    """
    medians, upper_bounds, lower_bounds = calculate_quantile_bounds(lice_counts)

    fig, ax = plt.subplots(figsize=(13, 4), constrained_layout=True)

    labels = ["a", "aA", "A"]

    for i in range(3):
        smoothed_med = scipy.ndimage.uniform_filter1d(medians[i], size=smooth_n, mode='nearest')
        smoothed_lower = scipy.ndimage.uniform_filter1d(lower_bounds[i], size=smooth_n, mode='nearest')
        smoothed_upper = scipy.ndimage.uniform_filter1d(upper_bounds[i], size=smooth_n, mode='nearest')

        ax.plot(timestamps, smoothed_med, label=labels[i],
                linestyle=LINESTYLES[i], color=COLORS[i], linewidth=LINEWIDTH)
        ax.fill_between(timestamps, smoothed_lower, smoothed_upper,
                        alpha=0.3, edgecolor=COLORS[i], facecolor=COLORS[i])

    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.set_xlim(timestamps[0], timestamps[-1])
    ax.grid(axis='y')
    ax.set_ylabel(r'Lice count', fontsize=13)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    plt.legend()
    plt.title(title, fontsize=18)
    plt.savefig(output_path)
    plt.close()


def plot_genotype_ratios(lice_counts, timestamps, num_farms, title, output_path, smooth_n=SMOOTH_N):
    """
    Plot normalized genotype ratios with confidence intervals.

    Args:
        lice_counts: array of shape (trials, days*farms, 3)
        timestamps: array of datetime values
        num_farms: number of farms in the simulation
        title: plot title
        output_path: path to save the figure
        smooth_n: smoothing window size (default: 7)
    """
    trials = {
        "a": lice_counts[:, :, 0],
        "Aa": lice_counts[:, :, 1],
        "A": lice_counts[:, :, 2]
    }

    total = trials["a"] + trials["A"] + trials["Aa"]

    ratios = {
        "a": np.reshape(trials["a"] / total, (lice_counts.shape[0] * num_farms, 730)),
        "Aa": np.reshape(trials["Aa"] / total, (lice_counts.shape[0] * num_farms, 730)),
        "A": np.reshape(trials["A"] / total, (lice_counts.shape[0] * num_farms, 730))
    }

    mean_ratios = {k: np.mean(v, axis=0) for k, v in ratios.items()}
    sorted_ratios = {k: np.sort(v, axis=0) for k, v in ratios.items()}

    conf = 0.95
    T = lice_counts.shape[0] * num_farms

    minimum = {k: sorted_ratios[k][int(T * (1 - conf))] for k in sorted_ratios}
    maximum = {k: sorted_ratios[k][int(T * conf)] for k in sorted_ratios}

    fig, ax = plt.subplots(figsize=(13, 4), constrained_layout=True)

    labels = ["AA", "Aa", "aa"]
    genotypes = ["A", "Aa", "a"]

    for i, geno in enumerate(genotypes):
        smoothed_mean = scipy.ndimage.uniform_filter1d(mean_ratios[geno], size=smooth_n, mode='nearest')
        smoothed_min = scipy.ndimage.uniform_filter1d(minimum[geno], size=smooth_n, mode='nearest')
        smoothed_max = scipy.ndimage.uniform_filter1d(maximum[geno], size=smooth_n, mode='nearest')

        ax.plot(timestamps, smoothed_mean, label=labels[i],
                linestyle=LINESTYLES[i], color=COLORS[i], linewidth=LINEWIDTH)
        ax.fill_between(timestamps, smoothed_min, smoothed_max,
                        alpha=0.2, edgecolor=COLORS[i], facecolor=COLORS[i])

    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.set_xlim(timestamps[0], timestamps[-1])
    ax.grid(axis='y')
    ax.set_ylabel(r'Ratio', fontsize=14)
    ax.tick_params(axis='x', labelsize=12)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    plt.legend()
    plt.title(title, fontsize=18)
    plt.savefig(output_path)
    plt.close()


def plot_farm_genotypes(lice_counts, timestamps, num_farms, farm_name, output_dir='./plots'):
    """
    Main function to generate both count and ratio plots for a farm.

    Args:
        lice_counts: array of shape (trials, days*farms, 3) with genotype counts
        timestamps: array of datetime values
        num_farms: number of farms in the simulation
        farm_name: name identifier for the farm (used in titles and filenames)
        output_dir: directory to save plots (default: './plots')
    """
    os.makedirs(output_dir, exist_ok=True)

    display_name = farm_name.replace("_", " ")

    # Plot absolute counts
    count_title = f"Lice population by genotype - {display_name}"
    count_path = os.path.join(output_dir, f'total_genotype_{farm_name}.pdf')
    plot_genotype_counts(lice_counts, timestamps, count_title, count_path)

    # Plot normalized ratios
    ratio_title = f"Normalised lice population by genotype - {display_name}"
    ratio_path = os.path.join(output_dir, f'total_genotype_sumto1_{farm_name}.pdf')
    plot_genotype_ratios(lice_counts, timestamps, num_farms, ratio_title, ratio_path)
