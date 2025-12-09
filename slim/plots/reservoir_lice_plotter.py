import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Plotting configuration
LINESTYLES = ["solid", "dashdot", "dashed", "dashdot", "solid", "dotted", "dashed", "dashdot"]
COLORS = ["tomato", "mediumseagreen", "orange", "darkmagenta", "olive", "orange"]
LINEWIDTH = 1.7
HATCHS = ["//", "OO", "XX", "\\", "oo", "//", "OO", "XX", "\\"]


def extract_reservoir_lice_data(table_df, farm_name="farm_0"):
    """
    Extract new reservoir lice ratios from a parquet table DataFrame.

    Args:
        table_df: pandas DataFrame from parquet file
        farm_name: name of the farm to filter (default: "farm_0")

    Returns:
        tuple: (reservoir_array, timestamps)
               reservoir_array shape: (days, 3) containing [A, Aa, a] ratios
               timestamps: unique timestamps from the table
    """
    farm_data = table_df.loc[table_df.farm_name == farm_name]

    if farm_data.shape[0] != 730:
        return None, None

    sorted_data = farm_data.sort_values(['farm_name', 'timestamp'])

    reservoir_ratios = sorted_data.apply(
        lambda row: (
            row.new_reservoir_lice_ratios["A"],
            row.new_reservoir_lice_ratios["Aa"],
            row.new_reservoir_lice_ratios["a"]
        ),
        axis=1
    )

    reservoir_array = np.array(list(map(np.array, list(reservoir_ratios))))
    timestamps = table_df.timestamp.unique()

    return reservoir_array, timestamps


def calculate_reservoir_bounds(reservoir_counts, days=730):
    """
    Calculate mean and confidence bounds for reservoir lice ratios.

    Args:
        reservoir_counts: array of shape (trials, days*farms, 3)
        days: number of days per farm cycle (default: 730)

    Returns:
        tuple: (means, upper_bounds, lower_bounds) - each is a list of 3 arrays
    """
    means = []
    upper_bounds = []
    lower_bounds = []

    for geno_idx in range(3):
        mean = []
        upper = []
        lower = []

        for day in range(days):
            day_data = reservoir_counts[:, day::days, geno_idx].flatten()
            mean.append(round(np.mean(day_data), 2))
            upper.append(round(np.quantile(day_data, 0.95), 2))
            lower.append(round(np.quantile(day_data, 0.05), 2))

        means.append(mean)
        upper_bounds.append(upper)
        lower_bounds.append(lower)

    return means, upper_bounds, lower_bounds


def plot_reservoir_lice(reservoir_counts, timestamps, farm_name, output_dir='./plots'):
    """
    Plot new reservoir lice by genotype with confidence intervals.

    Args:
        reservoir_counts: array of shape (trials, days*farms, 3) with genotype ratios
        timestamps: array of datetime values
        farm_name: name identifier for the farm (used in titles and filenames)
        output_dir: directory to save plots (default: './plots')
    """
    os.makedirs(output_dir, exist_ok=True)

    means, upper_bounds, lower_bounds = calculate_reservoir_bounds(reservoir_counts)

    fig, ax = plt.subplots(figsize=(13, 4), constrained_layout=True)

    labels = ["A", "aA", "a"]

    for i in range(3):
        ax.plot(timestamps, means[i], label=labels[i], linestyle=LINESTYLES[i],
                color=COLORS[i], linewidth=LINEWIDTH)
        ax.fill_between(timestamps, lower_bounds[i], upper_bounds[i], alpha=0.3,
                        edgecolor=COLORS[i], facecolor=COLORS[i],
                        label="CI " + labels[i], hatch=HATCHS[i])

    # Format axes
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.set_xlim(timestamps[0], timestamps[-1])
    ax.grid(axis='y')
    ax.set_ylabel(r'New reservoir lice', fontsize=13)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    plt.legend()

    display_name = farm_name.replace("_", " ")
    plt.title(f"New reservoir lice - {display_name}", fontsize=18)

    output_path = os.path.join(output_dir, f'total_new_reservoir_lice_{farm_name}.pdf')
    plt.savefig(output_path)
    plt.close()
