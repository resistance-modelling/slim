import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Plotting configuration
LICE_STAGES_TO_NAMES = {
    "L3": "Chalimuses",
    "L4": "Pre-adults",
    "L5m": "Adult males",
    "L5f": "Adult females",
    "all": "Sum"
}

LINESTYLES = ["solid", "dotted", "dashed", "dashdot", "solid", "dotted", "dashed", "dashdot"]
COLORS = ["tomato", "darkmagenta", "tomato", "mediumseagreen", "skyblue", "orange", "orange"]
LINEWIDTH = 1.7
HATCHS = ["//", "OO", "XX", "\\", "oo", "//", "OO", "XX", "\\"]


def extract_lice_stages_data(table_df, farm_name="farm_0"):
    """
    Extract lice stage counts from a parquet table DataFrame.

    Args:
        table_df: pandas DataFrame from parquet file
        farm_name: name of the farm to filter (default: "farm_0")

    Returns:
        tuple: (stages_array, timestamps)
               stages_array shape: (days, 6) containing [L1, L2, L3, L4, L5m, L5f]
               timestamps: unique timestamps from the table
    """
    farm_data = table_df.loc[table_df.farm_name == farm_name]

    if farm_data.shape[0] != 730:
        return None, None

    sorted_data = farm_data.sort_values(['farm_name', 'timestamp'])

    stages = sorted_data.apply(
        lambda row: (
            sum(row["L1"].values()),
            sum(row["L2"].values()),
            sum(row["L3"].values()),
            sum(row["L4"].values()),
            sum(row["L5m"].values()),
            sum(row["L5f"].values())
        ),
        axis=1
    )

    stages_array = np.array(list(map(np.array, list(stages))))
    timestamps = table_df.timestamp.unique()

    return stages_array, timestamps


def calculate_stage_bounds(lice_stages_counts, days=730):
    """
    Calculate median and confidence bounds for lice stage counts.

    Args:
        lice_stages_counts: array of shape (trials, days*farms, 6)
        days: number of days per farm cycle (default: 730)

    Returns:
        tuple: (medians, upper_bounds, lower_bounds) - each is a list of 6 arrays
    """
    medians = []
    upper_bounds = []
    lower_bounds = []

    for stage_idx in range(6):
        med = []
        upper = []
        lower = []

        for day in range(days):
            day_data = lice_stages_counts[:, day::days, stage_idx].flatten()
            med.append(round(np.quantile(day_data, 0.5), 2))
            upper.append(round(np.quantile(day_data, 0.95), 2))
            lower.append(round(np.quantile(day_data, 0.05), 2))

        medians.append(med)
        upper_bounds.append(upper)
        lower_bounds.append(lower)

    return medians, upper_bounds, lower_bounds


def plot_lice_stages(lice_counts, timestamps, farm_name, output_dir='./plots'):
    """
    Plot lice population by stage with confidence intervals.

    Args:
        lice_counts: array of shape (trials, days*farms, 6) with stage counts
        timestamps: array of datetime values
        farm_name: name identifier for the farm (used in titles and filenames)
        output_dir: directory to save plots (default: './plots')
    """
    os.makedirs(output_dir, exist_ok=True)

    medians, upper_bounds, lower_bounds = calculate_stage_bounds(lice_counts)

    # Calculate sum of stages L3, L4, and adults (L5f + L5m)
    include = [3, 4, 5]
    y_all = [sum([medians[i][j] for i in include]) for j in range(730)]
    y_all_u = [sum([upper_bounds[i][j] for i in include]) for j in range(730)]
    y_all_l = [sum([lower_bounds[i][j] for i in include]) for j in range(730)]

    fig, ax = plt.subplots(figsize=(13, 4), constrained_layout=True)

    # Plot sum
    ax.plot(timestamps, y_all, label="Sum", linestyle="solid", color="tomato", linewidth=LINEWIDTH)
    ax.fill_between(timestamps, y_all_l, y_all_u, alpha=0.15, edgecolor="tomato",
                    facecolor="r", label="CI Sum", hatch="//")

    labels = ["L1", "L2", "Chalimus", "Pre-adults", "Adult females", "Adult males", "Adults"]

    # Plot L3 (Chalimus) and combined adults (L5m + L5f)
    for i in [3, 6]:
        if i == 6:
            # Combine adult males and females
            y_adults = [sum(k) for k in zip(medians[4], medians[5])]
            y_adults_l = [sum(k) for k in zip(lower_bounds[4], lower_bounds[5])]
            y_adults_u = [sum(k) for k in zip(upper_bounds[4], upper_bounds[5])]

            ax.plot(timestamps, y_adults, label=labels[i], linestyle=LINESTYLES[i],
                    color=COLORS[i], linewidth=LINEWIDTH)
            ax.fill_between(timestamps, y_adults_l, y_adults_u, alpha=0.3,
                            edgecolor=COLORS[i], facecolor=COLORS[i],
                            label="CI " + labels[i], hatch=HATCHS[i])
        else:
            ax.plot(timestamps, medians[i], label=labels[i], linestyle=LINESTYLES[i],
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
    ax.set_ylabel(r'Lice count', fontsize=13)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    plt.legend()

    display_name = farm_name.replace("_", " ")
    plt.title(f"Average lice population by stage - {display_name}", fontsize=18)

    output_path = os.path.join(output_dir, f'total_lice_count_{farm_name}.pdf')
    plt.savefig(output_path)
    plt.close()
