import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from slim.log import logger

def extract_payoff_data(table_df, farm_names):
    """
    Extract total payoff data for all farms from a parquet table DataFrame.

    Args:
        table_df: pandas DataFrame from parquet file
        farm_names: list of unique farm names in the table

    Returns:
        list: total payoffs for each farm
    """
    total_payoffs = []

    for farm_name in farm_names:
        farm_data = table_df.loc[table_df.farm_name == farm_name]

        if farm_data.shape[0] == 730:
            payoff_sum = sum(farm_data.payoff)
            total_payoffs.append(payoff_sum)

    return total_payoffs


def plot_payoff_distribution(payoffs, farm_name, output_dir='./plots'):
    """
    Plot cumulative payoff distribution as a violin plot.

    Args:
        payoffs: list of total payoff values
        farm_name: name identifier for the farm (used in titles and filenames)
        output_dir: directory to save plots (default: './plots')
    """
    os.makedirs(output_dir, exist_ok=True)

    if not payoffs:
        logger.warning(f"Warning: No payoff data available for {farm_name}")
        return

    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)

    sns.violinplot(data=payoffs, ax=ax)

    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)
    ax.set_ylabel(r'GBP', fontsize=13)

    display_name = farm_name.replace("_", " ")
    plt.title(f'Cumulative payoff - {display_name}', fontsize=18)

    output_path = os.path.join(output_dir, f'total_cumulative_payoff_{farm_name}.pdf')
    plt.savefig(output_path)
    plt.close()

    # Print summary statistics
    logger.info(f"Payoff Statistics for {farm_name}:")
    logger.info(f"  Mean: £{np.mean(payoffs):,.2f}")
    logger.info(f"  Median: £{np.median(payoffs):,.2f}")
    logger.info(f"  Std Dev: £{np.std(payoffs):,.2f}")
    logger.info(f"  Min: £{np.min(payoffs):,.2f}")
    logger.info(f"  Max: £{np.max(payoffs):,.2f}")