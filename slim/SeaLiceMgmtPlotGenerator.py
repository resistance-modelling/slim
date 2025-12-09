#!/usr/bin/env python

"""
This script reads the simulation results and generates plots for publication.

To launch the script, from the root folder:
``python3 -m slim.SeaLiceMgmtPlotGenerator <path to the parquet file> --output_dir=outputs``
or ``uv run -m slim.SeaLiceMgmtPlotGenerator <path to the parquet file> --output_dir=outputs``
"""
import argparse
import sys
from pathlib import Path
import pyarrow.parquet as pq
import numpy as np
from slim.log import logger, create_logger
from slim.plots.total_genotypes import plot_farm_genotypes, extract_genotype_data
from slim.plots.lice_stages_plotter import plot_lice_stages, extract_lice_stages_data
from slim.plots.reservoir_lice_plotter import plot_reservoir_lice, extract_reservoir_lice_data
from slim.plots.payoff_plotter import plot_payoff_distribution, extract_payoff_data


def consolidate_farm_data(all_farm_data, data_type_name):
    """
    Consolidates a list of data blocks into the final counts array and returns metrics.
    """
    if not all_farm_data:
        logger.error(f"No valid {data_type_name} data found")
        return None, 0

    combined_data = np.concatenate(all_farm_data, axis=0)
    counts_array = np.expand_dims(combined_data, axis=0)

    return counts_array, len(all_farm_data)

if __name__ == "__main__":
    """Main function to process parquet file and generate all plots."""
    parser = argparse.ArgumentParser(description="SLIM plotter with comprehensive analysis")
    parser.add_argument(
        "results_file",
        type=str,
        help="Full path to the output parquet file. The base directory will be used for logging and output plots.",
    )
    parser.add_argument(
        "--gridded",
        action="store_true",
        help="If set, plots for individual farms will be saved in a single file in a grid format.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots. If not specified, uses the parent directory of the results file.",
    )

    args, unknown = parser.parse_known_args()

    # Set up paths
    output_file_path = Path(args.results_file)
    output_basename = output_file_path.parent
    output_filename = output_file_path.stem

    # Determine the output directory for plots
    if args.output_dir:
        plot_output_dir = Path(args.output_dir)
    else:
        plot_output_dir = output_basename / "plots"

    plot_output_dir.mkdir(parents=True, exist_ok=True)

    # Set up the logger
    create_logger()
    logger.info(f"Processing file: {output_file_path}")
    logger.info(f"Plots will be saved to: {plot_output_dir}")

    # Load parquet file
    try:
        table = pq.read_table(output_file_path).to_pandas()
        logger.info("Parquet file loaded successfully")
        logger.debug(f"{table.info()}")
    except Exception as e:
        logger.error(f"Failed to load parquet file: {e}")
        sys.exit(1)

    # Get unique farm names
    farm_names = table.farm_name.unique()
    logger.info(f"Found {len(farm_names)} farms: {list(farm_names)}")

    # Plot Generation
    all_genotype_data = []
    all_stages_data = []
    all_reservoir_data = []
    timestamps = None

    for farm_name in farm_names:
        logger.info(f"Processing data for farm: {farm_name}")

        # --- Genotype Data ---
        genotype_data, ts = extract_genotype_data(table, farm_name=farm_name)
        if genotype_data is not None:
            all_genotype_data.append(genotype_data)
        else:
            logger.warning(f"Could not extract genotype data for farm {farm_name}")

        # --- Lice Stages Data ---
        stages_data, _ = extract_lice_stages_data(table, farm_name=farm_name)
        if stages_data is not None:
            all_stages_data.append(stages_data)
        else:
            logger.warning(f"Could not extract lice stages data for farm {farm_name}")

        # --- Reservoir Lice Data ---
        reservoir_data, _ = extract_reservoir_lice_data(table, farm_name=farm_name)
        if reservoir_data is not None:
            all_reservoir_data.append(reservoir_data)
        else:
            logger.warning(f"Could not extract reservoir lice data for farm {farm_name}")

        # --- Timestamps (Assumed consistent) ---
        if timestamps is None:
            timestamps = ts

    # Generate Genotype Plots
    logger.info("Generating genotype plots...")

    lice_counts, num_farms = consolidate_farm_data(all_genotype_data, "genotype")
    if lice_counts is not None:
        try:
            plot_farm_genotypes(
                lice_counts=lice_counts,
                timestamps=timestamps,
                num_farms=num_farms,
                farm_name=output_filename,
                output_dir=str(plot_output_dir)
            )
            logger.debug("✓ Genotype plots generated successfully")
        except Exception as e:
            logger.error(f"Failed to generate genotype plots: {e}")
    else:
        logger.warning("Skipping genotype plots due to data processing errors")

    # Generate Lice Stages Plots
    logger.info("Generating lice stages plots...")

    stages_counts, _ = consolidate_farm_data(all_stages_data, "lice stages")
    if stages_counts is not None:
        try:
            plot_lice_stages(
                lice_counts=stages_counts,
                timestamps=timestamps,
                farm_name=output_filename,
                output_dir=str(plot_output_dir)
            )
            logger.debug("✓ Lice stages plots generated successfully")
        except Exception as e:
            logger.error(f"Failed to generate lice stages plots: {e}")
    else:
        logger.warning("Skipping lice stages plots due to data processing errors")

    # Generate Reservoir Lice Plots
    logger.info("Generating reservoir lice plots...")

    reservoir_counts, _ = consolidate_farm_data(all_reservoir_data, "reservoir lice")
    if reservoir_counts is not None:
        try:
            plot_reservoir_lice(
                reservoir_counts=reservoir_counts,
                timestamps=timestamps,
                farm_name=output_filename,
                output_dir=str(plot_output_dir)
            )
            logger.debug("✓ Reservoir lice plots generated successfully")
        except Exception as e:
            logger.error(f"Failed to generate reservoir lice plots: {e}")
    else:
        logger.warning("Skipping reservoir lice plots due to data processing errors")

    # Generate Payoff Plots
    logger.info("Generating payoff plots...")

    try:
        # Note: payoff extraction remains separate as it has a different output structure
        payoffs = extract_payoff_data(table, farm_names)
        plot_payoff_distribution(
            payoffs=payoffs,
            farm_name=output_filename,
            output_dir=str(plot_output_dir)
        )
        logger.debug("✓ Payoff plots generated successfully")
    except Exception as e:
        logger.error(f"Failed to generate payoff plots: {e}")

    logger.debug("Processing complete")
