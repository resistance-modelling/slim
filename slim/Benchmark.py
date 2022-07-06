#!/bin/env python

__all__ = []

import functools
import logging
import traceback
from collections import defaultdict
from pathlib import Path
import glob
from typing import List, Dict
import hashlib

import numpy as np
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.parquet as paq
import pandas as pd
import ray
import scipy.ndimage as ndimage
import scipy.stats as stats


from slim import common_cli_options, get_config
from slim.log import logger
from slim.simulation.simulator import get_simulation_path, Simulator
from slim.simulation.config import Config
from slim.simulation.lice_population import LicePopulation, geno_to_alleles
from slim.types.treatments import Treatment, treatment_to_class


@ray.remote
def launch(cfg, rng, out_path, **kwargs):
    trials = kwargs.pop("trials")
    quiet = kwargs.pop("quiet", False)
    orig_name = cfg.name
    out_path = out_path / cfg.name
    hash_ = hashlib.sha256()

    for t in range(trials):
        extra_args = kwargs.copy()
        extra_args["seed"] = rng.random()
        hash_.update(repr(extra_args).encode())

        sim_name = cfg.name = orig_name + str(hash_.digest().hex())
        artifact = get_simulation_path(out_path, cfg)[0]

        if artifact.exists():
            # Safety check: check if a run has completed.
            try:
                paq.read_table(str(artifact), columns=["timestamp"])
                logger.info(artifact, " passed the test")
                continue
            except ValueError:
                logger.info(f"{artifact} is corrupted, regenerating...")

        else:
            logger.info(f"Generating {sim_name}...")
        # defection_proba is farm-specific and needs special handling

        defection_proba = kwargs.pop("defection_proba", None)

        if defection_proba is not None:
            for farm in cfg.farms:
                farm.defection_proba = defection_proba

        sim = Simulator(out_path, cfg)
        sim.run_model(quiet=quiet)


@ray.remote
def load_worker(parquet_path, columns: List[str]):
    try:
        table = paq.read_table(parquet_path, columns=columns).to_pandas()
        return table
    except pa.lib.ArrowInvalid:
        return None


@ray.remote
def _remove_nulls(*tables):
    return [table for table in tables if table is not None]


def get_ci(matrix, ci_confidence=0.95):
    mean = np.mean(matrix, axis=0)
    n = len(matrix[0])

    se_interval = stats.sem(matrix, axis=0)
    h = se_interval * stats.t.ppf((1 + ci_confidence) / 2.0, n - 1)
    min_interval_ci, max_interval_ci = mean - h, mean + h

    return np.stack([mean, min_interval_ci, max_interval_ci])


@functools.lru_cache(maxsize=1 << 25)
def load_matrix(
    cfg: Config,
    out_path: Path,
    columns=None,
    preprocess=lambda x: x,
    ci=True,
    ci_confidence=0.95,
):
    out_path = out_path / cfg.name
    files = glob.glob(str(out_path / "*.parquet"))[:120]

    if columns is not None:
        columns = ["farm_name"] + list(columns)

    tasks: List[pd.DataFrame] = ray.get(
        _remove_nulls.remote(*[load_worker.remote(file, columns) for file in files])
    )
    columns = set(tasks[0].columns)
    columns -= {"farm_name"}

    results_per_farm = defaultdict(lambda: {})
    # extract group for each dataframe

    for df in tasks:
        for farm, sub_df in df.groupby("farm_name"):
            farm_dict = results_per_farm.setdefault(farm, {})
            for column in columns:
                rows = farm_dict.setdefault(column, [])
                rows.append(preprocess(sub_df[column].values))

    for column_groups in results_per_farm.values():
        for column_name, column_rows in column_groups.items():
            matrix = np.stack(column_rows)
            if ci:
                column_groups[column_name] = np.stack([*get_ci(matrix, ci_confidence)])
            else:
                column_groups[column_name] = matrix

    return results_per_farm


# now, we need to generate plots...
# TODO: this is largely copied from plots.py but largely simplified due to a (thankfully) better API.


def prepare_ax(
    ax,
    farm_name,
    ylabel="Lice Population",
    yscale="log",
    ylim=(1, 1e10),
    threshold=None,
    legend=True,
    grid=True
):
    ax.set_title(farm_name, fontsize=18)
    ax.grid(grid, axis="y")
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_yscale(yscale)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_xlabel("Days", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    if threshold:
        ax.axhline(
            threshold, linestyle="dashed", color="green", label="Controlled threshold"
        )

    if legend:
        # avoid plottings dups
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        # ax.legend()


def get_trials(dfs, func, columns):
    total_per_df = []
    for df in dfs:
        total_per_df.append(func(df, columns))
    return pd.concat(total_per_df, axis=1)


def ribbon_plot(ax, xs, trials, label="", conv=7, color=None):
    kernel = np.zeros((3, conv))
    kernel[1] = np.full((conv,), 1 / conv)
    mean_interval, min_interval_ci, max_interval_ci = ndimage.convolve(
        trials, kernel, mode="nearest"
    )

    plotted_line = ax.plot(mean_interval, linewidth=2.5, label=label, color=color)[0]
    ax.fill_between(
        xs, min_interval_ci, max_interval_ci, alpha=0.3, label=f"{label} 95% CI", color=plotted_line.get_color()
    )


def get_lice_pop(cfg, output_folder, stages):
    return load_matrix(
        cfg,
        output_folder,
        stages,
        preprocess=np.vectorize(lambda x: sum(x.values())),
        ci=False,
    )


def plot_pop(cfg: Config, axs, ax_idx, xs, output_folder):
    stages = tuple(LicePopulation.lice_stages)
    #stages_readable = LicePopulation.lice_stages_bio_labels
    #alleles = geno_to_alleles(0)

    lice_pop = get_lice_pop(cfg, output_folder, stages)

    for i in range(cfg.nfarms):
        farm_name = f"farm_{i}"
        lice_pop_farm = lice_pop[farm_name]
        data = get_ci(sum([lice_pop_farm[stage] for stage in stages]))
        gross_ax = axs[farm_name][1][ax_idx]

        ribbon_plot(gross_ax, xs, data, "Overall lice population")
        prepare_ax(gross_ax, "Lice population")


def plot_geno(cfg, axs, ax_idx, xs, output_folder):
    stages = tuple(LicePopulation.lice_stages)
    alleles = geno_to_alleles(0)

    lice_pop = get_lice_pop(cfg, output_folder, stages)

    def agg(x, allele):
        return x[allele]

    per_allele_pop = {
        allele: load_matrix(
            cfg,
            output_folder,
            stages,
            preprocess=np.vectorize(lambda x: agg(x, allele=allele)),
            ci=False
        ) for allele in alleles
    }

    for i in range(cfg.nfarms):
        farm_name = f"farm_{i}"
        for allele in alleles:
            per_allele_pop_farm = per_allele_pop[allele][farm_name]
            lice_pop_farm = lice_pop[farm_name]
            normalised = sum(per_allele_pop_farm[stage] for stage in stages) / (
                sum(lice_pop_farm[stage] for stage in stages)
            )
            geno_ax = axs[farm_name][1][ax_idx]
            ribbon_plot(geno_ax, xs, get_ci(normalised), allele)
            prepare_ax(geno_ax, "By geno", legend=True, yscale="linear", ylim=(0, 1))


def plot_fish_agg(cfg, axs, fish_ax_idx, agg_ax_idx, xs, output_folder):
    fish_pop_agg = load_matrix(
        cfg, output_folder, ("fish_population", "aggregation", "cleaner_fish")
    )
    for i in range(cfg.nfarms):
        farm_name = f"farm_{i}"
        fish_pop_agg_farm = fish_pop_agg[farm_name]
        # Fish population and aggregation
        fish_pop, fish_agg, cleaner_fish = (
            fish_pop_agg_farm["fish_population"],
            fish_pop_agg_farm["aggregation"],
            fish_pop_agg_farm["cleaner_fish"],
        )

        fish_ax = axs[farm_name][1][fish_ax_idx]
        cleaner_fish_ax = fish_ax.twinx()
        cleaner_fish_ax.set_ylabel("Cleaner fish population")
        cleaner_fish_ax.tick_params(axis='y')
        agg_ax = axs[farm_name][1][agg_ax_idx]

        ribbon_plot(fish_ax, xs, fish_pop, label="Fish population")
        ribbon_plot(cleaner_fish_ax, xs, cleaner_fish, label="Cleaner fish population", color="brown")
        ribbon_plot(agg_ax, xs, fish_agg, label="Aggregation rate")

        prepare_ax(
            fish_ax,
            "By fish population",
            ylabel="Fish population",
            ylim=None,  # Jess will not like this but not all farms are the same...
            yscale="linear",
            legend=False
        )
        prepare_ax(
            cleaner_fish_ax,
            "",
            ylabel="Cleaner fish population",
            ylim=None,
            yscale="linear",
            grid=False,
            legend=False
        )

        fish_ax.legend()
        cleaner_fish_ax.legend()

        prepare_ax(
            agg_ax, "By lice aggregation", ylim=(0, 10), yscale="linear", threshold=6.0
        )


def plot_payoff(cfg, axs, ax_idx, xs, output_folder):
    payoff = load_matrix(
        cfg, output_folder, ("payoff",), preprocess=lambda row: row.cumsum()
    )

    for i in range(cfg.nfarms):
        farm_name = f"farm_{i}"
        payoff_farm = payoff[farm_name]["payoff"]

        payoff_ax = axs[farm_name][1][ax_idx]

        ribbon_plot(payoff_ax, xs, payoff_farm.cumsum(axis=1), label="Payoff")
        # plot_regions(payoff_ax, xs, regions)

        prepare_ax(payoff_ax, "Payoff", "Payoff (pseudo-gbp)", ylim=None, yscale="linear")


def plot_reservoir(cfg, axs, ax_idx, xs, output_folder):
    alleles = geno_to_alleles(0)
    col = "new_reservoir_lice_ratios"

    def agg(allele):
        def _agg(x):
            return x[allele] / sum(x.values())

        return _agg

    data = {
        allele:
            load_matrix(
            cfg, output_folder, (col,), preprocess=np.vectorize(agg(allele)))
         for allele in alleles
    }
    # reservoir = load_matrix(cfg, output_folder, ["new_reservoir_lice"])

    for i in range(cfg.nfarms):
        farm_name = f"farm_{i}"
        reservoir_ax = axs[farm_name][1][ax_idx]

        for allele in alleles:
            data_farm = data[allele][farm_name][col]

            ribbon_plot(reservoir_ax, xs, data_farm, label=allele)

        prepare_ax(
            reservoir_ax,
            "Reservoir ratios",
            ylabel="Probabilities",
            ylim=(0, 1),
            yscale="linear",
        )


def plot_data(cfg: Config, extra: str, output_folder: Path):
    width = 26

    ax_dict = {
        f"farm_{i}": plt.subplots(nrows=3, ncols=2, figsize=(width, 12 / 16 * width))
        for i in range(cfg.nfarms)
    }
    xs = np.arange((cfg.end_date - cfg.start_date).days)

    # Basically plot the same parameter for all the farms
    plot_pop(cfg, ax_dict, (0,0), xs, output_folder)
    plot_geno(cfg, ax_dict, (0,1), xs, output_folder)
    plot_fish_agg(cfg, ax_dict, (1, 0), (1, 1), xs, output_folder)
    plot_payoff(cfg, ax_dict, (2, 0), xs, output_folder)
    plot_reservoir(cfg, ax_dict, (2, 1), xs, output_folder)

    for i in range(cfg.nfarms):
        farm_id = f"farm_{i}"
        farm_name = cfg.farms[i].name

        fig, _ = ax_dict[farm_id]
        fig.suptitle(farm_name, fontsize=30)
        fig.savefig(f"{output_folder}/{farm_name} {extra}.pdf")
        fig.savefig(f"{output_folder}/{farm_name} {extra}.png")
        fig.savefig(f"{output_folder}/{farm_name} {extra}.svg")

    plt.close(fig)


def main():
    parser = common_cli_options("SLIM Benchmark tool")
    bench_group = parser.add_argument_group(title="Benchmark-specific options")
    bench_group.add_argument(
        "description", type=str, help="A description to append to all the plots"
    )
    bench_group.add_argument(
        "--bench-seed",
        type=int,
        help="Seed to generate other benchmark seeds",
        default=0,
    )
    bench_group.add_argument(
        "--trials",
        type=int,
        help="How many trials to perform during benchmarking",
    )
    bench_group.add_argument(
        "--parallel-trials",
        type=int,
        help="How many trials to perform in parallel",
    )
    bench_group.add_argument(
        "--skip", help="(DEBUG) skip simulation running", action="store_true"
    )

    bench_group.add_argument(
        "--defection-proba",
        type=float,
        help="If using mosaic, the defection probability for each farm",
    )

    cfg, args, out_path = get_config(parser)
    if not args.skip and (args.trials is None or args.parallel_trials is None):
        parser.error("--trials and --parallel-trials are required unless --skip is set")

    if not args.skip:
        ss = np.random.SeedSequence(args.bench_seed)
        child_seeds = ss.spawn(args.parallel_trials)
        tasks = [
            launch.remote(cfg, np.random.default_rng(s), out_path, **vars(args))
            for s in child_seeds
        ]
        ray.get(tasks)

    plot_data(cfg, args.description, out_path)


if __name__ == "__main__":
    main()
