__all__ = []

import functools
import logging
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
    h = se_interval * stats.t.ppf((1 + ci_confidence) / 2., n - 1)
    min_interval_ci, max_interval_ci = mean - h, mean + h

    return np.stack([mean, min_interval_ci, max_interval_ci])


@functools.lru_cache(maxsize=1<<25)
def load_matrix(
    cfg: Config,
    out_path: Path,
    columns=None,
    preprocess=lambda x: x,
    ci=True,
    ci_confidence=0.95
):
    out_path = out_path / cfg.name
    files = glob.glob(str(out_path / "*.parquet"))

    if columns is not None:
        columns = ["timestamp", "farm_name"] + list(columns)

    tasks: List[pd.DataFrame] = ray.get(_remove_nulls.remote(*[load_worker.remote(file, columns) for file in files]))
    print(f"A total of {len(tasks)} files were loaded")
    columns = set(tasks[0].columns)
    columns -= {"timestamp", "farm_name"}

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

def prepare_ax(ax, farm_name, ylabel="Lice Population", yscale="log", ylim=(1, 1e10), threshold=None, legend=True):
    ax.set_title(farm_name, fontsize=18)
    ax.grid(True, axis='y')
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_yscale(yscale)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("Days", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    if threshold:
        ax.axhline(threshold, linestyle="dashed", color="green", label="Controlled threshold")

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


def ribbon_plot(ax, xs, trials, label="", conv=7):
    kernel = np.zeros((3, conv))
    kernel[1] = np.full((conv,), 1 / conv)
    mean_interval, min_interval_ci, max_interval_ci = data = ndimage.convolve(trials, kernel, mode="nearest")

    ax.plot(mean_interval, linewidth=2.5, label=label)
    ax.fill_between(xs, min_interval_ci, max_interval_ci, alpha=.3, label=f"{label} 95% CI")

    #ax.fill_between(xs, min_interval, max_interval, alpha=.1, label=label + " 100% CI",
    #                color=poly.get_facecolor())

    for trial in trials:
        ax.plot(trial, linewidth=1.0, color='gray', alpha=0.8)


# TODO: this is stupid and O(N^2)
def plot_farm(cfg: Config, output_folder: Path, farm_name: str, gross_ax, geno_ax, fish_ax, agg_ax, payoff_ax, reservoir_ax):
    stages = tuple(LicePopulation.lice_stages)
    stages_readable = LicePopulation.lice_stages_bio_labels
    alleles = geno_to_alleles(0)

    xs = np.arange((cfg.end_date - cfg.start_date).days)
    lice_pop = load_matrix(cfg, output_folder, stages,
                           preprocess=np.vectorize(lambda x: sum(x.values())), ci=False)[farm_name]
    data = get_ci(sum([lice_pop[stage] for stage in stages]))
    ribbon_plot(gross_ax, xs, data, "Overall lice population")
    prepare_ax(gross_ax, "Lice population")

    def agg(x, allele):
        return x[allele]

    for allele in alleles:
        per_allele_pop = load_matrix(
            cfg,
            output_folder,
            stages,
            preprocess=np.vectorize(lambda x: agg(x, allele=allele)),
            ci=False
        )[farm_name]
        normalised = sum(per_allele_pop[stage] for stage in stages) / (sum(lice_pop[stage] for stage in stages))
        ribbon_plot(geno_ax, xs, get_ci(normalised), allele)

    prepare_ax(geno_ax, "By geno", legend=True, yscale="linear", ylim=(0, 1))

    # plot_regions(geno_ax, xs, regions)

    # Fish population and aggregation
    fish_pop_agg = load_matrix(cfg, output_folder, ("fish_population", "aggregation", "cleaner_fish"))[farm_name]
    fish_pop, fish_agg, cleaner_fish = fish_pop_agg["fish_population"], fish_pop_agg["aggregation"], fish_pop_agg["cleaner_fish"]

    ribbon_plot(fish_ax, xs, fish_pop, label="Fish population")
    ribbon_plot(fish_ax, xs, cleaner_fish, label="Cleaner fish population")
    ribbon_plot(agg_ax, xs, fish_agg, label="Aggregation rate")

    prepare_ax(fish_ax, "By fish population", ylabel="Fish population",
               ylim=(0, 200000), yscale="linear")

    prepare_ax(agg_ax, "By lice aggregation",
               ylim=(0, 10), yscale="linear", threshold=6.0)

    payoff = load_matrix(cfg, output_folder, ("payoff",), preprocess=lambda row: row.cumsum())[farm_name]["payoff"]
    ribbon_plot(payoff_ax, xs, payoff.cumsum(axis=1), label="Payoff")
    # plot_regions(payoff_ax, xs, regions)

    prepare_ax(payoff_ax, "Payoff", "Payoff (pseudo-gbp)", ylim=None, yscale="linear")

    # reservoir = load_matrix(cfg, output_folder, ["new_reservoir_lice"])
    col = "new_reservoir_lice_ratios"

    def agg(allele):
        def _agg(x):
            return x[allele] / sum(x.values())
        return _agg

    for allele in alleles:
        data = load_matrix(cfg, output_folder, (col,), preprocess=np.vectorize(agg(allele)))[farm_name][col]
        ribbon_plot(reservoir_ax, xs, data, label=allele)

    prepare_ax(reservoir_ax, "Reservoir ratios", ylabel="Probabilities", ylim=(0, 1), yscale="linear")


def plot_data(cfg: Config, extra: str, output_folder: Path):
    width = 26

    logger.info("This function is quite slow. It could be optimised although we do not need to for now")
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(width, 12 / 16 * width))
    for i in range(cfg.nfarms):
        farm_name = cfg.farms[i].name
        fig.suptitle(farm_name, fontsize=30)
        plot_farm(cfg, output_folder, f"farm_{i}", axs[0][0], axs[0][1], axs[1][0], axs[1][1], axs[2][0], axs[2][1])
        logger.info(f"Generating plot to {output_folder}/{farm_name} {extra}.pdf")
        fig.savefig(f"{output_folder}/{farm_name} {extra}.pdf")
        # fig.clf() does not work as intended...
        for ax in axs.flatten():
            ax.cla()

    plt.close(fig)


def main():
    parser = common_cli_options("SLIM Benchmark tool")
    bench_group = parser.add_argument_group(title="Benchmark-specific options")
    bench_group.add_argument(
        "description",
        type=str,
        help="A description to append to all the plots"
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
        "--skip",
        help="(DEBUG) skip simulation running",
        action="store_true"
    )

    bench_group.add_argument("--defection-proba", type=float, help="If using mosaic, the defection probability for each farm")

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
