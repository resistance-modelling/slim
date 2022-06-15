__all__ = []

from collections import defaultdict
from pathlib import Path
import glob
from typing import List, Dict
import hashlib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pyarrow.parquet as paq
import pandas as pd
import ray
import scipy.ndimage, scipy.stats


from slim import common_cli_options, get_config
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
                print(artifact, " passed the test")
                continue
            except ValueError:
                print(f"{artifact} is corrupted, regenerating...")
                
        else:
            print(f"Generating {sim_name}...")
        # defection_proba is farm-specific and needs special handling

        defection_proba = kwargs.pop("defection_proba", None)

        if defection_proba is not None:
            for farm in cfg.farms:
                farm.defection_proba = defection_proba

        sim = Simulator(out_path, cfg)
        sim.run_model(quiet=quiet)


@ray.remote
def load_worker(parquet_path, columns: List[str]):
    table = paq.read_table(parquet_path, columns=columns).to_pandas()
    return table


def load_matrix(cfg: Config, out_path: Path, columns=None):
    out_path = out_path / cfg.name
    files = glob.glob(str(out_path / "*.parquet"))

    if columns != None:
        columns = ["timestamp", "farm_name"] + columns

    tasks: List[pd.DataFrame] = ray.get([load_worker.remote(file, columns) for file in files])
    columns = set(tasks[0].columns)
    columns -= {"timestamp", "farm_name"}

    results_per_farm = defaultdict(lambda: {})
    # extract group for each dataframe
    for df in tasks:
        for farm, sub_df in df.groupby("farm_name"):
            farm_dict = results_per_farm.setdefault(farm, {})
            for column in columns:
                rows = farm_dict.setdefault(column, [])
                rows.append(sub_df[column].values)

    for column_groups in results_per_farm.values():
        for column_name, column_rows in column_groups.items():
            column_groups[column_name] = np.stack(column_rows)

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


def get_treatment_regions(
        first_treatment_time, farm_df: pd.DataFrame
):
    # generate treatment regions
    # treatment markers
    # cm = pg.colormap.get("Pastel1", source="matplotlib", skipCache=True)
    # colors = cm.getColors(1)
    colors = cm.get_cmap("Pastel1")

    regions = []
    for treatment_idx in range(2):
        color = colors(treatment_idx, alpha=0.5)

        treatment_days_df = (
                farm_df[
                    farm_df["current_treatments"].apply(
                        lambda l: bool(l[treatment_idx])
                    )
                ]["timestamp"]
                - first_treatment_time
        )
        treatment_days = treatment_days_df.apply(lambda x: x.days).to_numpy()

        # Generate treatment regions by looking for the first non-consecutive treatment blocks.
        # There may be a chance where multiple treatments happen consecutively, on which case
        # we simply consider them as a unique case.
        # Note: this algorithm fails when the saving rate is not 1. This is not a problem as
        # precision is not required here.

        if len(treatment_days) > 0:
            treatment_ranges = []
            lo = 0
            for i in range(1, len(treatment_days)):
                if treatment_days[i] > treatment_days[i - 1] + 1:
                    range_ = (treatment_days[lo], treatment_days[i - 1])
                    # if range_[1] - range_[0] <= 2:
                    #    range_ = (range_[0] - 5, range_[0] + 5)
                    treatment_ranges.append(range_)
                    lo = i

            # since mechanical treatments are applied and effective for only one day we simulate a 10-day padding
            # This is also useful when the saving rate is not 1
            range_ = (treatment_days[lo], treatment_days[-1])
            # if range_[1] - range_[0] <= 2:
            #    range_ = (range_[0] - 5, range_[0] + 5)
            treatment_ranges.append(range_)

            regions.extend([(trange, treatment_idx, color) for trange in treatment_ranges])
    return regions


def plot_regions(ax, xs, regions):
    for region, treatment_idx, color in regions:
        label = treatment_to_class(Treatment(treatment_idx)).name + " treatment"
        if region[1] - region[0] <= 2:
            ax.axvline(region[0], 0, 1e20, linestyle=":", linewidth=3, label=label)
        else:
            ax.fill_between(xs, 0, 1e20, where=(region[0] <= xs) & (xs <= region[1]),
                            color=color, transform=ax.get_xaxis_transform(), label=label)


def get_trials(dfs, func, columns):
    total_per_df = []
    for df in dfs:
        total_per_df.append(func(df, columns))
    return pd.concat(total_per_df, axis=1)


def ribbon_plot(ax, xs, trials, label="", conf=0.95, conv=7):
    min_interval = trials.min(axis=1).values
    max_interval = trials.max(axis=1).values
    mean_interval = trials.mean(axis=1).values

    se_interval = trials.sem(axis=1).values
    n = len(trials)
    h = se_interval * scipy.stats.t.ppf((1 + conf) / 2., n - 1)
    min_interval_ci, max_interval_ci = mean_interval - h, mean_interval + h

    kernel = np.full((conv,), 1 / conv, )

    min_interval = scipy.ndimage.convolve(min_interval, kernel, mode='nearest')
    max_interval = scipy.ndimage.convolve(max_interval, kernel, mode='nearest')
    mean_interval = scipy.ndimage.convolve(mean_interval, kernel, mode='nearest')

    ax.plot(mean_interval, linewidth=2.5, label=label)
    poly = ax.fill_between(xs, min_interval_ci, max_interval_ci, alpha=.3, label=f"{label} {conf:.0%} CI")

    ax.fill_between(xs, min_interval, max_interval, alpha=.1, label=label + " 100% CI",
                    color=poly.get_facecolor())

    for trial in trials:
        ax.plot(trial, linewidth=1.0, color='gray', alpha=0.8)


def plot_farm(gross_ax, geno_ax, fish_ax, agg_ax, payoff_ax, reservoir_ax):
    stages = LicePopulation.lice_stages
    stages_readable = LicePopulation.lice_stages_bio_labels
    #first_day = farm_df.iloc[0]["timestamp"]
    # regions = get_treatment_regions(first_day, farm_df)
    #xs = np.arange(len(farm_df))
    #alleles = geno_to_alleles(0)

    # total population
    #def agg(df, stages):
    #    return sum(df[stage].apply(lambda x: sum(x.values())) for stage in stages)

    #lice_population
    total_per_df = get_trials(farm_dfs, agg, stages)

    ribbon_plot(gross_ax, xs, total_per_df, "Overall lice population")
    prepare_ax(gross_ax, "Lice population")

    # plot_regions(gross_ax, xs, regions)

    # Per allele
    def agg(df, col):
        return df[col[0]]

    for allele in alleles:
        total_per_df = get_trials(farm_dfs, agg, [allele])
        ribbon_plot(geno_ax, xs, total_per_df, label=allele)

    prepare_ax(geno_ax, "By geno", legend=True)

    # plot_regions(geno_ax, xs, regions)

    # Fish population and aggregation
    def agg(df, col):
        return df[col[0]].apply(lambda x: sum(x) / len(x))

    farm_population = get_trials(farm_dfs, agg, ["fish_population"])
    farm_agg = get_trials(farm_dfs, agg, ["aggregation"])

    ribbon_plot(fish_ax, xs, farm_population, label="Fish population")
    # plot_regions(fish_ax, xs, regions)

    prepare_ax(fish_ax, "By fish population", ylabel="Fish population",
               ylim=(0, 200000), yscale="linear")

    ribbon_plot(agg_ax, xs, farm_agg, label="Aggregation")
    # plot_regions(agg_ax, xs, regions)

    prepare_ax(agg_ax, "By lice aggregation",
               ylim=(0, 10), yscale="linear", threshold=6.0)

    # Payoff
    def agg(df, col):
        return df[col[0]].cumsum()

    payoff = get_trials(farm_dfs, agg, ["payoff"])
    ribbon_plot(payoff_ax, xs, payoff, label="Payoff")
    # plot_regions(payoff_ax, xs, regions)

    prepare_ax(payoff_ax, "Payoff", "Payoff (pseudo-gbp)", ylim=None, yscale="linear")

    # New eggs (reservoir)
    def agg(df, col):
        return pd.DataFrame(list(df[col[0]]))[col[1]]

    reservoir = get_trials(farm_dfs, lambda df, col: df[col[0]], ["new_reservoir_lice"])
    genos = ["a", "A", "Aa"]
    reservoir_ratios = [get_trials(farm_dfs, agg, ["new_reservoir_lice_ratios", geno]) for geno in genos]
    for i in range(3):
        ribbon_plot(reservoir_ax, xs, reservoir_ratios[i], label=genos[i])
    # plot_regions(reservoir_ax, xs, regions)

    prepare_ax(reservoir_ax, "Reservoir ratios", ylabel="Probabilities", ylim=(0, 1), yscale="linear")


def plot_data(cfg: Config, extra: str, output_folder: str):
    width = 26
    n = cfg.nfarms

    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(width, 12 / 16 * width))
    for i in range(cfg.nfarms):
        farm_name = cfg.farms[i].name
        fig.suptitle(farm_name, fontsize=30)
        plot_farm(axs[0][0], axs[0][1], axs[1][0], axs[1][1], axs[2][0], axs[2][1])
        plt.show()
        fig.savefig(f"{output_folder}/{farm_name} {extra}.pdf")
        # fig.clf() does not work as intended...
        for ax in axs.flatten():
            ax.cla()

    plt.close(fig)

def main():
    parser = common_cli_options("SLIM Benchmark tool")
    bench_group = parser.add_argument_group(title="Benchmark-specific options")
    #bench_group.add_argument("--fields", nargs="+")
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
        required=True,
    )
    bench_group.add_argument(
        "--parallel-trials",
        type=int,
        help="How many trials to perform in parallel",
        required=True,
    )

    bench_group.add_argument("--defection-proba", type=float, help="If using mosaic, the defection probability for each farm")

    cfg, args, out_path = get_config(parser)
    print(cfg.farms_per_process)
    ss = np.random.SeedSequence(args.bench_seed)
    child_seeds = ss.spawn(args.parallel_trials)

    tasks = [
        launch.remote(cfg, np.random.default_rng(s), out_path, **vars(args))
        for s in child_seeds
    ]
    ray.get(tasks)

    #plot_data(cfg, out_path)

if __name__ == "__main__":
    main()
