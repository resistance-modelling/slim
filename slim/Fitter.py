#!/usr/bin/env python

"""
Fit to real data
"""
import argparse
import pickle
from datetime import timedelta
import os

import numpy as np
import pandas as pd
import tqdm
from ray import tune

# note: requires hpbandster and ConfigSpace
# from ray.tune.schedulers import HyperBandForBOHB
# from ray.tune.suggest.bohb import TuneBOHB
# Also: it's a bit tricky to use early stopping in our case...
from ray.tune.suggest.bayesopt import BayesOptSearch

import slim
from slim.simulation.simulator import (
    get_env,
    PilotedPolicy,
    SimulatorPZEnv,
    load_counts,
)
from slim.simulation.config import Config

# Let us consider a run of 450 days from January 2018 to March 2019 on Loch Fyne
# (Then - extend on loch lynne)
from slim.types.treatments import Treatment, EMB, HeterozygousResistance


def load_strategy() -> PilotedPolicy:
    with open("config_data/piloted_fyne.pkl", "rb") as f:
        return pickle.load(f)


def load_report(config):
    return load_counts(config)


def trainable(environment):
    def _trainable(config: dict):
        os.chdir(
            os.environ["TUNE_ORIG_WORKING_DIR"]
        )  # See: https://github.com/ray-project/ray/pull/22803
        slim_config = Config(
            "config_data/config.json",
            environment,
            save_rate=0,
            farms_per_process=1,
        )
        # TODO: support for nested property updates would be nice...
        treatment = slim_config.get_treatment(Treatment.EMB)
        treatment.pheno_resistance = {
            HeterozygousResistance.DOMINANT: config[
                "EMB_pheno_resistance_homozygous_dominant"
            ],
            HeterozygousResistance.INCOMPLETELY_DOMINANT: config[
                "EMB_pheno_resistance_heterozygous_dominant"
            ],
            HeterozygousResistance.RECESSIVE: config[
                "EMB_pheno_resistance_homozygous_recessive"
            ],
        }

        days = 450

        slim_config.end_date = slim_config.start_date + timedelta(days=days)
        simulator = get_env(slim_config)
        strategy = load_strategy()
        simulator.reset()

        report_df = load_report(slim_config)
        counts_per_farm = (
            report_df.groupby("site_name")[["survived_fish", "lice_count", "date"]]
            .apply(lambda x: x.reset_index().to_dict())
            .to_dict()
        )

        loss = 0.0
        for _ in tqdm.trange(days):
            for agent in simulator.agents:
                action = strategy.predict(None, agent)
                simulator.step(
                    action
                )  # note: this assumes these two agent lists match...

            simulator_pz: SimulatorPZEnv = simulator.unwrapped

            last_fish_pop = simulator_pz._fish_pop[-1]
            last_lice_pop = simulator_pz._adult_females_agg[-1]

            for agent in range(simulator.num_agents):
                agent_name = slim_config.farms[agent].name
                farm_df = counts_per_farm.get(agent_name)

                day = simulator_pz.cur_day
                idx = pd.Series(farm_df["date"]).searchsorted(day)
                delta_fish = (
                    farm_df["survived_fish"][idx] - last_fish_pop[f"farm_{agent}"]
                )
                delta_lice = farm_df["lice_count"][idx] - last_lice_pop[f"farm_{agent}"]
                if np.isnan(delta_fish):
                    delta_fish = 0.0
                if np.isnan(delta_lice):
                    delta_lice = 0.0
                loss += delta_fish**2 + delta_lice**2

        tune.report(mean_loss=loss)

    return _trainable


search_space = {
    "fish_mortality_k": (5, 20),
    "fish_mortality_center": (0, 0.5),
    "reproduction_eggs_first_extruded": (100, 500),
    # "geno_mutation_rate": tune.loguniform(1e-5, 1e-2),
    "EMB_pheno_resistance_homozygous_dominant": (0.7, 1.0),
    "EMB_pheno_resistance_heterozygous_dominant": (0, 1.0),
    "EMB_pheno_resistance_homozygous_recessive": (0, 0.3),
    # TODO: fish mortality due to EMB
    # TODO: add thermolicer
    "male_detachment_rate": (0.5, 1.0),
}


def main(environment, logdir=None, num_samples=-1):
    """
    algo = TuneBOHB(metric="mean_loss", mode="min")
    scheduler = HyperBandForBOHB(
        time_attr="training_iteration",
        metric="mean_loss",
        mode="min",
    )
    """
    algo = BayesOptSearch(search_space, metric="mean_loss", mode="min")
    scheduler = None

    analysis = tune.run(
        trainable(environment),
        local_dir=logdir,
        name="SLIM Fitter",
        # config=search_space,
        scheduler=scheduler,
        search_alg=algo,
        num_samples=num_samples,
    )
    best_result = analysis.get_best_logdir("mean_loss", mode="min")
    print(best_result)


if __name__ == "__main__":

    parser = slim.common_cli_options("SLIM Fitter")
    parser.add_argument(
        "--num_samples",
        help="The number of iterations. -1 = infinite",
        type=int,
        default=-1,
    )

    parser.add_argument("--logdir", help="The output logging directory.")

    args = parser.parse_args()
    main(args.environment, logdir=args.logdir, num_samples=args.num_samples)
