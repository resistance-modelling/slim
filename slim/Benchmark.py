#!/usr/bin/env python

"""This module provides a standalone script to perform a number of simulations in parallel."""

__all__ = []

import datetime
import math
import hashlib

import numpy as np
import pyarrow.parquet as paq
import ray

from slim import common_cli_options, get_config
from slim.log import logger
from slim.simulation.simulator import get_simulation_path, Simulator
from slim.simulation.config import Config
from slim.types.treatments import Treatment


__all__ = ["main"]

@ray.remote
def launch(cfg: Config, rng, out_path, **kwargs):
    trials = kwargs.pop("trials")
    quiet = kwargs.pop("quiet", False)
    orig_name = cfg.name
    out_path = out_path / cfg.name
    hash_ = hashlib.sha256()

    defection_proba = kwargs.get("defection_proba")
    recurrent_treatment_type = kwargs.get("recurrent_treatment_type")
    recurrent_treatment_freq = kwargs.get("recurrent_treatment_frequency")

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

        if defection_proba is not None:
            for farm in cfg.farms:
                farm.defection_proba = defection_proba

        if not (recurrent_treatment_type is None or recurrent_treatment_freq is None):
            sim_start = cfg.start_date + datetime.timedelta(days=100)
            duration = (cfg.end_date - sim_start).days
            # TODO: if more treatments are applied this needs being extended.
            treatment_dict = {
                "emb": Treatment.EMB,
                "thermolicer": Treatment.THERMOLICER,
                "cleanerfish": Treatment.CLEANERFISH,
            }
            treatment = treatment_dict[recurrent_treatment_type]
            num_events = math.ceil(duration / recurrent_treatment_freq)
            print(num_events, recurrent_treatment_freq)
            for farm in cfg.farms:
                farm.treatment_dates = [
                    (
                        sim_start
                        + datetime.timedelta(days=i * recurrent_treatment_freq),
                        treatment,
                    )
                    for i in range(num_events)
                ]
                print(farm.treatment_dates)

        sim = Simulator(out_path, cfg)
        sim.run_model(quiet=quiet)


def main():
    """
    Main entry point of the benchmark script.
    """
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
        required=True,
    )
    bench_group.add_argument(
        "--parallel-trials",
        type=int,
        help="How many trials to perform in parallel",
        required=True,
    )

    bench_group.add_argument(
        "--defection-proba",
        type=float,
        help="If using bernoulli, the defection probability for each farm",
    )

    bench_group.add_argument(
        "--recurrent-treatment-type",
        type=str,
        help="Create regular treatments of the specified type",
    )

    bench_group.add_argument(
        "--recurrent-treatment-frequency",
        type=int,
        help="Frequency of the treatments in days",
    )

    cfg, args, out_path = get_config(parser)

    ss = np.random.SeedSequence(args.bench_seed)
    child_seeds = ss.spawn(args.parallel_trials)
    args.trials = math.ceil(args.trials / args.parallel_trials)
    tasks = [
        launch.remote(cfg, np.random.default_rng(s), out_path, **vars(args))
        for s in child_seeds
    ]
    ray.get(tasks)


if __name__ == "__main__":
    main()
