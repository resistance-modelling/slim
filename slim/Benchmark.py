__all__ = []

import random

import numpy as np
import ray

from slim import common_cli_options, get_config
from slim.simulation.simulator import get_simulation_path, Simulator, load_artifact


@ray.remote
def load_data(cfg, rng, trials, out_path, **kwargs):
    quiet = kwargs.pop("quiet", False)

    for t in range(trials):
        artifact = get_simulation_path(out_path, cfg)[0]
        extra_args = kwargs.copy()
        extra_args["seed"] = rng.random()
        sim_name = cfg.name = cfg.name + str(extra_args)

        if not artifact.exists():
            print(f"Artifact not yet generated, running {sim_name}")
            # defection_proba is farm-specific and needs special handling

            defection_proba = kwargs.pop("defection_proba", None)

            if defection_proba is not None:
                for farm in cfg.farms:
                    farm.defection_proba = defection_proba

            sim = Simulator(out_path, cfg)
            sim.run_model(quiet=quiet)

        return load_artifact(out_path, sim_name)


def main():
    parser = common_cli_options("SLIM Benchmark tool")
    bench_group = parser.add_argument_group(title="Benchmark-specific options")
    bench_group.add_argument(
        "--bench-seed",
        type=int,
        help="Seed to generate other benchmark seeds",
        default=0
    )
    bench_group.add_argument(
        "--trials",
        type=int,
        help="How many trials to perform during benchmarking",
        required=True)
    bench_group.add_argument(
        "--parallel-trials",
        type=int,
        help="How many trials to perform in parallel",
        required=True)

    cfg, args, out_path = get_config(parser)
    ss = np.random.SeedSequence(args.bench_seed)
    child_seeds = ss.spawn(args.parallel_trials)

    tasks = [load_data.remote(cfg, np.random.default_rng(s), out_path, **args) for s in child_seeds]
    ray.get(tasks)


if __name__ == "__main__":
    main()
