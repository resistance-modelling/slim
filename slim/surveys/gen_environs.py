#!/bin/env python

"""
This script generates a few artificial networks.
"""

import argparse
from pathlib import Path
import json

import numpy as np

preamble = {
    "name": "Loch Fyne ACME Ltd.",
    "start_date": "2018-01-01 00:00:00",
    "end_date": "2020-01-01 23:59:59",
    "ext_pressure": 100,
    "monthly_cost": 5000.00,
    "genetic_ratios": {
        "A": 0.0001,
        "a": 0.999,
        "Aa": 0.0009
    },
    "gain_per_kg": 4.39,
    "infection_discount": 1e-7,
    "genetic_learning_rate": 0.01,
    "agg_rate_suggested_threshold": 2.0,
    "agg_rate_enforcement_threshold": 6.0,
    "agg_rate_enforcement_strikes": 4,
    "treatment_strategy": "bernoulli",
}
farm = {
    "ncages": 2,
    "location": [
        190300,
        665300
    ],
    "num_fish": 40000,
    "start_date": "2017-10-01 00:00:00",
    "cages_start_dates": [
        "2017-10-01 00:00:00",
        "2017-10-08 00:00:00",
    ],
    "treatment_types": [
        "emb", "thermolicer", "cleanerfish"
    ],
    "treatment_dates": [],
    "max_num_treatments": 10,
    "sampling_spacing": 7,
    "defection_proba": 0.2
}

temperatures = np.array([[685715,8.2,7.55,7.45,8.25,9.65,11.35,13.15,13.75,13.65,12.85,11.75,9.85],
                [665300,8.4,7.8,7.7,8.6,9.8,11.65,13.4,13.9,13.9,13.2,12.15,10.2]])


def circular_path(N=9, directed=False, circular=False):
    eye = np.eye(N, dtype=np.float64)
    next_right = np.roll(eye, 1, axis=1) if circular else np.pad(eye, ((0, 0), (1, 0)))[:, :-1]
    next_down = np.roll(eye, 1, axis=0) if circular else np.pad(eye, ((1, 0), (0, 0)))[:-1, :]
    
    movement_matrix = eye + next_right
    if not directed:
        movement_matrix += next_down
    
    prob_matrix = movement_matrix * 1.5e-2
    time_matrix = movement_matrix * 7
    
    farms = [{**farm, "name": f"i"} for i in range(N)]

    return prob_matrix, time_matrix, {**preamble, "farms": farms}


def mesh(N=9):
    mesh_network = np.ones((N, N), dtype=np.float64)
    movement_matrix = mesh_network * 1e-2
    time_matrix = mesh_network * 7

    farms = [{**farm, "name": f"i"} for i in range(N)]
    return movement_matrix, time_matrix, {**preamble, "farms": farms}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name")
    parser.add_argument("command")
    parser.add_argument("-N", type=int, required=True)

    args = parser.parse_args()
    if args.command == "path":
        output = circular_path(args.N, directed=True, circular=False)
    elif args.command == "bipath":
        output = circular_path(args.N, directed=False, circular=False)
    elif args.command == "circle":
        output = circular_path(args.N, directed=True, circular=True)
    elif args.command == "bicircle":
        output = circular_path(args.N, directed=False, circular=True)
    elif args.command == "mesh":
        output = mesh(args.N)
    else:
        raise Exception("Wrong command")

    env_folder = Path(args.env_name)
    env_folder.mkdir(parents=True, exist_ok=True)
    interfarm_prob_csv = env_folder / "interfarm_prob.csv"
    interfarm_time_csv = env_folder / "interfarm_time.csv"
    temperatures_csv = env_folder / "temperatures.csv"
    param_json = env_folder / "params.json"

    np.savetxt(str(interfarm_prob_csv), output[0], delimiter=",")
    np.savetxt(str(interfarm_time_csv), output[1], delimiter=",")
    np.savetxt(str(temperatures_csv), temperatures, delimiter=",")
    with param_json.open("w") as f:
        json.dump(output[2], f)

if __name__ == "__main__":
    main()