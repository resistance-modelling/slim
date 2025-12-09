[![codecov](https://codecov.io/gh/resistance-modelling/slim/branch/master/graph/badge.svg?token=ykA9vESc7B)](https://codecov.io/gh/resistance-modelling/slim)
[![Documentation Status](https://readthedocs.org/projects/slim/badge/?version=latest)](https://slim.readthedocs.io/en/latest/?badge=latest)

<p>
<img src="https://github.com/resistance-modelling/slim/raw/master/res/logo_transparent.png" width="60%"/>
</p>


# SLIM: Sea Lice Modelling

## Introduction

**Sea lice** (*singular sea louse*) are a type of parasitic organisms that occur on salmonids (salmon, trout, char).

The main goal of this project is to find efficient treatment options that maximise a "payoff" function, calculated as
the difference between the number of healthy fish in a facility and the expenses incurred for treatment. Each agent can
select from a range of treatment options and decide to either collaborate with other farmers (i.e. by applying treatment
on the same day) or not.

This project thus includes the following components:

- [X] a simulator (the core is complete)
- [X] a visualisation tool
- [X] (NEW) fitting on an official report
- [ ] policy searching strategies
- [ ] a game theoretic optimisation framework

## Installation

 To run slim, we recommend uv, an extremely fast Python package installer and resolver, written in Rust. It handles Python versions, virtual environments, and dependencies in a single tool. Alternatively, a virtual environment can be used. For versions of python prior to 3.12, conda may be a better choice.

First download the repo:
```bash
git clone https://github.com/resistance-modelling/slim.git
```

### uv

If you don't have uv installed, use the standalone installer:

macOS/Linux:
```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
```


Windows:
```bash
powershell -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"
```
The repo contains the configuration files for uv, so no new packages need to be installed. To install packages using uv:
```bash
uv add <package_name>
```

### Virtual Environment

Python can create lightweight “virtual environments”, each with their own independent 
set of Python packages installed in their site directories. 
When used from within a virtual environment, common installation tools such as pip will install Python packages into a virtual environment,
allowing us to specify versions of libraries to use.

To create a virtual environment and configure it for use with slim:
```bash
python3 -m venv --without-pip <path_to_virtual_env>
source <path_to_virtual_env>/bin/activate
python3 -m pip install -r requirements.txt
```
where <path_to_virtual_env> is the name of your environment, e.g. slim_env, and 
the --without-pip command creates an environment without copying pip (which can make 
a bloated virtual environment).

### Conda 

Details on required packages and versions can be found in ```environment.yml``` which can be used to create a
[conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) environment for the project.

```bash
git clone https://github.com/resistance-modelling/slim slim-master
cd slim-master
# make sure conda is installed and the command is available at this point
conda env update
# This will make the slim module globally available.
pip install -e .
```

## Usage

Running slim on the command line using uv can be done as follows:

```bash
uv run -m slim.SeaLiceMgmt <simulation_name> <simulation_params_directory>
```

**Example**:
```bash
uv run -m slim.SeaLiceMgmt my_sim ./config_data/Fyne
```

This creates files named `simulation_data_my_sim.parquet` and `simulation_my_sim.pickle` which can can be used for restarting the simulation.

If using a virtual environment, you will need to activate it before running the simulation.
```bash
source <path_to_virtual_env>/bin/activate
```
and then run the simulation via
```bash
python3 -m slim.SeaLiceMgmt <simulation_name> <simulation_params_directory>
```
and when you are finished the simulation, you can `deactivate` the environment via
```bash
deactivate 
```

### GUI

We also provide a GUI for debugging and visualisation. Its support is still heavily experimental, so please use it with caution.

You can launch it via
```bash
uv run -m slim.SeaLiceMgmtGUI simulation_data_<simulation_name>.parquet <simulation_params_directory>
```
or, in a virtual environment,
```bash
python3 -m slim.SeaLiceMgmtGUI simulation_data_<simulation_name>.parquet <simulation_params_directory>
```

### Plotting

```# 4. Generate plots
uv run -m slim.SeaLiceMgmtPlotGenerator ../output/*_results.parquet
```

## Configuration Files

### Directory Structure

The configuration data directory requires the following file:

```
<simulation_params_directory>/
├── params.json            # Main simulation parameters
├── interfarm_prob.csv     # Probabilities of lice transmission between farms
├── interfarm_time.csv     # Time delays for lice transmission between farms  
├── temperatures.csv       # Water temperature data over time
└── report.csv             # (Optional) Reporting configuration
```

### 1. `params.json` - Main Configuration File

This is the primary configuration file validated against `params.schema.json`. It contains:

```json
{
  "name": "my_simulation",   (The simulation identifier, used in output filenames)
  "start_date": "2020-01-01 00:00:00",
  "end_date": "2022-01-01 00:00:00",
  "ext_pressure": 0.5,       (External infection pressure)
  "genetic_ratios": {        (Initial proportions of genotypes, must sum to 1.0)
    "a": 0.8,
    "Aa": 0.15,
    "A": 0.05
  },
  "genetic_learning_rate": 0.1,
  "monthly_cost": 50000.0,
  "gain_per_kg": 5.0,
  "infection_discount": 0.9,
  "agg_rate_suggested_threshold": 0.5,
  "agg_rate_enforcement_threshold": 1.0,
  "agg_rate_enforcement_strikes": 3,
  "treatment_strategy": "reactive",
  "farms": [    # Array of farm configurations
    {
      "name": "farm_0",
      "num_fish": 50000,
      "ncages": 10,
      "location": [0, 0],
      "start_date": "2020-01-01 00:00:00",
      "cages_start_dates": [
        "2020-01-01 00:00:00",
        "2020-02-01 00:00:00"
      ],
      "max_num_treatments": 6,
      "sampling_spacing": 7,
      "treatment_types": ["emb", "thermolicer", "cleanerfish"],
      "defection_proba": 0.1,
      "treatment_dates": [
        ["2020-06-01 00:00:00", "emb"],
        ["2020-09-01 00:00:00", "thermolicer"]
      ]
    }
  ]
}
```

### 2. `interfarm_prob.csv` - Transmission Probabilities

Square matrix (N x N) where N = number of farms. 
Entry (i,j) = probability of lice transmission from farm i to farm j.

```csv
0.0,0.2,0.1
0.3,0.0,0.15
0.1,0.2,0.0
```

For three farms, this is a 3x3 matrix. Diagonal is typically 0 (no self-transmission).

### 3. `interfarm_time.csv` - Transmission Time Delays

Square matrix (N x N) matching `interfarm_prob.csv`.
Entry (i,j) = days for lice to travel from farm i to farm j.

```csv
0,5,10
4,0,6
8,7,0
```

### 4. `temperatures.csv` - Water Temperature Data

Single column (or row) of temperature values, one per day of simulation.

```csv
8.5
8.4
8.3
8.6
...
```

Length should match the number of days in your simulation (end_date - start_date).

### 5. Configuration Files (in parent directory)

The simulator also needs these in the **parent directory** of your simulation directory:

```
parent_dir/
├── config.schema.json       # Schema for runtime parameters
├── params.schema.json       # Schema for params.json
└── <simulation_params_directory>/
    ├── params.json
    ├── interfarm_prob.csv
    └── ...
```

These schema files validate your JSON configuration.

## Output Files

After running the simulation, the output folder will contain:

```
output/
├── <simulation_name>_results.parquet  # Main results file
├── logs/                              # Log files
└── checkpoints/                       # (Optional) State checkpoints
```

The parquet file contains all simulation data:
- `farm_name`: Identifier for each farm
- `timestamp`: Date/time for each record
- `L1`, `L2`, `L3`, `L4`, `L5m`, `L5f`: Lice counts by life stage and genotype
- `new_reservoir_lice_ratios`: New lice entering from reservoir
- `payoff`: Economic payoff for the farm
- And more...

## Multiprocessing

Multiprocessing is enabled by default. By default, it will allocate one process per farm. To change this, you can set farms_per_process=N in the Config or by passing --farms-per-process=N in the CLI. N represents the maximum number of farms in a single process. The lower, the better (if you can afford it). If N=-1, multiprocessing is disabled.

Note that when running the simulator, an extra process is always created to dump the process output.

## Common Issues

1. **"No such file or directory: params.json"**
   - Make sure params.json is in your simulation directory
   - Check you're passing the correct directory path

2. **"Validation error"**
   - Your params.json doesn't match the schema
   - Check date formats, ensure genetic_ratios sum to 1.0
   - Verify all required fields are present

3. **"Shape mismatch"**
   - interfarm_prob.csv and interfarm_time.csv must be N x N (square matrices)
   - N must match the number of farms in params.json
   - temperatures.csv length must match simulation days

4. **Output folder confusion**
   - The simulation directory (with params.json) is the INPUT
   - The output folder is where RESULTS are saved
   - These are two different directories!

Please check our [quickstart](https://slim.readthedocs.io/en/stable/getting-started.html) guide for more information.

## Deprecated: run the original model

To run the original code, enter ```original``` directory and run:

```python model/slm.py param_module_name output_folder simulation_name```

For example:

```python model/slm.py dummy_vars /out/ 0```

*Note that at the moment dummy_vars is a copy of Fyne_vars and code execution takes a while.*
