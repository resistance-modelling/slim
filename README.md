[![codecov](https://codecov.io/gh/resistance-modelling/slim/branch/master/graph/badge.svg?token=ykA9vESc7B)](https://codecov.io/gh/resistance-modelling/slim)
[![Documentation Status](https://readthedocs.org/projects/slim/badge/?version=latest)](https://slim.readthedocs.io/en/latest/?badge=latest)

<p align="center">
<img src="https://github.com/resistance-modelling/slim/raw/master/res/logo_transparent.png" width="60%"/>
</p>


# SLIM: Sea Lice Modelling

## Introduction

**Sea lice** (*singular sea louse*) are a type of parasitic organisms that occur on salmonid species (salmon, trout, char).

The main goal of this project is to find efficient treatment options that maximise a "payoff" function, defined in terms
of the number of fish currently alive and well in a farm minus treatment costs. Each agent can decide a range of
treatment options and decide to either collaborate with other farmers (ie. by applying treatment on the same day) or not.

This project thus includes the following components:

- [X] a simulator (the core is complete)
- [X] a visualisation tool
- [X] (NEW) fitting on official report
- [ ] policy searching strategies
- [ ] a game theoretic optimisation framework

## Installation

We recommend using a virtual environment to run slim. For version 3.12 we commend using a virtual environment, for
older versions, conda may be a better choice.

### Virtual Environment

Python has the ability to create lightweight “virtual environments”, each with their own independent 
set of Python packages installed in their site directories. 
When used from within a virtual environment, common installation tools such as pip will install Python packages into a virtual environment,
allowing us to specify versions of libraries to use.

To create a virtual environment and configure it for use with slim:
```bash
python3 -m venv --without-pip <path_to_virtual_env>
source <path_to_virtual_env>/bin/activate
python3 -m pip install -r requirements.txt
```
where <path_to_virtual_env> is the name of your envoironment, e.g. slim_env, and 
the --without-pip command creates an environment without copying pip (which can make 
a bloated virtual environment).



### Conda 

Details on required packages and versions can be found in ```environment.yml``` which can be used to create a
[conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) environment for the project.

```
git clone https://github.com/resistance-modelling/slim slim-master
cd slim-master
# make sure conda is installed and the command is available at this point
conda env update
# This will make the slim module globally available.
pip install -e .
```

## Usage

Running slim on the command line is simply

```
python3 -m slim.SeaLiceMgmt output_folder/simulation_name simulation_params_directory```
```

For example:
```slim run out/0 config_data/Fyne```

This will create 2 files; simulation_data_<simulation_name>.parquet that contains the results of the simulation
and a .pickle file can can be used for restarting the simulation

The configuration data directory requires config.json, params.json that define 
the details of the simulations parameters and the farms, and their management.

The configuration data directory also needs the following data files;
interfarm_prob.csv, interfarm_time.csv, report.csv,  temperatures.csv.


When you are finished the simulation, you can `deactivate` the environment via
```
deactivate 
```

To use the virtual environment in the future, you can do so via
```aiignore
source <path_to_virtual_env>/bin/activate
```

### GUI

We also provide a GUI for debugging and visualisation. Its support is still heavily experimental so please
use with caution.

You can launch it via

```
python3 -m slim.SeaLiceMgmtGUI output_folder/simulation_name simulation_params_directory
```


Please check our [quickstart](https://slim.readthedocs.io/en/stable/getting-started.html) guide for more information.

## Deprecated: run the original model

To run the original code, enter ```original``` directory and run:

```python model/slm.py param_module_name output_folder simulation_name```

For example:

```python model/slm.py dummy_vars /out/ 0```

*Note that at the moment dummy_vars is a copy of Fyne_vars and code execution takes a while.*
