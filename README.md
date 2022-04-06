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
- [ ] policy searching strategies
- [ ] a game theoretic optimisation framework

## Installation

Details on required packages and versions can be found in ```environment.yml``` which can be used to create a
[conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) environment for the project.

```bash
git clone https://github.com/resistance-modelling/slim slim-master
cd slim-master
# make sure conda is installed and the command is available at this point
conda env update
```

## Usage

To run the model run the following from the root folder:

```python -m slim.SeaLiceMgmt output_folder/simulation_name simulation_params_directory```

For example:
```python -m slim.SeaLiceMgmt out/0 config_data/Fyne```

### GUI
We also provide a GUI for debugging and visualisation. Its support is still heavily experimental so please
use with caution.

You can launch it via `python -m slim.SeaLiceMgmtGUI` and provide your run data (generated via the `--save-rate` option mentioned
above) from the given menu.

Please check our [quickstart](https://slim.readthedocs.io/en/stable/getting-started.html) guide for more information.

## Deprecated: run the original model

To run the original code, enter ```original``` directory and run:

```python model/slm.py param_module_name output_folder simulation_name```

For example:

```python model/slm.py dummy_vars /out/ 0```

*Note that at the moment dummy_vars is a copy of Fyne_vars and code execution takes a while.*
