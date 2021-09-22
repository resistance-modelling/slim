[![codecov](https://codecov.io/gh/resistance-modelling/slim/branch/master/graph/badge.svg?token=ykA9vESc7B)](https://codecov.io/gh/resistance-modelling/slim)

<p align="center">
<img src = "https://user-images.githubusercontent.com/6224231/126653948-4d698656-b22f-4dbd-9bee-e919b56407aa.png" width="60%"/>
</p>


# Sea Lice Modelling

# Introduction

**Sea lice** (*singular sea louse*) are a type of parasitic organisms that occur on salmonid species (salmon, trout, char).
We differentiate between farm **salmon** (in the **sea cages**) and **wild fish** (wild salmonid species in the **reservoir**,
that is external environment). Salmon in sea cages can be infected when the sea lice crosses into the sea cage from the
reservoir where it occurs on the wild fish.

Chemical or biological **treatment** can be applied to the sea cages against the sea lice. Currently the treatment in the
model is modelled after a chemical treatment (Emamectin Benzoate, *EMB*). Unfortunately, genetic resistance naturally
develops after a few years, hence farmers have resorted to a wide range of alternatives.

In the past few years, biological treatment via cleaner fish (typically lumpfish and ballan wrasse) has been introduced
with mixed results. Another solution deployed is a time break between the farming cycles (growing a salmon to be ready
for harvest) so that the sea lice population dies down before populating the cages with salmon again.

The main goal of this project is to find efficient treatment options that maximise a "payoff" function, defined in terms
of the number of fish currently alive and well in a farm minus treatment costs. Each agent can decide a range of
treatment options and decide to either collaborate with other farmers (ie. by applying treatment on the same day) or not.

This project thus includes the following components:

- [X] a simulator (the core is complete)
- [ ] a visualisation tool
- [ ] policy searching strategies
- [ ] a game theoretic optimisation framework

## Requirements to run
Details on required packages and versions can be found in ```environment.yml``` which can be used to create a
[conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) environment for the project.
In short, all you need to do is install a conda distribution (see earlier link) and run `conda env update`.

## Usage

It is estimated that our refactored model has fully reimplemented the original model under `original`.
Until a final QA  decision is released, please use the source code under `refactor`.

To run the refactored model, enter the ```refactor``` directory and run:

```python -m src.SeaLiceMgmt output_folder/simulation_name simulation_params_directory```

For example:
```python -m src.SeaLiceMgmt out/0 config_data/Fyne```

### Simulation parameters and CLI override

An _environmental setup_ consists of a subfolder containing three files:

- `params.json` with simulation parameters specific to the organisation
- `interfarm_time.csv` with travel time of sealice between any two given farms (as a dense CSV matrix)
- `interfarm_prob.csv` with probability of sealice travel between any two given farms (as a dense CSV matrix)

See `refactor/config_data/Fyne` for examples.

Additionally, global simulation constants are provided inside `config_data/`.

### Advanced features

#### CLI overridal

If one wishes to modify a runtime option without modifying those files an extra CLI parameter can be passed to the command.
In general, for  each key in the format `a_b_c` an automatic parameter in the format "--a-b-c" will be generated.
For example:

```python -m src.SeaLiceMgmt out/0 config_data/Fyne --seed=0 --genetic-mechanism=discrete```

For now, nested and list properties are not yet supported.

TODO: add help CLI option

#### Debugging

For efficiency reasons, this only saves a snapshot of the model at the end of the simulation. To generate
a dump every `n` days add the `--save-rate=n` option. 

To _resume_ a session one can instead pass the `--resume` parameter, e.g. 

`python -m src.SeaLiceMgmt outputs/sim_1 config_data/Fyne --end-date "2018-01-01 00:00:00" --resume="2017-12-05 00:00:00"`

### GUI
We also provide a GUI for debugging and visualisation. Its support is still heavily experimental so please
use with caution.

You can lauch it via `python -m src.SeaLiceMgmtGUI` and provide your run data (generated via the `--save-rate` option mentioned
above) from the given menu.

## Testing

The refactored model is being thoroughly tested thanks to unit testing, integration testing and (intelligent) type
checking.

To test, also from ```refactor``` directory, run:

```pytest```

To enable coverage reporting, run:

```pytest --cov=src/ --cov-config=.coveragerc  --cov-report html --cov-context=test```

The last two parameters generate a human-friendly report.

For type checking, install pytype and run (from the root folder):

```pytype --config pytype.cfg refactor```


## Deprecated: run the original model

To run the original code, enter ```original``` directory and run:

```python model/slm.py param_module_name output_folder simulation_name```

For example:

```python model/slm.py dummy_vars /out/ 0```

*Note that at the moment dummy_vars is a copy of Fyne_vars and code execution takes a while.*
