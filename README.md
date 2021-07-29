[![codecov](https://codecov.io/gh/resistance-modelling/slim/branch/master/graph/badge.svg?token=ykA9vESc7B)](https://codecov.io/gh/resistance-modelling/slim)

<p align="center">
<img src = "https://user-images.githubusercontent.com/6224231/126653948-4d698656-b22f-4dbd-9bee-e919b56407aa.png" width="60%"/>
</p>


# Sea Lice Modelling
**Sea lice** (*singular sea louse*) are a type of parasitic organisms that occur on salmonid species (salmon, trout, char). We differentiate between farm **salmon** (in the **sea cages**) and **wild fish** (wild salmonid species in the **reservoir**, that is external environment). Salmon in sea cages can be infected when the sea lice crosses into the sea cage from the reservoir where it occurs on the wild fish. 

Chemical or biological **treatment** can be applied to the sea cages against the sea lice. Currently the treatment in the model is modelled after a chemical treatment (Emamectin Benzoate, *EMB*). In past few years, biological treatment has also been deployed (small fish that eats the sea lice is put into the sea cages). Another solution deployed is a time break between the farming cycles (growing a salmon to be ready for harvest) so that the sea lice population dies down before populating the cages with salmon again.


## Salmon
Salmon fish live in a sea cage in a farm. Over time it grows and has a chance of dying.

### Growth rate
- ```10000 / (1 + exp(-0.01 * (t - 475)))``` based  on a fitted logistic curve to data from [FAO](http://www.fao.org/fishery/affris/species-profiles/atlantic-salmon/growth/en/)

### Death rate
- constant background daily rate ```0.00057``` (based on [Scottish fish farm production survey 2016](www.gov.scot/Resource/0052/00524803.pdf)) multiplied by lice coefficient 
- TODO: *see surv.py (compare to threshold of 0.75 lice/g fish)*

## Sea lice
The life cycle of sea lice is modelled as 5 stages:

**L1: Nauplius**

The sea lice nauplii hatch from the eggs and drift in the water with some very limited ability to move. This stage is also called planctonic.

**L2: Copepodid**

Nauplii develop further into copepodids. They also drift in water and have limited movement but can perform initial attachment to the fish (infection). Note that a sea lice needs to be attached to a fish to develop further into the chalimus.

**L3 & L4: Chalimus & Pre-adult**

Chalimus performs further attachement to the fish. They develop into pre-adult sea lice and are now mobile and can move on the host and swim in water column. They cannot yet reproduce.

**L5: Adult**

Adult sea lice can reproduce. They are further split into adult male (L5m) and adult female (L5f).

See [here](https://www.marine.ie/Home/site-area/areas-activity/aquaculture/sea-lice/life-cycle-salmon-louse) for more details and [here](https://oar.marine.ie/bitstream/handle/10793/1590/Irish%20Fisheries%20Bulletin%20No.%2050.pdf?sequence=1&isAllowed=y) for a summary graph (page 6, Figure I).

Additionally:
* Adult male and adult female attached to the same fish can reproduce.
* The sea lice development depends on the water temperature (warmer -> faster)
* The sea lice can develop resistance to a chemical treatment and pass it on to their offspring
* Sea lice have a chance of dying depending on their development stage and gender (background death) and the treatment (treatment death)

### Sea Lice Events
#### Development
Progression in the life cycle of the sea lice. Based on [Aldrin et al 2017](https://doi.org/10.1016/j.ecolmodel.2017.05.019).

#### Infection
Attachement to the host fish. Based on [Aldrin et al 2017](https://doi.org/10.1016/j.ecolmodel.2017.05.019).

#### Mating
Originally devised: sea lice grows on a given fish obtained by this formula: ```adult_males * adult_females * (1/4) * (2.5 / avg_adult_female_lifetime) ```

TODO: source.

New approach: use a sigmoid function on the current adult female population.
Based on [Aldrin et al. 2017](https://doi.org/10.1016/j.ecolmodel.2017.05.019).

Advantages:

- Simple and straightforward

Disadvantages/Opinionated?

- Simplistic: does not take into account the _available_ lice. In particular, it does not consider that lice
  are usually unavailable for 3 days after mating (see assumptions below). TODO: Does this matter?
- Assumes a bias-free sex distribution. Some papers discuss that sex biases may be plausible as lice appear to have
  promiscuous behaviour and propose better models. See [Cox et al., 2017](https://doi.org/10.1002/ecs2.2040).

#### Egg production

Based on Aldrin, but may change according to the assumptions below.

##### Old notes

6-11 broods of 500 ova on average extrude a pair every 3 days (alternatively a constant production of 70 nauplii/day for 20 days post-mating), based on [Costello 2006](https://doi.org/10.1016/j.pt.2006.08.006).

##### Assumptions

- lice is unavailable for 3 days after mating 
- a female will mate as soon as a male is available
- only adult (5F and 5M) sea lice can reproduce
- the offspring's inherited resistence is calculated as linear function of the parents' resistances

#### Background death
Constant rates based on [Stien et al 2005](http://dx.doi.org/10.3354/meps290263).

#### Treatment death
<!-- Treatment death in a stage: ```(1 - individual resistance for individuals in appropriate stage) * treatment effect * (dose presence / decay)```. -->
The treatment death in a stage depends on three factors:

- the type of treatment
- the genotype distribution of the lice
- the temperature

The first and second factor are heavily intertwined: different genetic mechanisms influence resistance in different
ways that are not fully understood to this day. For example, Deltamethrin resistance is mainly affected by mithocondrial
haplotype, while Emabectine resistance is modelled by heterozygous monogenic allele. For more information:


- see [this issue](https://github.com/resistance-modelling/slim/issues/36#issuecomment-883195962)
- additionally, [Jensen et al. (2017)](https://doi.org/10.1371/journal.pone.0178068) discussed the relationship between
  some genes and related resistance.
  
The third factor is mainly taken into account to determine the length for which the treatment has noticeable effects.
For example, it is known that once EMB is applied the effect can range between 10 and 28 days according to the temperature
(the higher, the shorter). These are discussed by Aldrin et al. (2017) (see §2.2.3, equations 19-20).

## Farms, Sea cages & Reservoir
Farm consists of multiple sea cages. Reservoir is the external environment that is also modelled as a constant inflow of sea lice into farms.

### Model outline
Given sea cages populated with fish (based on farm data) at each development stage, at each considered timestep perform the following updates:

- get dead sea lice from background death
- get dead sea lice from treatment death
- progress the development of the sea lice
- progress fish growth
- get dead fish (natural death or from sea lice)
    * remove lice associated with dead fish
- perform infection
- get new offspring from sea lice mating
- update the population deltas (changes in population of fish and sea lice)

Also:
- indicate farm that will be reached by the sea lice in reservoir using probability matrix from Salama et al 2013 and sample cage within the farm randomly
    - TODO: confirm the right citation
    - https://doi.org/10.1016/j.prevetmed.2012.11.005 
    - or https://doi.org/10.3354/aei00077 
    - or https://doi.org/10.1111/jfd.12065
- apply the treatment (if treatment should be applied on a given timestep)

## Treatment
### EMB Treatment Model
TODO: source of data (f_meanEMB, f_sigEMB, EMBmort, env_meanEMB, env_sigEMB)

We follow Jensen et al (2017) and model resistance according to genotype population. Unfortunately
the paper does not provide "survival rates per genotype" but rather expected lifespan under treatment. We currently
approximate with a simple geometric distribution (every day only a fixed fraction of lice belonging to a certain stage
and genetic group can die).

## Usage
### Refactored
To run the WIP refactored model, enter the ```refactor``` directory and run:

```python -m src.SeaLiceMgmt output_folder simulation_name config_json_path simulation_params_directory```

For example:
```python -m src.SeaLiceMgmt out 0 config_data/config.json config_data/Fyne```

#### Simulation parameters directory
The program expects three files in the simulation parameters directory:
- `params.json` with simulation parameters
- `interfarm_time.csv` with travel time of sealice between two given farms
- `interfarm_prob.csv` with probability of sealice travel between two given farms
See `refactor/config_data/Fyne` for examples.

#### Testing

To test, also from ```refactor``` directory, run:

```pytest```

To enable coverage reporting, run:

```pytest --cov=src/ --cov-config=.coveragerc  --cov-report html --cov-context=test```

For type checking, install pytype and run (from the root folder):

```pytype --config pytype.cfg refactor```


### Original
To run the original code, enter ```original``` directory and run:

```python model/slm.py param_module_name output_folder simulation_name```

For example:

```python model/slm.py dummy_vars /out/ 0```

*Note that at the moment dummy_vars is a copy of Fyne_vars and code execution takes a while.*

### Requirements
Details on required packages and versions can be found in ```environment.yml``` which can be used to create a [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) environment for the project. 




