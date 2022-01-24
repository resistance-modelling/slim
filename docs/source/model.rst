.. _Model Overview:

Model Overview
==============

SLIM is a *probabilistic*, *statistical* epidemiological simulator. Compared to IBM approaches
we simulate overall populations and associated distributions and perform statistical operations on them rather than
emulating individual lice or fish.

At the same time, SLIM is also a *game* simulator where each agent (farmers) seeks to maximise
their own profit and may choose to collaborate or defect (more on this later), which treatment to apply and so on.

This simulation includes explicit encoding
of genomic effects into the evolution of the pandemic. Treatments may encourage resistant
traits to proliferate and thus make the former less and less effective at each cycle.

.. note::
   This section assumes you have some basic knowledge of salmon aquaculture. If
   you don't we suggest having a look at :ref:`Rationale` or [1]_ first.

The project models an environment in which an :py:class:`slim.simulation.simulator.Organisation` of salmon farmers - which reside
and operate on the same *loch* - run their own :py:class:`slim.simulation.farm.Farm` s which are in turn divided into
:py:class:`slim.simulation.cage.Cage` s.

A *cage* is the physical location of salmons. Initially all cages within the same farm are filled
with *smolts* which are then left to grow up and harvested after 1.5-2 years.

Sea lice stay in the reservoir and, either thanks to water currents or wild salmon, leak into
salmon cages. Once they find a host to attach to they evolve from *recruitment* or *copepopid* to
*chalimus* and beyond. SLIM models cage-specific, stage-specific lice populations and assumes concentration
and pollution effects as farms are operative for long periods of time.

As lice attach to fish they cause a number of diseases which result in fish death.
If the *lice aggregation* ratio exceeds a given threshold treatment must be performed. However,
other managers not reaching that threshold may be faring better and opt to `defect` by failing to treat.

Whether egoism backfires or not is not an easy question as it
depends on the current stage of the epidemics, lake pollution and how water currents work in the specific simulation
environment.

Note that farm managers are individual agents, meaning they have full freedom to stock cages in
any season they prefer and treat when needed, and also to decide which treatment to apply.

.. note::
   The exact formulae behind this are documented in our paper. When it comes out please check it out!

Sea Lice Population
*******************

Our model is built on 3 main papers [#Aldrin17]_ [#Cox17]_ [#Jensen17]_:

Just like in [#Aldrin17]_, we model the lice lifecycle as a compartmentalised model of 6 stages: *recruitment* (R),
*copepopids* (CO), *chalimus* (CH), *preadults* (PA) and *adults*, the latter divided in males (AM) and females (AF).
Differently from [#Aldrin17]_ we omit the explicit age encoding but instead group the population
by its genotype. Nevertheless, the age distribution can be simulated at will via
:py:meth:`slim.simulation.cage.Cage.get_stage_ages_distrib`.

A genotype distribution of the lice population as related to resistance to treatment is modelled by the
:py:class:`slim.simulation.lice_population.GenoDistrib` class. A genotype distribution
is, ultimately, a discretised dictionary with where the keys are allele combinations (see
:py:attr:`slim.simulation.lice_population.GenoDistrib.alleles`) and the values are the actual number of lice in that group.

In a similar way to [#Aldrin17]_, each day we estimate how many lice die or evolve at every stage.

All stages are subject to the following:

* background mortality;
* treatment-induced mortality;
* cleaner fish mortality (*not yet implemented*);
* evolution (can be either additive from the previous stage or subtractive to the successive stage);
* age mortality (i.e. lice that could not evolve will die).

In all stages but AM or AF lice can evolve. Evolution from PA will result in roughly 50% split between AM and AF.

Treatment
*********

Treatment affects the lice population in different ways, depending on the genotype of the population and the
type of treatment. We model two types of treatment: chemical and non-chemical treatments. The full list
of supported treatments can be found in :py:class:`slim.types.TreatmentTypes.Treatment`.

When a treatment is administered some delay occurs before effects are noticeable (non-chemical
treatments have a virtual delay of one day). The mortality rate (a multiplicative factor of the given
population) is computed in
:py:meth:`slim.simulation.cage.Cage.get_lice_treatment_mortality_rate`. In the case of EMB
it is the following:

.. math::
   \mu^{EMB}_{tfcg} = \begin{cases}
    1 - \phi^{EMB}(g) &\text{if t } \in [t_{fcb} + \delta^{EMB}, t_{fcb} + \Delta^{dur}]\\
    0 &\text{otherwise}
    \end{cases}

where:

* :math:`t, f, c` represent the current time, farm and cage;
* :math:`g` is the chosen genotype;
* :math:`\phi^EMB` is the phenotype resistance corresponding to the given genotype. The codomain is in :math:`[0,1]`
* :math:`t_{fcb}` is the time when a treatment was started;
* :math:`\delta^{EMB}` is the delay of the treatment;
* :math:`\Delta^{dur}` is the efficacy duration, computed as :math:`\delta^{dur} / T_{t^0}` where :math:`T_{t^0}`
  is the average water temperature when the treatment is applied and :math:`\delta^{dur}` is a constant.

In other words, if the current time falls within the efficacy timeframe of a treatment
the mortality rate is computed as the inverse of the resistance rate provided by :math:`\phi^{EMB}`.

Once the mortality rates are computed for each genotype, we use a Poisson distribution to generate the mortality
events and a hypergeometric distribution to choose from which stages to remove lice.

For the sake of notation the stage has been omitted but not all stages are taken into account. For examples,
since EMB affects lice attachment only the stages from Chalimus onward are affected.

For more information check :py:mod:`slim.types.TreatmentTypes`.
.. _Reproduction:

Reproduction
************

During mating alleles are recombined according to a Mendelian approach. The number of
reproduction events is calculated on the *estimated* number of matings that can happen on a single
host. We assume a scenario in which one female lice can mate with only one male lice
before being fecundated. As in [#Cox17]_ we estimate such number via a negative multinomial
distribution.

The number of produced eggs is defined in a similar way to [#Aldrin17]_ and follows a power law
parametrised on the (virtual) age distribution.

TODO: expand on this.

Footnote
--------

.. [1] https://www.marine.ie/Home/site-area/areas-activity/aquaculture/sea-lice
.. [#Aldrin17] `"A stage-structured Bayesian hierarchical model for salmon lice populations at individual salmon farms – Estimated from multiple farm data sets" by Aldrin et al. 2017 <https://doi.org/10.1016/j.ecolmodel.2017.05.019>`_
.. [#Cox17] `"Mate limitation in sea lice infesting wild salmon hosts: the influence of parasite sex ratio and aggregation" by Cox et al. 2017_ <https://doi.org/10.1002/ecs2.2040>`_
.. [#Jensen17] `"A selection study on a laboratory-designed population of salmon lice (Lepeophtheirus salmonis) using organophosphate and pyrethroid pesticides" by Jensen et al. 2017 <https://doi.org/10.1371/journal.pone.0178068>`_
