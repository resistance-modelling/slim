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

The project models an environment in which an :py:class:`slim.simulation.organisation.Organisation` of salmon farmers - which reside
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

Gene encoding
*************

Let :math:`\mathbb{G}` be the set of genes. Each gene :math:`g` can appear in at 3 forms (*alleles*): *dominant*
(represented as :math:`G`), *recessive* (:math:`g`) and *heterozygous dominant* (:math:`Gg`), typically grouped in
the set :math:`\mathbb{A_g}`. :math:`A2G(G) = g` gives the representative gene of an allele.

We represent the genotype distribution of the population :math:`m`, time :math:`t`, farm :math:`f` and cage :math:`c`
as :math:`G^m_{tfc}`.

This function is defined as :math:`\mathbb{Allele} \rightarrow \mathbb{N}`, that is it associate
alleles to a discrete number.

The frequency of a single allele :math:`x` is represented by :math:`G^m_{tfc}(X=x)`, with the following constraints:

* All the frequencies of the alleles belonging to the same gene must sum to the same gross count
* All the frequencies of distinct genes must coincide - i.e. they provide different yet consistent views on the same
  distribution
* All the frequencies must be positive integers.

In symbols:

.. math::
     \begin{split}
     &S_g(G^m_{tfc}) = \sum_{x \in \mathbb{A_g}} G^m_{tfc}(X=x) = N^m_{tfc} \\
     &\forall g \in \mathbb{G}. S_g(G^m_{tfc}) = N^m_{tfc} \\
     &\forall x \in A. G^m_{tfc} \in \mathbb{N}
     \end{split}

One can see the different genes as independently identically distributed (i.i.d.) variables with no correlation or
whatsoever.

Treatment
*********

Treatment affects the lice population in different ways, depending on the genotype of the population and the
type of treatment. We model two types of treatment: chemical and non-chemical treatments. The full list
of supported treatments can be found in :py:class:`slim.types.treatment_types.Treatment`.

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

Evolution
*********

Similarly to [#Stien02]_ we avoid to model age dependencies explicitly to keep the model complexity simple
and rather use explicit rates across stages. However, due to interesting interactions between fish population,
weight and lice infection we preferred to use [#Aldrin17]_'s approach to infection.

+------------+--------------------------+--------------+
| Stage      | Approach                 | Source       |
+============+==========================+==============+
| Eggs       | Explicit hatching queue  | [#Aldrin17]_ |
+------------+--------------------------+--------------+
| Nauplius   | Evolution rate           | [#Stien02]_  |
+------------+--------------------------+--------------+
| Copepopids | Explicit infection       | [#Aldrin17]_ |
+------------+--------------------------+--------------+
| Preadults  | Evolution rate           | [#Stien02]_  |
+------------+--------------------------+--------------+
| Adults     | Evolution rate           | [#Stien02]_  |
+------------+--------------------------+--------------+

.. _Reproduction:

Reproduction
************

During mating alleles are recombined according to a Mendelian approach. The number of
reproduction events is calculated on the *estimated* number of matings that can happen on a single
host. We assume a scenario in which one female lice can mate with only one male lice
before being fecundated. As in [#Cox17]_ we estimate such number via a negative multinomial
distribution.

Assuming a louse can attach to either fish, we reduce this problem to finding the
probability that :math:`k` lice land to :math:`n` fish, and then find such expectation.

The expectation can be found to be: [#ExpInfectedFish]_

.. math::
   \mu = \mathbb{E}[X] = n\left[ 1 - \left(\frac{n-1}{n}\right)^k \right]

To use the negative multinomial distribution we also need a VMR: variance-to-mean ratio.
The variance in this case refers to the variance of the *lice occupation*
:math:`Y \sim Multinomial(\textbf{p} = \left(\frac{1}{n}, \ldots \right))` which we
reduced to a multinomial problem. Thus the formula for the variance is the usual (with a notation abuse for X):

.. math::
   \nu = \mathbb{Var}[Y] = k^2 \frac{n-1}{n^2}

The probability of a matching occurring between two lice on the same host is thus:

.. math::
   p_{mating} = 1 - \left(1 + \frac{N_{AM}}{\left(N_{AM} + N_{AF'}\right)\gamma}\right)^{-(1 + \gamma)}

where:

.. math::
    \mu_l &= \frac{N_{AM} + N_{AF'}}{\mu}\\
    VMR &= \frac{\nu}{\mu} \\
    \gamma &= \frac{\mu_l}{VMR - 1}


The number of matings is thus just a proportion on the number of free females (denoted with :math:`N_AF'`)

The number of produced eggs is defined in a similar way to [#Aldrin17]_ and follows a power law
parametrised on the (virtual) age distribution. For details see :meth:`slim.simulation.cage.Cage.get_num_eggs`

Once an adult female has been breed she enters a refractory period that lasts up to 3 days.

Matings and gene inheritance
----------------------------

Explicit mating modelling would be pointless without addressing the genomics.

In principle, inheritance follows a Mendelian scheme: because of heterozygosity there are only a handful
of valid pairings that can yield to 3 distinct traits, each of them with different probabilities
depending on the parents' genes.

To accommodate the need for fast processing, we use double counting to consider the number of possible
matches, then use a multinomial distribution to generate the bespoke genotype distribution given
such (normalised) pairings.


Offspring distribution
**********************

There are two mechanisms that distribute the offspring: farm-to-farm movements and external pressure integration.

Farm-to-Farm movements
----------------------

The majority of lice offsprings is typically lost, some are transmitted to neighbouring farms and very few are
reintegrated into the reservoir. For each pair of farms :math:`f_i` and :math:`f_j` we have data on the travel distance
:math:`d_{i,j}` and lice rate :math:`r_{i,j}`.

If a farm :math:`f_i` generates :math:`N^{Egg}_i` eggs with genotype distribution :math:`G^{Egg}_i` these will be distributed to the farm :math:`f_j`
with the following formulae:

.. math::
    \begin{split}
    &t' = t + \lambda (d_{i,j}) \\
    &n_j = \min (\lambda (N^{Egg}_i * r_{i,j}), N^{Egg}_i) \\
    &n_{jk} = Multinomial(n_j) \\
    &G^{Eggs}_{t'jk} = G^{Egg}_i \cdot \frac{n_{jk}}{N^{Egg}_i}
    \end{split}

Where :math:`\lambda` is the Poisson distribution.

External Pressure
-----------------

Quite differently from previous authors, we only model the external pressure without modelling a reservoir
*as a special cage*. Instead, we consider the reservoir as an infinite, _dynamic_ generator of new lice.

Indeed, we assume that a fraction of the eggs extruded during mating will be reabsorbed by the loch rather than
just disappearing, for example by wild salmons. The external pressure therefore emits a dynamic number
of lice depending on two things:

1. the average offspring throughput over the last 30 days;
2. the genotype distribution of such offspring.

As for the first requirement, the overall number of lice is: :math:`N_{EXT}^t = N_{EXT}^0 + k{N_{Eggs}^{t-30...t-1}}`.
As for the second, we use a Dirichlet-Multinomial Bayesian process to infer the genotype ratio of the new lice.
The objective is to guarantee a _reserve_ of each genotype (even rare ones) while favouring the most prolific trait.

Appendix
........

.. [1] https://www.marine.ie/Home/site-area/areas-activity/aquaculture/sea-lice
.. [#Aldrin17] `"A stage-structured Bayesian hierarchical model for salmon lice populations at individual salmon farms – Estimated from multiple farm data sets" by Aldrin et al. 2017 <https://doi.org/10.1016/j.ecolmodel.2017.05.019>`_
.. [#Cox17] `"Mate limitation in sea lice infesting wild salmon hosts: the influence of parasite sex ratio and aggregation" by Cox et al. 2017_ <https://doi.org/10.1002/ecs2.2040>`_
.. [#Jensen17] `"A selection study on a laboratory-designed population of salmon lice (Lepeophtheirus salmonis) using organophosphate and pyrethroid pesticides" by Jensen et al. 2017 <https://doi.org/10.1371/journal.pone.0178068>`_
.. [#Stien02] `Population dynamics of salmon lice Lepeophtheirus salmonis on Atlantic salmon and sea trout <http://dx.doi.org/10.3354/meps290263>`_
.. [#ExpInfectedFish] `Stack Exchange <https://stats.stackexchange.com/a/296053/328123>`_
