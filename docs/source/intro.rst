Introduction
============

SLIM is an FOSS sea lice lifecycle simulation tool.
Its intended goal is to simulate sea lice infestations
in salmonid farms.

However, it is more than a simple epidemic simulator. The tool
is intended to answer *What if?* questions in terms of treatment strategies.
For example, *What will be our loss if we apply treatment as soon as the adult female aggregation ratio reaches x?*
*How long does it take before treatment resistance becomes significant? (regardless of what we mean by that)*

.. _Rationale:

Rationale
*********

**Sea lice** (singular *sea louse*) are a type of parasitic organism that occurs on salmonid species (salmon, trout, char).
We differentiate between **farm salmon** (in the sea cages) and **wild fish** (wild salmonid species in the *reservoir*,
that is, external environment). Salmon in sea cages can be infected when the sea lice cross into the sea cage from the
reservoir where it occurs on wild fish.

Chemical or biological *treatment* can be applied to the sea cages against the sea lice. Currently, the only
chemical treatment modelled in here is *Emamectin Benzoate* or *EMB*. Regardless of the adopted pesticide, there is
extensive proof that **genetic resistance to treatment developed after a few years**,
making each treatment cycle less and less effective. Therefore, managers of salmon farms have resorted to a wide range of alternatives.

In the past few years, biological treatment via *cleaner fish* (typically lumpfish and ballan wrasse) has been introduced
with mixed results. Another solution deployed is a time break between the farming cycles, typically known as
*fallowing*, in order to reduce the number of lice surrounding the cages before repopulating them again.

All treatments have a non-null cost both in economical terms, collateral damage (intoxication, increased stress
levels etc.) and/or ecological impact on other cultures. While farmers and managers in Scotland typically belong to cooperatives or consortiums, and have cooperation agreements requiring treatment coordination, they agreements can lack legal force. Therefore, defection and non-coordination is possible.  

SLIM is the Sea Lice Model associated with a funded BBSRC project on the evolution to resistance to treatment in sea
lice (BBR009309). The project kicked off at the University of Stirling, Scotland and is now co-developed by
the University of Stirling and University of Glasgow. 

Features of this Program
************************

SLIM is, in essence, a statistical model of sea lice epidemics that aims to be `the opposite of this <https://phdcomics.com/comics/archive.php?comicid=1689>`_.
That is, we aim for this project to provide a battery-ready developing and usage experience.

Overall, this project offers the following tools:

* A simulator (See :py:mod:`slim.SeaLiceMgmt`)
* A visualisation tool based on PyQt5 (See :py:mod:`slim.SeaLiceMgmtGUI`)
* A (deprecated) strategy optimiser (See :py:mod:`slim.Optimiser`)
* A benchmark tool (See :py:mod:`slim.Benchmark`)
* A fitter for reports (See :py:mod:`slim.Fitter`)

There are thus two designed workflows:

1. Use the simulator to generate a run artifact, then run the GUI to visualise such data.
2. Use the fitter to perform a fit on recorded behaviour
3. Find an optimal strategy for a given configuration.

The simulator is typically executed as a standalone command, but it provides a relatively easy Python API for
embedding purposes.

The simulator is also battle-tested (see :py:mod:`tests`).

What makes this simulator enticing, however, is the underlying model. For a longer description of what and how we
simulate please visit :ref:`Model Overview`.