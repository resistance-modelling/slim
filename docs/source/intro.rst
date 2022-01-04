Introduction
============

SLIM is an FOSS sea lice lifecycle simulation tool.
Its intended goal is to simulate sea lice pandemics
in salmonid farms.

However, it is much more than a simple epidemic simulator. The tool
is intended to answer *What if?* questions in terms of treatment strategies.
For example, *What will be our loss if we apply treatment as soon as the AF aggregation ratio reaches x?*
*How long does it take before treatment resistance becomes significant? (regardless of what we mean by that)*

Features
********

SLIM is, in essence, a statistical model of sea lice epidemics that aims to be `the opposite of this <https://phdcomics.com/comics/archive.php?comicid=1689>`_.
That is, we aim for this project to provide a battery-ready developing and usage experience.

Overall, this project offers the following tools:

- A simulator (See :py:mod:`src.SeaLiceMgmt`)
- A visualisation tool based on PyQt5 (See :py:mod:`src.SeaLiceMgmtGUI`)
- A strategy optimiser (See :py:mod:`src.Optimiser`)

There are thus two designed workflows:

1. Use the simulator to generate a *session dump*, then run the GUI to visualise such data.
2. Use the strategy optimiser to obtain the best treatment parameters (more on this later), save the resulting configuration artifact.

The simulator is typically executed as a standalone command, but it provides a relatively easy Python API for
embedding purposes.

The simulator is also battle-tested (see :py:mod:`tests`).

Our novel model, soon to be documented in a paper, relies on a hybrid approach stemming from `Aldrin et al. 2017 <https://doi.org/10.1016/j.ecolmodel.2017.05.019>`_,
`Jensen et al. 2017 <https://doi.org/10.1371/journal.pone.0178068>`_, `Cox et al., 2017 <https://doi.org/10.1002/ecs2.2040>`_ and a few others,
with original additions (better explained below).

Motivation and history of this project
**************************************

Salmon farms have been affected by sea lice for the last few decades,
causing multi-million dollar losses in revenue.

SLIM is the Sea Lice Model associated with a funded BBSRC project on the evolution to resistance to treatment in sea lice (BBR009309).
We aim to integrate an epidemiological and genetic model of sea lice with a model of treatment decision-making by different salmon farms.
