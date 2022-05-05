Reinforcement Learning Guide
============================

SLIM is, in principle, a game with different agents optimising their profits. We support
this by encapsulating the simulator state within a RL environment. However as Gym's
API was not designed for `MARL <https://arxiv.org/pdf/1911.10635>`_ scenarios we decided to
adopt PettingZoo.

AEC and PettingZoo
******************

*Alternated Environment Cycle* or AEC is a type of MARL in which multiple agents act in turns
exactly once before the turn is over. The order in which they act is irrelevant.

.. note::
   Currently, we adopt a simple AEC scheme which does not allow for parallel agent execution.

Each agent performs two actions, in the given order:

* samples from the observation space
* performs an action

Only after all agents have performed an action farms' spaces will be updated. Each agent
can only predicate about its own space, and only has access to a limited subset of what the simulator
models. In particular, the simulator exposes to an agent the following:

* current lice aggregation;
* fish population;
* which treatments are being used;
* how many treatments can still be used within the year;
* whether the organisation has asked to treat;

The action space is made of :math:`T+2` actions with :math:`T` being the number of available
treatments. The two extra options are fallowing and inaction.

The main logic is implemented in :class:`slim.simulation.simulator.SimulatorPZEnv`.

Policies
********

A number of policies are defined in :mod:`slim.simulation.simulator`. These are namely:

* No treatment policy (:class:`slim.simulation.simulator.UntreatedPolicy` )
* Bernoullian policy, i.e. each farm will randomly cooperate and randomly apply a treatment of choice (:class:`slim.simulation.simulator.BernoullianPolicy` )
* Mosaic policy, i.e. each farm will apply a different treatment whenever requested to (:class:`slim.simulation.simulator.MosaicPolicy` )

Additionally, any policy within the stable-baselines package should be supported although
they have not been tested yet.

The main policy prediction loop is performed inside :class:`slim.simulation.simulator.Simulator`.

To select a policy one needs to set the ``treatment_strategy`` option in the configuration.


For example:

.. tabs::
    .. group-tab:: Command Line

        .. code-block:: bash

            slim run \
                output_folder/Loch_Fyne \
                config_data/Fyne \
                --treatment-strategy=bernoulli

    .. group-tab:: Python

        .. code-block:: python

            from slim.simulation.config import Config
            from slim.simulation.simulator import Simulator

            cfg = Config("config_data/config.json", "config_data/Fyne", treatment_strategy="bernoulli")
            sim = Simulator("output", "Fyne_foobar", cfg)
            sim.run_model()

See :ref:`Environment Config` for details.
