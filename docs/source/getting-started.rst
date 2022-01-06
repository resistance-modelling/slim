Getting Started
===============

This guide assumes you have already installed slim and its dependencies. If you haven't please check
the  `README <https://github.com/resistance-modelling/slim/blob/master/README.md>`_.

Run the simulator
*****************

.. tabs::
    .. group-tab:: Command Line

        The easiest way to run this project is from the command line.

        The main script you'll need to run is SeaLiceMgmt. The general way to launch
        it is the following:

        .. code-block:: bash

            python -m slim.SeaLiceMgmt output_folder/simulation_name simulation_params_directory

        For example:

        .. code-block:: bash

            python -m slim.SeaLiceMgmt output/Loch_Fyne config_data/Fyne

    .. group-tab:: Python

        The entry point of this program is the :py:class:`slim.Simulator.Simulator` class.

        All you have to do is to instantiate an object of that class and provide a :py:class:`slim.Config.Config`
        instance. For example:

        .. code-block:: python

            from slim.simulation.config import Config
            from slim.simulation.Simulator import Simulator

            cfg = Config("config_data/config.json", "config_data/Fyne")
            sim = Simulator("output", "Fyne_foobar", cfg)
            sim.run_model()

Runtime Environment
*******************

An *environmental setup* consists of a subfolder containing three files:

- ``params.json`` with simulation parameters specific to the organisation;
- ``interfarm_time.csv`` with travel time of sea lice between any two given farms (as a dense CSV matrix);
- ``interfarm_prob.csv`` with probability of sea lice travel between any two given farms (as a dense CSV matrix);

See ``config_data/Fyne`` for examples.

Additionally, global simulation constants are provided inside ``config_data/config.json``.

.. warning::
   Do not change the values inside ``config.json``! Here are two reasons:

   * changing those values will impact *any* simulation;
   * those values are determined via experimentation and fitting on real world data.

   Currently, the only way to overwrite these parameters is via manual
   overriding, for example as described in :ref:`Parameter Override`.

Advanced features
*****************

.. _Parameter Override:

Parameter Override
""""""""""""""""""

.. tabs::

    .. group-tab:: Command Line

        If one wishes to modify a runtime option without modifying those files an extra CLI parameter can be passed to the command.
        In general, for  each key in the format ``a_b_c`` an automatic parameter in the format ``--a-b-c`` will be generated.
        For example:

        ``python -m slim.SeaLiceMgmt out/0 config_data/Fyne --seed=0 --genetic-mechanism=discrete``

        For now, nested and list properties are not yet supported.

    .. group-tab:: Python

        :py:class:`slim.Config.Config` allows you to override the default parameter configuration
        of either global or environment-specific variables, assuming there is no name clash.

        For example:

        .. code-block:: python

            override = {
                "seed": 42,
                "gain_per_kg": 5.0,
                "dam_unavailability": 3,
                "start_date": "2017-10-01 00:00:00",
                "end_date": "2019-10-01 00:00:00",
            }
            cfg = Config("config_data/config.json", "config_data/Fyne")
            sim = Simulator("output", "Fyne_foobar", cfg)
            sim.run_model()


.. note::
    The format of the override options must be consistent with the schema.
    This also means that overriding with the schema. See ``config_data/config.schema.json``
    and ``config_data/params.schema.json``.


Artifact Saving
"""""""""""""""

The artifacts are saved inside the output folder under the name ``simulation_data_${NAME}.pickle``.

For efficiency reasons, slim only saves a checkpoint of the simulation state at the end of the simulation.

This has two uses:

* introspecting the simulation output with the GUI;
* resuming an interrupted simulation.


.. tabs::
    .. group-tab:: Command Line

        To generate a dump every ``n`` days add the ``--save-rate=n`` option. For example:

        ``python -m slim.SeaLiceMgmt outputs/sim_1 config_data/Fyne --save-rate=1"``

        This will save the output every day.

        To *resume* a session one can instead pass the `--resume` parameter. Via CLI:

        ``python -m slim.SeaLiceMgmt outputs/sim_1 config_data/Fyne --resume="2017-12-05 00:00:00"``

        If you only know ``n`` days have elapsed since the start use the `--resume-after=n` option. For example:

        ``python -m slim.SeaLiceMgmt outputs/sim_1 config_data/Fyne --resume-after=365``

    .. group-tab:: Python

        To generate a dump every ``n`` days set up an instance of :py:class:`slim.Config.Config` and pass the extra
        parameter ``save_rate``. The rest follows as usual.

        .. code-block:: python

            from slim.simulation.Config import Config
            from slim.simulation.Simulator import Simulator

            n = 10 # every 10 days

            cfg = Config("config_data/config.json", "config_data/Fyne", save_rate=n)
            sim = Simulator("output", "Fyne_foobar", cfg)
            sim.run_model()

            # Press Ctrl+C before the end to stop it prematurely

        To resume the session you need to know either a timestamp
        or the number of elapsed days.

        .. code-block:: python

            from slim.simulation.config import Config, to_dt
            from slim.simulation.simulator import Simulator

            timestamp = to_dt("2018-12-05 00:00:00")
            sim = Simulator.reload("output", "Fyne_foobar", timestamp=timestamp)
            # or alternatively
            sim = Simulator.reload("output", "Fyne_foobar", resume_after=365)
            sim.run_model()

        Additionally, one can override the config parameters.

.. note::

    Artifacts are opened in read-only mode when resuming. It is not allowed to
    combine resuming and dumping.

Run the GUI
***********

We also provide a GUI for debugging and visualisation. Its support is still heavily experimental so please
use with caution.

To run the GUI you need to launch :py:mod:`slim.SeaLiceMgmtGUI`, for example via:
```python -m slim.SeaLiceMgmtGUI``` and provide your run data (generated via the `--save-rate` option mentioned
above) from the given menu.

TODO: expand on this?