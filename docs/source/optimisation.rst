.. currentmodule:: slim.simulation

Internals Guide
===============

This document describes how the internals are structured and why some choices have been made.

It is currently a stub and will hopefully be improved in the next few days.

Statistical modelling
*********************

The most evident characteristic of SLIM is that it is a statistical model. In order for this
to work efficiently we need to rely on fast operations such as:

* distribution arithmetic;
* efficient sampling;
* fast discretisation.

The first cornerstone is obviously the optimised :class:`.lice_population.GenoDistrib`
and numpy/scipy's sampling algorithms.

Multi-threading and Numba
*************************

When the project started we tried to make our codebase as pythonic as possible, while
relying on numerical tools such as numpy for fast calculation. However, statistical distributions
allow for very compact representations which are barely advantaged by vectorised instructions,
meaning using numpy functions over builtin ones does not result into performance speed-ups.
Furthermore, because of `Python's GC limitations <https://wiki.python.org/moin/GlobalInterpreterLock>`_
it is not possible to write parallelised python code in pure python.

This leaves a few possibilities:

1. Rewrite everything in a better language with support for parallelism. Notable examples of fast (jit-)compiled languages
   include Julia, R, Go, Scala and C/C++. Some of these options offer integration to OpenMP, CUDA and
   similar tools.
2. Rewrite a few core components in Numba or Cython

The first approach was frowned upon for a number of reasons:

* the complexity of the codebase
* the authors are not fully familiar in other languages but Python and C++
* inferior library support (PyData...);

We are experimenting with numba to achieve fast performance without being forced to rewrite
the entire codebase, however during benchmarking we verified that it is not a silver bullet:
performance hits mainly arise from the continuous context switch from Python to numba/c.
For example, converting from a tensor generated via Numba and Python (numpy) takes a non-negligible
amount of time.

Thus, our roadmap is to convert *all* the computing code to numba wherever possible. Unfortunately,
much of this code relies on the :mod:`slim.simulation.cage` and :mod:`slim.simulation.farm` modules,
meaning a lot of work still needs to be done.

See `this issue <https://github.com/resistance-modelling/slim/issues/209>`_ for details.

Because Numba support is still heavily experimental we chose to keep it disabled by default.
Since Numba can only be enabled or disabled by setting the ``NUMBA_DISABLE_JIT``
env var to 1 (see `<https://numba.pydata.org/numba-doc/dev/reference/envvars.html#envvar-NUMBA_DISABLE_JIT>`_
we came up with a similar flag ``SLIM_ENABLE_NUMBA`` which does the opposite.
During debugging sessions it is automatically disabled.


Writing style
*************

The main crux with numba is that many python-esque features are not available. For example,
operator overloading, method overloading, OOP and so on. In particular, the ``jitclass`` feature
is quite hard to use and presents a number of challenges.

In short, refrain from using dictionaries, generators and wrappers whenever possible.

.. _Multiprocessing:

Multiprocessing on farms
************************

While the lack of true multithreading is a limiting factor, multiprocessing is still usable for long-running tasks
like farm updates.

Internally, the :class:`.organisation.Organisation` spawns a number of ``FarmActor`` s, each of them owning a variable
number of farms to update. The driver (in this case, the organisation) schedules step updates and sometimes also
alerts farms about treatment cooperation opportunities. Each ``FarmActor`` returns a collation of outputs in the form
of gym spaces, rewards and other metrics.

Multithreading on cages
***********************

As farms can have a lot of cages each (e.g. between 6 and 20) multithreading is therefore a nice feature to have.
As explained earlier, multithreading on pure python cages is ineffective unless the GIL is released.

However, one could try to make large cage actor pools as it was done for farms - although that might require some
effort.