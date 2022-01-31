.. _Testing:

Testing
=======

Unit testing
************

The refactored model is being thoroughly tested thanks to unit testing, integration testing and (intelligent) type
checking.

To test, from either the root folder or within ``tests/`` run:

``pytest``

To enable coverage reporting, run:

``pytest --cov=slim/ --cov-config=.coveragerc  --cov-report html --cov-context=test``

The last two parameters generate a human-friendly report.

Type checking
*************

For type checking, install pytype and run (from the root folder):

``pytype --config pytype.cfg``

Profiling
*********

This project relies on cProfile for profiling. To profile an execution pass the `--profile` option.

To visualise the final profiling artifact (released as ``{output_folder}/profile_{sim_name}``) one could use `snakeviz <https://jiffyclub.github.io/snakeviz/>`_ or any other tooling of choice.

Note that cProfile does not provide line coverage. For that purpose `line-profiler <https://github.com/pyutils/line_profiler>`_
is available. Unfortunately it requires manual annotation of the source code.

.. tip::
    It is recommended to minimise any I/O operation while profiling. Thus avoid
    using options like ``--save-rate``.


Assertion failures
******************

Sometimes the simulation crashes due to some invariants being broken. If you believe that
it shouldn't be concerning you could pass the ``-O`` option to the interpreter to ignore the assertion
failures. That said, assertion failures are good indicators of simulation bug.
Please report them via Github.
