Contributor's Guide
===================

Thank you for considering to contribute to the project.

To ensure the project upholds high quality standards please make sure to:

- ensure your code is documented with proper ReST syntax as required for sphinx;
- use ``black`` to ensure consistent code quality;
- use ``pytype`` to check for type safety; this also includes writing type
  annotations where reasonable;
- ensure every new feature is tested (where applicable). Check :ref:`Testing` for details.
- document in the code where formulae/ideas have been taken from;
  constants should be written inside the provided configuration files,
  together with a comment providing the source.

To our knowledge, the majority of linting tools out there are too verbose, unsound and incomplete
for our purposes so we are not recommending to comply to anyone in particular.