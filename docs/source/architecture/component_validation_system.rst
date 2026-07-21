Component validation
====================

Component declarations describe semantic microscope dimensions such as well,
site, channel, Z, and time point. ``AllComponents``, ``VariableComponents``,
``SequentialComponents``, and ``GroupBy`` are typed views used at different
boundaries; equal string values do not make the declarations interchangeable.

Validation converts external values to the exact owning enum and rejects
unknown or semantically invalid combinations. It checks, among other things,
that:

- configured variable components exist in the source universe;
- grouping is allowed by every callable contract in the pattern;
- required callable axes are present;
- sequential and multiprocessing axes do not create an invalid execution
  topology.

Consumers should use the shared conversion/validation helpers and nominal enum
members. They must not maintain copied component-name sets. See
:doc:`../concepts/data_dimensions` and :doc:`processing_semantics`.

