Function patterns
=================

A function pattern describes callable composition inside one ``FunctionStep``.
It does not own source discovery, stack-axis meaning, artifact identity, or
materialization.

Single callable
---------------

.. code-block:: python

   FunctionStep(func=normalize)
   FunctionStep(func=(normalize, {"percentile": 99.0}))

Callable chain
--------------

A list executes its items in order inside one step:

.. code-block:: python

   FunctionStep(
       func=[
           (normalize, {"percentile": 99.0}),
           (smooth, {"sigma": 2.0}),
           threshold,
       ],
       name="preprocess",
   )

Dictionary routing
------------------

A dictionary selects a function pattern for each compiled group key. The
``group_by`` value belongs to ``ProcessingConfig``:

.. code-block:: python

   from openhcs.constants import GroupBy, VariableComponents
   from openhcs.core.config import LazyProcessingConfig, ProcessingConfig

   by_channel = ProcessingConfig(
       variable_components=(VariableComponents.SITE,),
       group_by=GroupBy.CHANNEL,
   )
   step = FunctionStep(
       func={
           "1": (segment_nuclei, {}),
           "2": (trace_neurites, {}),
       },
       name="channel-specific analysis",
       processing_config=LazyProcessingConfig.from_config(by_channel),
   )

Dictionary keys are validated against compiled grouping/source identity. They
are not arbitrary filename matches.

Grouping is not stacking
------------------------

``variable_components`` declares what changes along the assembled 3D stack
axis. ``group_by`` partitions already assembled arrays. It creates callable
fan-out only for a dictionary pattern. With a single callable pattern, the same
callable is applied to every assembled group.

Callable contracts
------------------

Every normalized invocation has a ``CallableContract``. All invocations in a
step must agree on execution scope and satisfy declared axes, grouping, runtime
parameters, image mode, and artifacts. Compilation turns the public pattern into
a typed compiled pattern; workers consume that compiled form.

See :doc:`../architecture/processing_semantics` and
:doc:`../architecture/artifact_contract_system`.
