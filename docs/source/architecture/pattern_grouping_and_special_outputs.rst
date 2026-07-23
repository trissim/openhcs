Grouping and output terminology (transition)
============================================

.. important::

   The previous page combined stack construction, function-pattern dispatch,
   output-slot names, artifact declarations, and VFS paths. Those are independent
   systems in the current architecture.

Use these canonical pages:

- :doc:`processing_semantics` for ``variable_components``, ``group_by``,
  dictionary pattern fan-out, and ``ProcessingContract``;
- :doc:`artifact_contract_system` for semantic inputs/outputs and
  materialization;
- :doc:`pipeline_compilation_system` for one-time resolution and typed plans;
- :doc:`runtime_value_system` for runtime values and grouping scope.

In particular, ``group_by`` groups already assembled arrays and creates callable
fan-out only for a dictionary function pattern. It does not define the semantic
stack axis, and compiler stages do not re-resolve it from ObjectState.
