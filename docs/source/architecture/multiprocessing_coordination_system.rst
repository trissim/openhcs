Multiprocessing coordination
============================

Multiprocessing is coordinated from compiled execution facts rather than from
a second task-planning model.

1. The microscope/source model discovers execution-axis values.
2. ``PipelineCompiler`` produces a frozen ``ProcessingContext`` for each value.
3. Bundle assembly validates the contexts and worker-safe steps.
4. The worker-start policy chooses one multiprocessing context for the entire
   execution.
5. The executor submits one task per compiled axis and collects typed
   ``ExecutionResult`` values.

Worker processes install the ``CompiledExecutionBundle`` through the worker-lane
authority. Runtime stores and adapters are process-local; cross-process state is
limited to serialized bundle data, progress/cancellation channels, and result
messages.

Generic code must not ask the orchestrator for hard-coded component names.
Component identities come from the source model and nominal component
declarations. See :doc:`component_system_integration` and
:doc:`concurrency_model`.
