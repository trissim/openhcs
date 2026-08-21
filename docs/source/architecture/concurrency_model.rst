Concurrency model
=================

OpenHCS compiles the selected execution axes before starting workers. A
``CompiledExecutionBundle`` carries the worker-safe steps, frozen
``ProcessingContext`` instances, and environment decisions for one plate.

Execution units
---------------

The configured multiprocessing axis normally yields one context per well.
Source discovery determines the available axis values; compiler stages create
the corresponding typed step plans. Workers receive one compiled context at a
time and never share mutable compiler state.

``compiled_plate_execution`` owns the plate-level execution lifecycle.
``worker_execution`` owns executor construction, bundle installation, task
submission, result collection, cancellation, and cleanup. ``worker_lanes``
provides the process-local bundle authority used by worker code.

Start-method policy
-------------------

``resolve_worker_start_context`` is the single policy for ``spawn``, ``fork``,
and ``forkserver`` selection. It considers the requested method, platform,
server mode, and whether compiled plans require a GPU. GPU execution and macOS
default to ``spawn`` unless an explicit unsafe override is allowed.

Progress and cancellation
-------------------------

Workers emit typed progress events to the execution's queue. The parent and ZMQ
runtime project those events for UI consumers. ZMQRuntime owns the terminal
execution status and applies cancellation only to the requested execution.
OpenHCS owns an execution-scoped cancellation authority: inline and threaded
lanes check it between contexts, steps, and axes, while process execution stops
owned worker processes. A callable already in progress reaches its next safe
boundary before stopping. Cancellation never becomes success or failure when
late worker completion or an exception reaches the server, and every path still
runs cleanup.
Worker ``AXIS_COMPLETED`` events close only the axis-scoped portion. Terminal
plate steps, output consolidation, viewer settlement, and final state projection
remain parent-owned; :doc:`progress_runtime_projection_system` defines the
separate final ``SUCCESS`` boundary.

Invariants
----------

- compile all selected axes before executing any of them;
- create every multiprocessing primitive from the same resolved context;
- pass typed bundles and values, not reconstructed path dictionaries;
- keep compiler state in the parent and immutable runtime state in workers;
- make success, failure, cancellation, and broken-pool paths converge on cleanup.

See :doc:`pipeline_compilation_system`,
:doc:`progress_runtime_projection_system`, and
:doc:`orchestrator_cleanup_guarantees`.
