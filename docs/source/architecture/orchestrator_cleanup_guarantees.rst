Execution cleanup guarantees
============================

The compiled plate execution boundary makes cleanup converge across success,
failure, cancellation, and a broken process pool.

Parent responsibilities
-----------------------

The parent shuts down the executor, releases progress/cancellation resources,
cleans parent GPU framework state, and applies viewer persistence policy. It
does not terminate persistent viewers that remain healthy by configuration.

Worker responsibilities
-----------------------

Each worker clears process-local runtime stores and ArrayBridge GPU framework
state after its task. Worker failures are returned as typed execution results
when possible; pool failure still enters the parent cleanup path.

Ordering
--------

1. stop accepting or waiting for new work;
2. collect/cancel outstanding futures;
3. shut down the executor using the execution's multiprocessing context;
4. clean runtime and GPU resources;
5. reconcile viewer lifecycle and progress state;
6. surface the original execution outcome.

Cleanup errors are logged without replacing the primary failure unless they
prevent a safe terminal state. Direct callers must provide the required
progress context so resource ownership is explicit.

See :doc:`concurrency_model` and
:doc:`streaming_boundary_and_wrappers`.

