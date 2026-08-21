Execution cleanup guarantees
============================

The compiled plate execution boundary makes cleanup converge across success,
failure, cancellation, and a broken process pool.

Parent responsibilities
-----------------------

The parent shuts down the executor, closes the exact execution cancellation
scope, releases progress resources, cleans parent GPU framework state, and
applies viewer persistence policy. It does not terminate persistent viewers
that remain healthy by configuration.

Worker responsibilities
-----------------------

Each worker clears process-local runtime stores and ArrayBridge GPU framework
state after each context, including when a cooperative cancellation boundary or
failure is reached. Worker failures are returned as typed execution results when
possible; pool failure still enters the parent cleanup path.

Ordering
--------

1. record terminal cancellation at the generic execution lifecycle when requested;
2. stop submitting work and interrupt only the targeted execution;
3. collect or cancel outstanding futures and let an active callable reach a safe boundary;
4. shut down the executor using the execution's multiprocessing context;
5. clean runtime and GPU resources;
6. reconcile viewer lifecycle and progress state;
7. surface the original execution outcome without replacing cancellation.

Cleanup errors are logged without replacing the primary failure unless they
prevent a safe terminal state. Direct callers must provide the required
progress context so resource ownership is explicit.

See :doc:`concurrency_model` and
:doc:`streaming_boundary_and_wrappers`.
