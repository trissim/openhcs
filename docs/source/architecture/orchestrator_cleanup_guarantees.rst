Execution cleanup guarantees
============================

The compiled plate execution boundary makes cleanup converge across success,
failure, cancellation, and a broken process pool.

Parent responsibilities
-----------------------

The parent shuts down the executor, closes the exact execution cancellation
scope, releases progress resources, and applies viewer persistence policy. For
thread-backed execution, the parent waits for every lane to join before it
resets shared in-memory storage and cleans the merged compiled framework/device
footprint. It does not terminate persistent viewers that remain healthy by
configuration.

Worker responsibilities
-----------------------

An inline or process-isolated worker clears its process-local runtime stores and
exact ArrayBridge framework/device footprint after each context, including when
a cooperative cancellation boundary or failure is reached. Thread-backed lanes
defer both operations to the parent boundary because sibling lanes share the
same process. Worker failures are returned as typed execution results when
possible; pool failure still enters the parent cleanup path.

Ordering
--------

1. record terminal cancellation at the generic execution lifecycle when requested;
2. stop submitting work and interrupt only the targeted execution;
3. collect or cancel outstanding futures and let an active callable reach a safe boundary;
4. shut down the executor using the execution's multiprocessing context;
5. clean runtime and framework resources at the owning process boundary;
6. reconcile viewer lifecycle and progress state;
7. surface the original execution outcome without replacing cancellation.

Cleanup errors are logged without replacing the primary failure unless they
prevent a safe terminal state. Direct callers must provide the required
progress context so resource ownership is explicit.

See :doc:`concurrency_model` and
:doc:`streaming_boundary_and_wrappers`.
