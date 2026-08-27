Troubleshooting and FAQ
=======================

Plate does not initialize
-------------------------

Confirm that the selected directory is the plate root and contains the metadata
or filename layout expected by a registered microscope handler. Read the first
initialization error in the log; later disabled buttons are consequences.

Pipeline does not compile
-------------------------

Compilation intentionally stops before execution when a source, component,
callable, artifact, memory, or output declaration cannot be satisfied. Open the
named step and correct the first error. Recompile after changing configuration.

No GPU functions appear
-----------------------

GPU functions require the relevant optional dependencies and compatible device
runtime. ``openhcs --no-gpu`` and ``OPENHCS_CPU_ONLY=true`` deliberately hide
GPU-backed declarations. Do not install the GPU extra on an unsupported system.

Viewer does not open
--------------------

Verify that the viewer extra is installed, the step's streaming configuration
is enabled, and compilation succeeded. Check the detached viewer log for
dependency, port/readiness, Java/ImageJ, or Qt errors.

Execution server is unavailable
-------------------------------

Use the server browser/status surface to distinguish connection, readiness, and
execution failure. Record the server endpoint and execution identifier, then
inspect both client and server logs. Restarting may restore availability but
does not diagnose an incompatible protocol or invalid payload.

If a locally launched server fails before becoming ready, OpenHCS retains its
``.startup.jsonl`` journal beside the server log in the configured OpenHCS log
directory. The final journal entry identifies the last reported startup phase.

When the UI connects, it compares its OpenHCS version with the version reported
by the execution server's handshake. If they differ, or an older server does
not report a version, the UI offers to replace the server and restart OpenHCS.
Accepting preserves and restores the current session and edit history; cancelling
leaves the current UI and server running.

An MCP/UI bridge mismatch is a separate process-boundary error. Restart the MCP
client and OpenHCS from the same installed OpenHCS environment; descriptors from
another checkout or version are intentionally rejected before they can change
the running UI.

For a headless MCP submission timeout, read the reported outcome before
retrying. Submission preparation has its own longer deadline because a cold
start can include connection, progress registration, and pipeline
serialisation. If that preparation expired before the execute request was sent,
increase ``submit_timeout_ms``. If the request was sent but no reply arrived,
poll execution status first because the server may have accepted it.

Function selector is still loading
----------------------------------

The function catalogue is supplied by the execution server, which may need to
import optional processing libraries the first time it starts. After the main
window appears, OpenHCS starts or attaches to the configured server and prewarms
the catalogue used by both the selector and local MCP authoring tools. The
selector remains interactive while the shared request completes and reports an
endpoint error in its status line if loading fails. Inspect that error and the
execution-server log rather than repeatedly reopening the selector. A client
timeout measures an interval with no server or MCP progress, not the catalogue's
total build time. Closing the server cancels and joins its exact preparation
process; it does not leave discovery running as an unowned background import.

After changing the desktop execution host, port, or transport, restart the MCP
client so its newly launched local server snapshots the same saved endpoint
configuration.

Where are outputs?
------------------

Only explicitly materialized image outputs and declared analysis artifacts are
persisted. Inspect the resolved step materialization and analysis-results
configuration; intermediate main-flow values may exist only in the runtime
backend.

For developer diagnosis, see
:doc:`../development/pipeline_debugging_guide`.
