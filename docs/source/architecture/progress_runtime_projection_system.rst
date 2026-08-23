Progress and Debugger Runtime Projection System
================================================

Scope
-----

This page documents the OpenHCS implementation path only.

The generic progress registry and projection mechanisms live in
``zmqruntime.progress`` and are documented in the ``zmqruntime`` docs. OpenHCS
owns the microscopy-pipeline event vocabulary, execution-tree semantics, and
retained debugger state layered over those mechanisms.

OpenHCS Implementation Path
---------------------------

OpenHCS-specific modules:

- ``openhcs.core.progress.types`` owns the registered channel, phase, and status
  semantics carried by ``ProgressEvent``.
- ``openhcs.core.progress.registry`` owns OpenHCS semantic keying over the generic
  registry.
- ``openhcs.core.progress.projection`` and
  ``openhcs.core.progress.runtime_tree`` own the OpenHCS execution and tree
  projections.
- ``openhcs.core.progress.emitters`` owns orchestrator and step event emission.

Consumption path:

1. Emit typed ``ProgressEvent`` values from compile/execute flows.
2. Route events through explicit queue wiring via ``set_progress_queue(...)``.
3. Register events in ``ProgressRegistry`` with semantic channel keying.
4. Reconcile typed server-heartbeat running/queued entries by execution and
   plate identity in ``ExecutionRuntimeProjection``.
5. Build runtime and execution-tree projections for status derivation.
6. Coalesce Plate Manager registry mutations through its explicitly
   GUI-thread-owned timer.
7. Render the same projection in Plate Manager and ZMQ server browser services.

OpenHCS Invariants
------------------

- Canonical execution tree path is ``plate -> worker -> well -> step``.
- ``PATTERN_GROUP`` is step-detail only and does not set well pipeline percent.
- Plate/server status is derived from projection snapshots, not mutable cache.
- Event retention is monotonic per semantic key. Stale or duplicate timestamps
  are rejected before listeners and typed side-effect consumers run.
- Topology is reconstructed from the retained event snapshot. A late subscriber
  may render a shallow lifecycle node when the initialization event is absent.
- ``PlateRuntimeStateDeclarationBase`` owns state labels, count labels,
  terminal policy, and live-server transition policy. ``state_counts`` is the
  only retained count projection; per-state count fields are forbidden.
- Runtime identity is ``(execution_id, plate_id)``. A new queued execution may
  replace an older visible execution for the same plate without conflating their
  retained histories.
- A background progress callback may only dirty the retained projection. Qt
  timers and row rendering are application-thread lifecycle responsibilities.

Debugger ownership
------------------

Debugger execution uses the normal progress transport but does not reduce its
retained state to status text:

- ``DebugCommandDeclarationBase`` owns command semantics, while the registered
  execution and stop strategies implement command-specific runtime behavior.
- ``DebugBoundaryEventDeclarationBase`` owns the progress status, boundary
  outcome, timeline state, and artifact-reporting policy for each boundary event.
- ``DebugCursor`` and ``DebugSession`` retain the exact invocation and controller
  identity. ``DebugSnapshot`` retains boundary evidence, and ``DebugArtifactRef``
  plus ``DebugArtifactIdentity`` retain typed artifact provenance.
- ``DebugExecutionPolicy`` installs the selected event sink and controls bounded
  execution, replay, axis selection, and the debugger-specific execution shape.

These are nominal authorities in ``openhcs.core.debug``. Consumers query their
registries and typed values; they do not maintain command lists, event tables, or
parallel debugger state machines.

Debugger projection path
------------------------

``ProgressDebugEventSink`` encodes a ``DebugProgressContext`` in an ordinary
``ProgressEvent``. ``RuntimeProjectionBuilder`` then builds the normal execution
projection and the debugger projection from the same immutable event snapshot.
``DebugRuntimeProjectionBuilder`` filters events by debug session, resolves their
snapshots, and derives current, last, and timeline frames while retaining the
normal execution projection beside them.

The runtime projection is not the session-state machine. Given the selected
target, active session, manager execution state, terminal summary, pause state,
and snapshots, ``DebugSessionPhaseDeclarationBase`` selects the current debugger
phase. Toolbar, inspector, UI-bridge, agent, and MCP-facing adapters consume that
typed session context and projection instead of re-deriving phase semantics.

Snapshots, replay, and inspection
---------------------------------

``DebugSnapshotStore`` is the registered storage boundary for snapshot metadata;
its concrete stores select local or FileManager-backed persistence. Snapshots
refer to artifacts rather than embedding bulk runtime arrays. Artifact reads and
exports therefore remain explicit, bounded operations.

Warm replay validates skipped outputs through
``DebugWarmReplayArtifactReusePlan``. It matches the exact compiled artifact
plans and stable ``DebugArtifactIdentity`` values, then either finds or hydrates
the required values through the runtime store. It does not infer reusable outputs
from generated names. Persistent paused-worker mode instead retains the live
``RuntimeValueStore`` behind ``DebugPausedWorkerController`` and exposes a typed,
read-only inspection view while the worker is paused.

Other typed-progress consumers
------------------------------

Live measurement previews are a sibling consumer of the progress stream, not a
debugger projection. ``LiveMeasurementProgressPayload`` places bounded typed
measurement previews in event context, and the live-measurement notification and
window services retain their own UI projection. They do not consume debugger
command declarations, debugger snapshots, or debugger session phase state.

Axis completion is not plate completion
---------------------------------------

Workers execute only axis-scoped steps. ``AXIS_COMPLETED`` reports the number
of completed axis positions and deliberately stops at the first terminal
plate-scoped step. It is a successful terminal event for that worker/axis
channel, but it does not assert that the plate execution has finished.

After every worker result succeeds, the parent merges runtime records and calls
``execute_plate_scoped_steps()`` once for each terminal plate step. The parent
then materializes those outputs, runs analysis consolidation, settles viewer
state, and projects the orchestrator result. Only after those stages succeed
does it emit the pipeline ``SUCCESS`` event at 100 percent. Failure,
cancellation, viewer settlement, or terminal-step errors therefore cannot be
hidden behind an earlier worker ``AXIS_COMPLETED`` event.

Canonical Abstraction Docs
--------------------------

See the published ``zmqruntime`` owner documentation for abstraction internals:

- `Progress Registry and Projection
  <https://github.com/OpenHCSDev/ZMQRuntime/blob/v0.2.8/docs/source/architecture/progress_registry_projection.rst>`_
- `ZMQ Execution System
  <https://github.com/OpenHCSDev/ZMQRuntime/blob/v0.2.8/docs/source/architecture/zmq_execution_system.rst>`_

Related OpenHCS Pages
---------------------

- :doc:`batch_workflow_service`
- :doc:`zmq_server_browser_system`
- :doc:`concurrency_model`
- :doc:`runtime_value_system`
