ZMQ Server Browser System
=========================

Overview
--------

OpenHCS implements a thin wrapper over the generic pyqt-reactive browser:

- Generic base: ``pyqt_reactive.widgets.shared.ZMQServerBrowserWidgetABC``
- OpenHCS adapter: ``openhcs.pyqt_gui.widgets.shared.zmq_server_manager.ZMQServerManagerWidget``

This split keeps UI infrastructure generic while preserving OpenHCS-specific
progress semantics and topology validation. For execution progress, the browser
is a projection consumer and does not own a second subscriber. For endpoint
observation, the generic browser owns one immutable
``EndpointObservationSnapshot``. The OpenHCS main window derives the configured
endpoint's status-bar presentation from that snapshot. Client connection
ownership remains with ZMQRuntime's ``EndpointClientSession`` rather than being
inferred from browser rows.

Boundary
--------

This page documents OpenHCS-owned browser behavior:

- progress topology validation
- OpenHCS tree construction and presentation

Generic widget infrastructure (tree rebuild/state sync/poll scheduling and
endpoint actions) remains owned by ``pyqt-reactive``. ZMQRuntime owns the
typed shutdown modes, execution, and outcomes used by those actions.

OpenHCS Browser Components
--------------------------

- ``EndpointObservationSnapshot``:
  canonicalises responsive heartbeats and in-progress startup observations by
  port, then derives endpoint status without persistent connection flags.
- ``RuntimeExecutionTopology``:
  derives worker/well ownership and step names from each retained event
  snapshot and validates worker claims without persistent browser state.
- ``RuntimeTreeProjectionBuilder``:
  builds typed core runtime trees with recursive aggregation policies.
- ``ProgressTreeBuilder``:
  is the thin Qt adapter over the core runtime-tree projection.
- ``ExecutionProgressProjection``:
  supplies the typed running/queued heartbeat entries to the shared core
  lifecycle projection.
- ``ServerRowPresenter``:
  type-dispatched rendering for execution/viewer/generic servers.
- ``LiveServerTreeSync``:
  renders live scan results, startup observations, and active launching-viewer
  lifecycle records without storing another endpoint state. A launching row is
  rebuilt from the current ZMQRuntime viewer snapshot, refreshed in place while
  that snapshot changes, and removed when the viewer is neither launching nor
  discovered. ``ServerRowPresenter`` supplies the corresponding execution,
  viewer, and generic rows.
- ``EndpointShutdownService``:
  generic graceful/force shutdown execution with typed per-endpoint outcomes
  and progress-tracker retirement.

Canonical Tree Path
-------------------

Execution subtree path is invariant:

``plate -> worker -> well -> step``

Compilation path is plate -> compilation entries until execution events begin.

Aggregation Semantics
---------------------

Tree percentages are recursive and policy-driven:

- ``mean`` for parent aggregates (plate, worker)
- ``explicit`` for leaf/detail nodes (well, step, compilation)

Policies are enforced per registered node type in
``RuntimeTreeProjectionBuilder``.

Refresh/State Preservation
--------------------------

The Plate Manager's execution client is the sole subscriber that registers
progress events. The browser observes accepted registry mutations, marshals the
notification through a queued Qt signal, and coalesces redraws at the interval
declared by ``ProgressUIConfig``. Server scans independently supply the current
typed running/queued heartbeat entries. Expansion/selection preservation is
delegated to the generic browser base.

Server scans and startup events both produce a new
``EndpointObservationSnapshot`` through the generic browser's single commit
boundary. One snapshot emission updates the server tree, the central status
text, and the right-hand status indicator. Startup callbacks only commit
observations and request a scan; they do not update any of those projections
directly. When the configured endpoint is absent from the snapshot, its browser
row disappears and both status projections report that it is not connected.

Primary Modules
---------------

- ``openhcs/pyqt_gui/widgets/shared/zmq_server_manager.py``
- ``openhcs/pyqt_gui/widgets/shared/server_browser/progress_tree_builder.py``
- ``openhcs/pyqt_gui/widgets/shared/server_browser/presentation_models.py``
- ``openhcs/pyqt_gui/widgets/shared/server_browser/live_tree_sync.py``
- ``external/pyqt-reactive/src/pyqt_reactive/services/zmq_server_scan_service.py``
- ``external/pyqt-reactive/src/pyqt_reactive/widgets/shared/zmq_server_browser_widget.py``
- ``external/zmqruntime/src/zmqruntime/shutdown.py``

See Also
--------

- :doc:`batch_workflow_service`
- :doc:`progress_runtime_projection_system`
