ZMQ Server Browser System
=========================

Overview
--------

OpenHCS implements a thin wrapper over the generic pyqt-reactive browser:

- Generic base: ``pyqt_reactive.widgets.shared.ZMQServerBrowserWidgetABC``
- OpenHCS adapter: ``openhcs.pyqt_gui.widgets.shared.zmq_server_manager.ZMQServerManagerWidget``

This split keeps UI infrastructure generic while preserving OpenHCS-specific
progress semantics and topology validation. The browser is a projection
consumer; it does not own a second execution-progress subscriber.

Boundary
--------

This page documents OpenHCS-owned browser behavior:

- progress topology validation
- OpenHCS tree construction and presentation
- OpenHCS server/process actions

Generic widget infrastructure (tree rebuild/state sync/poll scheduling) remains
owned by ``pyqt-reactive`` and is documented there.

OpenHCS Browser Components
--------------------------

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
  reconciles live scan results and startup observations without duplicating
  endpoint state; ``ServerRowPresenter`` supplies the corresponding execution,
  viewer, and generic rows.
- ``ServerKillService``:
  kill-plan execution (graceful/force) with logging hooks.

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

Primary Modules
---------------

- ``openhcs/pyqt_gui/widgets/shared/zmq_server_manager.py``
- ``openhcs/pyqt_gui/widgets/shared/server_browser/progress_tree_builder.py``
- ``openhcs/pyqt_gui/widgets/shared/server_browser/presentation_models.py``
- ``openhcs/pyqt_gui/widgets/shared/server_browser/live_tree_sync.py``
- ``openhcs/pyqt_gui/widgets/shared/server_browser/server_kill_service.py``

See Also
--------

- :doc:`batch_workflow_service`
- :doc:`progress_runtime_projection_system`
