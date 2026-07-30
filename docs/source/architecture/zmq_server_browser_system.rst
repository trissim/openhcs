ZMQ Server Browser System
=========================

Overview
--------

OpenHCS implements a thin wrapper over the generic pyqt-reactive browser:

- Generic base: ``pyqt_reactive.widgets.shared.ZMQServerBrowserWidgetABC``
- OpenHCS adapter: ``openhcs.pyqt_gui.widgets.shared.zmq_server_manager.ZMQServerManagerWidget``

This split keeps UI infrastructure generic while preserving OpenHCS-specific
progress semantics and topology validation.

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

- ``ProgressTopologyState``:
  validates worker/well ownership claims from progress events.
- ``ProgressTreeBuilder``:
  builds ``ProgressNode`` trees with recursive aggregation policies.
- ``ServerRowPresenter``:
  type-dispatched rendering for execution/viewer/generic servers.
- ``ServerTreePopulation``:
  composes live scan results, launching viewers, and busy fallback rows.
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

Policies are enforced per node type in ``ProgressTreeBuilder``.

Refresh/State Preservation
--------------------------

OpenHCS coalesces progress updates on a timer and only rebuilds when dirty.
Expansion/selection preservation is delegated to the generic browser base.

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
