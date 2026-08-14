Plate Manager services
======================

The Plate Manager widget delegates workflow policy to
``PlateManagerBatchWorkflow``. The facade owns shared context and lifecycle;
``BatchWorkflowComponents`` owns lazy construction of focused services.

Ownership map
-------------

``openhcs/pyqt_gui/services/plate_manager_batch_workflow.py``
  Workflow facade and compile-all-then-execute-all sequence.

``openhcs/pyqt_gui/widgets/shared/services/batch_workflow_components.py``
  Lazy component owner for compile, submit, control, progress, debug, and live
  measurement services.

``openhcs/pyqt_gui/widgets/shared/services/compile_batch_workflow_service.py``
  Batch compile policy and exact compile-artifact collection.

``openhcs/pyqt_gui/widgets/shared/services/compile_workflow_service.py``
  One compile request, its nominal identity, submission, and wait behavior.

``openhcs/pyqt_gui/widgets/shared/services/plate_pipeline_request_builder.py``
  Projection of one selected plate into ``CompileJob`` or ``RunSpec``.

``openhcs/pyqt_gui/widgets/shared/services/execution_submission_service.py``
  Execution submission, completion polling, and terminal callbacks.

``openhcs/pyqt_gui/widgets/shared/services/execution_control_service.py``
  Stop, cancel, disconnect, and failure convergence.

``openhcs/pyqt_gui/widgets/shared/services/progress_workflow_service.py``
  Coalesced progress projection and server-information refresh.

``openhcs/pyqt_gui/widgets/shared/services/plate_manager_workflows.py``
  Code-document mutation scope and application. A selected document may synchronise
  only the plate graphs named by its read-time scope; an all-plates document
  synchronises the complete visible collection.

``openhcs/pyqt_gui/services/pipeline_object_state_binding.py``
  Pipeline declaration reconciliation and saved-baseline commits for the exact
  active editor, step, and nested-function state graph.

``openhcs/pyqt_gui/services/function_catalog_projection.py``
  Shared asynchronous projection of the execution endpoint's callable
  catalogue. The endpoint remains the authority for callable availability.

Supporting presentation owners include
``execution_server_status_presenter.py``, ``progress_batch_reset.py``, and
``plate_config_resolver.py`` in the same services package.

Invariants
----------

- Every selected run compiles before the first execution submission.
- Execution requests carry the exact compiled artifact ID returned for their
  plate.
- Plate, progress, and terminal state have host-owned stores; services project
  updates rather than maintain parallel semantic caches.
- Generic submit/wait/status-polling mechanics belong to ZMQRuntime. OpenHCS
  owns how those mechanics map to plate compilation and UI state.
- Complete pipeline documents reconcile their step and nested-function
  occurrences by declaration-owned authority. Unchanged and unambiguously
  edited occurrences retain identity across reordering; ambiguous duplicates
  receive new scopes rather than inheriting identity by position.
- Applying a Pipeline Editor code document commits the reconciled root, step,
  and nested-function states as the editor's saved baseline. The Pipeline
  Editor has no second Save action after code apply.
- Code-document selection is proof-bearing apply authority. Applying a selected
  Plate Manager document preserves every unselected plate and rejects a payload
  whose plate IDs differ from the scope that was read.
- Deferred desktop startup uses the batch workflow's persistent endpoint policy,
  then prewarms the endpoint-owned callable catalogue without blocking Qt.
- Cleanup removes listeners and timers owned by the workflow facade.

See :doc:`batch_workflow_service`, :doc:`progress_runtime_projection_system`,
and :doc:`zmq_server_browser_system`.
