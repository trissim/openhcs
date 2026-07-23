Compilation Service (Legacy)
============================

Status
------

``CompilationService`` has been removed from OpenHCS.

Reason
------

Compilation and execution orchestration now live in one service boundary to
enforce a single workflow invariant and eliminate duplicated state logic.

Replacement
-----------

Use :doc:`batch_workflow_service` for all compile and run orchestration.

Also see
--------

- :doc:`plate_manager_services`
- :doc:`progress_runtime_projection_system`
