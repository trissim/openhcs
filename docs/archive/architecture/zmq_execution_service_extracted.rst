ZMQ Execution Service (Legacy)
==============================

Status
------

``ZMQExecutionService`` has been removed from OpenHCS.

Reason
------

Execution polling and compile orchestration are no longer split across two
services. OpenHCS now uses a unified workflow service plus typed projection and
status presenter abstractions.

Replacement
-----------

- :doc:`batch_workflow_service` for compile + execute orchestration
- :doc:`progress_runtime_projection_system` for runtime status projection
- :doc:`zmq_server_browser_system` for tree/browser rendering
