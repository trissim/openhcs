Log viewer
==========

Use the desktop Log Viewer to inspect application, compiler, worker, ZMQ, and
viewer messages without leaving the active workflow. Filter by level or source
and keep the relevant execution identifier when reporting a failure.

.. openhcs-gallery:: execution-log-viewer

For compilation failures, start at the earliest error tied to the plate and
step. Later worker or UI messages are often consequences. For runtime failures,
record the execution axis, step name, worker process, and the typed plan field or
artifact named in the error.

Detached viewers and subprocesses may also write dedicated log files. Their
paths are reported by the launching process. Do not infer success solely from a
viewer window appearing; readiness and acknowledgments are recorded separately.

While the Log Viewer is visible, its selector refreshes live server discovery
without overlapping scans. A server that finishes starting after the window was
created appears with its own process status and opens its corresponding log.

Developer diagnosis is covered in
:doc:`../development/pipeline_debugging_guide`.
