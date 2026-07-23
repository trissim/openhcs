Orchestrator configuration
==========================

``PipelineOrchestrator`` holds one ``PipelineConfig`` for a plate. Global,
pipeline, and step declarations are ObjectState-backed scopes; lazy nested
configs inherit until compilation resolves them.

Resolution boundary
-------------------

Compilation registers the relevant scopes, resolves saved values and
inheritance once, and snapshots the resulting configuration. Compiler stages
and runtime contexts consume that resolved snapshot. They must not walk the UI,
thread-local state, or fallback chains independently.

The orchestrator delegates editable ObjectState fields to ``pipeline_config``
through its declared ObjectState boundary. The desktop forms and code projection
edit the same state. Assignment or restoration changes the declaration; it does
not mutate already compiled contexts.

Ownership
---------

ObjectState owns generic scope, resolution, snapshot, and provenance mechanics.
OpenHCS owns the configuration topology, domain dataclasses, compiler injection,
and runtime projection. See :doc:`external_foundations`,
:doc:`pipeline_compilation_system`, and
:doc:`code_ui_interconversion`.
