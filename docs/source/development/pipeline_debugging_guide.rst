Debugging compilation and execution
===================================

Diagnose the first boundary that fails: declaration, compilation, bundle
assembly, or runtime. Avoid reconstructing compiler state from paths, logs, or
private dictionaries.

Declaration failures
--------------------

Inspect the owning declaration:

- ``FunctionStep`` and its nested configuration;
- the invocation's ``CallableContract``;
- for CellProfiler lowering, its ``CellProfilerModuleArtifactContracts`` mixin,
  active ``SettingToKeywordBinding`` declarations, and module leaf hooks;
- source bindings and source-universe declarations;
- artifact specifications and runtime-bound parameters.

If metadata is missing, add it at that declaration boundary. Do not add a
consumer-side name test or fallback list.

Compilation failures
--------------------

Find the affected ``CompiledStepPlan`` typed field and the compiler stage that
owns it. ``CompilationSession`` is the cross-step authority. Useful questions
are:

1. Was ObjectState configuration resolved and snapshotted once?
2. Did the callable contract, including module-owned derivation when applicable,
   lower successfully?
3. Was the source or artifact satisfied by an exact declared producer?
4. Did axis, grouping, memory, and device validation agree?
5. Did bundle assembly validate all runtime-required fields?

Use the debug execution policy and the desktop debug inspector when a bounded
run-to-step observation is needed. Direct diagnostic scripts must install a
progress queue/context before invoking execution paths that emit progress.

Agent workflow for compiled intent
----------------------------------

``openhcs_inspect_pipeline_source_artifact_plan`` compiles one complete
``PipelineDocument`` without running its functions and returns bounded,
structured evidence for the source workspace, execution axes, steps, callable
groups, artifact inputs and outputs, and persistent-materialization candidates.
Use it before a full run. A clean plan proves compiled intent; it does not prove
that a runtime value was produced.

For a visible desktop workflow, read ``pipeline_debug_toolbar.session`` after
selecting and compiling the real Plate Manager row. That state surface owns the
debug phase, session and execution identities, selected source group, cursor,
current and last runtime frames, snapshot-store identity, and typed disabled
reasons. List the toolbar actions rather than guessing availability.

To stop at a specific step, set ``debug_pause=True`` on that ``FunctionStep``,
compile again, and invoke the declared ``Run to Pause`` action. ``Step`` advances
one debug boundary. ``Inspect Runtime`` is meaningful only while a debug worker
is active and paused.

Runtime failures
----------------

Runtime workers consume ``CompiledExecutionBundle`` and typed runtime values.
Check the failing execution axis, step plan, artifact store, runtime adapter,
and progress event. A runtime workaround that redoes source selection,
conversion planning, or artifact matching belongs in the compiler instead.

The runtime server exposes the same renderer-independent inspection used by the
desktop inspector through ``openhcs_inspect_debug_runtime_values`` (the
``runtime-debug-values`` developer command). Pass the active
``debug_session_id`` and the declared runtime-server connection from the debug
session state. The bounded result identifies invocation parameters, runtime
value records, measurements, relationships, and artifact locations without
copying bulk image arrays into the control response.

A mutation receipt and a debug/workflow state are separate evidence. Wait once
for a returned UI operation id, then re-read the debug and Plate Manager state
surfaces. Do not repeatedly invoke a pending command.

Choose the evidence boundary deliberately:

* use the compiled artifact plan for what should exist;
* use paused runtime inspection for what exists in the active worker;
* use ``StepMaterializationConfig`` for a persistent ordinary main-flow
  checkpoint;
* use typed artifact materialization for named outputs; and
* use viewer payload, sample, and ROI validation for visual presentation.

Normal execution status is lifecycle evidence and output-plate inventory is
persistent-file evidence. Neither is a hidden ``RuntimeValueStore`` dump.

Verification
------------

Run the focused unit test for the owner, then a compiler test and an integration
test when lowering changed. See :doc:`compiler_extension`,
:doc:`runtime_value_extension`, and
:doc:`../architecture/pipeline_compilation_system`.
