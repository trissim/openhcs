Pipeline compilation
====================

OpenHCS compiles all selected execution axes before running any of them. The
compiler is organized around resolved declarations, sessions, and typed plan
fields—not a public promise of a fixed number of phases.

Inputs and outputs
------------------

Input
  ``PipelineConfig`` plus ``list[FunctionStep]`` and an initialized
  ``PipelineOrchestrator``.

Output
  A ``CompiledExecutionBundle`` owning runtime and transport contexts, worker
  assignments, and the compiled runtime environment.

Resolve once
------------

The normal compiler path materializes the pipeline and each step from ObjectState
once. ``ResolvedPipelineDefinition`` groups the resolved steps, their ObjectState
map, and a ``StepSnapshot`` tuple shared by every axis compilation.

A ``StepSnapshot`` contains:

- stable step index
- ObjectState scope identity
- the already-resolved step object

All step semantics remain on that resolved step and its authoritative callable
or module declarations. Later stages must not call ObjectState again to recover
``group_by``, source bindings, or other semantic fields.

Callable catalog and preparation boundary
-----------------------------------------

Pipeline-source parsing and code/transport normalization are not compilation.
``FunctionStepTransportAuthority`` normalizes callable identity for those
boundaries. When the callable's registry owner is already loaded,
``RegistryService.registered_callable()`` projects that owner directly; it does
not initialize unrelated registry families or import execution-only runtimes.
Cold normalization and later catalog discovery must return the same registered
wrapper identity rather than create a second callable projection.

Native module families may declare both their owned callable names and the
catalogue module that exposes them. Catalogue discovery projects those nominal
claims, rejects duplicate canonical names, and does not maintain a parallel
module-to-function list.

Compilation owns the deferred warmup. Callable leaves become typed
``FunctionReference`` and ``CallableContract`` values. Resolving those
references initializes the function catalog, and
``prepare_compiled_context_callables()`` resolves and caches every compiled
invocation and runs its declaration-owned module or callable preparation hook
before the execution bundle is emitted. A first compilation may therefore pay
catalog and backend preparation costs; function-step orchestration does not own
another preparation phase in its hot path. A spawn worker may still resolve a
reference and rebuild a process-local callable cache because parent-process
caches cannot cross that process boundary.

This ordering is an ownership boundary, not merely a startup optimization.
Source inspection must stay safe and bounded, while compilation must fail if a
selected callable cannot be resolved or prepared. Do not move catalog-wide
discovery into source loading, defer preparation until the first image batch,
or maintain a second source-only callable registry.

Compilation sessions
--------------------

``AxisCompilationRequest`` carries the resolved ``PipelineConfig`` into axis
fanout. It projects the pipeline's source-image-set identity policy onto each
new ``ProcessingContext`` before constructing the narrower session boundary.
``CompilationSession`` does not retain a second pipeline-configuration owner.

Each axis receives a ``CompilationSession`` joining:

- the axis ``ProcessingContext``
- resolved steps and snapshots
- the step ObjectState map
- global and orchestrator configuration
- the virtual source-workspace projection
- the mutable ``CompiledStepPlan`` map
- plate and transport execution facts

The session validates one-to-one step, state, snapshot, and plan identity. A
compiler stage accepts the session or a narrower typed request derived from it;
it does not pass loosely related dictionaries between phases.

Typed step plans
----------------

``CompiledStepPlan`` is deliberately mutable during compilation and becomes the
single plan authority for a step. Important fields include:

- input/output paths and storage backends
- source binding, source universe, and source load plans
- main-flow dependency
- artifact input/output plans and execution group scope
- compiled function pattern and callable contracts
- variable components, grouping, and sequential filters
- input, output, and execution memory roles plus framework-local device bindings
- materialization plans and enabled streaming declarations

Stages mutate these fields directly. Code should not recreate
``context.step_plans[index]["special_outputs"]``-style semantic dictionaries.
Callable execution scope remains on the compiled invocation contract. Worker
assignments and runtime-environment decisions remain on
``CompiledExecutionBundle``.

Planning order and invariants
-----------------------------

The implementation may split or combine internal stages, but these dependencies
remain stable:

1. declaration configuration is resolved before plan semantics are consumed;
2. source workspace and source universes exist before source-bound inputs are
   validated;
3. callable/module contracts exist before the artifact graph and paths are
   finalized;
4. artifact satisfaction is decided before runtime input edges are required;
5. memory, runtime projection, materialization, worker start, and
   framework-device facts are validated before the execution bundle is emitted.

Compile completeness and runtime ownership
------------------------------------------

Before a function step can execute, its plan requires an axis identity, input
and output paths, variable components, read/write backends, pipeline position,
compiled function pattern, and other execution-owned fields. Failure is reported
as a compilation error rather than repaired with runtime fallback logic.

Values that only a producer can create are different from compile-time facts.
They become ``RuntimeValue`` instances during execution and fail as typed
``RuntimeValueStore`` resolution errors if a required producer did not publish
them. Viewer process readiness, render progress, settlement, evidence capture,
and cleanup are runtime service/protocol responsibilities. They are not fields
on ``CompiledStepPlan``.

Execution bundle
----------------

``CompiledExecutionBundle`` includes ``CompiledRuntimeEnvironmentPlan``. That
plan records the selected multiprocessing start method and its reason, threading
mode, server mode, GPU enablement facts, and configured worker count. The bundle
also owns the axis-to-worker assignments and projects its axis identities from
the compiled ``ProcessingContext`` values. Runtime-context mapping keys identify
individual contexts and must not be parsed as a second axis authority.
Framework-local device bindings remain on the compiled step plans that derived
them. Workers consume these compiled decisions instead of resolving global
configuration again.

Execution scopes
----------------

Callable contracts declare either ``AXIS`` or ``PLATE`` execution scope.
Ordinary image, segmentation, and measurement steps are axis-scoped: workers
execute them for each compiled plate axis and record typed observations.

A plate-scoped step is terminal. Workers skip it during axis execution. After
all axes complete, the parent runtime merges exact observations from their
``RuntimeValueStore`` instances, constructs the declared ``RuntimeArtifactBatch``
inputs, and executes the step once for the plate. Spreadsheet and CPA database
exports use this route so they see the complete plate rather than one well or
axis. A plate-scoped step in a non-terminal position, missing merged inputs, or
with ambiguous artifact scope is a compile/runtime error rather than a fallback
to per-axis execution.

After axis and plate-scoped work completes, the parent refreshes each populated
OpenHCS metadata target from its completed storage contents before viewer
settlement. Step plans declare those output targets; the final projection is
deduplicated across axes so a multi-well plate records every materialized image
without creating a second metadata owner.

Adding compiler behavior
------------------------

1. Identify the authoritative declaration that owns the new fact.
2. Add a typed field or typed nested plan to ``CompiledStepPlan`` or the runtime
   environment as appropriate.
3. Populate it in a stage with explicit prerequisites.
4. Validate it at the earliest complete boundary and require it at runtime.
5. Test session identity, missing prerequisites, serialization, and worker use.

Do not add a parallel context dictionary, hardcoded backend-name switch, or
fallback lookup chain.

Related pages
-------------

- :doc:`system_overview`
- :doc:`source_model`
- :doc:`artifact_contract_system`
- :doc:`runtime_value_system`
- :doc:`streaming_boundary_and_wrappers`
