Runtime system assembly rules
=============================

Use the compiled execution product as the only bridge from mutable declarations
to workers.

Assembly order
--------------

.. code-block:: text

   PipelineConfig + FunctionStep declarations
       -> one ObjectState resolution
       -> StepSnapshot + CompilationSession
       -> typed CompiledStepPlan objects
       -> CompiledExecutionBundle
       -> workers + RuntimeValueStore
       -> materialization and export

Do not skip compilation by reconstructing paths, source bindings, function
metadata, or artifact identities in an adapter. The typed plans already own
those decisions.

Owner map
---------

``FunctionStep`` and nested configuration
   User-authored callable pattern, variable components, grouping, and requested
   materialization.

``CallableContract``
   Callable ABI, processing requirements, semantic artifact inputs/outputs, and
   runtime behavior for one compiled invocation.

``CellProfilerModuleArtifactContracts``
   CellProfiler-only mixin that resolves active setting bindings and module leaf
   hooks into the invocation's ordinary ``CallableContract``. It is not a
   parallel generic runtime contract.

``CompilationSession`` and ``CompiledStepPlan``
   Axis-scoped compiler invariants and the decisions produced by compilation.

``CompiledExecutionBundle``
   Contexts, worker-safe steps, plans, and the resolved runtime environment.

``RuntimeValueStore``
   Validated runtime values, artifact keys, locations, current bindings, and
   observation history.

``RuntimeSliceProjection``
   Nominal, provenance-aware slice projection and aggregation.

Source and artifacts
--------------------

``SourceBindingsConfig`` is the public source authority. Compilation produces a
source universe and a source-binding plan for each step. Source-satisfied inputs
can be absent from the runtime artifact plan; selectors validate exact plans
that are present.

Artifact identity is an ``ArtifactSpecRef`` at declaration time and an
``ArtifactKey`` plus explicit execution scope at runtime. A filename, Python
parameter name, or return position is not semantic identity.

Application boundaries
----------------------

The GUI, MCP services, and execution transports project typed compiler/runtime
results. They own lifecycle, status, cancellation, and presentation, but not
pipeline semantics. ZMQRuntime owns generic transport; pyqt-reactive owns
generic UI reaction; OpenHCS adapters own domain integration.

Extension rule
--------------

Add a new fact to its declaration or nominal strategy family, compile it once,
and consume the typed result at runtime. Stop if implementation starts requiring
duplicated predicates, copied feature lists, module-name branches, or a generic
component that imports a concrete backend.
