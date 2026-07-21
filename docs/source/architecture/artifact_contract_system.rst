Artifact contracts and planning
===============================

Artifacts are typed semantic values that can enter a callable from sources,
metadata, main flow, or a prior runtime producer. Their architecture is separate
from Python parameter names and storage paths.

Declarations
------------

``ArtifactType``
  A nominal semantic family such as image, object labels, measurements,
  relationships, spatial grids, tables, or files.

``ArtifactSpec`` and ``ArtifactSpecRef``
  A named input or output declaration and its stable semantic reference.

``CallableContract``
  Declares callable-level artifact inputs and outputs alongside processing and
  runtime behavior. Its ordered ``ArtifactSpecCollection`` values are the
  generic callable ABI authority.

``CellProfilerModuleArtifactContracts``
  A CellProfiler module mixin, not a second contract object. It resolves active
  ``SettingToKeywordBinding`` declarations and module leaf hooks for one parsed
  invocation, then returns a ``CallableContract`` containing the resulting
  exact input and output specs.

The callable contract validates artifact relations. CellProfiler's module mixin
also validates its setting-derived names, source availability, grouping
sources, and module-specific relations before the invocation reaches generic
compilation.

Callable ABI versus semantic artifacts
--------------------------------------

Legacy decorators and loaders may expose ``special_inputs`` or
``special_outputs`` names. Those names describe callable ABI slots or output
positions. They do not by themselves declare artifact types, producers,
materialization, or runtime-store identity.

Semantic ownership comes from ``artifact_inputs`` and ``artifact_outputs`` on
the callable. A CellProfiler module derives those declarations through its
nominal module type; it does not attach a parallel artifact contract. Storage
paths are a compiled consequence, not the artifact's identity.

Forward planning
----------------

For each step, ``ArtifactDeclarationStepContext`` carries source declarations,
available producers, main-flow identity, and step identity. Invocation contract
providers may replace a public callable with a compile-only runtime contract
while retaining the ordinary public declaration at the boundary.

Artifact extraction advances an ``ArtifactGraph``. The graph is the source of
truth for producer identity, artifact kind, invocation ownership, grouping
scope, and materialization relationships across the pipeline.

Satisfaction and exact selection
--------------------------------

The path planner decides how each input is satisfied:

- source binding or source metadata
- main-flow payload
- metadata already present in the context
- a runtime artifact plan backed by a prior producer

``ArtifactPlanKeySelector`` selects declared plans in declaration order. An
input satisfied outside the runtime store is correctly absent from the runtime
plan map. A plan that is present must match the declaration's exact artifact
reference; type/name mismatches fail compilation.

Source binding, metadata, main-flow, and runtime-store satisfaction are
compiler decisions represented by source plans and invocation input edges.
They are not declaration partitions. Compiled invocation plans, rather than a
second declaration object, also own which declared outputs are active.

Compiled plans
--------------

``CompiledStepPlan.artifact_inputs`` and ``artifact_outputs`` contain typed
``ArtifactInputPlan`` and ``ArtifactOutputPlan`` objects. Plans add runtime
addresses, producer edges, group scopes, and materialization targets while
preserving semantic references.

Runtime
-------

Workers record validated ``RuntimeValue`` instances in ``RuntimeValueStore``
under ``ArtifactKey`` and an explicit backend location. Consumers resolve typed
queries derived from compiled input edges. Repeated producers replace the
current binding explicitly while the observation stream retains history.

Materialization is a plan over an artifact declaration. It is not a side effect
of naming a Python return value or applying ``@special_outputs``.

Runtime availability versus persistence
----------------------------------------

An output contract first makes a typed value available to the compiled graph.
If a later step consumes it, the runtime store supplies that value under its
exact artifact identity. This is runtime dataflow, not a request to write a
user-facing file.

Persistence is a separate compiled decision. An ``ArtifactSpec`` may carry an
explicit materialization contract; nominal artifact-type strategies may add
terminal or viewer-only materialization; and the global runtime-artifact
materialization setting controls whether persistent runtime-artifact storage is
enabled. ``StepMaterializationConfig`` separately checkpoints the ordinary
main-flow image result. Inspect ``ArtifactOutputPlan.materialization`` and the
step's compiled materialization plans to learn what will actually persist.

Extension rule
--------------

Add a new artifact behavior to the owning artifact type, callable contract,
CellProfiler module leaf or mixin, or nominal planning/runtime strategy. Do not
add a copied name table to the compiler, executor, exporter, or UI.
