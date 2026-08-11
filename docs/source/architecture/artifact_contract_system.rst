Artifact contracts and planning
===============================

Artifacts are typed semantic values that can enter a callable from sources,
metadata, main flow, or a prior runtime producer. Their architecture is separate
from Python parameter names and storage paths.

Declarations
------------

``ArtifactType``
  A nominal semantic family. The registered core families are special
  side-channel values, images, object labels, measurements, object lineage and
  relationships, tables, spatial grids, spatial graphs, and metadata. An
  external resource path is a typed setting/source binding, not an artifact
  family. A file bundle is a materialisation format over a declared special
  artifact.

``ArtifactSpec`` and ``ArtifactSpecRef``
  A named input or output declaration and its stable semantic reference.

``CallableContract``
  Declares callable-level artifact inputs and outputs alongside processing and
  runtime behaviour. Its ordered ``ArtifactSpecCollection`` values are the
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

Relations and cross-artifact subject identity
---------------------------------------------

``ArtifactSpecRelation`` is the nominal owner for semantics that connect one
target artifact declaration, whether input or output, to an exact
``ArtifactSpecRef``. Relation leaves expose hooks for group scope, source
context, materialisation identity, measurement subject, and artifact-member
subject. Generic planners and materializers call those hooks; they do not branch
on a callable name, artifact filename, feature-column spelling, or concrete
relation subclass.

Object-scoped tables and feature-bearing artifacts use the same declared object
subject without flattening their different local schemas:

``ObjectMeasurementSubjectRelation``
  Binds one measurement row's local identifier to an exact object-label output.
  Declarations without an ``id_field`` still own the measurement subject but do
  not project a row-identity binding for runtime viewer linkage.

``ObjectArtifactMemberSubjectRelation``
  Binds one feature-bearing artifact member's local identifier to that same
  object-label output. A self-owned object-label output can declare the relation
  without adding a dependency edge to itself.

When a relation declares a local identifier, it projects an
``ObjectArtifactSubjectBinding``. The binding derives one producer-scoped
subject token from the referenced object declaration and keeps each target's
local identifier field on its own declaration. Thus a measurement row and
several disconnected path members can represent one object without copying
aggregate measurements onto every path or fabricating combined geometry.

``ArtifactOutputPlan.relations`` preserves these declarations through
compilation. Runtime artifact materialisation passes the exact output plan into
the registered writer, which may project the binding into framework-owned
metadata beside the artifact's ordinary features. ROI persistence and viewer
streaming carry that metadata without exposing the framework keys as biological
table columns. A viewer can then join native rows by the exact subject token and
subject identifier while leaving geometry, feature values, selection, colours,
and layer order on the native layer owner.

This is a generic relation path, not ROI- or neurite-specific dispatch. A new
cross-artifact linkage belongs on an ``ArtifactSpecRelation`` leaf and its
owning declarations; it must not be recovered later by matching column names,
display labels, output paths, or assay vocabulary.

Measurement source provenance
-----------------------------

Measurement rows retain the exact biological coordinates of the runtime planes
they represent. ``SourceImageProvenance`` is the authority for source image
names and ``AllComponents`` values such as well, site, channel, z-index, and
timepoint. A row carrying the canonical ``slice_index`` receives the matching
plane coordinates. A row without a slice axis receives only coordinates common
to the whole represented stack; OpenHCS does not label an aggregate row with a
representative first plane. Producer-declared coordinate columns are retained
only after exact consistency validation against runtime provenance.

The same rule controls persistence names. An aggregate measurement or artifact
uses an aggregate descriptor containing only fixed execution-scope components;
varying site/channel values remain on rows rather than being copied into the
filename. Compile-time previews expose a ``shared_output_stem`` plus the
concrete candidate paths for each applicable materialisation format. They do
not present an ROI-shaped base path as though it were also the CSV output.

Callable ABI versus semantic artifacts
--------------------------------------

Legacy decorators and loaders may expose ``special_inputs`` or
``special_outputs`` names. Those names describe callable ABI slots or output
positions. They do not by themselves declare artifact types, producers,
materialisation, or runtime-store identity.

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
scope, and materialisation relationships across the pipeline.

When a native callable produces ordinary main flow without a named image
artifact, ``unnamed_main_flow_artifact_name`` gives that producer a
deterministic compiler-only identity. The identity lets a later artifact-owned
consumer inherit the exact producer and group scope without generated-name
prefix matching or a CellProfiler-specific branch.

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

Invocation occurrence identity
------------------------------

``ArtifactSpecRef`` identifies a semantic artifact type and name. It does not
identify repeated occurrences of that artifact in one function pattern.
``InvocationArtifactInputProjectionKey`` adds the exact
``FunctionInvocationKey`` and zero-based input index. One
``InvocationArtifactInputEdgePlan`` is stored for each such occurrence and
records:

- the exact declared ``ArtifactSpec``
- an optional prior-producer ``ArtifactInputPlan`` and runtime projection
- whether the occurrence consumes ordinary main flow
- which ``MainFlowInputProjection`` applies

This is why two callable parameters can consume the same semantic artifact
without collapsing into one edge. ``CompiledStepPlan.artifact_inputs`` remains
a semantic-reference-keyed plan map; exact duplicate occurrence identity lives
on the compiled invocation edges.

Implicit main-flow projection
-----------------------------

An invocation edge with ``consumes_main_flow=True`` cannot also carry a storage
plan. Its projection is one of two explicit modes:

``MainFlowInputProjection.COMPLETE_PAYLOAD``
  The invocation has one main-flow reference and consumes the current payload
  as a whole.

``MainFlowInputProjection.DECLARED_SOURCE_IMAGE``
  The current payload represents multiple declared images and the invocation
  selects the image named by its artifact declaration.

The end-to-end regression in
``tests/unit/test_implicit_main_flow_compilation.py`` compiles grouped native
``percentile_normalize`` output into a CellProfiler ``Threshold`` consumer. It
asserts the step dependency, channels ``1``/``2``/``4`` group scope,
``consumes_main_flow=True``, absence of a storage plan, and axis execution
scope.

Compiled plans
--------------

``CompiledStepPlan.artifact_inputs`` and ``artifact_outputs`` contain typed
``ArtifactInputPlan`` and ``ArtifactOutputPlan`` objects. Plans add runtime
addresses, producer edges, group scopes, and materialisation targets while
preserving semantic references. ``InvocationArtifactInputEdgePlan`` supplies
the occurrence-level authority described above.

Runtime
-------

Workers record validated ``RuntimeValue`` instances in ``RuntimeValueStore``
under ``ArtifactKey`` and an explicit backend location. Consumers resolve typed
queries derived from compiled input edges. Repeated producers replace the
current binding explicitly while the observation stream retains history.

Materialisation is a plan over an artifact declaration. It is not a side effect
of naming a Python return value or applying ``@special_outputs``.

Runtime availability versus persistence
----------------------------------------

An output contract first makes a typed value available to the compiled graph.
If a later step consumes it, the runtime store supplies that value under its
exact artifact identity. This is runtime dataflow, not a request to write a
user-facing file.

Persistence is a separate compiled decision. An ``ArtifactSpec`` may carry an
explicit materialisation contract; nominal artifact-type strategies may add
terminal or viewer-only materialisation; and the global runtime-artifact
materialisation setting controls whether persistent runtime-artifact storage is
enabled. ``StepMaterializationConfig`` separately checkpoints the ordinary
main-flow image result. Inspect ``ArtifactOutputPlan.materialization`` and the
step's compiled materialisation plans to learn what will actually persist.

Authoring boundary
------------------

Artifact behaviour belongs to the owning artifact type, callable contract,
CellProfiler module leaf or mixin, or nominal planning/runtime strategy. The
compiler, executor, exporter, and UI consume those declarations rather than
owning copied name tables. See
:doc:`../development/callable_artifact_authoring` for the corresponding
extension workflow.
