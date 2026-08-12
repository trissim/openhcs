Runtime values, stores, and projection
======================================

OpenHCS runtime data is not limited to ndarrays. Images, object labels,
measurements, relationships, spatial grids, sparse labels, and tables carry
different invariants and projection behavior. Each family has a nominal value
type rather than a shared dictionary payload.

Value families
--------------

``RuntimeArrayPayload``
  Nominal array protocol with explicit NumPy interoperation.

image payloads
  Image data plus mask, plane-axis declaration, source metadata, spatial domain,
  and source-plane provenance.

object-label values
  Dense or sparse labeled-object domains, grouped sets, and storage/building
  strategies.

``MeasurementTable`` and ``ColumnarRows``
  Typed measurement and tabular values with row/column semantics.

``ObjectRelationship``
  Directed parent/child or other object relationships with endpoint identity.

``SparseIJVLabelRows`` and ``SpatialGrid``
  Sparse label and grid domains used by segmentation and measurement modules.

Artifact identity
-----------------

``RuntimeValue`` pairs a payload with an ``ArtifactKey``. The key contains the
semantic artifact name/type, execution-axis scope, optional component group, and
semantic identity. Persistence location is represented separately by
``RuntimeArtifactLocation``.

Invocation-selected output plans remain keyed by ``ArtifactSpecRef`` when they
enter ``RuntimeAdapterRequest``. ``RuntimeReturnedOutputMatcher`` binds returned
positions to the exact plan/spec refs, and adapter lookup accepts the
declaration-derived ref whose key agrees with its plan. Two outputs may
therefore share a name when their artifact types differ: selection, matching,
and request lookup preserve both identities rather than collapsing to the name.

RuntimeValueStore
-----------------

Each processing context owns a ``RuntimeValueStore``. It is the source of truth
for:

- validating and recording typed values
- exact current binding by artifact key and backend location
- explicit replacement when a later producer supersedes a binding
- typed queries derived from compiled artifact edges
- append-only observations and revision-safe query caches

Runtime consumers resolve an exact ``RuntimeArtifactQuery``. Missing,
ambiguous, incorrectly typed, or incorrectly scoped results are errors; the
store does not guess from filenames or search a fallback chain.

Axis and group scope
--------------------

``RuntimeExecutionAxisScope`` identifies the compiled execution axis and optional
component group. ``ComponentGroupScope`` is used at declaration/plan time and
can represent ungrouped, fixed, or dynamic group selection. These scopes are
part of artifact identity rather than encoded into ad hoc path strings.

Runtime slice projection
------------------------

``RuntimeSliceProjectionStrategy`` is a nominal-type registry family. A strategy
declares the runtime value type it owns. Selection follows the value's MRO, and
projection uses explicit plane-axis and source-identity declarations.

Registered strategies cover image payloads, aligned image collections, object
labels, measurement tables, columnar rows, sparse labels, relationships, and
identity-projectable semantic values. Primitive/pass-through types are declared
explicitly. If a new value family can carry a runtime slice, its projection and
aggregation behavior must be registered with the value type.

Function-output contextualization
---------------------------------

A callable return does not inherit context from whichever runtime value happens
to be nearby. ``FunctionOutputContextStrategy`` first selects behavior from the
exact compiled output artifact type. For an image output,
``ImageOutputSourceContextStrategy`` then selects behavior from the nominal type
of the source chosen by the output's compiled ``source_context_source``
relation. This is a two-stage owner dispatch, not a table of callable names or
payload shapes.

An image result that already carries context may bypass derivation only when its
source identity is complete and, for a declared artifact output, its represented
source identities exactly align with the source surfaces. Otherwise the selected
source strategy attaches context and validates the plane projection. Aligned
value sets preserve per-slice context and require their count to match the
declared runtime-slice axis.

Object-label sources have a specialized image-output rule. A rendered label
volume with no selected plane must acquire the invocation's declared plane axis,
including when the volume depth is one. Its source-plane provenance count and
rendered leading dimension must match that axis. A genuine 2D label value does
not acquire a synthetic plane axis. These decisions use the nominal
``ObjectLabelValue`` source plus declared provenance; rank alone is not an
identity authority.

Contextualizers attach source and plane context. The compiled artifact plan owns
the artifact name and type, and the runtime store owns the resulting artifact
key. A contextualizer must not rename a value, select a same-named plan, or
coerce one artifact family into another.

Declared ``SpecialArtifactType`` outputs are side-channel values rather than
image slices. Their nominal ``UnchangedFunctionOutputContextStrategy`` returns
the value unchanged and does not invoke the image plane projector. The artifact
declaration therefore determines projection behaviour; individual backend
result classes do not need a copied exemption registry.

Provenance invariant
--------------------

Projection cannot infer semantic slice identity from array shape. A projected
stack requires a declared runtime plane axis and complete per-slice component or
source provenance when source identity matters. Cardinality mismatches fail at
the projection boundary.

Adding a runtime value
----------------------

1. Add the nominal value type at the owning semantic boundary.
2. Add its artifact type/contract declaration if it is a new artifact family.
3. Register slice projection, aggregation, serialization, and materialization
   behavior that the value requires.
4. Add store/query and worker-transport tests.
5. Keep backend storage mechanics in PolyStore and array conversion mechanics in
   ArrayBridge.
