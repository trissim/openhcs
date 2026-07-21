Runtime-value extensions
========================

Add a nominal payload type when new data has distinct semantic invariants.
Do not encode the family as a dictionary tag or infer it from ndarray shape.

Required decisions
------------------

1. Define the payload and its artifact type/spec boundary.
2. Decide how it is validated and wrapped in ``RuntimeValue``.
3. Register ``RuntimeSliceProjectionStrategy`` behavior if it can be projected
   or aggregated across runtime slices.
4. Add serialization, worker transport, materialization, equivalence, or viewer
   strategies only where the value participates in those systems.
5. Keep persistence location separate from its ``ArtifactKey``.

Projection strategies are selected by nominal type and MRO specificity. A value
that preserves a runtime slice must carry explicit plane-axis and source
identity declarations; raw shape is insufficient.

``RuntimeValueStore`` records validated values under an artifact key and
explicit backend location. Consumers resolve typed queries derived from
compiled artifact edges. Missing or ambiguous results must fail rather than
searching paths or fallback names.

Verify store record/replace/query behavior, slice projection and aggregation,
transport round trips, and materialization for every supported backend.

See :doc:`../architecture/runtime_value_system`.
