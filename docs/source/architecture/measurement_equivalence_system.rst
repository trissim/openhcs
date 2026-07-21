Measurement and runtime equivalence
===================================

Runtime equivalence compares semantic outputs rather than requiring byte-for-byte
identity. It is used to validate backend and CellProfiler parity while accounting
for measurement dialects, row identity, object-label domains, relationships, and
declared numeric tolerances.

Inputs
------

Equivalence operates on runtime observations and snapshots built from typed
artifacts:

- images and masks
- measurement tables and columnar rows
- object labels and object-instance catalogs
- directed object relationships
- spatial grids and sparse label rows

Artifact names, types, execution scopes, source provenance, and component groups
remain part of the comparison identity.

Measurement identity
--------------------

``RuntimeMeasurementFeatureKey`` separates subject, feature, source
qualification, and aggregate identity. ``RuntimeMeasurementDialect`` declares
how a producer encodes source names, qualifiers, aliases, and row layout. This
lets two outputs be normalized without erasing meaningful distinctions.

Row projection derives stable identities for image-, object-, and
relationship-scoped measurements. Wide and long-form tables project into the
same semantic fact model when the dialect explicitly supports that mapping.

Feature semantics
-----------------

``RuntimeMeasurementFeatureSemanticProfile`` is a most-derived context strategy
family. Feature markers and declarations select behavior for counts,
identifiers, locations, calculated values, shapes, intensity, and other roles.
The profile owns value comparison, row-identity stability, and special
derivations for that feature family.

Numeric policy
--------------

``RuntimeEquivalencePolicy`` owns non-negative tolerances, measurement dialect,
name normalization, missing-value behavior, and stability rules. Feature- and
relationship-specific tolerances extend the nominal policy surface rather than
being collected in an external feature-name table.

Object and relationship alignment
---------------------------------

Object-label comparison accounts for plane/object domains and derives required
object measurements from the label values. Relationship comparison preserves
parent and child identities and applies registered alignment strategies when
object instance keys are projected across slices.

Output
------

``RuntimeEquivalenceReport`` contains typed difference records for artifact
counts, measurement features/content, tables, and images. An empty difference
tuple means the compared outputs are semantically equivalent.

Extension rule
--------------

New semantics belong on the authoritative measurement feature, artifact type,
dialect, relationship declaration, or registered strategy. Generic comparison
code must not hardcode concrete CellProfiler feature names or copy tolerances
into a second registry.
