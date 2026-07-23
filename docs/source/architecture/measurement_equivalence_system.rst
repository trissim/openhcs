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

Official30 evidence boundary
----------------------------

The portable acceptance definition is
``benchmark/manifests/official30_portable_axis1.json`` plus the two integration
tests in ``tests/integration/test_cellprofiler_official30_zmq.py``. The headless
test loads exactly 30 cases, requires native references, runs every case with
``continue_on_error=True``, and requires every native/OpenHCS execution to
succeed and every observation to be equivalent. The Napari test exercises the
same cases one at a time with non-persistent viewer lifecycle and functional
viewer-state checks.

Run the headless acceptance boundary under the repository-wide runtime lock:

.. code-block:: bash

   flock /home/ts/.cache/openhcs/official30-runtime.lock \
     env OPENHCS_CPU_ONLY=true QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg \
     OPENHCS_CP_NATIVE_REFERENCE_ROOT=<durable-native-reference-root> \
     pytest -q \
     tests/integration/test_cellprofiler_official30_zmq.py::test_official30_compile_execute_and_match_native_references_over_zmq

Use the sibling
``test_official30_nonpersistent_napari_isolated_per_case`` target for the Napari
route. Different ZMQ/viewer ports avoid endpoint conflict, but overlapping runs
remain diagnostic only and must not be reported as canonical acceptance timing.

Compared modalities and policy
------------------------------

The OpenHCS benchmark adapter builds typed runtime/output snapshots and compares:

- images and materialized label images when image comparison is enabled
- CSV/table outputs, including measurement and relationship facts
- CPA SQLite table projections
- CellProfiler Analyst ``.properties`` values

The strict CellProfiler policy sets numeric and image absolute/relative
tolerances to ``1e-6``, permits zero differing image fraction, disables the
broad feature-specific relaxations used by less strict compatibility modes, and
applies the same policy to database export comparison. A successful observation
therefore means semantic equivalence under that declared policy, not universal
byte identity.

Minimum durable receipt
-----------------------

The runner emits ``observations.jsonl``, ``observations.csv``,
``phase_timing.csv``, ``summary.csv``, and ``suite_metadata.json``. Those files
record case/suite identity, success, equivalence, difference count, numeric
tolerances, output paths, timing, platform, and native-reference root. They do
not currently record every source identity required for a durable publication
claim.

Retain the generated files together with a receipt containing at least:

- OpenHCS Git commit and whether the worktree was dirty
- SHA-256 of the exact Official30 manifest
- native-reference root identity plus an inventory or digest
- native CellProfiler executable/version identity
- exact command, environment flags, test target, and ZMQ/viewer ports
- suite id, start/end timestamps, host isolation/lock state, and exit status
- compared modalities and the complete equivalence-policy values
- per-case success, ``equivalent``, and ``difference_count`` fields
- for Napari, the uninterrupted-run case set and any separately targeted reruns

``summary.csv`` alone, a transient ``/tmp`` path, a test definition without a
recorded invocation, or a compatibility-matrix report is not a durable parity
receipt. If Napari cases are closed by targeted reruns, report that topology
explicitly rather than describing it as one uninterrupted all-case run.

Extension rule
--------------

New semantics belong on the authoritative measurement feature, artifact type,
dialect, relationship declaration, or registered strategy. Generic comparison
code must not hardcode concrete CellProfiler feature names or copy tolerances
into a second registry.
