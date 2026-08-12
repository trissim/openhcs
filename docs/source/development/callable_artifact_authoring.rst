Callable and artifact authoring
===============================

A processing callable declares its ABI and semantic contract at the callable
boundary. ``CallableContract.from_callable`` snapshots those declarations once
for the compiler.

Callable contract
-----------------

The contract can carry input/output memory types, artifact inputs and outputs,
runtime-bound parameters, required variable components, allowed grouping,
processing contract, execution scope, runtime adapter, preparation hook, and a
callable request binding. Use the existing decorators and declaration helpers;
do not teach compiler phases to inspect backend names. A CellProfiler module
resolves its dynamic artifact names into this same contract rather than
attaching another contract object.

Processing semantics
--------------------

Every callable that participates in normal image processing should declare a
``ProcessingContract``. It describes whether a call is local to a 2D plane or
depends on a wider stack. It does not select the stack axis: the step's
``ProcessingConfig.variable_components`` does that.

Semantic artifacts
------------------

Declare each semantic input/output as a role-neutral ``ArtifactSpec`` with a
nominal ``ArtifactType``. The enclosing ``artifact_inputs`` or
``artifact_outputs`` decorator binds that term to ``ArtifactInputPlan`` or
``ArtifactOutputPlan``; callers should not duplicate an otherwise identical
spec merely to assign its role. ``ArtifactSpec.input()`` and
``ArtifactSpec.output()`` remain explicit constructors when a pre-bound
reference is required. Relations and group/materialization sources point to
exact ``ArtifactSpecRef`` identities.

For CellProfiler declarations, use ``SettingToKeywordBinding.input()`` and
``SettingToKeywordBinding.output()`` for setting-backed artifact roles. Put
module-specific derivation in the leaf hooks supplied by
``CellProfilerModuleArtifactContracts``. The mixin resolves those declarations
into the invocation's ``CallableContract``. Source versus runtime satisfaction,
main-flow publication, and the active output subset are then represented by
compiled source, edge, and artifact plans; do not copy them into declaration
partitions.

Callable names such as ``special_inputs`` and ``special_outputs`` describe ABI
positions only. They are not a substitute for artifact types, producer edges,
or materialization declarations.

Typed table and object-label outputs
------------------------------------

A callable declaring ``MeasurementsArtifactType`` returns a schema-bearing
``ColumnarRows`` payload in the declared output position. Its ``fields`` and
physical ``columns`` must agree exactly in name and order; the runtime wraps the
payload in the measurement value carrying subject, source, and feature-owner
context. A measurement-producing declaration must also put its nominal
``RuntimeMeasurementFeatureOwner`` on the output ``ArtifactSpec``. Later
measurement consumers query that owner; they must not infer it from the
producer invocation name. Returning a bare list of dictionaries or relying on
a filename is not a measurement contract.

A callable declaring ``ObjectLabelsArtifactType`` returns the complete integer
label payload for that output. Runtime contextualization produces the nominal
``ObjectLabelValue``/``ObjectLabelSet`` with its domain, plane axis, source
spatial domain, and provenance. Do not return only an ROI sidecar or infer label
identity from the output tuple position. Multiple declared outputs must retain
their exact declaration order and artifact identities so compiled matching can
associate each runtime value with its producer.

Verification
------------

- Build ``CallableContract.from_callable`` and assert its typed declarations.
- For a CellProfiler module, derive its invocation ``CallableContract`` and
  assert the setting-resolved specs and relations.
- Compile a minimal ``FunctionStep`` and inspect its artifact input/output plans.
- Execute a focused case when runtime-bound parameters or an adapter changed.

See :doc:`../architecture/artifact_contract_system`.
