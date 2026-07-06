# CellProfiler Measurement Semantic Authority Refactor Plan

Date: 2026-07-03

This records Aristotle and Faraday's findings and turns them into an
implementation plan. The target pattern is module-owned semantic authority
through inheritance:

- CP module classes inherit semantic authority mixins.
- The inheritance is the tag.
- `CellProfilerModule.__registry__.values()` is the CP module registry.
- Generic consumers query registered module classes and their MRO.
- Authority classes are the semantic tokens.
- Runtime semantics are read from owning module declarations, registered profile
  classes, and enum-member payloads.
- Existing sidecar authority classes, role tuples, local feature-name sets,
  fallback maps, and mirrored enum/string roles are deletion targets.

The established OpenHCS pattern to follow is already present in
`openhcs/interop/cellprofiler/module_declarations.py`:

- `CellProfilerModule.measurement_feature_types()` discovers nested feature
  enum declarations through MRO.
- `ArtifactContractModule.declared_artifact_capabilities()` discovers declared
  capabilities through MRO.

Review rule: every time an implementation introduces an iteration over semantic
objects, stop and name the nominal `__registry__` that owns that iterable. Use
the existing owning registry when it exists. Create a registered semantic family
when the iterated semantic lacks a registry owner. Replace parallel provider
lists, sidecar registries, and tuples of semantic identities with that
registry-owned query.

Implementation readiness rule: the end state must be statically classifiable by
file boundary before coding starts.

- `openhcs/core/runtime_semantics.py` may define identities, payload carriers,
  parser/render declarations, relations, row axes, row values, subjects, and
  artifact-free runtime names.
- `openhcs/core/runtime_semantics.py` must not import runtime-equivalence policy
  types, numeric tolerance profiles, CP modules, CP backend helpers, or value
  comparison functions.
- Equivalence behavior lives in `openhcs/core/equivalence/*` roots and in
  backend-specific profile implementations that inherit those roots.
- CP parser/render names live in CP modules or CP support files owned by those
  modules. CP value comparison and row-stability behavior enters core only
  through registered `RuntimeMeasurementFeatureSemanticProfile` subclasses.
- Before implementation, create a non-commit checkpoint of the dirty worktree.
  Do not make a real git commit for intermediate work. The checkpoint is a
  patch/tar safety copy plus an optional local branch label pointing at the
  current base commit; the working tree remains dirty and continues from the
  same state.

Concrete non-commit checkpoint:

```bash
checkpoint_dir="/tmp/openhcs_measurement_semantics_checkpoint_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$checkpoint_dir"
git status --short > "$checkpoint_dir/status.txt"
git diff --binary HEAD > "$checkpoint_dir/tracked.diff"
git ls-files --others --exclude-standard -z > "$checkpoint_dir/untracked.files"
tar --null -T "$checkpoint_dir/untracked.files" -czf "$checkpoint_dir/untracked.tgz"
git branch "wip/cp-measurement-semantic-refactor-$(date +%Y%m%d_%H%M%S)" HEAD
```

The branch records the base commit only. The patch and untracked tarball record
the dirty state. Do not `git add` or commit these intermediate changes unless
the full parity/refactor run is passing and the final semantic batch is ready.

## Running Checklist

This section is the active implementation tracker. Update it after each batch
with concrete evidence from the current worktree.

- [x] Non-commit checkpoint created:
  `/tmp/openhcs_measurement_semantic_refactor_20260703_034657`.
- [x] Zernike parser/render ownership moved out of core into
  `openhcs/processing/backends/cellprofiler/zernike.py`.
- [x] CP marker lookup uses marker classes as semantic authority through
  `CellProfilerModule.measurement_feature_marker_types_for_key(...)`.
- [x] Focused tests passed after marker-provider correction:
  `pytest -q tests/unit/test_runtime_equivalence.py tests/unit/test_measureobjectintensitydistribution.py -q`.
- [x] Remove the remaining `ObjectMeasurementFeatureRole` strategy/authority
  mirror from `openhcs/core/equivalence/measurement_features.py`.
- [x] Remove the remaining `CurrentObjectFeatureVectorProvider` sidecar and
  move shape current-vector logic onto `MeasureObjectSizeShapeModule`.
- [x] Remove CP default feature-specific numeric tolerance exceptions:
  strict parity passes `feature_numeric_tolerances=()`, pipeline execution does
  not consume runtime-equivalence policy, and the CP default exception list had
  no product/parity purpose. Focused equivalence tests passed after removing
  the dead default-behavior assertions:
  `pytest -q tests/unit/test_cellprofiler_interop_namespace.py tests/unit/test_runtime_equivalence_package.py tests/unit/test_runtime_equivalence.py -q`.
- [x] Move remaining central CP dialect mirrors to owning modules:
  Haralick texture prefixes and module-specific feature aliases/category
  prefixes. Direct aliases, category prefixes, source-feature prefixes,
  calculated-feature prefixes, scale-qualified prefixes, and numbered prefix
  aliases are now declared by owning modules and aggregated through
  `CellProfilerModule.__registry__.values()`.
- [x] Replace static CSV field-list globals with dataclass-derived field names:
  classification, granularity, maxima, projection, and skeleton.
- [x] Let executable CP modules inherit the base `ProcessingContract.PURE_2D`
  default; only non-executable declarations keep `contract = None`, and
  flexible/3D/volumetric modules override explicitly.
- [x] Re-run the validation `rg` checks in Batch 9. The semantic-mirror
  checks are empty, including the deleted CP numeric-tolerance default names.
- [x] Keep selected-image axis semantics compile-time/source-schema owned:
  `AlignModule` no longer declares `default_variable_components = CHANNEL`;
  generated Align steps infer the axis from selected source aliases through
  `PipelineImageSchema` / `SourceProcessingComponentSemantics`. Runtime-artifact
  consumers also inherit source metadata axes when no explicit source-stack axis
  exists. Generator tests cover both ChannelNumber-selected Align inputs and
  Site-selected Align inputs, proving the axis rule is source-schema derived and
  orthogonal to the Align module.
- [x] Collapse absorbed CellProfiler backend facade helpers into
  `CellProfilerFunctionCatalog`: production, generated pipelines, and tests now
  call `CellProfilerFunctionCatalog.get_function(...)`,
  `.require_function(...)`, `.list_functions()`, and `.runtime_metadata(...)`
  directly instead of one-line package wrapper functions.
- [x] Collapse module-owned measurement-row declaration lookup into
  `CellProfilerModule.declared_authority_types(...)`: measurement stat-field
  and feature-template enum declarations inherit `CellProfilerModuleAuthority`,
  and `ModuleOwnedResultMeasurementRows` no longer maintains a separate MRO
  scanner.
- [x] Preserve catalog authority across OpenHCS registry discovery:
  `OpenHCSRegistry` no longer mutates `OpenHCSFunctionCatalogModule` globals
  with registry wrapper controls. The CellProfiler catalog remains the public
  function/documentation authority after registry scans.
- [x] Make direct source-binding group-by fallback consistent with module
  declarations: source inference still wins when it resolves a group axis, but
  direct source-bound modules now fall back to the module default
  `group_by=GroupBy.CHANNEL` with the existing stack-axis collision
  normalization.
- [x] Collapse indexed descriptor discovery onto the descriptor declaration
  registry: `RuntimeMeasurementIndexedDescriptorDeclaration.__registry__` now
  owns descriptor matching and suffix-width lookup. The CP dialect no longer
  calls through `CellProfilerModule.indexed_descriptor_suffix_token_width`, and
  `CellProfilerDescriptorSemanticProfile` no longer rediscover descriptors via
  module feature enums.
- [x] Preserve same-table runtime image-feature multiplicity while still
  deduping duplicate measurement artifacts: image-feature fact dedupe is scoped
  to distinct runtime records, so repeated rows inside one measurement record
  remain load-bearing facts.
- [ ] Run targeted tests, generated pipeline tests, and the strict 30-pipeline
  parity benchmark.

Current validation status from the dirty worktree:

```text
rg -n "role = ObjectMeasurementFeatureRole|object_measurement_feature_roles|feature_families_for_object_measurement_role|matches_object_measurement_feature_role" openhcs
# empty

rg -n "ObjectMeasurementFeatureRole\.(SHAPE_DESCRIPTOR|ZERNIKE_DESCRIPTOR|INTENSITY)" openhcs
# empty

rg -n "IndexedObjectZernikeDescriptor|ObjectZernikeDescriptorFeature|indexed_object_intensity_zernike_feature_name" openhcs/core openhcs/interop
# empty

rg -n "CELLPROFILER_HARALICK_TEXTURE_FEATURE_PREFIXES|CLASSIFICATION_RESULT_FIELDS|GRANULARITY_FIELDS|CurrentObjectFeatureVectorProvider|FeatureRoleAuthority" openhcs benchmark tests -g '*.py'
# empty for deleted production mirrors/providers

rg -n "CurrentObjectFeatureVectorProvider|FeatureRoleAuthority" openhcs/processing/backends/cellprofiler openhcs/core/equivalence
# empty
```

Recent verification:

```text
pytest -q tests/unit/test_measuregranularity.py tests/unit/test_cellprofiler_module_execution.py::test_classification_rows_include_unclassified_objects -q
# passed

pytest -q tests/unit/test_cellprofiler_processing_backend.py -q
# passed

pytest -q tests/unit/test_cellprofiler_module_execution.py::test_classification_rows_include_unclassified_objects tests/unit/test_cellprofiler_processing_backend.py::test_cellprofiler_processing_backend_exports_absorbed_function -q
# passed after CellProfiler package exports moved to typed OpenHCSFunctionCatalogModule catalog access

pytest -q tests/unit/test_runtime_equivalence.py tests/unit/test_measureobjectintensitydistribution.py -q
# passed

pytest -q tests/unit/test_runtime_equivalence.py tests/unit/test_measureobjectintensitydistribution.py tests/unit/test_cellprofiler_source_schema.py tests/unit/test_cellprofiler_generated_pipeline_execution.py -q --maxfail=5 --tb=short
# passed after dialect provider hooks moved module-owned prefixes/aliases out of measurement_dialect.py

AST check: executable modules with explicit `contract = None`
# empty; remaining explicit None declarations have no concrete function

pytest -q tests/unit/test_cellprofiler_source_schema.py tests/unit/test_cellprofiler_generated_pipeline_execution.py -q --maxfail=5 --tb=short
# passed after compile-time source-axis lineage fix

pytest -q tests/unit/test_cellprofiler_source_schema.py::test_align_source_images_infer_axis_from_selected_source_aliases tests/unit/test_cellprofiler_source_schema.py::test_align_source_images_adapt_to_site_axis_from_selected_source_aliases tests/unit/test_cellprofiler_source_schema.py::test_codegen_preserves_source_timepoint_lineage_for_runtime_artifact_steps -q --tb=short
# passed; Align infers CHANNEL or SITE from source alias selectors, not module defaults

pytest -q tests/integration/test_cellprofiler_generated_pipeline.py::test_official_example_untangleworms_cppipe_executes_via_source_schema_workspace tests/integration/test_cellprofiler_generated_pipeline.py::test_official_example_untangleworms_brightfield_cppipe_executes_overlay tests/integration/test_cellprofiler_generated_pipeline.py::test_official_example_colocalization_cppipe_executes_relationship_exports -q --maxfail=3 --tb=short
# passed after runtime overlay stack assertion was updated to site-stack semantics

pytest -q tests/unit/test_cellprofiler_processing_backend.py::test_cellprofiler_processing_backend_exports_absorbed_function tests/unit/test_cellprofiler_backend_public_api.py tests/unit/test_cellprofiler_generated_pipeline_execution.py::test_generator_uses_absorbed_function_contract_for_unknown_registry_contract tests/unit/test_pycodify_formatters.py tests/unit/test_function_step_transport.py::test_transport_authority_accepts_stripped_compiled_function_steps tests/unit/test_cellprofiler_runtime_callable_introspection.py --tb=short
# passed after deleting CellProfilerFunctionCatalog package wrapper aliases

pytest -q tests/unit/test_cellprofiler_module_execution.py::test_classification_rows_include_unclassified_objects tests/unit/test_cellprofiler_backend_public_api.py tests/unit/test_cellprofiler_generated_pipeline_execution.py::test_generator_uses_absorbed_function_contract_for_unknown_registry_contract --tb=short
# passed after routing ModuleOwnedResultMeasurementRows through CellProfilerModule.declared_authority_types(...)

pytest -q tests/unit/test_cellprofiler_processing_backend.py tests/unit/test_cellprofiler_backend_public_api.py tests/unit/test_cellprofiler_generated_pipeline_execution.py tests/unit/test_cellprofiler_module_execution.py::test_classification_rows_include_unclassified_objects tests/unit/test_cellprofiler_runtime_callable_introspection.py tests/unit/test_function_step_transport.py::test_transport_authority_accepts_stripped_compiled_function_steps --maxfail=3 --tb=short
# 109 passed after catalog-module mutation boundary, direct-source group_by fallback, and execution-plan helper update

pytest -q tests/unit/test_cellprofiler_source_schema.py::test_align_source_images_infer_axis_from_selected_source_aliases tests/unit/test_cellprofiler_source_schema.py::test_align_source_images_adapt_to_site_axis_from_selected_source_aliases tests/unit/test_cellprofiler_source_schema.py::test_codegen_keeps_source_binding_channel_out_of_runtime_artifact_scope tests/unit/test_cellprofiler_source_schema.py::test_codegen_preserves_source_timepoint_lineage_for_runtime_artifact_steps tests/unit/test_cellprofiler_library_loading.py::test_cellprofiler_threshold_diagnostics_matches_reference_formula tests/unit/test_runtime_values.py::test_object_label_set_preserves_payload_parent_image_spacing --maxfail=3 --tb=short
# 6 passed after direct-source group_by fallback

pytest -q tests/unit/test_runtime_equivalence.py::test_runtime_reference_artifact_equivalence_ignores_duplicate_measurement_artifacts tests/unit/test_runtime_equivalence.py::test_runtime_reference_artifact_equivalence_ignores_duplicate_image_feature_rows tests/unit/test_runtime_equivalence.py::test_runtime_reference_artifact_equivalence_preserves_same_table_image_feature_rows --tb=short
# 3 passed after same-record image-feature rows stopped being deduped against themselves

pytest -q tests/unit/test_runtime_equivalence.py tests/unit/test_measureobjectintensitydistribution.py --maxfail=3 --tb=short
# 198 passed after descriptor discovery moved to RuntimeMeasurementIndexedDescriptorDeclaration.__registry__
```

## Executed AST Audit: 2026-07-03

This audit was run over the current `/home/ts/code/projects/openhcs` worktree.
It found the concrete semantic mirrors that must be removed, not future places to
look.

### Role System Mirrors

Current leaking classes:

- `openhcs/core/equivalence/measurement_features.py:47`
  `ObjectMeasurementFeatureRoleAuthority`
- `openhcs/core/equivalence/measurement_features.py:127`
  `ObjectMeasurementFeatureRoleStrategy`
- `openhcs/core/equivalence/measurement_features.py:187`
  `ObjectCountFeatureRoleStrategy`
- `openhcs/core/equivalence/measurement_features.py:207`
  `ObjectIdentifierFeatureRoleStrategy`
- `openhcs/core/equivalence/measurement_features.py:232`
  `ObjectMeasuredObjectAnchorFeatureRoleStrategy`
- `openhcs/core/equivalence/measurement_features.py:257`
  `ObjectLocationFeatureRoleStrategy`
- `openhcs/core/equivalence/measurement_features.py:285`
  `ObjectIntensityFeatureRoleStrategy`
- `openhcs/core/equivalence/measurement_features.py:312`
  `ObjectCalculatedFeatureRoleStrategy`
- `openhcs/core/equivalence/measurement_features.py:343`
  `ObjectShapeDescriptorFeatureRoleStrategy`
- `openhcs/core/equivalence/measurement_features.py:368`
  `ObjectZernikeDescriptorFeatureRoleStrategy`

These classes mirror semantic roles through `ObjectMeasurementFeatureRole`
instead of consuming feature-member marker payloads and descriptor declarations.
The replacement is a registered semantic-profile family that answers the same
queries by iterating `CellProfilerModule.__registry__.values()`,
`module_type.measurement_feature_types()`, and marker/descriptor payloads on the
actual `RuntimeMeasurementFeature` members.

Downstream call sites currently importing this role system:

- `openhcs/core/equivalence/measurement_facts.py:21`
- `openhcs/core/equivalence/measurement_requirements.py:19`
- `openhcs/core/equivalence/object_label_measurements.py:41`
- `openhcs/core/runtime_equivalence.py:53`

These consumers must move to the semantic-profile query surface instead of
passing an `ObjectMeasurementFeatureRole`.

### CP Module Sidecars

Current CP module sidecars:

- `openhcs/processing/backends/cellprofiler/intensity.py:296`
  `MeasureObjectIntensityFeatureRoleAuthority`
- `openhcs/processing/backends/cellprofiler/zernike.py:327`
  `ObjectZernikeDescriptorFeatureRoleAuthority`
- `openhcs/processing/backends/cellprofiler/shape.py:87`
  `MeasureObjectSizeShapeModule` inherits `ObjectMeasurementFeatureRoleAuthority`

The intensity sidecar duplicates the module it names. Its semantics belong on
`MeasureObjectIntensityModule.MeasurementFeature` members through marker
payloads. The Zernike sidecar belongs in the same file as registered descriptor
declarations consumed from feature-member payloads. The shape module should
inherit broad marker authorities and attach the Zernike descriptor declaration to
`MeasurementFeature.ZERNIKE`; it should not implement role dispatch.

### Zernike Ownership Leak

Current core-owned CP Zernike declarations:

- `openhcs/core/runtime_semantics.py:1762`
  `ObjectZernikeDescriptorFeature`
- `openhcs/core/runtime_semantics.py:1786`
  `ObjectIntensityZernikeFeatureNameStrategy`
- `openhcs/core/runtime_semantics.py:1830`
  `indexed_object_intensity_zernike_feature_name`
- `openhcs/core/runtime_semantics.py:1845`
  `ObjectIntensityZernikeMeasurementRows`
- `openhcs/core/runtime_semantics.py:2829`
  `IndexedObjectZernikeDescriptor`
- `openhcs/core/runtime_equivalence.py:2146`
  `ShapeZernikeDescriptorFeatureSemantics`
- `openhcs/core/runtime_equivalence.py:2176`
  `ObjectZernikeDescriptorStabilityContract`
- `openhcs/core/runtime_equivalence.py:3759`
  `_zernike_descriptor_values_equivalent`

These are CellProfiler-specific. The concrete owner is
`openhcs/processing/backends/cellprofiler/zernike.py`, which already owns
Zernike kernels, `IntensityZernikeMeasurementRowsRequest`,
`ObjectIntensityZernikeMeasurementColumnarRows`, backend strategy selection, and
debug traces. Move parser/render declarations there behind registered
`RuntimeMeasurementIndexedDescriptorDeclaration` implementations. Move
stability/equivalence behavior there through the equivalence-side
`RuntimeMeasurementIndexedDescriptorEquivalence` mixin consumed by CP semantic
profiles.

Current CP files importing leaked core Zernike declarations:

- `openhcs/processing/backends/cellprofiler/shape.py:11`
- `openhcs/processing/backends/cellprofiler/zernike.py:26`

`shape.py` should stop importing Zernike parser types entirely. It should only
attach `ShapeObjectZernikeDescriptorDeclaration` to
`MeasureObjectSizeShapeModule.MeasurementFeature.ZERNIKE`.

### Relation Payload Mirror

Current role-qualified relation mirror:

- `openhcs/core/runtime_semantics.py:1409`
  `ObjectMeasurementFeatureRole`
- `openhcs/core/runtime_semantics.py:1567`
  `RoleQualifiedRuntimeMeasurementFeatureFamilyRelation`
- `openhcs/processing/backends/cellprofiler/intensity.py:1878`
  `MAX_INTENSITY_X` declares source/target roles and target member by string
- `openhcs/processing/backends/cellprofiler/intensity.py:1888`
  `MAX_INTENSITY_Y` declares source/target roles and target member by string
- `openhcs/processing/backends/cellprofiler/intensity.py:1898`
  `MAX_INTENSITY_Z` declares source/target roles and target member by string

The relation is real, but the payload is not the right owner. The owning enum
members should carry marker payloads; `MeasureObjectIntensityModule` should
derive the relation from marker-tagged members in its own `MeasurementFeature`
enum, using axis suffix semantics from the registered axis strategy. The relation
declaration remains a `RuntimeMeasurementFeatureRelationDeclaration`, but it is
derived from the module-owned enum rather than hand-authored as role/string
payloads on each coordinate member.

### Central CP Dialect Mirrors

Current central CP dialect mirrors:

- `openhcs/interop/cellprofiler/measurement_dialect.py:26`
  `CELLPROFILER_MEASUREMENT_CATEGORY_PREFIXES`
- `openhcs/interop/cellprofiler/measurement_dialect.py:45`
  `CELLPROFILER_MEASUREMENT_FEATURE_PART_ALIASES`
- `openhcs/interop/cellprofiler/measurement_dialect.py:61`
  `CELLPROFILER_HARALICK_TEXTURE_FEATURE_PREFIXES`

These contain module-specific entries. Keep dialect-wide normalization in the
dialect file; move module-specific feature aliases and prefixes to the owning
`CellProfilerModule` declarations and aggregate them through
`CellProfilerModule.__registry__.values()`. Delete CP default feature-specific
numeric tolerance exceptions instead of moving them; strict parity disables
them and pipeline execution does not use runtime-equivalence policy.

### Static Row Field Lists

Current static CP row-field globals:

- `openhcs/processing/backends/cellprofiler/classification.py:675`
  `CLASSIFICATION_RESULT_FIELDS`
- `openhcs/processing/backends/cellprofiler/granularity.py:154`
  `GRANULARITY_FIELDS`
- `openhcs/processing/backends/cellprofiler/maxima.py:24`
  `MAXIMA_RESULT_FIELDS`
- `openhcs/processing/backends/cellprofiler/projection.py:22`
  `PROJECTION_STATS_FIELDS`
- `openhcs/processing/backends/cellprofiler/skeleton.py:19`
  `SKELETON_MEASUREMENT_FIELDS`
- `openhcs/processing/backends/cellprofiler/skeleton.py:20`
  `OBJECT_SKELETON_MEASUREMENT_FIELDS`

Each field list has an existing owner. Do not keep any passive field-list global:

- `CLASSIFICATION_RESULT_FIELDS` mirrors `ClassificationResult` and the existing
  `ClassifyObjectsSingleMeasurementModule.MeasurementStatField` /
  `MeasurementRows` declarations. Delete the global and derive CSV fields from
  `dataclasses.fields(ClassificationResult)`; CP measurement rows keep using the
  nested module declarations.
- `GRANULARITY_FIELDS` mirrors `GranularityMeasurement` for image rows and
  `ObjectGranularityMeasurement` for object rows. Delete the global and derive
  CSV fields from those dataclasses at the `csv_materializer(...)` call sites.
- `MAXIMA_RESULT_FIELDS` mirrors `MaximaResult`. Delete the global and derive CSV
  fields from `dataclasses.fields(MaximaResult)`.
- `PROJECTION_STATS_FIELDS` mirrors `ProjectionStats`. Delete the global and
  derive CSV fields from `dataclasses.fields(ProjectionStats)`.
- `SKELETON_MEASUREMENT_FIELDS` mirrors `SkeletonMeasurement`; derive CSV fields
  from `dataclasses.fields(SkeletonMeasurement)`.
- `OBJECT_SKELETON_MEASUREMENT_FIELDS` mirrors `ObjectSkeletonMeasurement`; derive
  CSV fields from `dataclasses.fields(ObjectSkeletonMeasurement)`.

The associated pattern is existing structured declaration, not a new registry:
for CP measurement materialization use nested
`ModuleOwnedResultMeasurementRows` / `FieldDerivedMeasurementFeatureModule`
declarations; for backend CSV materialization use the result dataclass as the
single source of field order.

## Existing Findings

### Core Runtime Semantics Owns CP Syntax

`openhcs/core/runtime_semantics.py` still owns CellProfiler-specific feature
syntax:

- `ObjectLocationMeasurementFeature`
- `ObjectZernikeDescriptorFeature`
- `indexed_object_intensity_zernike_feature_name`
- `IndexedObjectZernikeDescriptor`
- `ObjectIntensityZernikeMeasurementRows`

Core can own generic measurement axes, identity, row/value contracts, and
generic object-location semantics. CP feature spelling belongs to CP module
declarations or CP support owned by those declarations.

### Core Runtime Equivalence Knows Module Semantics

`openhcs/core/runtime_equivalence.py` still contains CP/module-specific
equivalence logic:

- Zernike phase vs magnitude tolerance.
- Shape descriptor gating feature lists.
- `ShapeZernikeDescriptorFeatureSemantics`.
- `ObjectZernikeDescriptorStabilityContract`.
- `_zernike_descriptor_values_equivalent`.

Replace these with generic registered feature-equivalence consumers. The owning
declarations live on modules or authority mixins inherited by modules:

- `MeasureObjectSizeShapeModule` for broad shape descriptor membership and
  shape-gating semantics.
- `openhcs/processing/backends/cellprofiler/zernike.py` for all Zernike
  descriptor parsing, rendering, stability, and phase/magnitude behavior.
- `MeasureObjectIntensityDistributionModule` for declaring which emitted
  feature members use the Zernike declarations owned by `zernike.py`.
- `MeasureObjectIntensityModule` for intensity feature families.

AST audit targets before implementation:

- `openhcs/core/equivalence/measurement_features.py` defines the role strategy
  root and role-strategy leaves that must collapse into registered semantic
  profiles and marker payloads.
- `openhcs/processing/backends/cellprofiler/intensity.py` defines
  `MeasureObjectIntensityFeatureRoleAuthority`; move its semantics onto
  `MeasureObjectIntensityModule` marker payloads and relation hooks.
- `openhcs/processing/backends/cellprofiler/zernike.py` defines
  `ObjectZernikeDescriptorFeatureRoleAuthority`; replace it with registered
  descriptor declarations in the same file.
- `openhcs/processing/backends/cellprofiler/shape.py` imports
  `IndexedObjectZernikeDescriptor`, `ObjectZernikeDescriptorFeature`, and
  `ObjectMeasurementFeatureRole`; delete those imports by moving shape-Zernike
  parser/render to `zernike.py` descriptor declarations, moving
  stability/equivalence to the equivalence-side descriptor mixin implemented by
  the same `zernike.py` declaration classes, and moving broad shape membership
  to `MeasurementFeature` marker payloads.

### Central Measurement Dialect Hardcodes Module Semantics

`openhcs/interop/cellprofiler/measurement_dialect.py` hardcodes
module-specific prefixes and aliases.

Move hardcoded feature declarations to owning modules:

- `frac_at_d`, `mean_frac`, `radial_cv`: `MeasureObjectIntensityDistributionModule`.
- Haralick texture prefixes: `MeasureTextureModule`.
- `weighted_variance`, `final_threshold`, `orig_threshold`: threshold owner.
- classify bin suffixes: `ClassifyObjectsSingleMeasurementModule`.

### Current Object Vector Runtime Knows AreaShape

`openhcs/interop/cellprofiler/runtime/object_measurement_vectors.py` contains
AreaShape-specific vector derivation and status names. Replace it with a generic
consumer of a current-object-vector authority inherited by
`MeasureObjectSizeShapeModule`.

### Row Schema And Field Strings Remain Centralized

Central row-schema/string-field logic remains in:

- `openhcs/interop/cellprofiler/runtime/measurement_materialization.py`
- `openhcs/interop/cellprofiler/runtime/processing_contracts.py`
- `openhcs/interop/cellprofiler/runtime/object_measurement_row_policies.py`

These must query nominal row-axis/value declarations or module-owned row
declarations. Delete local field-name sets as each owning declaration is wired.

## Target API

Keep CP module registry access explicit at the CP interop boundary. A caller in
`openhcs/interop/cellprofiler` must iterate
`CellProfilerModule.__registry__.values()` directly. Delete helper functions
that rename that registry lookup.

Add the authority validation and MRO declaration pieces that encode real
invariants in `openhcs/interop/cellprofiler/module_declarations.py`.

```python
AuthorityT = TypeVar("AuthorityT", bound="CellProfilerModuleAuthority")


class CellProfilerModuleAuthority(ABC):
    """Nominal semantic capability inherited by CP module declarations."""


class CellProfilerModule(...):
    @classmethod
    def require_authority_type(
        cls,
        authority_type: type[AuthorityT],
    ) -> type[AuthorityT]:
        if not isinstance(authority_type, type) or not issubclass(
            authority_type,
            CellProfilerModuleAuthority,
        ):
            raise TypeError(
                f"{cls.__name__} authority must inherit CellProfilerModuleAuthority."
            )
        return authority_type

    @classmethod
    def declared_authority_types(
        cls,
        authority_root: type[AuthorityT],
    ) -> tuple[type[AuthorityT], ...]:
        cls.require_authority_type(authority_root)
        matching_authority_types = tuple(
            dict.fromkeys(
                candidate_type
                for candidate_type in cls.__mro__
                if candidate_type is not cls
                and candidate_type is not authority_root
                and candidate_type is not CellProfilerModuleAuthority
                and issubclass(candidate_type, authority_root)
            )
        )
        return tuple(
            candidate_type
            for candidate_type in matching_authority_types
            if not any(
                other_type is not candidate_type
                and issubclass(other_type, candidate_type)
                for other_type in matching_authority_types
            )
        )
```

CP interop aggregators and profile classes use exact authority classes or root
aggregations by iterating `CellProfilerModule.__registry__.values()` directly.
Do not add a CP default numeric-tolerance aggregator; strict parity and pipeline
execution do not consume it.

Core code iterates its own semantic profile registry. CP-specific profile
subclasses contain the CP module registry loops internally. The core strategy
root has no CellProfiler discovery package, so CP interop imports its concrete
profile module before returning a CP runtime equivalence policy. Put those
profiles in `openhcs/interop/cellprofiler/measurement_semantic_profiles.py` and
import that module from `openhcs/interop/cellprofiler/measurement_dialect.py`.

## Core Semantic Profile Pseudocode

Core owns the registered semantic-profile family as a context-keyed strategy
root. Match the existing `MostDerivedContextStrategyMixin` pattern used by
runtime equivalence strategy families. The iterable remains the root registry,
and dispatch goes through `for_context(...)`.

```python
@dataclass(frozen=True, slots=True)
class RuntimeMeasurementFeatureSemanticContext:
    key: RuntimeMeasurementFeatureKey
    policy: RuntimeEquivalencePolicy


class RuntimeMeasurementFeatureSemanticProfile(
    MostDerivedContextStrategyMixin[RuntimeMeasurementFeatureSemanticContext],
    ABC,
):
    """Registered semantic behavior for runtime measurement features."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.STRATEGY_KEY)
    strategy_key: ClassVar[str | None] = None

    @abstractmethod
    def matches(
        self,
        context: RuntimeMeasurementFeatureSemanticContext,
    ) -> bool:
        """Return whether this profile owns behavior for ``context.key``."""

    @classmethod
    def for_feature_key(
        cls,
        key: RuntimeMeasurementFeatureKey,
        policy: RuntimeEquivalencePolicy,
    ) -> "RuntimeMeasurementFeatureSemanticProfile":
        context = RuntimeMeasurementFeatureSemanticContext(key, policy)
        strategy = cls.for_context(
            context,
            required=False,
            error_subject="Runtime measurement feature semantic profile",
        )
        if strategy is None:
            return DefaultRuntimeMeasurementFeatureSemanticProfile()
        return strategy

    @abstractmethod
    def values_equivalent(
        self,
        key: RuntimeMeasurementFeatureKey,
        left: object,
        right: object,
        policy: RuntimeEquivalencePolicy,
    ) -> bool:
        """Return semantic value equivalence for this feature."""

    def row_identity_stable(
        self,
        key: RuntimeMeasurementFeatureKey,
        row_identity: RuntimeMeasurementRowIdentity,
        policy: RuntimeEquivalencePolicy,
    ) -> bool:
        return True

    def current_object_vector(
        self,
        key: RuntimeMeasurementFeatureKey,
        label_array: NDArray,
    ) -> NDArray | None:
        return None


class RuntimeMeasurementDescriptorSemantics(RuntimeMeasurementFeatureSemanticProfile):
    """Registered equivalence profile for indexed descriptor-like features."""

    @abstractmethod
    def descriptor_identity(
        self,
        key: RuntimeMeasurementFeatureKey,
        dialect: RuntimeMeasurementDialect,
    ) -> object:
        """Return an opaque descriptor identity owned by this profile."""
```

Descriptor parser/render declarations are not equivalence declarations.
`RuntimeMeasurementIndexedDescriptorDeclaration` lives in
`runtime_semantics.py` and stays pure. The equivalence-side contract lives in
`core/equivalence/measurement_features.py`:

```python
class RuntimeMeasurementIndexedDescriptorEquivalence(ABC):
    """Equivalence behavior for an indexed descriptor declaration."""

    @classmethod
    @abstractmethod
    def descriptor_values_equivalent(
        cls,
        feature: RuntimeMeasurementFeature,
        descriptor: object,
        key: RuntimeMeasurementFeatureKey,
        left: object,
        right: object,
        policy: RuntimeEquivalencePolicy,
    ) -> bool:
        """Return value equivalence for this descriptor."""

    @classmethod
    def descriptor_row_identity_stable(
        cls,
        feature: RuntimeMeasurementFeature,
        descriptor: object,
        key: RuntimeMeasurementFeatureKey,
        row_identity: RuntimeMeasurementRowIdentity,
        policy: RuntimeEquivalencePolicy,
    ) -> bool:
        del feature, descriptor, key, row_identity, policy
        return True
```

The default behavior is a registered profile class too:

```python
class DefaultRuntimeMeasurementFeatureSemanticProfile(
    RuntimeMeasurementFeatureSemanticProfile
):
    strategy_key = "default"

    def matches(self, context) -> bool:
        del context
        return False

    def values_equivalent(self, key, left, right, policy) -> bool:
        return generic_numeric_equivalent(key, left, right, policy)
```

Core equivalence uses the registered profile family:

```python
def feature_values_equivalent(
    key: RuntimeMeasurementFeatureKey,
    left: object,
    right: object,
    policy: RuntimeEquivalencePolicy,
) -> bool:
    profile = RuntimeMeasurementFeatureSemanticProfile.for_feature_key(
        key,
        policy,
    )
    return profile.values_equivalent(key, left, right, policy)
```

This is the key boundary: core sees generic `RuntimeMeasurement...`
interfaces. Move any `CellProfilerModule`, CP backend module, CP authority, or
CP feature-enum import out of core and into CP-specific profile subclasses.

## CP Marker And Descriptor Pseudocode

CP marker membership is not a registered semantic-profile subclass. The marker
type is the semantic authority. CP modules inherit marker classes and their
`RuntimeMeasurementFeature` members carry the same marker classes in the enum
payload. Consumers ask the dialect for actual marker classes, then compare with
`issubclass(actual_marker_type, requested_marker_type)`.

The CP dialect owns one marker-provider hook:

```python
class CellProfilerModule(...):
    @classmethod
    def measurement_feature_marker_types_for_key(
        cls,
        key: RuntimeMeasurementFeatureKey,
        dialect: RuntimeMeasurementDialect,
    ) -> tuple[type[RuntimeMeasurementFeatureSemanticMarker], ...]:
        return tuple(
            dict.fromkeys(
                authority_type
                for module_type in cls.__registry__.values()
                for authority_type in module_type.__mro__
                if authority_type is not CellProfilerMeasurementFeatureMarker
                and authority_type is not RuntimeMeasurementFeatureSemanticMarker
                and isinstance(authority_type, type)
                and issubclass(authority_type, CellProfilerMeasurementFeatureMarker)
                if authority_type.matches_feature_key(module_type, key, dialect)
            )
        )
```

`declared_authority_types(...)` remains useful for most-derived module
capabilities, but marker membership must walk the MRO directly because parent
markers are also true. For example, `ShapeZernikeFeatureAuthority` must imply
both `ShapeZernikeFeatureAuthority` and `ShapeDescriptorFeature`.

Core marker matching is additive:

```python
def object_measurement_feature_matches_marker(key, marker_type, policy) -> bool:
    declared_markers = (
        policy.measurement_dialect.measurement_feature_marker_types(key)
    )
    if any(issubclass(actual, marker_type) for actual in declared_markers):
        return True
    profile = RuntimeMeasurementFeatureSemanticProfile.for_feature_key(key, policy)
    return profile.matches_marker(key, marker_type, policy)
```

The profile path remains for generic core markers and descriptor behavior. Do
not add a `CellProfilerObjectFeatureSemanticProfile` that mirrors marker
membership.

CP interop contributes a descriptor profile only for indexed descriptor
behavior. It iterates `CellProfilerModule.__registry__.values()`, then
`module_type.measurement_feature_types()`, then each feature member's
`indexed_descriptor_declarations()` payload:

```python
class CellProfilerDescriptorSemanticProfile(RuntimeMeasurementDescriptorSemantics):
    """Registered profile backed by feature-member descriptor declarations."""

    strategy_key = "cellprofiler_descriptors"

    def matching_descriptor_declarations(self, key):
        return tuple(
            (declaration_type, descriptor)
            for module_type in CellProfilerModule.__registry__.values()
            for feature_type in module_type.measurement_feature_types()
            for feature in feature_type
            for declaration_type in feature.indexed_descriptor_declarations()
            for descriptor in (
                declaration_type.from_feature_name(key.feature_name),
            )
            if descriptor is not None
        )

    def matches(self, context) -> bool:
        return bool(self.matching_descriptor_declarations(context.key))

    def descriptor_identity(self, key, dialect) -> object:
        del dialect
        matches = self.matching_descriptor_declarations(key)
        if len(matches) != 1:
            raise ValueError(...)
        _declaration_type, descriptor = matches[0]
        return descriptor

    def values_equivalent(self, key, left, right, policy) -> bool:
        matches = self.matching_descriptor_declarations(key)
        if len(matches) != 1:
            raise ValueError(...)
        declaration_type, descriptor = matches[0]
        if not issubclass(
            declaration_type,
            RuntimeMeasurementIndexedDescriptorEquivalence,
        ):
            raise TypeError(...)
        return declaration_type.descriptor_values_equivalent(
            descriptor,
            key,
            left,
            right,
            policy,
        )
```

Descriptor declarations are load-bearing only for parser/render/equivalence of
indexed descriptors. Marker classes remain the load-bearing semantic tokens for
ordinary feature membership and gating.

## Pattern Review

The current pseudocode now matches the strongest local OpenHCS patterns:

- Semantic behavior dispatch uses
  `RuntimeMeasurementFeatureSemanticProfile.for_context(...)` through
  `MostDerivedContextStrategyMixin`.
- Iterating CP modules uses the existing `CellProfilerModule.__registry__.values()`.
- CP module feature ownership stays on nested `RuntimeMeasurementFeature` enums,
  module-owned attributes, or module authority bases.
- MRO discovery follows the shape of
  `CellProfilerModule.measurement_feature_types()` and
  `ArtifactContractModule.declared_artifact_capabilities()`.
- CP default feature-specific numeric tolerances are deleted, not moved.
- Row schemas use `ModuleOwnedResultMeasurementRows` and nested
  `MeasurementStatField` / `MeasurementFeatureTemplate`.

Before implementing any loop from this plan, re-run the review rule: if there is
an iteration, identify the owning `__registry__`. Convert loops over mirrored
tuples, dicts, sets, and sidecar providers into loops over a registry or over an
existing owner declaration inside a registry item.

## Authority Roots

Add authority roots as small generic hook surfaces. Make every authority root
carry behavior through abstract hooks or generic methods over module-owned
payloads. Convert existing role enums, `role = ...`, string keys, priority
numbers, and sidecar registries into inherited authority types or enum-member
payloads.

### `CellProfilerMeasurementFeatureMarker`

Owns object feature-family matching as a generic operation over module-owned
feature declarations. The source of truth for a feature family is the nested
`RuntimeMeasurementFeature` member payload. Add semantic markers to
`RuntimeMeasurementFeature` using the same payload style as existing relations.

Do not create a second authority class that points at a marker through
`feature_marker = ...`. The marker class is the authority. CP modules inherit
the same marker class that their `MeasurementFeature` enum members carry, so
module MRO identity and enum-member payload identity cannot diverge.

```python
_FeatureT = TypeVar(
    "_FeatureT",
    bound=RuntimeMeasurementFeature,
    covariant=True,
)


class AxisFeatureProjectionStrategy(
    EnumKeyedStrategyMixin[_FeatureT], ABC
):
    """Registry-backed strategy root for one semantic axis feature."""

    __enum_member_attr__ = "axis_feature"
    axis_feature: ClassVar[_FeatureT]


class ObjectLocationCoordinateProjectionStrategy(
    AxisFeatureProjectionStrategy[ObjectCoreMeasurementFeature],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Project dense-label coordinates for one nominal object-location feature."""

    axis_feature: ClassVar[ObjectCoreMeasurementFeature]

    @abstractmethod
    def coordinate_values(
        self, axis_centers: Sequence[Any], counts: Any
    ) -> ObjectLocationCoordinateValues:
        """Return dense label-indexed coordinate values for this feature."""


class RuntimeMeasurementFeatureSemanticMarker(ABC):
    """Nominal marker carried by a RuntimeMeasurementFeature member."""

    family_qualifier: ClassVar[str | None] = None

    @classmethod
    @final
    def matches_feature(cls, feature: "RuntimeMeasurementFeature") -> bool:
        return any(
            issubclass(marker_type, cls)
            for marker_type in feature.semantic_markers
        )

    @classmethod
    def qualified_family(cls, feature: "RuntimeMeasurementFeature") -> str:
        qualifier = cls.family_qualifier
        if qualifier is None:
            raise ValueError(f"{cls.__name__} does not declare family_qualifier.")
        return normalize_runtime_identifier(f"{qualifier}_{feature.value}")


class AxisSuffixedFeatureMarker(RuntimeMeasurementFeatureSemanticMarker, ABC):
    """Marker whose feature family denotes an axis-suffixed source."""

    @classmethod
    @abstractmethod
    def axis_strategy_type(
        cls,
    ) -> type[AxisFeatureProjectionStrategy[RuntimeMeasurementFeature]]:
        """Return the registry root that declares this marker's axis features."""

    @classmethod
    @final
    def axis_features(cls) -> tuple[RuntimeMeasurementFeature, ...]:
        return tuple(
            strategy_type.axis_feature
            for strategy_type in cls.axis_strategy_type().registered_strategy_types()
        )

    @classmethod
    @final
    def axis_tokens(cls) -> frozenset[str]:
        return frozenset(
            feature.feature_family().rsplit("_", 1)[-1]
            for feature in cls.axis_features()
        )

    @classmethod
    @final
    def source_stem(cls, feature: RuntimeMeasurementFeature) -> str:
        family = feature.feature_family()
        stem, separator, axis_token = family.rpartition("_")
        if not separator or axis_token not in cls.axis_tokens():
            raise ValueError(
                f"{feature!r} must end in one of {tuple(sorted(cls.axis_tokens()))!r} "
                f"for {cls.__name__}."
            )
        return stem


class ObjectCountFeature(RuntimeMeasurementFeatureSemanticMarker):
    family_qualifier = "count"


class ObjectIdentifierFeature(RuntimeMeasurementFeatureSemanticMarker):
    family_qualifier = "identifier"


class ObjectLocationFeature(AxisSuffixedFeatureMarker):
    family_qualifier = "location"

    @classmethod
    def axis_strategy_type(cls) -> type[ObjectLocationCoordinateProjectionStrategy]:
        return ObjectLocationCoordinateProjectionStrategy


class ObjectCalculatedFeature(RuntimeMeasurementFeatureSemanticMarker):
    family_qualifier = "calculated"


class RuntimeMeasurementIndexedDescriptorDeclaration(
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered parser/renderer declaration for indexed feature names."""

    __registry_key__ = "descriptor_key"
    __skip_if_no_key__ = True
    descriptor_key: ClassVar[str | None] = None

    @classmethod
    def require_registered(
        cls,
        declaration_type: type["RuntimeMeasurementIndexedDescriptorDeclaration"],
    ) -> type["RuntimeMeasurementIndexedDescriptorDeclaration"]:
        if not isinstance(declaration_type, type) or not issubclass(
            declaration_type,
            RuntimeMeasurementIndexedDescriptorDeclaration,
        ):
            raise TypeError(
                "Indexed descriptor declaration must inherit "
                "RuntimeMeasurementIndexedDescriptorDeclaration."
            )
        if declaration_type not in cls.__registry__.values():
            raise TypeError(
                f"{declaration_type.__name__} is not registered in {cls.__name__}."
            )
        return declaration_type

    @classmethod
    @abstractmethod
    def from_feature_name(
        cls,
        feature: "RuntimeMeasurementFeature",
        feature_name: str,
    ) -> object | None: ...

    @classmethod
    @abstractmethod
    def feature_name(
        cls,
        feature: "RuntimeMeasurementFeature",
        descriptor: object,
    ) -> str: ...


class RuntimeMeasurementFeature(str, Enum):
    def __new__(
        cls,
        value: str,
        relations: Iterable[RuntimeMeasurementFeatureRelation] = (),
        semantic_markers: Iterable[type[RuntimeMeasurementFeatureSemanticMarker]] = (),
        indexed_descriptor_declarations: Iterable[
            type[RuntimeMeasurementIndexedDescriptorDeclaration]
        ] = (),
    ):
        descriptor_declarations = tuple(
            RuntimeMeasurementIndexedDescriptorDeclaration.require_registered(
                declaration_type
            )
            for declaration_type in indexed_descriptor_declarations
        )
        member = str_enum_member_with_payload(
            cls,
            value,
            payload_attribute="_relations",
            payload=tuple(relations),
        )
        member.__dict__["_semantic_markers"] = tuple(semantic_markers)
        member.__dict__["_indexed_descriptor_declarations"] = (
            descriptor_declarations
        )
        return member

    semantic_markers = AliasProperty[
        tuple[type[RuntimeMeasurementFeatureSemanticMarker], ...]
    ]("_semantic_markers")

    _indexed_descriptor_declaration_types = AliasProperty[
        tuple[type[RuntimeMeasurementIndexedDescriptorDeclaration], ...]
    ]("_indexed_descriptor_declarations")

    def indexed_descriptor_declarations(
        self,
    ) -> tuple[type[RuntimeMeasurementIndexedDescriptorDeclaration], ...]:
        return self._indexed_descriptor_declaration_types


class CellProfilerMeasurementFeatureMarker(
    RuntimeMeasurementFeatureSemanticMarker,
    CellProfilerModuleAuthority,
    ABC,
):
    @classmethod
    def feature_members(
        cls,
        module_type: type[CellProfilerModule],
    ) -> tuple[RuntimeMeasurementFeature, ...]:
        return tuple(
            feature
            for feature_type in module_type.measurement_feature_types()
            for feature in feature_type
            if any(
                issubclass(marker_type, cls)
                for marker_type in feature.semantic_markers
            )
        )

    @classmethod
    def feature_families(
        cls,
        module_type: type[CellProfilerModule],
    ) -> tuple[str, ...]:
        return tuple(
            feature.feature_family()
            for feature in cls.feature_members(module_type)
        )

    @classmethod
    def matches_feature_key(
        cls,
        module_type: type[CellProfilerModule],
        key: RuntimeMeasurementFeatureKey,
        dialect: RuntimeMeasurementDialect,
    ) -> bool:
        if key.subject.scope is not MeasurementScope.OBJECT:
            return False
        if key.statistic != MeasurementStatistic.VALUE.value:
            return False
        return any(
            key.feature_name == family or key.feature_name.startswith(f"{family}_")
            for family in cls.feature_families(module_type)
        )

    @classmethod
    def values_equivalent(
        cls,
        module_type: type[CellProfilerModule],
        key: RuntimeMeasurementFeatureKey,
        left: object,
        right: object,
        policy: RuntimeEquivalencePolicy,
    ) -> bool:
        del module_type
        return generic_numeric_equivalent(key, left, right, policy)
```

Concrete semantic tokens are classes:

```python
class ShapeDescriptorFeature(CellProfilerMeasurementFeatureMarker):
    family_qualifier = "shape"


class IntensityFeature(CellProfilerMeasurementFeatureMarker):
    family_qualifier = "intensity"


class MeasuredObjectAnchorFeature(CellProfilerMeasurementFeatureMarker):
    family_qualifier = "object"
```

Indexed descriptor behavior is declared on the feature member and implemented
by the real semantic owner. For Zernike, that owner is
`openhcs/processing/backends/cellprofiler/zernike.py`, because that module
already owns Zernike kernels, intensity-Zernike row materialization, Zernike
feature-name rendering, and the existing Zernike role shim.

The generic parser/render interface is small and lives with runtime measurement
feature declarations. Concrete Zernike declarations live in `zernike.py`. Shape
and intensity-distribution modules do not implement Zernike parser/render or
equivalence/stability switches; their feature members only point at the
registered Zernike declaration types.

```python
class MeasureObjectSizeShapeModule(
    ...,
    MeasuredObjectAnchorFeature,
    ShapeDescriptorFeature,
    CurrentObjectFeatureVectorAuthority,
):
    class MeasurementFeature(RuntimeMeasurementFeature):
        AREA = ("Area", (), (ShapeDescriptorFeature,))
        PERIMETER = ("Perimeter", (), (ShapeDescriptorFeature,))
        ZERNIKE = (
            "Zernike",
            (),
            (ShapeDescriptorFeature,),
            (ShapeObjectZernikeDescriptorDeclaration,),
        )
        OBJECT_NUMBER = ("ObjectNumber", (), (MeasuredObjectAnchorFeature,))


class ShapeObjectZernikeDescriptorDeclaration(
    RuntimeMeasurementIndexedDescriptorDeclaration,
    RuntimeMeasurementIndexedDescriptorEquivalence,
):
    """Shape-Zernike descriptor declaration owned by cellprofiler/zernike.py."""

    descriptor_key = "cellprofiler_shape_zernike"

    @classmethod
    def from_feature_name(
        cls,
        feature: RuntimeMeasurementFeature,
        feature_name: str,
    ) -> ShapeZernikeDescriptor | None:
        return ShapeZernikeDescriptor.from_feature_name(feature_name)

    @classmethod
    def feature_name(
        cls,
        feature: RuntimeMeasurementFeature,
        descriptor: ShapeZernikeDescriptor,
    ) -> str:
        return feature.indexed_name(descriptor.degree, descriptor.repetition)

    @classmethod
    def descriptor_values_equivalent(
        cls,
        feature: RuntimeMeasurementFeature,
        descriptor: ShapeZernikeDescriptor,
        key,
        left,
        right,
        policy,
    ) -> bool:
        del feature, descriptor
        return SparseNumericCounterToleranceProfile.SHAPE_DESCRIPTOR.equivalent(
            left,
            right,
            policy,
        )
```

This extends the existing `RuntimeMeasurementFeature` payload pattern used for
relations. It keeps feature-role identity on the enum member and keeps the
authority root as a generic query over `module_type.measurement_feature_types()`.
Descriptor parsing is a registered declaration consumed by a feature-member
payload, not module-level dispatch. Broad semantics remain broad markers such as
`ShapeDescriptorFeature` and `IntensityFeature`. Zernike-specific
parsing/rendering/stability belongs to concrete registered descriptor
declarations in `cellprofiler/zernike.py`. The registered
`CellProfilerDescriptorSemanticProfile` is the runtime strategy that iterates
`CellProfilerModule.__registry__.values()`, then `module_type.measurement_feature_types()`,
then each feature member's `indexed_descriptor_declarations()`.

### Numeric Tolerance Declarations

Do not keep CP default feature-specific numeric tolerance exceptions. The strict
benchmark policy explicitly passes `feature_numeric_tolerances=()`, and pipeline
execution does not consume runtime-equivalence policy. Keep the generic
`RuntimeMeasurementFeatureNumericTolerance` type only as an explicit policy knob
for direct callers/tests; do not wire a CP module-owned default exception list.

### `CurrentObjectFeatureVectorAuthority`

Owns current-object vector derivation.

```python
class CurrentObjectFeatureVectorAuthority(CellProfilerModuleAuthority):
    @classmethod
    @abstractmethod
    def current_object_feature_vector(
        cls,
        module_type: type[CellProfilerModule],
        feature_name: str,
        label_array: NDArray,
    ) -> NDArray | None: ...
```

### Dialect Declarations

Use declarations already on `CellProfilerModule` for dialect aliases and
prefixes. The existing owner pattern is visible in
`module_declarations.py`:

```python
def alternative_measurement_feature_part_aliases(cls):
    for module_type in cls.__registry__.values():
        for source, alternatives in module_type.measurement_feature_part_aliases.items():
            ...


```

The concrete edit is:

- move module-specific values out of `measurement_dialect.py`;
- put them on the owning `CellProfilerModule` subclass using the existing class
  attributes:
  `measurement_feature_part_aliases`,
  `directional_pair_feature_aliases`,
  `scale_qualified_measurement_feature_prefixes`,
  `pair_correlation_feature_name`,
  `pair_regression_slope_feature_name`,
  `undirected_pair_feature_names`,
  `threshold_sensitive_pair_feature_names`;
- add `haralick_texture_feature_prefixes: ClassVar[tuple[str, ...]] = ()` to
  `CellProfilerModule`;
- add `haralick_texture_feature_prefix_declarations()` beside the existing
  aggregation methods. It must iterate `CellProfilerModule.__registry__.values()`
  and collect `module_type.haralick_texture_feature_prefixes`;
- aggregate by iterating `CellProfilerModule.__registry__.values()` in
  `module_declarations.py`, matching the functions above.

Represent static dialect data as module class attributes. Represent polymorphic
dialect behavior as an inherited authority root with abstract hooks.

### Row Schema Declarations

Use the existing `ModuleOwnedResultMeasurementRows` and
`FieldDerivedMeasurementFeatureModule` patterns for CP measurement row schemas.
The established declaration shape is:

- nested `MeasurementStatField`;
- nested `MeasurementFeatureTemplate`;
- nested `MeasurementRows`;
- dataclass `MeasurementRecord` fields where present.

The concrete edit shape is:

```python
class OwningModule(CellProfilerModule):
    class MeasurementStatField(CellProfilerMeasurementStatField):
        ...

    class MeasurementFeatureTemplate(FormattingMeasurementFeatureTemplate):
        ...

    @dataclass(frozen=True, slots=True)
    class MeasurementRows(ModuleOwnedResultMeasurementRows):
        registry_key = "owning_module"

        @classmethod
        def for_request(cls, module_type, request):
            return cls(request.output_value, module_type=module_type, ...)

        def rows(self):
            stat_field = self.stat_field_type
            feature_template = self.feature_template_type
            ...
```

`ModuleOwnedResultMeasurementRows` discovers nested declarations by walking the
owning module MRO. `FieldDerivedMeasurementFeatureModule` derives feature rows
from dataclass fields and `MeasurementRowAxisField` annotations.

For backend CSV artifacts whose rows are concrete dataclasses rather than CP
measurement rows, the result dataclass is the field authority. Use:

```python
def dataclass_field_names(row_type: type[object]) -> tuple[str, ...]:
    return tuple(field.name for field in fields(row_type))
```

Then pass `fields=dataclass_field_names(ResultDataclass)` to
`csv_materializer(...)`. The implementation can inline the `fields(...)`
expression or add one private helper in the backend module when multiple
dataclasses are used. Delete local tuples such as
`CLASSIFICATION_RESULT_FIELDS`, `GRANULARITY_FIELDS`, and
`SKELETON_MEASUREMENT_FIELDS`; they are parallel copies of dataclass fields.

## File-By-File Changes

### `openhcs/interop/cellprofiler/module_declarations.py`

Add:

- `CellProfilerModuleAuthority`
- `CellProfilerMeasurementFeatureMarker`
- `CellProfilerModule.require_authority_type(...)`
- `CellProfilerModule.declared_authority_types(...)`

Change:

- Delete the CP numeric-tolerance aggregation point and module-owned tolerance
  calls. Keep only explicit `RuntimeEquivalencePolicy.feature_numeric_tolerances`
  support for callers that deliberately pass tolerances.
- `measurement_feature_relation_declarations()` remains the aggregation point
  for CP measurement-feature relations. Keep its existing enum-member
  `feature.relation_declarations()` collection, then have it also ask each
  registered module for derived relation declarations owned by that module's
  nested `RuntimeMeasurementFeature` enums.
- add a strict module hook for derived measurement-feature relations. The base
  implementation returns `()`. Mixins that derive relations from marker-tagged
  features must raise during relation collection when a source marker has zero
  targets, multiple targets, or a target whose marker/source-family contract is
  incomplete. The hook returns `RuntimeMeasurementFeatureRelationDeclaration`
  values, so the existing dialect provider and
  `RuntimeMeasurementFeatureRelationDeclarationCollection` remain the only
  runtime relation query surface.

Concrete reuse shape:

```python
class CellProfilerModule(...):
    @classmethod
    def derived_measurement_feature_relation_declarations(
        cls,
    ) -> tuple[RuntimeMeasurementFeatureRelationDeclaration, ...]:
        return ()

    @classmethod
    def measurement_feature_relation_declarations(
        cls,
    ) -> tuple[RuntimeMeasurementFeatureRelationDeclaration, ...]:
        return tuple(
            (
                relation
                for module_type in cls.__registry__.values()
                for feature_type in module_type.measurement_feature_types()
                for feature in feature_type
                for relation in feature.relation_declarations()
            )
        ) + tuple(
            relation
            for module_type in cls.__registry__.values()
            for relation in (
                module_type.derived_measurement_feature_relation_declarations()
            )
        )
```

`CELLPROFILER_MEASUREMENT_DIALECT` keeps the existing provider:

```python
measurement_feature_relation_provider=(
    CellProfilerModule.measurement_feature_relation_declarations
)
```

This is the reuse boundary: new relation behavior enters through
`RuntimeMeasurementFeatureRelationDeclaration` values collected by the existing
CellProfiler module registry provider. Do not add another relation registry,
provider, or lookup path.
- dialect feature aggregation to keep using module-owned class attributes and
  explicit `CellProfilerModule.__registry__.values()` loops. For each moved
  dialect value, use the existing class attribute listed in "Dialect
  Declarations"; add
  `haralick_texture_feature_prefixes: ClassVar[tuple[str, ...]] = ()` and
  `haralick_texture_feature_prefix_declarations()` for Haralick prefixes.

Keep:

- `measurement_feature_types()` as-is; it is already the correct MRO pattern.
- `source_qualified_measurement_feature_types()` as-is for current
  source-qualified feature enum discovery. Add concrete module overrides
  for source-qualified subsets.

### `openhcs/core/equivalence/measurement_features.py`

Add:

- `RuntimeMeasurementFeatureSemanticContext`
- `RuntimeMeasurementFeatureSemanticProfile`
- `RuntimeMeasurementDescriptorSemantics`
- `DefaultRuntimeMeasurementFeatureSemanticProfile`

Delete:

- `ObjectMeasurementFeatureRoleAuthority`
- `ParsedObjectMeasurementFeatureRoleAuthority`
- `ObjectMeasurementFeatureRoleStrategy`
- `ObjectMeasurementFeatureRole`
- role-specific strategy subclasses for CP-owned descriptor/intensity/shape
  semantics
- `feature_families_for_object_measurement_role(...)`
- `matches_object_measurement_feature_role(...)`

Replace with:

```python
@dataclass(frozen=True, slots=True)
class TieSensitiveLocationValueFeatureRelation(RuntimeMeasurementFeatureRelation):
    target_feature: RuntimeMeasurementFeature
    source_marker: type[RuntimeMeasurementFeatureSemanticMarker]
    target_marker: type[RuntimeMeasurementFeatureSemanticMarker]

    def __post_init__(self) -> None:
        if self.target_marker not in self.target_feature.semantic_markers:
            raise ValueError(
                f"{self.target_feature!r} must carry {self.target_marker.__name__}."
            )

    def source_family_names(
        self,
        source_feature: RuntimeMeasurementFeature,
    ) -> tuple[str, ...]:
        if self.source_marker not in source_feature.semantic_markers:
            raise ValueError(
                f"{source_feature!r} must carry {self.source_marker.__name__}."
            )
        return (
            source_feature.feature_family(),
            self.source_marker.qualified_family(source_feature),
        )

    def target_family_name(
        self,
        source_feature: RuntimeMeasurementFeature,
        source_family_name: str,
        feature_type: type[RuntimeMeasurementFeature],
    ) -> str | None:
        del feature_type
        normalized_source_family = normalize_runtime_identifier(source_family_name)
        if normalized_source_family == source_feature.feature_family():
            return self.target_feature.feature_family()
        if normalized_source_family == self.source_marker.qualified_family(source_feature):
            return self.target_marker.qualified_family(self.target_feature)
        return None


def object_measurement_feature_matches(
    key: RuntimeMeasurementFeatureKey,
    policy: RuntimeEquivalencePolicy,
) -> RuntimeMeasurementFeatureSemanticProfile:
    return RuntimeMeasurementFeatureSemanticProfile.for_feature_key(
        key,
        policy,
    )
```

Keep core semantics for generic count, identifier, and location features in core
files. CP module-specific feature-family membership enters through
`CellProfilerModule.measurement_feature_marker_types_for_key(...)`; do not mirror
marker membership in CP-specific `RuntimeMeasurementFeatureSemanticProfile`
subclasses.

### `openhcs/interop/cellprofiler/measurement_semantic_profiles.py`

Add the CP concrete descriptor profile:

- `CellProfilerDescriptorSemanticProfile`

This class is a registered subclass of the core
`RuntimeMeasurementFeatureSemanticProfile` family and contains the
`CellProfilerModule.__registry__.values()` scan for indexed descriptor
declarations. Broad object-feature matching uses the dialect marker-provider and
feature-member marker payloads, not a profile mirror. Import this module from
`openhcs/interop/cellprofiler/measurement_dialect.py` so CP profiles are
registered before a CP runtime equivalence policy reaches core equivalence.

### `openhcs/core/runtime_semantics.py`

Keep generic runtime concepts. Move CP-owned names out:

- add `RuntimeMeasurementFeatureSemanticMarker`;
- add `RuntimeMeasurementFeature.semantic_markers` payload support beside the
  existing `relations` payload;
- add `RuntimeMeasurementFeature.indexed_descriptor_declarations()` payload
  support beside the existing `relations` payload. The payload must validate
  each entry with
  `RuntimeMeasurementIndexedDescriptorDeclaration.require_registered(...)`;
- add core generic marker classes for count, identifier, location, and
  calculated object features;
- add the generic `RuntimeMeasurementIndexedDescriptorDeclaration` registered
  root. Concrete Zernike declarations do not live in core;
- remove `ObjectMeasurementFeatureRole`;
- keep the existing `RuntimeMeasurementFeatureRelation`,
  `RuntimeMeasurementFeatureRelationDeclaration`, and
  `RuntimeMeasurementFeatureRelationDeclarationCollection` relation authority
  path. Replace role-qualified relation payloads by deriving ordinary
  `RuntimeMeasurementFeatureRelationDeclaration` values from marker-tagged
  feature enum members inside the existing
  `CellProfilerModule.measurement_feature_relation_declarations()` provider.
- move `ObjectZernikeDescriptorFeature` to `cellprofiler/zernike.py`
  declaration support;
- move `indexed_object_intensity_zernike_feature_name(...)` to
  `cellprofiler/zernike.py` declaration support;
- move `IndexedObjectZernikeDescriptor` to `cellprofiler/zernike.py`
  declaration support;
- move `ObjectIntensityZernikeMeasurementRows` to `cellprofiler/zernike.py`, beside
  `ObjectIntensityZernikeMeasurementColumnarRows`.

Keep remaining classes that name generic axes, statistics, row values, or
subjects. Move classes that name CellProfiler feature spelling into CP-owned
module support.

### `openhcs/core/runtime_equivalence.py`

Replace CP-specific branches with registered semantic-profile queries:

- shape descriptor checks use a registered `RuntimeMeasurementFeatureSemanticProfile`;
- shape Zernike parsing/stability uses a registered
  `RuntimeMeasurementDescriptorSemantics`;
- intensity Zernike phase/magnitude tolerance uses the same registered
  descriptor profile surface;
- sparse tolerance profiles use `RuntimeEquivalencePolicy`'s existing
  `feature_numeric_tolerances_provider`.

Target shape:

```python
def feature_values_equivalent(key, left, right, policy) -> bool:
    profile = RuntimeMeasurementFeatureSemanticProfile.for_feature_key(
        key,
        policy,
    )
    return profile.values_equivalent(key, left, right, policy)
```

Remove imports of concrete CP backend modules and `CellProfilerModule` from this
file. Route shape, texture, granularity, threshold, and intensity semantics
through CP-specific profile subclasses registered on
`RuntimeMeasurementFeatureSemanticProfile`.

### `openhcs/interop/cellprofiler/measurement_dialect.py`

Shrink this file to generic dialect construction plus registry aggregation.

Move:

- module-specific category prefixes and aliases to existing module-owned class
  attributes on the owning `CellProfilerModule` subclass.
- `CELLPROFILER_HARALICK_TEXTURE_FEATURE_PREFIXES` values to
  `MeasureTextureModule.haralick_texture_feature_prefixes`. Add the
  `CellProfilerModule.haralick_texture_feature_prefix_declarations()`
  aggregation method and use it in dialect construction.

Target remaining shape:

```python
CELLPROFILER_MEASUREMENT_DIALECT = RuntimeMeasurementDialect(
    ...,
)
```

Do not add a CP feature-tolerance aggregation surface back to the dialect.

### `openhcs/interop/cellprofiler/runtime/object_measurement_vectors.py`

Delete `CurrentObjectFeatureVectorProvider` and provider subclasses. Replace
provider lookup with direct CP module registry lookup:

```python
for module_type in CellProfilerModule.__registry__.values():
    for authority_type in module_type.declared_authority_types(
        CurrentObjectFeatureVectorAuthority
    ):
        vector = authority_type.current_object_feature_vector(
            module_type,
            feature_name,
            label_array,
        )
```

Move AreaShape current-object vector logic onto `MeasureObjectSizeShapeModule`
through `CurrentObjectFeatureVectorAuthority`.

### `openhcs/processing/backends/cellprofiler/shape.py`

Make `MeasureObjectSizeShapeModule` inherit:

- `MeasuredObjectAnchorFeature`
- `ShapeDescriptorFeature`
- `CurrentObjectFeatureVectorAuthority`

Delete:

- `object_measurement_feature_roles(...)`
- `feature_families_for_object_measurement_role(...)`
- `matches_object_measurement_feature_role(...)`
- imports of `ObjectMeasurementFeatureRole` used for shape semantics
- `ObjectShapeCurrentFeatureVectorProvider` after its behavior moves to the
  module hook

Tag the owning `MeasurementFeature` enum members directly:

```python
class MeasurementFeature(RuntimeMeasurementFeature):
    AREA = ("Area", (), (ShapeDescriptorFeature,))
    PERIMETER = ("Perimeter", (), (ShapeDescriptorFeature,))
    ZERNIKE = (
        "Zernike",
        (),
        (ShapeDescriptorFeature,),
        (ShapeObjectZernikeDescriptorDeclaration,),
    )
    OBJECT_NUMBER = ("ObjectNumber", (), (MeasuredObjectAnchorFeature,))
```

Do not add Zernike parser/name hooks to `MeasureObjectSizeShapeModule`. The
module owns the emitted `MeasurementFeature.ZERNIKE` member and broad shape
membership; `ShapeObjectZernikeDescriptorDeclaration` lives in
`openhcs/processing/backends/cellprofiler/zernike.py` and owns Zernike
parser/render plus equivalence-side stability behavior for that member.

### `openhcs/processing/backends/cellprofiler/intensity.py`

Make `MeasureObjectIntensityModule` inherit:

- `IntensityFeature`

Put tie-sensitive max-location semantics on the module's `MeasurementFeature`
enum members by marker payload only. Location coordinate members carry
`ObjectLocationFeature`; value members carry `IntensityFeature`. The existing
`CellProfilerModule.measurement_feature_relation_declarations()` provider asks
the module hook for derived `RuntimeMeasurementFeatureRelationDeclaration`
values, and the hook derives source-to-value relations by scanning the owning
`MeasurementFeature` enum and matching the normalized coordinate stem. Do not
repeat feature strings or target member names in relation payloads.

Before adding the relation hook, rename the existing
`ObjectLocationCoordinateProjectionStrategy` enum-member attribute from
`coordinate_feature` to `axis_feature`, including its generated leaf specs.
Do not keep a compatibility alias. `ObjectLocationFeature` implements only the
typed `axis_strategy_type()` hook, and all axis enumeration flows through
final `AxisSuffixedFeatureMarker.axis_features()` over that strategy root's
`__registry__`. Module code must not name `CENTER_X`, `CENTER_Y`, `CENTER_Z`,
or any equivalent private axis tuple.

Use AST to append marker payloads to the existing `MeasurementFeature` members
in place. Preserve the current enum member names and values as the only source
of CP spelling. The migration identifies coordinate members by existing enum
members whose normalized feature family has an axis suffix and whose stem
matches one marker-tagged intensity value member in the same enum.

The hook contract is strict: each marked coordinate source must resolve exactly
one marker-tagged value target in the same feature enum. Missing or ambiguous
targets raise during relation declaration collection. The hook returns ordinary
`RuntimeMeasurementFeatureRelationDeclaration` values, so downstream dialect
and runtime-equivalence code continue to use the existing relation collection.

Concrete hook shape:

```python
@classmethod
def derived_measurement_feature_relation_declarations(
    cls,
) -> tuple[RuntimeMeasurementFeatureRelationDeclaration, ...]:
    feature_type = cls.MeasurementFeature
    features = tuple(feature_type)
    return tuple(
        RuntimeMeasurementFeatureRelationDeclaration(
            source_feature=source_feature,
            relation=TieSensitiveLocationValueFeatureRelation(
                target_feature=cls._single_target_feature_for_source(
                    source_feature,
                    features,
                    source_marker=ObjectLocationFeature,
                    target_marker=IntensityFeature,
                ),
                source_marker=ObjectLocationFeature,
                target_marker=IntensityFeature,
            ),
        )
        for source_feature in features
        if ObjectLocationFeature.matches_feature(source_feature)
    )
```

`MeasureObjectIntensityModule._single_target_feature_for_source(...)` operates
only on `features = tuple(cls.MeasurementFeature)`. It derives the target by
normalized feature-family stem supplied by the source marker and raises
`ValueError` if the source marker, axis suffix, target marker, or target
cardinality contract is not exactly satisfied. It never stores a separate map
of feature names and never names coordinate enum members; axis membership comes
from the marker's registry-backed `AxisSuffixedFeatureMarker`.

Concrete helper:

```python
@classmethod
def _single_target_feature_for_source(
    cls,
    source_feature: RuntimeMeasurementFeature,
    features: tuple[RuntimeMeasurementFeature, ...],
    *,
    source_marker: type[RuntimeMeasurementFeatureSemanticMarker],
    target_marker: type[RuntimeMeasurementFeatureSemanticMarker],
) -> RuntimeMeasurementFeature:
    if not source_marker.matches_feature(source_feature):
        raise ValueError(
            f"{cls.__name__}.{source_feature.name} must carry "
            f"{source_marker.__name__}."
        )
    if not issubclass(source_marker, AxisSuffixedFeatureMarker):
        raise TypeError(
            f"{source_marker.__name__} must inherit AxisSuffixedFeatureMarker "
            "to derive a value relation."
        )
    source_stem = source_marker.source_stem(source_feature)
    matches = tuple(
        target_feature
        for target_feature in features
        if target_marker.matches_feature(target_feature)
        and target_feature.feature_family() == source_stem
    )
    if len(matches) != 1:
        raise ValueError(
            f"{cls.__name__}.{source_feature.name} relation requires exactly "
            f"one {target_marker.__name__} target with family {source_stem!r}, "
            f"got {[feature.name for feature in matches]!r}."
        )
    return matches[0]
```

Delete:

- `MeasureObjectIntensityFeatureRoleAuthority`
- private sets/lists of max-intensity location fields
- role enum imports used to classify intensity features
- `source_role` / `target_role` relation payloads
- `target_member_name` payloads for relations whose target is derivable from
  marker-tagged enum members

### `openhcs/processing/backends/cellprofiler/intensity_distribution.py`

Make `MeasureObjectIntensityDistributionModule` inherit:

- `IntensityFeature`

Do not add intensity-distribution numeric tolerances to this module. The CP
default tolerance surface is removed.

Add intensity Zernike feature members only as declarations of emitted feature
families, and point them at the Zernike declarations owned by
`cellprofiler/zernike.py`:

```python
class MeasurementFeature(RuntimeMeasurementFeature):
    FRACTION_AT_DISTANCE = "FracAtD"
    MEAN_FRACTION = "MeanFrac"
    RADIAL_CV = "RadialCV"
    ZERNIKE_MAGNITUDE = (
        "ZernikeMagnitude",
        (),
        (IntensityFeature,),
        (IntensityMagnitudeObjectZernikeDescriptorDeclaration,),
    )
    ZERNIKE_PHASE = (
        "ZernikePhase",
        (),
        (IntensityFeature,),
        (IntensityPhaseObjectZernikeDescriptorDeclaration,),
    )
```

Do not add intensity-Zernike parser/render hooks to
`MeasureObjectIntensityDistributionModule`; those hooks live in `zernike.py`.

### `openhcs/processing/backends/cellprofiler/zernike.py`

Make this file the concrete owner of all Zernike descriptor semantics. Move
these CP-specific types and helpers here from core:

- `ObjectZernikeDescriptorFeature`
- `IndexedObjectZernikeDescriptor`
- `ObjectIntensityZernikeFeatureNameStrategy`
- `indexed_object_intensity_zernike_feature_name(...)`
- `ObjectIntensityZernikeMeasurementRows`
- Zernike descriptor stability contracts currently in
  `openhcs/core/runtime_equivalence.py`

Replace the existing `ObjectZernikeDescriptorFeatureRoleAuthority` role shim
with registered descriptor declarations that implement
`RuntimeMeasurementIndexedDescriptorDeclaration` and
`RuntimeMeasurementIndexedDescriptorEquivalence`:

- `ShapeObjectZernikeDescriptorDeclaration`
- `IntensityMagnitudeObjectZernikeDescriptorDeclaration`
- `IntensityPhaseObjectZernikeDescriptorDeclaration`

Those declarations own Zernike parse/render plus the equivalence-side
stability/equivalence hooks. The existing `ShapeZernikeBackendStrategy`, backend
leaf classes, moment functions, debug trace types, and
`ObjectIntensityZernikeMeasurementColumnarRows` stay in this file and become
consumers of the same declarations instead of calling core Zernike helpers.

### `openhcs/processing/backends/cellprofiler/texture.py`

Move `CELLPROFILER_HARALICK_TEXTURE_FEATURE_PREFIXES` values to
`MeasureTextureModule.haralick_texture_feature_prefixes`. Use existing
module-owned class attributes for tolerances. Add
`CellProfilerModule.haralick_texture_feature_prefix_declarations()` and have it
iterate `CellProfilerModule.__registry__.values()`.

### `openhcs/processing/backends/cellprofiler/thresholding.py`

Keep `ThresholdSettingsModule`, `ThresholdMeasurementRecordRowsMixin`, and
`ThresholdModule` for their distinct owned behavior.

Required result:

- threshold feature templates are derived from the field declaration class or a
  module hook;
- central string template maps for threshold fields are deleted;
- local row-field mirrors are deleted after the row declaration owns the fields;
- threshold tolerances are declared on the threshold module owner through the
  same single tolerance declaration path used by the rest of CP modules.

### `openhcs/processing/backends/cellprofiler/granularity.py`

Replace `GRANULARITY_FIELDS` with dataclass-derived field lists:

- image CSV rows use `dataclass_field_names(GranularityMeasurement)`;
- object CSV rows use `dataclass_field_names(ObjectGranularityMeasurement)`.

Change the two `csv_materializer(...)` call sites to consume those dataclass
field names and delete the free-standing `GRANULARITY_FIELDS` global and
`__all__` export. Do not create a second row declaration for these CSV artifacts;
the dataclasses already own the schema.

### `openhcs/processing/backends/cellprofiler/classification.py`

Delete `CLASSIFICATION_RESULT_FIELDS`. The CP measurement-row path already owns
semantic row projection through
`ClassifyObjectsSingleMeasurementModule.MeasurementStatField`,
`MeasurementFeatureTemplate`, and `MeasurementRows`. The backend CSV result path
uses `ClassificationResult`; update the three `csv_materializer(...)` call sites
to pass `dataclass_field_names(ClassificationResult)`. Remove the exported
global.

### `openhcs/processing/backends/cellprofiler/area_occupied.py`

Move `ImageAreaOccupiedMeasurementFeature` and related category/tolerance
semantics onto `MeasureImageAreaOccupiedBinaryModule`. Move generated enum specs
inside the module that owns the measurement.

### `openhcs/processing/backends/cellprofiler/tracking.py`

Move tracking measurement feature names, result row fields, and tolerance/name
authorities onto `TrackObjectsModule`. Replace string keys such as
`"final_age"` or `"trajectory_x"` with module-owned field declarations.

### `openhcs/processing/backends/cellprofiler/alignment.py`

Move `align_xshift` and `align_yshift` semantics and tolerances onto
`AlignModule`.

### `openhcs/processing/backends/cellprofiler/grid.py`

Move defined-grid spacing/location feature and tolerance declarations onto
`DefineGridManualModule`. Give `IdentifyObjectsInGridModule` its own authority
inheritance for separate output row semantics.

### `openhcs/processing/backends/cellprofiler/maxima.py`

Delete `MAXIMA_RESULT_FIELDS`. The result schema is `MaximaResult`; update both
`csv_materializer(...)` call sites to pass `dataclass_field_names(MaximaResult)`.
Remove the `extra_names=("MAXIMA_RESULT_FIELDS",)` export.

### `openhcs/processing/backends/cellprofiler/projection.py`

Delete `PROJECTION_STATS_FIELDS`. The result schema is `ProjectionStats`; update
the `csv_materializer(...)` call site to pass
`dataclass_field_names(ProjectionStats)`. Remove the exported global.

### `openhcs/processing/backends/cellprofiler/skeleton.py`

Delete `SKELETON_MEASUREMENT_FIELDS` and `OBJECT_SKELETON_MEASUREMENT_FIELDS`.
The image skeleton schema is `SkeletonMeasurement`; the object skeleton schema is
`ObjectSkeletonMeasurement`. Update the three `csv_materializer(...)` call sites
to pass dataclass-derived field names from the corresponding row type. Remove
both exported globals.

### `openhcs/interop/cellprofiler/runtime/measurement_materialization.py`
### `openhcs/interop/cellprofiler/runtime/processing_contracts.py`
### `openhcs/interop/cellprofiler/runtime/object_measurement_row_policies.py`

Replace local field-name sets with nominal row-axis/value declarations or
module-owned row declarations. These files must classify generic
measurement table shape. CP module-specific output schemas live on module
declarations.

Concrete target:

```python
rows_type = ModuleOwnedResultMeasurementRows.single_nested_declaration(
    module_type,
    ModuleOwnedResultMeasurementRows,
)
rows = rows_type.for_request(module_type, request)
```

For modules with an existing row/record dataclass, route materializers through
that declaration. For modules that need one, add a nested module-owned
`MeasurementRows` or `MeasurementRecord` declaration and route materializers
through it.

## Mechanical AST Rewrite Batches

AST executes the known migration after this ownership model is selected.
Each batch below transforms explicit node shapes and then runs validation.

### Batch 1: Install Discovery And Authority Roots

Direct edit or AST insertions:

- add the generic authority API to `module_declarations.py`;
- add authority roots for object feature matching, indexed descriptors, and
  current vectors;
- delete CP default numeric tolerances instead of routing them;
- route dialect aliases/prefixes through module class attributes and
  `CellProfilerModule.__registry__.values()` aggregators;
- route row schemas through `ModuleOwnedResultMeasurementRows` and
  `FieldDerivedMeasurementFeatureModule`;
- import roots at consumption and inheritance sites.

### Batch 2: Add Module Authority Bases

Use AST `ClassDef.bases` rewrites for the module classes listed above. Insert
the authority base on the owning module class. Move sidecar class behavior into
the owning module declaration during the same batch.

Pseudo-transform:

```python
authority_base_insertions = authority_base_insertions_from_file_plan(
    plan_targets=FILE_BY_FILE_MODULE_AUTHORITY_TARGETS,
    project_ast=project_ast,
)
for class_def in module_ast.body:
    if class_def.name in authority_base_insertions:
        class_def.bases = append_missing_bases(
            class_def.bases,
            authority_base_insertions[class_def.name],
        )
```

`FILE_BY_FILE_MODULE_AUTHORITY_TARGETS` is the concrete module list from this
plan's file-by-file section. The migration script verifies each target class
exists once, rewrites the class bases, and reports any unresolved class instead
of inventing a fallback. Delete the migration script or keep it outside runtime
code after the rewrite lands.

### Batch 3: Remove Role Mirrors

Use AST to delete methods and classes matching these names:

- `ObjectMeasurementFeatureRoleAuthority`
- `ParsedObjectMeasurementFeatureRoleAuthority`
- `ObjectMeasurementFeatureRoleStrategy`
- `object_measurement_feature_roles`
- `feature_families_for_object_measurement_role`
- `matches_object_measurement_feature_role`
- sidecar `*FeatureRoleAuthority` classes in CP backend modules

Use AST to replace CP interop call sites:

```python
ObjectMeasurementFeatureRole.SHAPE_DESCRIPTOR
    -> ShapeDescriptorFeature
ObjectMeasurementFeatureRole.ZERNIKE_DESCRIPTOR
    -> feature-member indexed descriptor declarations owned in cellprofiler/zernike.py
ObjectMeasurementFeatureRole.INTENSITY
    -> IntensityFeature
```

In core, replace role dispatch with
`RuntimeMeasurementFeatureSemanticProfile.for_feature_key(...)`. In CP interop,
replace role dispatch with inherited authority types and feature-member
descriptor declarations. When local context is insufficient, inspect the owning
CP module and wire broad membership to the module's `MeasurementFeature` marker
payload; wire indexed descriptor parser/render by attaching a registered
`RuntimeMeasurementIndexedDescriptorDeclaration` from `cellprofiler/zernike.py`
to that feature member, and require that declaration to implement
`RuntimeMeasurementIndexedDescriptorEquivalence` when descriptor-specific
stability or tolerance behavior is needed.

### Batch 4: Delete CP Default Numeric Tolerance Calls

Use AST/grep to remove CP default feature-specific numeric tolerance calls and
their aggregator/export:

- delete `CELLPROFILER_FEATURE_NUMERIC_TOLERANCES` from
  `measurement_dialect.py`;
- delete the default insertion into `cellprofiler_runtime_equivalence_policy`;
- delete `CellProfilerModule.measurement_feature_numeric_tolerance_declarations`;
- delete `measurement_feature_numeric_tolerances` and
  `declared_measurement_feature_numeric_tolerances` from CP module classes.

Keep the generic `RuntimeMeasurementFeatureNumericTolerance` policy type and
tests for explicit caller-supplied tolerances.

In the same batch, move `CELLPROFILER_HARALICK_TEXTURE_FEATURE_PREFIXES` values
to `MeasureTextureModule.haralick_texture_feature_prefixes`, add
`CellProfilerModule.haralick_texture_feature_prefix_declarations()`, and replace
the dialect reference with that registry aggregation.

### Batch 5: Move Zernike Descriptor Semantics

Use AST to move CP Zernike descriptor enum/parser/name construction out of
`runtime_semantics.py` and into `cellprofiler/zernike.py`:

- `MeasureObjectSizeShapeModule.MeasurementFeature.ZERNIKE` keeps the broad
  `ShapeDescriptorFeature` marker and adds
  `ShapeObjectZernikeDescriptorDeclaration`;
- `MeasureObjectIntensityDistributionModule.MeasurementFeature` declares the
  emitted intensity-Zernike families and attaches
  `IntensityMagnitudeObjectZernikeDescriptorDeclaration` and
  `IntensityPhaseObjectZernikeDescriptorDeclaration`;
- reusable numeric/kernel helpers stay in `zernike.py`.

After this batch, `core/runtime_equivalence.py` must use the registered
`RuntimeMeasurementFeatureSemanticProfile` family for descriptor
identity/stability/tolerance behavior. CP-specific profile classes implement
that by iterating
`CellProfilerModule.__registry__.values()`,
`module_type.measurement_feature_types()`, and each feature member's
`indexed_descriptor_declarations()`.

### Batch 6: Replace Current-Vector Sidecar Registry

Use AST to inline `ObjectShapeCurrentFeatureVectorProvider` behavior into the
`MeasureObjectSizeShapeModule` authority hook and delete
`CurrentObjectFeatureVectorProvider`. Replace provider-registry lookup loops
with explicit `CellProfilerModule.__registry__.values()` loops in
`object_measurement_vectors.py`.

### Batch 7: Replace Static Row Field Lists

Use AST to replace static CSV field-list globals with their existing row
dataclass owners:

- `CLASSIFICATION_RESULT_FIELDS` -> `ClassificationResult`
- `GRANULARITY_FIELDS` image rows -> `GranularityMeasurement`
- `GRANULARITY_FIELDS` object rows -> `ObjectGranularityMeasurement`
- `MAXIMA_RESULT_FIELDS` -> `MaximaResult`
- `PROJECTION_STATS_FIELDS` -> `ProjectionStats`
- `SKELETON_MEASUREMENT_FIELDS` -> `SkeletonMeasurement`
- `OBJECT_SKELETON_MEASUREMENT_FIELDS` -> `ObjectSkeletonMeasurement`

Use the nested `ModuleOwnedResultMeasurementRows` /
`FieldDerivedMeasurementFeatureModule` pattern only for CP measurement rows such
as threshold, tracking, alignment, and classification measurement-row emission.
Do not wrap backend CSV dataclasses in new row declaration classes; their
dataclass field order is the schema.

Pseudo-transform:

```python
for call in csv_materializer_calls(project_ast):
    fields_expr = call.keyword("fields")
    resolved_field_names = evaluate_literal_field_sequence(fields_expr)
    owner_type = exactly_one(
        dataclass_type
        for dataclass_type in dataclasses_declared_in(call.module)
        if dataclass_field_names(dataclass_type) == tuple(resolved_field_names)
    )
    replace(
        fields_expr,
        f"dataclass_field_names({owner_type.__name__})",
    )
    delete_schema_alias_globals_used_only_by(fields_expr)
```

The transform also removes each deleted global from `public_names_from_objects`
or `extra_names`. The script stops if a field-list global is referenced outside
`csv_materializer(...)` or `public_names_from_objects(...)`, because that means
the row schema has another consumer that needs an explicit owner decision.

### Batch 8: Remove Central Dialect Mirrors

Use AST to delete central CP tuples once their module-owned replacements exist:

- `CELLPROFILER_HARALICK_TEXTURE_FEATURE_PREFIXES`
- module-specific entries in `CELLPROFILER_MEASUREMENT_CATEGORY_PREFIXES`
- module-specific entries in `CELLPROFILER_MEASUREMENT_FEATURE_PART_ALIASES`

The dialect keeps generic dialect-wide entries plus registry aggregation.
CP feature-specific numeric tolerance exceptions were removed instead of moved,
because they do not affect strict parity or pipeline execution.

### Batch 9: Validation Queries

After each batch, these checks must be empty or explain generic identities owned
outside CP modules:

```text
rg -n "role = ObjectMeasurementFeatureRole|object_measurement_feature_roles|feature_families_for_object_measurement_role|matches_object_measurement_feature_role" openhcs
rg -n "ObjectMeasurementFeatureRole\\.(SHAPE_DESCRIPTOR|ZERNIKE_DESCRIPTOR|INTENSITY)" openhcs
rg -n "IndexedObjectZernikeDescriptor|ObjectZernikeDescriptorFeature|indexed_object_intensity_zernike_feature_name" openhcs/core openhcs/interop
rg -n "CELLPROFILER_HARALICK_TEXTURE_FEATURE_PREFIXES|CLASSIFICATION_RESULT_FIELDS|GRANULARITY_FIELDS" openhcs
rg -n "CurrentObjectFeatureVectorProvider|FeatureRoleAuthority" openhcs/processing/backends/cellprofiler openhcs/core/equivalence
```

Purity exit checks for `runtime_semantics.py`:

```text
rg -n "RuntimeEquivalencePolicy|RuntimeMeasurementIndexedDescriptorEquivalence|generic_numeric_equivalent|SparseNumericCounterToleranceProfile|CellProfilerModule|openhcs\\.processing\\.backends\\.cellprofiler|openhcs\\.interop\\.cellprofiler" openhcs/core/runtime_semantics.py
rg -n "values_equivalent|row_identity_stable|descriptor_values_equivalent|descriptor_row_identity_stable" openhcs/core/runtime_semantics.py
```

Both checks must be empty. If a name looks generic but fails this check, move it
to `openhcs/core/equivalence/measurement_features.py` or another equivalence
module and keep only the runtime identity/payload declaration in
`runtime_semantics.py`.

Then run an AST check for `ClassDef` nodes whose names end in `Authority` and
whose bodies contain a `role`, `authority_key`, or concrete module-name
payload. Rewrite each match into an authority class with abstract hooks consumed
through MRO or generic behavior over hooks declared by the owning module.

### Batch 10: Behavioral Verification

After structural cleanup:

- run targeted unit tests for runtime equivalence and CP module execution;
- run generated pipeline tests for CP imports;
- run the 30-pipeline parity benchmark with strict parity tolerances;
- treat any relaxed tolerance introduced during the migration as a failure until
  the owning module authority justifies it.
