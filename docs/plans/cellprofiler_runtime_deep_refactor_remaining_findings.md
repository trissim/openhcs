# CellProfiler Runtime Deep Refactor Plan: Remaining Risky Boundaries

Date: 2026-05-16

## Purpose

This plan captures the remaining non-trivial refactors after the 2026-05-16 advisor cleanup checkpoint.

Current checkpoint:

- Latest pushed checkpoint before this plan: `0dde31fe Remove callable output specs forwarding wrapper`.
- Targeted advisor scan on:
  - `openhcs/core/runtime_semantics.py`
  - `openhcs/core/runtime_values.py`
  - `openhcs/interop/cellprofiler/runtime/module_execution.py`
  - `openhcs/interop/cellprofiler/runtime/invocation.py`
- Current advisor count: 68 findings.
- Full unit suite at checkpoint: `1485 passed, 10 warnings`.

The easy cleanup is done. The remaining high-value work is risky because it touches runtime execution, measurement materialization, CellProfiler parity, and registry/metaclass policy families. That risk is acceptable only if each sequence is isolated, benchmarked, and pushed as a revertible checkpoint.

## Current Finding Profile

Advisor counts at the checkpoint:

- `trivial_forwarding_wrapper`: 24
- `semantic_inheritance_family_ssot`: 21
- `repeated_hardcoded_strings`: 6
- `metadata_only_class_family`: 3
- `repeated_builder_calls`: 3
- `prefixed_role_field_bundle`: 2
- `sparse_constructor_variant_family`: 2
- `dangling_private_method`: 2
- `repeated_private_methods`: 1
- `projection_builder_authority`: 1
- `class_role_quotient`: 1
- `fail_soft_effect_pipeline`: 1
- `under_amortized_infrastructure`: 1

Do not optimize for the raw count alone. Several findings are known analyzer noise or reflect intentionally public compatibility wrappers. The valuable work is the risky load-bearing refactor work below.

## Sequence 1: Split `CellProfilerModuleExecutor`

Primary file:

- `openhcs/interop/cellprofiler/runtime/module_execution.py`

Advisor signal:

- `class_role_quotient`
- `repeated_builder_calls`
- `projection_builder_authority`
- several `trivial_forwarding_wrapper` entries that exist because executor is acting as a broad facade.

Problem:

`CellProfilerModuleExecutor` still owns too many roles:

- invocation request construction
- primary image input resolution
- measurement image resolution
- per-object measurement execution
- per-image measurement execution
- runtime artifact input resolution
- output recording order
- global image-number projection setup
- runtime profiling event emission

The prior cleanup removed loose private helpers, but the executor is still a broad orchestration hub. The next step is not more helper renaming. It should be a staged split into cohesive collaborators.

Target shape:

- `CellProfilerInvocationRequestBuilder`
  - Owns `_image_request(...)`, `_invocation_request(...)`, primary image input resolution, source-image naming, runtime input kwargs, and special execution-mode options.
- `CellProfilerMeasurementImageResolver`
  - Owns independent/composed/label-aligned measurement image construction.
- `CellProfilerPerObjectMeasurementRunner`
  - Owns per-object measurement execution, row completion, measurement target scope, and per-object timing labels.
- `CellProfilerPerImageMeasurementRunner`
  - Owns per-image measurement execution, image-owned measurement rows, and per-image timing labels.
- `CellProfilerMeasurementProjectionMaterializer`
  - Extends the existing `CellProfilerMeasurementMaterializer` so global image-number projection and field projection are not constructed in multiple execution paths.
- `CellProfilerRuntimeProfiler`
  - Owns repeated `_log_module_profile(...)` builder calls and standard payload fields.
- `CellProfilerModuleExecutor`
  - Becomes a thin facade that composes those collaborators and preserves public `run(...)`, `prepare(...)`, and contract accessors.

Migration strategy:

1. Extract immutable request/result records first. No behavior change.
2. Move measurement image resolution out of executor with focused tests for image/object/measured-label combinations.
3. Move per-object runner. Preserve existing tests plus run CP generated-pipeline tests.
4. Move per-image runner. Preserve current image-number projection behavior.
5. Move profiler event construction last, because it is low semantic risk but high call-site volume.
6. Only after the split, reassess `primary_image_inputs`, `image_outputs`, and other forwarding wrappers. Some wrappers should disappear naturally when collaborators own the behavior.

Verification gate for each commit:

- `tests/unit/test_cellprofiler_module_execution.py`
- `tests/unit/test_cellprofiler_generated_pipeline_execution.py`
- full `tests/unit`
- targeted advisor scan
- official30 parity rerun after both runner extractions are complete.

## Sequence 2: Make Global Image-Number Projection a Materializer Authority

Primary file:

- `openhcs/interop/cellprofiler/runtime/module_execution.py`

Advisor signal:

- `projection_builder_authority`

Problem:

`CellProfilerGlobalImageNumberProjection(...)` is constructed in several paths:

- normal measurement output recording
- per-image measurement execution
- per-object/image-owned measurement paths

Earlier attempts to hide this behind a trivial executor forwarding method made advisor results worse. The right fix is not a wrapper; it is to make measurement materialization own projection setup completely.

Target shape:

- `CellProfilerMeasurementMaterializer.record(...)` receives a typed `CellProfilerMeasurementMaterializationRequest`.
- That request includes adapter, measurement name, rows, fields, object/source ownership, source payload, and a projection policy.
- All call sites construct request records, not projection builders.
- `CellProfilerGlobalImageNumberProjection` becomes an internal implementation detail of the materializer/projection policy.

Migration strategy:

1. Add `CellProfilerMeasurementMaterializationRequest`.
2. Change existing materializer call sites to pass the request record.
3. Move projection construction inside materializer.
4. Add tests asserting source image name, object name, fields, row projection, and image-number behavior.
5. Remove direct projection construction from executor paths.

Risk:

High. This touches CellProfiler parity and image-number semantics. Do not combine with executor runner extraction in the same commit.

Verification gate:

- focused module execution tests
- generated-pipeline execution tests
- official30 parity before claiming complete.

## Sequence 3: Normalize Policy Families with a Generated Leaf Registry Substrate

Primary files:

- `openhcs/interop/cellprofiler/runtime/module_execution.py`
- `openhcs/core/runtime_semantics.py`
- `openhcs/core/runtime_values.py`

Advisor signal:

- `semantic_inheritance_family_ssot`
- `repeated_hardcoded_strings`
- `metadata_only_class_family`

Problem:

Many policy families are already nominal and metaclass-registered, but advisor still sees repeated registry-key strings and metadata-only leaves. The issue is not that the families are invalid. The issue is that the repository lacks a consistent generated-leaf declaration substrate for all simple registered policy leaves.

Existing partial substrate:

- `CellProfilerModulePolicyLeafSpec`
- `GeneratedLeafClassSpec`
- several ad hoc generated leaf loops.

Target shape:

- One reusable generated leaf spec for registry-keyed CellProfiler module policies.
- One reusable generated leaf spec for enum-keyed runtime strategies.
- One declarative table per policy family.
- Hand-written leaf classes only when they contain behavior.
- Registry key literals remain in root classes unless metaclass support changes; do not extract them into constants if that breaks registry behavior.

Candidate families:

- `CellProfilerInvocationExecutionModePolicy`
- `CellProfilerMainFlowReplacementPolicy`
- `CellProfilerMeasurementRecordBuilder`
- `CellProfilerObjectInputPolicy`
- `CellProfilerObjectMeasurementRowPolicy`
- `CellProfilerPrimaryImageInputPolicy`
- `CellProfilerSpecialInputPolicy`
- `RuntimePlaneAxisProjectionStrategy`
- `ObjectLabelPlaneDomainStrategy`
- `ObjectLabelIdDomainStrategy`
- object-label runtime-value strategy families.

Migration strategy:

1. Pick one CellProfiler module policy family and convert metadata-only leaves to generated specs.
2. Run advisor and tests. If the count improves without hurting readability, repeat for the next family.
3. For enum/stat-field families such as `ClassifyObjectsMeasurementStatField`, `AlignMeasurementStatField`, and measurement feature enums, treat advisor findings skeptically. These are typed closed vocabularies, not boilerplate unless a real repeated behavior emerges.
4. Do not collapse behaviorful classes into data tables.

Risk:

Medium-high. Registry-family declarations are load-bearing. Mistakes can silently change policy lookup.

Verification gate:

- focused tests for the converted family
- registry lookup smoke tests
- generated-pipeline execution tests
- advisor scan confirming no new semantic alias or broken family findings.

## Sequence 4: Make Spatial Grid Axes Nominal

Primary file:

- `openhcs/core/runtime_values.py`

Advisor signal:

- `prefixed_role_field_bundle`

Problem:

`SpatialGridTopology` and `SpatialGrid` carry parallel X/Y fields:

- `x_spacing`, `y_spacing`
- `x_origin`, `y_origin`
- `x_locations`, `y_locations`

This is a real modeling smell. Grid axes are semantic objects, not prefixed scalar bundles.

Target shape:

- `SpatialAxisTopology`
  - `spacing`
  - `origin`
  - `locations`
- `SpatialGridTopology`
  - `x: SpatialAxisTopology`
  - `y: SpatialAxisTopology`
- `SpatialGrid`
  - either owns `topology: SpatialGridTopology`, or owns `x/y` axis topology directly plus existing compatibility properties.

Compatibility policy:

- Keep read-only compatibility properties for current field names during migration.
- Update constructors/factories first.
- Remove compatibility fields only after all call sites and serialized artifacts are migrated.

Migration strategy:

1. Add `SpatialAxisTopology` and constructor helpers.
2. Update internal runtime construction to populate axes.
3. Add compatibility properties for old field names.
4. Update direct internal call sites to use `grid.topology.x.spacing` style.
5. Run full unit tests.
6. Only then consider serialization changes.

Risk:

High. Spatial grids affect CP `DefineGrid`, object measurements, and downstream artifacts. Keep this separate from measurement projection refactors.

Verification gate:

- unit tests for spatial grid runtime values
- CP module execution tests touching `DefineGrid`/grid consumers
- official30 parity if any benchmark pipelines use grid modules.

## Sequence 5: Replace Sparse Constructor Variants with Named Request Records

Primary files:

- `openhcs/interop/cellprofiler/runtime/module_execution.py`
- `openhcs/core/runtime_semantics.py`

Advisor signal:

- `sparse_constructor_variant_family`

Problem:

Some constructors encode semantic variants through sparse classmethods:

- `CellProfilerObjectMeasurementVectorBinding.for_object_input(...)`
- `CellProfilerObjectMeasurementVectorBinding.for_object_name(...)`
- `ObjectLabelDomain.from_metadata(...)`
- `ObjectLabelDomain.plane_domain(...)`

These are not all bad. The risk is that variant meaning is split across constructor names instead of a single request/domain object.

Target shape:

- `ObjectMeasurementVectorBindingRequest`
  - object name
  - optional artifact spec
  - feature name
  - labels
  - image number
  - source strategy
  - object identity source enum: `DECLARED_INPUT`, `RUNTIME_NAME`, `EXPLICIT_LABELS`
- `ObjectLabelDomainRequest`
  - declared count
  - declared ids
  - plane domains
  - scope

Migration strategy:

1. Add request records.
2. Keep existing classmethods as thin compatibility constructors that build requests.
3. Move implementation to `from_request(...)`.
4. Update internal call sites to construct requests directly where that improves clarity.
5. Decide later whether public compatibility constructors should remain.

Risk:

Medium. Safer than executor split, but touches measurement vector binding and object-domain semantics.

Verification gate:

- object measurement vector tests
- CalculateMath tests
- ClassifyObjects/DisplayDataOnImage special-input tests
- runtime semantics tests.

## Sequence 6: Fail Loud on Unsupported Current-Label Shape Features

Primary file:

- `openhcs/interop/cellprofiler/runtime/module_execution.py`

Advisor signal:

- `fail_soft_effect_pipeline`

Problem:

`CurrentObjectShapeFeatureVectorSourceStrategy.current_label_shape_vector(...)` returns `None` when:

- labels are not 2D
- the feature name is not an object-shape feature
- the computed feature array is absent

Returning `None` means the caller falls back to persisted runtime measurements. That is correct for runtime measurement lookup, but dangerous when the source strategy has explicitly selected `CURRENT_OBJECT_SHAPE_FEATURE`. If the selected source cannot compute the feature, the behavior should be explicit.

Target shape:

- Distinguish "not my source" from "my source failed".
- Add `CellProfilerCurrentShapeFeatureRequest`.
- Add a result type:
  - `ComputedMeasurementVector`
  - `FeatureNotCurrentShape`
  - `UnsupportedCurrentShapeFeature`
- Only the strategy dispatcher may ignore `FeatureNotCurrentShape`.
- Once `CURRENT_OBJECT_SHAPE_FEATURE` is selected, unsupported dimensionality or missing arrays should fail loudly unless there is a documented compatibility exception.

Migration strategy:

1. Add result objects and tests for each branch.
2. Preserve fallback only when the feature is not a shape feature.
3. Fail loudly for shape-feature requests that cannot be computed from current labels.
4. Run generated-pipeline tests and parity-sensitive CP tests.

Risk:

Medium-high. This may expose hidden parity dependencies where fallback was masking missing current-label feature support. That is exactly why it is worth doing.

Verification gate:

- focused tests for shape/current-label vector source
- CP module execution tests
- failing official30 cases, if any, triaged before broad benchmark rerun.

## Sequence 7: Decide What Is Analyzer Noise and Encode It

Primary files:

- advisor configuration, if available
- code comments only where unavoidable

Advisor signals likely to remain:

- `dangling_private_method` on `CellProfilerFunctionContractExecutor._execute_flexible`
- `dangling_private_method` on `_execute_volumetric_to_slice`
- some `trivial_forwarding_wrapper` on public compatibility or strategy methods
- some `semantic_inheritance_family_ssot` where metaclass registration is already correct
- repeated `__registry_key__` literals in registered strategy roots.
- repeated builder calls to existing authorities such as `image_payload_with_context(...)` or `ImageStackLayout.for_slices(...).stack(...)` when the proposed extraction would only create a forwarding shell.

Known reasoning:

- `_execute_flexible` and `_execute_volumetric_to_slice` are dynamic `ProcessingContract` dispatch targets. Renaming/deleting them would break enum-driven execution.
- Several trivial wrappers are required by ABC contracts or preserve public API.
- Registry root literals are recognized by `AutoRegisterMeta`; previous constant extraction attempts risk breaking registration.
- The aligned-payload cleanup validated that replacing repeated `ImageStackLayout.for_slices(...).stack(...)` calls with a `stack_slices(...)` classmethod made advisor output worse because it introduced an identity keyword-forwarding shell. Keep those calls direct until the extracted type carries real policy, state, or validation.
- Repeated calls to `image_payload_with_context(...)` are already routed through the runtime image payload builder. A local wrapper would be a compatibility helper, not a nominal fix.
- `NestedAlignedImageStackKwargResolutionStrategy.resolve(...)` remains a thin strategy method because the strategy is the dispatch owner for nested `AlignedImageStack` values. Moving the body elsewhere without changing resolver ownership just relocates the wrapper smell.

Target shape:

- Add a local advisor suppression/allowlist only if the project has an accepted mechanism.
- Otherwise keep a short `docs/plans/advisor_known_noise.md` section or append to this plan after validation.
- Do not refactor stable metaclass roots only to satisfy repeated-string findings.

Verification gate:

- Advisor count can remain nonzero if every remaining finding is classified as:
  - architectural work queued in this plan
  - public compatibility wrapper
  - dynamic dispatch target
  - registry/metaclass analyzer limitation.

## Sequence 8: Unify Source-Binding Assignment Authority

Primary files:

- `openhcs/core/source_bindings.py`
- `openhcs/core/pipeline_image_schema.py`
- `openhcs/core/source_bindings_view.py`
- `openhcs/interop/cellprofiler/source_schema.py`
- `openhcs/interop/cellprofiler/pipeline_generator.py`

Advisor signal:

- `existing_nominal_authority_reuse` on `NamedSourceBinding`
- `semantic_inheritance_family_ssot` on `SourceAssignmentBase`

Problem:

`SourceAssignmentBase` already owns the source-assignment identity contract:

- alias normalization
- selector type
- origin type
- artifact-kind participation semantics

`NamedSourceBinding` repeats the same core fields in `openhcs/core/source_bindings.py` because the modules are split around two historical concerns:

- `pipeline_image_schema.py` owns pipeline-level image/source assignments.
- `source_bindings.py` owns step-local runtime binding declarations.

That split now creates two authorities for the same source-binding identity. The correct fix is not a local inheritance tweak in `NamedSourceBinding`; that would likely introduce a circular import or preserve two parallel models. The load-bearing refactor is to move the shared assignment identity to a neutral core module, then make both pipeline schema assignments and step-local named bindings depend on that one authority.

Target shape:

- Create a neutral source-assignment contract module, for example `openhcs/core/source_assignment.py`, that owns:
  - `SourceSelector` or a dependency-safe selector protocol/record, if the selector can be moved without cycles.
  - `SourceBindingOrigin`.
  - `SourceAssignmentIdentity` / `SourceAssignmentBase` with alias, selector, origin normalization.
  - artifact-kind participation helpers.
- Make `NamedSourceBinding` inherit or wrap that shared identity instead of re-declaring `alias`, `selector`, and `origin` semantics.
- Make `ImageAssignment` and `SourceArtifactAssignment` use the same identity base.
- Preserve serialized dataclass field names and constructor compatibility for:
  - `NamedSourceBinding(alias=..., artifact_kind=..., selector=..., origin=..., required=...)`
  - existing `StepSourceBindingsConfig` pickle/state restoration.
- Keep CellProfiler import/generation logic source-binding-first. Do not convert source-bound images into artifact consumers.

Migration strategy:

1. Map import dependencies between `source_bindings.py`, `pipeline_image_schema.py`, and `source_bindings_view.py`.
2. Move only the dependency-neutral pieces first. If `SourceSelector` cannot move cleanly, introduce a neutral identity base that accepts the existing selector type through a nominal import direction.
3. Convert `NamedSourceBinding` to reuse the shared identity while preserving its public dataclass fields and `required` semantics.
4. Convert `SourceAssignmentBase` users to the same shared identity.
5. Delete duplicated alias/origin/selector validation after both sides route through the same authority.
6. Re-run source-binding GUI/model tests before touching CellProfiler generator code.

Verification gate:

- `tests/unit/test_source_bindings.py`
- `tests/unit/test_source_bindings_view.py`
- `tests/unit/test_cellprofiler_source_schema.py`
- `tests/unit/test_cellprofiler_generated_pipeline_execution.py`
- full `tests/unit`
- advisor scan on the source-binding files.

Risk:

Medium-high. This is a core model refactor under source-binding GUI, CP pipeline import, generated pipeline execution, and debug replay. It should be an isolated commit series, not mixed with runtime-equivalence or benchmark changes.

Stop condition:

- Stop and diagnose if any serialized config/pickle compatibility test fails.
- Do not add compatibility helper shims unless they are explicitly temporary and queued for deletion in the same sequence.

Status:

- First implementation slice completed after this plan was written:
  - `SourceAssignmentBase` moved to `openhcs/core/source_bindings.py`.
  - `NamedSourceBinding`, `ImageAssignment`, and `SourceArtifactAssignment` now share that authority.
  - `SourceAssignmentBase` is an `AutoRegisterMeta` family keyed by `assignment_kind`.
  - `CompiledSourceBindingPlan` now has a real default empty state.
  - Focused source-binding/CP generated-pipeline tests and full `tests/unit` passed before commits.

Remaining work:

- Decide whether the remaining `SourceRole` metadata-only family in `pipeline_image_schema.py` should become generated leaf declarations or stay as explicit registered leaves.
- Consolidate repeated validation loops only if the extracted validator owns real typed semantics; do not add private list-check helpers.

## Sequence 9: Split Runtime Equivalence Fact Extraction

Primary file:

- `openhcs/core/runtime_equivalence.py`

Advisor signal:

- Multiple `bare_function_method_family` findings:
  - table snapshot fact extraction around table/identity/static-wide measurements
  - measurement feature stability predicates
  - object measurement fact extraction by context
  - runtime-row cached schema/fact projection
- `latent_nominal_function_family` around cell/numeric measurement fact helpers.

Problem:

`runtime_equivalence.py` has accumulated several parallel function families that project facts from tables, rows, and measurement feature metadata. The advisor is not pointing to a cosmetic issue: these helpers encode several independent semantic domains in one large module:

- table snapshot classification
- measurement feature stability
- object/cell/identifier/count/location fact extraction
- runtime row schema caching
- long-form vs wide-form measurement fact projection

The current shape makes it hard to reason about parity failures because feature-domain semantics are distributed across similarly named private functions instead of owned by nominal projectors.

Target shape:

- `RuntimeTableSnapshotFactExtractor`
  - Owns table-level measurement/identity/static-wide fact extraction from immutable table snapshots.
- `MeasurementFeatureStabilityPolicy`
  - Owns object count, shape descriptor, and role-specific stability checks.
- `RuntimeMeasurementFactExtractor`
  - Owns object/cell/count/location/identifier measurement fact extraction from a typed context.
- `RuntimeMeasurementRowFactProjector`
  - Owns cached row schema, row-to-fact projection, long-form fact projection, and numeric-value projection.
- Keep `runtime_equivalence.py` as the public orchestration module only after these collaborators exist; do not split public API first.

Migration strategy:

1. Add dataclass request/context records for the current function parameter bundles without moving behavior.
2. Extract table snapshot fact extraction first; it is the smallest of the top findings and should have focused tests in `tests/unit/test_runtime_equivalence.py` or equivalent.
3. Extract row schema/fact projection second, preserving cache keys and `lru_cache` behavior.
4. Extract feature stability predicates third, because this can affect parity classification and needs focused fixtures.
5. Extract object/cell measurement fact families last, after row projection and stability semantics are centralized.
6. After each slice, run advisor on `runtime_equivalence.py` and new collaborators. Do not chase count if the remaining finding is queued to a later slice.

Verification gate:

- Existing runtime-equivalence tests.
- Measurement/table parity tests that cover CP outputs.
- Full `tests/unit`.
- If a slice changes parity classification rather than just moving code, rerun the affected official30 cached cases before continuing.

Risk:

High. Runtime equivalence is parity-critical. Each slice must be small, committed, and revertible. Do not combine this with CellProfiler executor splitting or benchmark graph work.

Stop condition:

- Stop on the first semantic test failure and inspect before stacking further extraction.
- If cache identity changes, add a regression test for the old key semantics before proceeding.

Status:

- Implemented and pushed:
  - `RuntimeTableSnapshotFactExtractor` owns exported table snapshot fact projection.
  - `RuntimeMeasurementRowFactProjector`, `RuntimeMeasurementRowSchemaProjector`, and `RuntimeMeasurementLongFormFactProjector` own cached runtime-row projection.
  - `MeasurementFeatureStabilityPolicy` owns object-count, shape-geometry, and role-gated feature stability checks.
  - `RuntimeObjectLabelMeasurementFactProjector` owns implicit object-label count/location facts.
  - `RuntimeRowMeasurementFactRecorder` owns row fact emission, row-merge deferral, aggregate input recording, and object-row domain emission.
  - `ArtifactKind.participates_in_axis_plane_identity` owns the measurement/relationship plane-identity policy.
- Focused runtime-equivalence tests and full `tests/unit` passed before each pushed checkpoint.

Remaining work:

- Split the now-nominal row projection and table snapshot projectors into dedicated modules so `runtime_equivalence.py` stops acting as the storage site for all private runtime-row cohorts.
- Recover the remaining latent feature-key/cell-value authority around `_cell_measurement_facts`, `_numeric_cell_measurement_values`, `_aggregate_mean_key`, and source-qualified aggregate feature naming.
- Decide whether `RuntimeMeasurementFactProjectionContract` should remain a static utility namespace or become a keyed projection family; do not keep `AutoRegisterMeta` on it unless it has a real stable key axis.
- Normalize `RuntimeObjectLocationRowMergeContract` into a registered family or remove the inheritance if the two leaves are better represented as explicit projection policies.

## Commit and Benchmark Policy

Each risky sequence must be independently revertible:

1. Make one cohesive structural change.
2. Run focused tests.
3. Run full `tests/unit`.
4. Run advisor.
5. Commit and push.
6. Run official30 parity only after runtime/planner behavior changes, not after pure docs or generated-leaf-only cleanup.

Minimum benchmark gates after behavior-affecting runtime changes:

- official30 parity remains green.
- 1-core execution remains above the 4x floor unless the change is intentionally only architectural and benchmark-neutral.
- If any pipeline regresses, preserve logs and stop to diagnose before stacking more refactors.

## Recommended Next Commit

Start with Sequence 2, not Sequence 1.

Reason:

`CellProfilerGlobalImageNumberProjection` construction is a narrow, high-value behavior boundary. It is risky enough to matter but smaller than the full `CellProfilerModuleExecutor` split. A typed materialization request will also make the later executor split easier because per-image/per-object runners can call the same materializer API.

Expected first commit:

- Add `CellProfilerMeasurementMaterializationRequest`.
- Convert `CellProfilerMeasurementMaterializer.record(...)` to accept the request.
- Update all call sites.
- Add or adjust tests around projected image-number rows.
- Do not yet split the executor.
