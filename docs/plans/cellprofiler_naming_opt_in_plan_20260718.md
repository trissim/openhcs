# CellProfiler Automatic Typed Artifact Identity Plan

**Date:** 2026-07-18
**Status:** Production implementation in progress as a disjoint importer workstream
**Scope:** Import lowering and focused architecture verification; canonical ZMQ acceptance remains parent-serialized

## Progress

- 2026-07-19 00:18 EDT: Read root `AGENTS.md`, this plan in full, and every
  active `.agents` ownership/plan/parent-note section. The assigned importer and
  focused-test files do not overlap the active path-planner, generic output,
  neighbor, primary-object, or worm workstreams. A stdlib AST inventory found
  `_sparse_kwargs_for_contract()` as the sole local subset-search owner with two
  production call sites and six direct tests. The existing nominal authorities
  are `SettingToKeywordBinding`, module-owned public contract reconstruction,
  exact `ArtifactSpecRef`/relation dependencies, `RuntimeExportExpectation`, and
  `SourceBindingDeclarationsMixin.for_artifact_refs()`.
- 2026-07-19 00:18 EDT, corrected 00:41 EDT: Inspected the AdvancedSegmentation
  generated source under
  `/home/ts/.cache/openhcs/parity-cp-advanced-source-conflict-20260718-fixed/`.
  It contains repeated enabled step-local source subsets on lineage-only
  `PREVIOUS_STEP` consumers, which remains valid API evidence for implementation
  change 2. It is not causal evidence for the later channel-1 failure. The 23.7 MB
  canonical compiled execution bundle under
  `/home/ts/.cache/openhcs/parity-cp-advanced-source-conflict-20260719-plan-classification/`
  proves step 37 already has exact channel-2 execution scope, channel-2/4 compiled
  bindings, and `ER@2`/`Cells@4` edges. That failure belongs to runtime filter
  ordering, outside this importer workstream.
- 2026-07-19 00:25 EDT: Owner inventory is complete. Parsed setting-to-artifact
  projection is owned by `SettingToKeywordBinding` plus
  `CellProfilerModuleArtifactContracts.artifact_names_for_binding()`; canonical
  omission is owned by `module_blocks_for_invocation()`; exact relations are
  owned by each `ArtifactSpecRelation`'s `dependency_refs()`,
  `measurement_subject()`, and `materialization_source()` methods; explicit
  export observation is owned by `RuntimeExportExpectation.from_output_specs()`;
  direct source projection is owned by
  `StepSourceBindingsConfig.for_artifact_refs()`; and normal-versus-dict emission
  remains owned by `_lower_module_batch()`. Active peer plans own only
  path-planner source output resolution, generic runtime output normalization,
  and backend kernels, so this workstream has no write conflict.
- 2026-07-19 00:34 EDT: Corrected a plan/implementation ownership contradiction
  before testing. `ArtifactSpec` owns the relation collection but does not own
  `measurement_subject()` or `materialization_source()`; those polymorphic APIs
  belong to each `ArtifactSpecRelation`. Observation now traverses
  `spec.relations`, deduplicates the exact non-null subjects and source refs
  returned by those methods, and follows each relation's `dependency_refs()`.
  No convenience method, relation-kind branch, or mirrored table is introduced.
  The implementation remains in the first production batch; the next gate is
  still AST parsing followed by focused importer tests.
- 2026-07-19 00:43 EDT: Searched production for an existing nominal parsed
  invocation/module-unit record; none exactly owns the importer's parsed module,
  resolved contract, split public kwargs, processing/source context, and source
  position. `_ParsedTargetUnit` is therefore one private frozen slots dataclass
  replacing the former ten-field tuple completely. All consumers use named fields;
  AST/text postconditions find no positional `unit[...]` access. The one-pass
  target graph is implemented, `_sparse_kwargs_for_contract()` and its two call
  sites are deleted, and source lowering now projects only exact direct refs with
  `for_artifact_refs()`. A static-gate correction also removed every importer-side
  `ArtifactSpecRef(...)` construction: active binding-owned names now select exact
  input/output specs already carried by each unit's `CallableContract`, input
  selection additionally matches the binding's runtime `parameter_name` when
  declared, and all refs come from `spec.ref()`. `py_compile`, the no-constructor
  and no-positional-index postconditions, and the full static deletion suite pass
  (`26 passed`). Implementation now proceeds to focused importer behavior; the
  next gate is the full pipeline-import suite, followed by transport/pycodify and
  function-pattern tests. Official30 remains parent-owned.
- 2026-07-19 01:10 EDT: Superseded the initial global-ref selection draft after
  independent architecture review. Same-type candidate counts, available-name
  counts, and target/candidate name sets are not selection authorities;
  `module_artifact_contracts.py` owns cardinality and ordered scoped expansion.
  Selection analysis now reconstructs the all-input-identities-omitted candidate
  through `module_blocks_for_invocation()`, canonical numbering, and
  `invocation_callable_contract()`, then compares ordered target/candidate
  `ArtifactSpec.ref()` sequences by artifact-type domain. Proven mismatches are
  stored immutably on each `_ParsedTargetUnit` and projected only through that
  unit's owning input binding. `_lower_module_batch()` performs both the analysis
  pattern decision and the final normal-versus-dict decision; analysis stops
  before public graph advancement, while the single final emission pass retains
  observed outputs and performs exact public-contract reconstruction.
- 2026-07-19 01:10 EDT: Removed the scope-free global
  `output_by_input_ref` closure. Each parsed unit now retains the exact
  `ArtifactProducer` objects published for its contract outputs. Selected
  consumer refs resolve only through that consumer unit's
  `available_artifact_producers`; the backward worklist carries a nominal
  producer occurrence `(target position, producer)` and resolves relation
  dependencies relative to the unit that owns that occurrence. Reused artifact
  names therefore cannot become last-writer-wins. The next gate is focused
  internal producer-consumer rename, subset, reorder, and reused-name occurrence
  coverage, followed by the complete importer and static suites.
- 2026-07-19 01:42 EDT: Corrected the remaining analysis-stage over-retention
  before resuming tests. Comparing combined contracts by artifact type is not an
  ownership boundary: it retains unrelated scalar bindings whenever any sibling
  binding of the same type differs. The existing nominal owner is
  `artifact_inputs_for_binding()`, which already resolves scalar, repeated, and
  multiple-scalar cardinality in declaration order. Candidate reconstruction will
  therefore retain each numbered candidate module occurrence and its exact
  invocation context, compare each active binding's ordered target specs against
  only that binding's candidate specs, and mark refs solely on target units whose
  owning binding differs. Bindings with `runtime_parameter_name` match candidate
  contract specs by that exact runtime parameter; bindings without one resolve
  through the candidate module block and the same binding owner. No artifact-type
  proxy, count test, name set, or combined-contract selection fact remains. The
  implementation is at this binding-exact correction; the next gate is
  `py_compile` plus the focused internal-rename/subset/reorder/import tests before
  the complete importer and static suites.
- 2026-07-19 01:51 EDT: Corrected selection-fact ownership before continuing the
  context investigation. A bare per-unit ref set still conflates two active
  bindings that intentionally consume the same exact artifact, such as `labels`
  and `neighbor_labels` both consuming `Cells`. Each parsed target unit will now
  store private frozen binding occurrences containing the authoritative
  `SettingToKeywordBinding` object and that binding's ordered exact refs. Backward
  observation may flatten those occurrences to refs because it follows payload
  dependencies; public kwarg retention must match the exact binding occurrence and
  never infer ownership by ref intersection or copied parameter strings. A focused
  shared-ref/two-binding regression will prove only the mismatching binding is
  retained.
- 2026-07-19 01:58 EDT: Corrected occurrence lookup to preserve the exact
  `ArtifactSpecRef.plan_type`. The draft's `include_current_outputs` flag coerced
  every dependency to input role and preferred a current same-name output, which
  can self-cycle an in-place transform instead of resolving its prior producer.
  Output refs will resolve only against the current unit's exact output producers;
  input refs will resolve only against that unit's prior
  `available_artifact_producers`; unsupported plan roles fail. The boolean and
  cross-role coercion are deleted. Focused coverage now includes an in-place
  same-name input/output relation and the existing reused-name-across-steps
  occurrence case.
- 2026-07-19 02:04 EDT: Replaced the ref-only selection field with
  `_SelectedInputBindingOccurrence`; no ref-intersection kwarg-retention path
  remains. Replaced boolean occurrence priority with strict input/output role
  dispatch and removed all observation-side plan-role coercion. Focused tests now
  prove two bindings sharing `Nuclei` retain only the selected binding, an in-place
  same-name input dependency resolves the prior producer rather than self-cycling,
  and a later same-name consumer resolves the producer in its own context
  (`3 passed`). `py_compile` and scoped diff checks pass. The next implementation
  gate remains the exact grouped candidate-context correction exposed by
  `MeasureObjectIntensity` and `Crop`, followed by the internal-name and full
  importer suites.
- 2026-07-19 02:24 EDT: The full focused importer suite exposed that the
  all-input-identities-omitted analysis is not the canonical proof boundary
  (`5 failed, 24 passed`). One unresolved binding can prevent reconstruction of
  otherwise omittable siblings: grouped `MeasureObjectIntensity` lacks a scoped
  object-label candidate for one group, while cross-group `Crop` presents three
  images to two scalar bindings when both identities disappear together.
  Selection analysis is therefore corrected to one owning binding at a time.
  For each active binding, reconstruction omits only that binding and retains
  every other target binding's exact identity through its existing
  `SettingToKeywordBinding`; the resulting module blocks and callable contracts
  are compared only against that binding's ordered target specs. Selection facts
  remain `_SelectedInputBindingOccurrence` values, so a non-derivable binding
  cannot block proving sibling omissions and shared refs cannot conflate binding
  ownership. No combined omission, exception-controlled fallback, count test,
  name set, or importer-owned cardinality rule remains. `module_blocks_for_invocation()`
  remains the reconstruction owner. The direct `SaveImages` expectation will be
  updated to assert omission for its one exact canonical input and exact contract
  round-trip. The next gate is this per-binding implementation followed by the
  complete focused importer suite.
- 2026-07-19 02:43 EDT: Reproduced the exact focused baseline after the plan
  correction (`5 failed, 24 passed`). Repository-wide API search found no
  non-throwing exact omission-candidate query. The sole owner,
  `CellProfilerModule.module_blocks_for_invocation()`, delegates cardinality to
  `_artifact_input_record_groups_for_bindings()` and raises when an omitted
  binding has zero scoped candidates. That is correct for final public
  reconstruction but incomplete for importer analysis, where zero alternatives
  is data proving that the binding must remain explicit. Group-2
  `MeasureObjectIntensity` demonstrates the defect exactly: retaining its image
  identity permits sibling analysis, while omitting its cross-group `Nuclei`
  object binding has no scoped candidate and raises. Importer-side exception
  control, copied availability/cardinality logic, and group/type predicates are
  prohibited. The next production edit therefore requires the nominal module
  owner to expose exact omission alternatives (including an empty result), after
  which importer can compare each binding independently and final reconstruction
  can continue requiring a valid block. No importer patch will mirror that owner.
- 2026-07-19 02:50 EDT: Parent's adjacent required-suite run adds three exact
  regressions (`116 passed, 3 failed`) to the per-binding gate. The
  BBBC021Illumination probe proves that "every other identity" includes active
  output bindings: omitting `Name the averaged image` while probing an input
  changes `CorrectIlluminationCalculate.active_artifact_bindings()` before the
  candidate exists. Each probe must therefore retain all exact target artifact
  identities except the one input binding under analysis. BBBC021Analysis proves
  candidate contracts must never be combined for omission analysis: the same
  `Nuclei` ref has distinct group-local lineage from `OpeningDAPI` and `CorrDAPI`,
  so each candidate occurrence is compared only with its corresponding parsed
  target occurrence. ExampleHuman reproduces the zero scoped object-label
  alternative already seen in grouped `MeasureObjectIntensity`. Required
  acceptance is now the complete importer suite followed by the five-file
  invocation/pycodify/function-pattern/corpus/generated-execution batch; no
  official30 run belongs to this workstream.
- 2026-07-19 02:50 EDT: Per-binding omission reconstruction is complete.
  `SettingToKeywordBinding` now distinguishes ordinary artifact domains from
  exact sidecar domains, and `ArtifactSidecarSourceRelation` owns the primary
  identity that selects a sidecar. Crop's leaf resolver is deleted; the same
  binding declaration now drives both transient module reconstruction and exact
  callable inputs. A unique prior crop omits its name, while a later branch with
  multiple crop masks retains `CropBlue` because it selects a noncanonical
  earlier producer.
- 2026-07-19 02:50 EDT: Group reconstruction now scopes only source declarations
  that own the current main flow. Exact supplemental source edges remain
  available across groups, which preserves SaveImages' `GrayTumor` processing
  group and separate `ColorLung` filename source without copying pipeline source
  declarations into the step. `ArtifactDeclarationStepContext.with_source_declarations()`
  now honors its declared source subset when reseeding pipeline-start main flow.
- 2026-07-19 02:50 EDT: Behavior-owned repeated input reconstruction is complete.
  `MeasureImageQualityModule.artifact_inputs_for_binding()` resolves "All loaded
  images" from the typed main-flow image domain and the parallel
  `artifact_contract_inputs()` path is deleted. Image and object measurement MRO
  owners now attach nominal measurement-subject relations, allowing the existing
  observation closure to retain externally visible producer vocabulary without
  an importer-side measurement rule.
- 2026-07-19 02:50 EDT: Focused acceptance is green. The importer/source/sidecar/
  measurement-subject batch passes (`69 passed`), and the broader declaration,
  activation, settings, provider, and static-deletion batch passes (`257 passed`).
  Identical complete-domain channel modules lower to a plain pattern; real
  group-specific behavior or selection lowers to one dict pattern. The required
  public source/corpus batch is `117 passed, 2 failed`; both remaining failures
  are the same pre-existing IdentifyTertiaryObjects output-ABI declaration gap
  and are assigned to the owning leaf module before canonical ZMQ acceptance.
- 2026-07-19 03:50 EDT: Exact multi-input reconstruction and plain-pattern
  lowering are complete for the two-channel illumination chain. Candidate input
  domains now come from current main flow or produced runtime-parameter
  artifacts according to the existing binding declaration, and paired inputs
  join only through their nominal source-stack lineage relation. The generated
  ExampleColocalization pipeline now emits one plain
  `CorrectIlluminationCalculate`, one plain `CorrectIlluminationApply`, and a
  behavior-only `MeasureColocalization`; all routine image identities disappear
  from their public kwargs. The real two-channel importer regression and focused
  declaration/import/provider batch pass (`109 passed`). Broad corpus and
  canonical ZMQ acceptance remain next.

## Outcome

CellProfiler-backed pipelines use the same public authoring model as native
OpenHCS pipelines:

- `PipelineConfig` owns pipeline-wide source and processing defaults.
- `FunctionStep` owns callable choice, behavior kwargs, execution configuration,
  and the rare explicit artifact identity needed by that invocation.
- Typed callable and module declarations own input/output artifact roles.
- Missing input identities resolve from the exact typed artifact scope.
- Missing output identities receive deterministic canonical names.
- The compiler reconstructs exact contracts and removes compile-only identity
  kwargs before runtime invocation.

Automatic canonical naming is already the default contract behavior. The required
implementation removes importer behavior that preserves every parsed CellProfiler
workspace name merely because the setting exists.

Repeated source declarations and duplicate per-channel steps are separate cleanup
concerns. They remain on the existing generic config-inheritance and
function-pattern paths rather than becoming artifact naming rules.

## Governing Invariants

1. A `.cppipe` is an import source, never a runtime or compile-time dependency.
2. A public `PipelineConfig` plus `list[FunctionStep]` completely determines the
   compiled pipeline.
3. Artifact type, role, cardinality, relations, and canonical identity come from
   existing nominal declarations and exact contract reconstruction.
4. Parsed setting presence carries no semantic weight after import.
5. An omitted artifact identity means canonical typed derivation.
6. An explicit artifact identity means exact selection from a larger same-type
   scope or preservation of externally observed CellProfiler vocabulary.
7. Compile-only artifact identity never reaches a backend callable.
8. Pipeline-level source declarations are inherited. A step declares only a
   direct source input needed beside produced or main-flow inputs.
9. A normal callable applies to the complete third-axis stack. A dict function
   pattern exists only for a source-group subset or group-specific
   callable/kwargs.
10. The implementation adds no naming mode, policy type, configuration field,
    registry, declaration table, or backend-specific runtime path.

## Verified Current Authorities

A stdlib AST inventory parsed 119 production files under
`openhcs/interop/cellprofiler` and
`openhcs/processing/backends/cellprofiler`. It found 181 canonical
`SettingToKeywordBinding.input()`/`.output()` declarations across 39 files: 111
inputs and 70 outputs. No second naming owner exists in current production.

### Module declaration and artifact roles

Retain these owners without adding naming state:

- `openhcs/interop/cellprofiler/module_declarations.py`
  - `CellProfilerModule` combines `CellProfilerModuleCallableABI`,
    `CellProfilerModuleArtifactContracts`, `CellProfilerModuleSettings`, and the
    measurement owners through its existing MRO.
  - Its `AutoRegisterMeta` registry resolves module and callable ownership.
  - `module_blocks_for_invocation()` reconstructs transient setting rows from one
    public invocation.
  - `invocation_callable_contract()` returns the exact callable contract and the
    names of authored kwargs consumed during compilation.
- `openhcs/interop/cellprofiler/module_artifact_declarations.py`
  - Module roots and leaf declarations own artifact roles, relations, cardinality,
    and module-specific specialization.
- `openhcs/interop/cellprofiler/settings_binder.py`
  - `SettingToKeywordBinding.input()` and `.output()` connect one CellProfiler
    setting to an exact artifact plan type, artifact type, public parameter name,
    runtime parameter name, and repeated cardinality.
  - `require_parameter_name()` is the sole spelling authority for an explicit
    public identity kwarg.
  - `records_from_kwargs()` lowers that public identity into transient module
    setting rows.

The binding declaration already contains every fact needed to distinguish
behavior parameters from compile-only artifact identity. It gains no naming
field.

### Canonical contract reconstruction

Retain `CellProfilerModuleArtifactContracts` in
`openhcs/interop/cellprofiler/module_artifact_contracts.py` as the sole
CellProfiler contract reconstruction owner:

- `declared_artifact_bindings()` and `active_artifact_bindings()` collect exact
  binding declarations through the module MRO.
- `_available_artifact_input_names()` obtains exact typed candidates from main
  flow, prior producers, and source declarations in declared flow order.
- `_artifact_input_record_groups()` applies binding cardinality to missing input
  rows.
- `require_available_artifact_input()` validates an explicit identity against one
  exact source, main-flow artifact, or producer.
- `_derived_identity_setting_records()` fills missing output rows.
- `canonical_output_artifact_name()` supplies deterministic output identity.
- `callable_contract()` binds artifact parameters, outputs, and relations into the
  canonical callable contract.

These methods define public omission semantics:

- One repeated input binding consumes the complete matching typed domain.
- One scalar input binding expands over the complete matching typed domain.
- Multiple scalar bindings consume an equal-cardinality matching domain in
  declaration order.
- Explicit input kwargs narrow or reorder that canonical domain and are validated
  by exact typed reference.
- Missing output kwargs always receive canonical identities.

### Public step and compiler boundary

The existing public identity API is ordinary function-pattern kwargs:

```python
FunctionStep(
    func=(cellprofiler_callable, {
        binding.require_parameter_name(): exact_artifact_name,
    }),
)
```

Concrete source uses the literal parameter name declared by the owning
`SettingToKeywordBinding`; the snippet shows ownership, not a new runtime API.
Grouped patterns carry the same kwargs under the selected group key.

Retain the complete public transport and compiler path:

- `FunctionStepTransportAuthority` serializes and reloads ordinary public steps.
- `CellProfilerInvocationContractProviderFactory.provider_for_session()` in
  `openhcs/interop/cellprofiler/compile_time_contracts.py` rebuilds the forward
  artifact context from resolved public `FunctionStep` snapshots.
- `InvocationContractPlan.consume_authored_kwargs()` in
  `openhcs/core/invocation_artifacts.py` proves each consumed identity was authored
  and removes it.
- `_compile_invocation()` in `openhcs/core/function_patterns.py` selects exact
  input/output plans and validates only the remaining runtime kwargs.

Compile-time validation remains exact and total:

- Canonical omission expands according to declared binding cardinality.
- Explicit selection resolves to one exact typed source, main-flow artifact, or
  producer.
- Missing, conflicting, and wrong-type selections fail compilation.
- Output relations reference artifacts present in the reconstructed graph.
- Consumed compile-only names equal the names authored in the public step.

### Source binding inheritance

Retain the generic source-binding authorities:

- `SourceBindingsConfig` and `StepSourceBindingsConfig` are registered pipeline
  configs in `openhcs/core/config.py` and implemented in
  `openhcs/core/source_bindings.py`.
- `LazyStepSourceBindingsConfig` resolves through normal config inheritance.
- `bindings_for_artifact_specs()` follows exact `ArtifactSpec.relations` group
  lineage.
- `for_artifact_refs()` projects declarations to exact direct source refs.
- `PathPlannerArtifactStage.source_bindings_for_contracts()` in
  `openhcs/core/pipeline/path_planner.py` projects inherited source declarations
  to compiled contract inputs.

Artifact naming does not own source discovery, component coordinates, or source
grouping.

## Current Dataflow and Defect

### Pre-implementation import flow

1. `import_cellprofiler_pipeline()` parses numbered `ModuleBlock` values.
2. `_public_pipeline()` creates one pipeline `PipelineConfig`, resolves inherited
   step defaults, and walks the forward artifact context.
3. `_lower_module_batch()` obtains each exact parsed target contract, separates
   behavior kwargs from binding-owned identity kwargs, and chooses normal or dict
   function-pattern syntax.
4. `_sparse_kwargs_for_contract()` repeatedly reconstructed a local candidate and
   retains the smallest identity subset that remains exactly string-equal to the
   parsed target contract.
5. `_public_step_source_bindings()` recursively projected source lineage into a
   step-local source configuration.
6. The generated public source reloads into `PipelineConfig` and
   `list[FunctionStep]`, then compiles through the same provider used by
   hand-authored pipelines.

### Defect

Local full-name equality makes every parsed producer name appear significant. A
producer and its downstream consumers cannot adopt canonical identities together,
so internal image/object names survive in generated source. Recursive lineage
projection also copies pipeline source declarations into many steps even though
the compiler already traces that lineage. Retained incidental identity kwargs then
keep otherwise identical per-channel modules in dict patterns or separate steps.

The compiler and runtime are not missing a naming feature. The importer preserves
more identity than the public artifact graph exposes.

## Exact Identity Boundary

### Automatic default

No public identity kwarg means the module contract owner performs canonical typed
derivation. This is complete public semantics, not missing metadata.

For inputs, binding cardinality determines the complete exact typed domain. For
outputs, `canonical_output_artifact_name()` supplies identity. Relations are built
against those reconstructed specs by the owning module declaration.

### Explicit input identity

An explicit input identity remains only when the parsed invocation selects a
strict subset or different order from canonical binding-cardinality expansion over
the exact same-type scope. The owning public API is the ordinary
`FunctionStep.func` kwarg named by the corresponding
`SettingToKeywordBinding.require_parameter_name()`.

Source setting rows remain source declarations in pipeline config. They do not
become step naming policy.

### Explicit output identity

An explicit output identity remains only when the exact output ref participates in
externally observed CellProfiler vocabulary. Observation is derived from current
nominal facts:

- a measurement output relation returns a non-null `measurement_subject()` and
  names its exact source in `dependency_refs()`;
- a relationship declaration exposes exact source and target refs through its
  relation implementation;
- `RuntimeExportExpectation.from_output_specs()` identifies explicitly
  materialized outputs;
- `ArtifactSpecRelation.materialization_source()` identifies the exact source
  whose identity participates in the materialized result.

The observation closure follows `ArtifactSpecRelation.dependency_refs()` backward
through the parsed target graph. It is transient derived import state, not another
declaration owner.

Internal main-flow images, object labels, intermediate measurements, and sidecars
receive canonical identities outside that closure. Behavior values such as
measurement feature strings, output column vocabulary, threshold choices, and
algorithm options remain public callable kwargs and are not rewritten.

## Target Dataflow

1. Parse `.cppipe` modules and source declarations.
2. Build the exact parsed target artifact graph once through the current module
   MRO, setting bindings, `callable_contract()`, and
   `advance_artifact_context()`.
3. Reconstruct each unit/group with all input identities omitted and retain exact
   refs only for active bindings whose ordered candidate specs differ from that
   binding's ordered parsed-target specs.
4. Store those selection facts on their owning parsed target units.
5. Derive observed producer occurrences from selected consumers, measurement,
   relationship, and materialization relations, then follow relation dependencies
   backward through each occurrence's owning parsed context.
6. Lower each module using behavior kwargs plus binding-owned identity kwargs only
   for selection and observation refs.
7. Let `module_blocks_for_invocation()` derive every omitted input/output name from
   the candidate public graph.
8. Let `_lower_module_batch()` use its existing full-domain normal-callable rule.
   Dict patterns remain only for group-specific work.
9. Serialize and reload the public pipeline.
10. Compile the reloaded `PipelineConfig` and `FunctionStep` list. Exact contract
    reconstruction validates selection, relations, source ownership, and consumed
    kwargs.
11. Execute with compile-only identity kwargs removed and runtime behavior
    unchanged.

No runtime workspace name resolver participates in this flow.

## Exact Implementation Changes

### 1. Replace local identity subset search

Change `openhcs/interop/cellprofiler/pipeline_import.py`:

- Move the current parsed target contract/context construction out of the local
  identity search and make `_public_pipeline()` perform one complete forward
  target-graph pass before public emission.
- Derive retained identity refs from exact target contracts and nominal relations.
  Use `ArtifactSpec.ref()` plus each relation's `dependency_refs()`,
  `measurement_subject()`, and `materialization_source()`, together with
  `RuntimeExportExpectation.from_output_specs()` as existing observation
  authorities.
- Project retained refs back to their owning `SettingToKeywordBinding`. Emit that
  binding's exact public parameter only for selected inputs and observed outputs.
- Reconstruct every emitted public contract through
  `module_blocks_for_invocation()` and `invocation_callable_contract()` once.
- Report reconstruction failures with module number, invocation key, binding
  parameter, expected typed refs, and reconstructed typed refs.
- Delete `_sparse_kwargs_for_contract()` and its exception-controlled subset search
  after both call sites use exact retained-ref projection.
- Keep `_lower_module_batch()` as the sole normal-versus-dict function-pattern
  lowering owner. Add no post-lowering pattern optimizer.

Names are compared only as members of exact `ArtifactSpecRef` values declared by
bindings and relations. The importer performs no string classification.

Selection and observation are occurrence-aware. A scope-free `ArtifactSpecRef`
never keys a global producer map; each selected consumer resolves its producer
from that unit's exact parsed context, and relation traversal carries the resolved
producer occurrence backward.

### 2. Remove repeated step source declarations

Change `_public_step_source_bindings()` in
`openhcs/interop/cellprofiler/pipeline_import.py`:

- Keep all discovered source declarations once in the pipeline
  `SourceBindingsConfig` produced by `_public_pipeline()`.
- For `InputSource.PIPELINE_START`, rely on the inherited resolved step config.
- For `InputSource.PREVIOUS_STEP`, collect only contract input refs directly owned
  by `source_bindings.binding_for_artifact_ref()` and project them with
  `for_artifact_refs()`.
- Emit an enabled step override only for direct source refs loaded beside
  produced/main-flow artifacts.
- Do not copy declarations reached solely through group-lineage relations.

`PathPlannerArtifactStage.source_bindings_for_contracts()` remains responsible for
tracing inherited declarations through exact artifact lineage during compilation.
This cleanup uses generic OpenHCS config inheritance and source-binding semantics;
it adds no CellProfiler source mechanism.

### 3. Collapse redundant per-channel emissions

Retain the existing `_lower_module_batch()` decisions:

- A single callable/behavior set covering the complete source group domain emits
  a normal callable pattern.
- Different callable behavior, a selected source-group subset, or explicit
  group-specific artifact selection emits a dict pattern.
- Adjacent module blocks remain separate only when their processing config,
  callable, direct step source override, or exact group behavior differs.

Removing incidental identity kwargs and copied source declarations allows the
existing equality checks to collapse repeated channel blocks. No batching
metadata or CellProfiler-specific execution rule is introduced.

`variable_components` continues to define third-axis meaning. `group_by` continues
to select dict-pattern branches after stack construction.

### 4. Retain compiler and module APIs unchanged

No naming-driven changes belong in:

- `module_artifact_contracts.py`
- `module_artifact_declarations.py`
- `settings_binder.py`
- `module_declarations.py`
- `compile_time_contracts.py`
- `core/invocation_artifacts.py`
- `core/function_patterns.py`
- `core/artifacts.py`
- `core/pipeline/path_planner.py`

Implementation evidence may expose a defect in one of these owners. Such a defect
is corrected at that owner under its own focused change, never by adding importer
state.

## Deletions

Delete from the prior proposal and do not implement:

- the naming opt-in concept;
- a naming policy enum or per-binding naming field;
- parsed-setting presence as an identity decision;
- deleted artifact partition and port concepts;
- a UI naming mode or toggle;
- module-name checks and module catalogs;
- runtime workspace-name reconstruction;
- dual native/imported naming behavior;
- migration shims and deprecated paths.

Delete from production during implementation:

- `_sparse_kwargs_for_contract()` after exact retained-ref projection replaces its
  two call sites;
- importer branches whose sole purpose is preserving local target-contract string
  equality;
- step-local source declarations produced solely by recursive lineage projection.

Delete or rewrite tests that assert incidental internal CellProfiler names,
identity-subset search order, or copied lineage source declarations. Retain tests
that assert exact explicit selection and externally observed vocabulary.

## Test Plan

### Contract reconstruction

Add focused tests in `tests/unit/test_cellprofiler_artifact_declarations.py`:

- A CP callable with no input/output identity kwargs derives exact typed inputs and
  canonical outputs from a public `FunctionStep`.
- One scalar binding over multiple same-type candidates expands according to the
  current canonical cardinality rule.
- Explicit input selection resolves the declared exact producer.
- Missing, wrong-type, and conflicting explicit selections fail during compile.
- Explicit output identity appears in the compiled artifact graph and is absent
  from runtime kwargs.
- Measurement and relationship relations reference the exact retained object/image
  identity.

### Import lowering

Update `tests/unit/test_cellprofiler_pipeline_import.py`:

- Replace direct `_sparse_kwargs_for_contract()` tests with public import and
  compile assertions.
- Internal producer/consumer names disappear together while exact relations remain
  valid.
- Measurement subjects and explicitly materialized names preserve parsed
  CellProfiler vocabulary.
- A same-type subset selection retains only the owning input/output identity kwargs.
- A same-type reorder retains the exact owning identities in parsed order.
- An internal producer and consumer canonicalize together when omission
  reconstructs the same ordered typed input domain.
- Reused producer names resolve through the selected consumer occurrence rather
  than a global last-writer-wins ref map.
- Identical full-domain channel modules emit one normal callable.
- Group-specific behavior and source subsets emit a dict pattern.
- Pipeline source declarations occur once; produced-artifact lineage emits no step
  copy.
- A mixed previous-step/direct-source contract emits only the direct source step
  override.
- Pycodified source executes to the same `PipelineConfig` and `FunctionStep`
  declarations and compiles to the same exact contract graph.

Retain and adapt current exact-selection and grouping coverage:

- `test_missing_runtime_producer_requires_explicit_artifact_identity`
- `test_explicit_runtime_artifact_identity_requires_exact_producer`
- `test_explicit_runtime_artifact_identity_uses_exact_cross_group_producer`
- `test_adjacent_channel_specific_modules_lower_to_one_grouped_step`
- `test_channel_outputs_with_custom_identities_lower_to_sparse_grouped_step`
- `test_previous_step_lowering_enables_only_contract_selected_source_bindings`
- `test_repeated_natural_measurement_images_use_group_local_source_identity`

Run these generic public-path suites:

- `tests/unit/test_invocation_contract_provider.py`
- `tests/unit/test_pycodify_formatters.py`
- `tests/unit/test_function_patterns.py`
- `tests/unit/test_cppipe_corpus.py`
- `tests/unit/test_cellprofiler_generated_pipeline_execution.py`

### Public source round trip

For every supported pipeline, require:

1. `.cppipe` import produces only public `FunctionStep` values plus
   `PipelineConfig`.
2. `FunctionStepTransportAuthority.source_from_pipeline()` emits executable
   source.
3. `pipeline_steps_from_namespace()` reconstructs equivalent step names, configs,
   callable patterns, and exact contracts.
4. Rendering reconstructed declarations is source-stable.
5. Compilation receives no private importer state.

### Static architecture gates

Extend `tests/unit/test_cellprofiler_static_deletion_gates.py` with AST assertions
that reject:

- a naming policy declaration or naming config field;
- reintroduction of deleted partition/port APIs;
- setting-presence identity branches;
- module-name identity branches;
- runtime access to compile-only artifact identity kwargs;
- a second function-pattern lowering or source-binding projection path.

Static gates inspect declaration and call ownership rather than source-text
spelling.

## Canonical Acceptance

Run gates in this order after the current parity integration is green:

1. Focused artifact declaration, pipeline import, source binding, function pattern,
   transport, and static architecture tests.
2. Public-source round trip for representative image, object, measurement,
   relationship, export, multi-channel, and mixed-source pipelines.
3. Compile those round-tripped declarations over the canonical ZMQ
   compile-then-execute path.
4. Run the baseline parameter of
   `tests/integration/test_cellprofiler_official30_zmq.py` with
   `OPENHCS_CP_NATIVE_REFERENCE_ROOT` pointing to the existing cached native
   reference root.
5. Load all 30 cases from
   `benchmark/manifests/official30_portable_axis1.json`.
6. Import, pycodify, reload, compile, and execute generated public source over ZMQ.
7. Require 30/30 successful execution and equivalence at the benchmark's strict
   `1e-6` numeric/image tolerances.
8. Inspect generated ExampleColocalization and other multi-channel pipelines for:
   - no incidental internal image/object identity kwargs;
   - pipeline-level source declarations without lineage copies;
   - normal callable patterns for identical full-domain work;
   - retained explicit names only at selection and observation boundaries.

Focused tests alone do not accept this change. The official-30 public-source path
is the end-to-end gate.

## Non-Goals and Boundary

- No new public configuration or invocation-options type.
- No changes to generic OpenHCS stack-axis, grouping, special input/output,
  materialization, or runtime artifact semantics.
- No inference from upstream CellProfiler Python setting classes.
- No change to source discovery or source component identity.
- No rewriting of algorithm kwargs or measurement feature vocabulary.
- No importer metadata attached to `FunctionStep` or `PipelineConfig`.
- No `.cppipe` access after public pipeline construction.
- No parallel semantic authority, reflective field access, silent recovery path,
  or name-based module dispatch.

## Completion Definition

The work is complete when a user constructs the same CP-module pipeline from
scratch with public `PipelineConfig` and `FunctionStep` declarations, omits routine
input/output image and object names, compiles exact typed artifact edges, preserves
only deliberately selected or externally observed CellProfiler identity, executes
without compile-only kwargs reaching callables, and passes official-30 parity over
the canonical ZMQ path.

## Implementation Progress

### 2026-07-19 04:40 EDT

- [x] Reconstructed direct FilterObjects runtime parameters from its nominal
  object-label bindings; the generic public-path batch passes with 120 tests.
- [x] Corrected Align omission probing so an incomplete inferred input domain
  rejects the candidate without weakening final contract validation.
- [x] Removed the importer candidate path that rebuilt total contracts while
  probing one input binding; each binding now queries its owning declaration.
- [x] Preserved externally exported measurement identities through the shared
  `ArtifactExportModule` provenance relation. `ExamplePercentPositive` now
  imports without exposing `PH3PosNuclei` as hidden importer state.
- [x] Reconstructed observed channel outputs as one exact dict-pattern step when
  output identities differ by group.
- [x] Collapsed identical complete-domain module runs to one plain pattern using
  exact contract group-scope lineage rather than a second main-flow-only
  projection.
- [x] Made complete-domain comparison independent of `.cppipe` module order only
  for the exact one-artifact-per-unit proof shape. Ordered repeated inputs remain
  ordered.
- [x] Audited every self-contained cached CellProfiler3 pipeline: no identical
  complete-domain dict pattern remains. Remaining identical dicts select strict
  channel subsets.
- [x] `tests/unit/test_cellprofiler_pipeline_import.py`: 30 passed.
- [x] Generic public-path batch: 120 passed, 1 upstream rank-filter warning.
- [x] Cached CellProfiler3 corpus import: every self-contained pipeline imports;
  only the three flat-cache worm pipelines fail because their declared model XML
  resources are absent from that directory.
- [ ] Resolve the three worm model paths through the deterministic source-root/VFS
  path work without filename search or ambient process state.
- [ ] Run representative public-source round trips and canonical ZMQ execution.
- [ ] Run official-30 strict parity, performance, and Napari acceptance gates.
