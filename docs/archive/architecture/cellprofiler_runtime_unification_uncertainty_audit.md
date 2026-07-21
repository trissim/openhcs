# Superseded: CellProfiler Runtime Unification Implementation Dry Run

## Audit Verdict

The previous runtime-unification plan was not implementation-ready. Its
governing invariants required existing OpenHCS artifact and nominal declaration
machinery to remain authoritative, while its target API introduced a second
artifact-port hierarchy, a second compile-time artifact flow, new special-I/O
descriptor classes, and CellProfiler setting descriptors. Those additions
re-encoded information already owned by `ArtifactSpec`,
`ModuleArtifactContractItem`, callable special-I/O metadata, and
`CellProfilerModule`.

The corrected plan removes those proposals. The dry run found no missing public
configuration type and no missing artifact taxonomy. It did find four generic
value types absent from the current execution/materialization API:
axis-versus-plate callable scope, an exact plate artifact batch, one-image-file
writer options, and named-file-bundle writer options. Each has one behavior
owner and none mirrors CP module semantics. It also found owner-level defects
that require no new public model: provider claims are first-match, public kwargs
are revalidated and filtered at runtime, compiled artifact selection uses
string keys, and FileManager reconstruction loses its execution-local registry
while generic virtual loading rewrites backend addresses. The required work is
deletion, reconnection, four new value types, and method-level corrections on
those existing owners:

- connect the existing invocation-contract provider to nominal module
  declarations and existing artifact collections;
- make the existing `SourceBindingsConfig` complete, collapse retained
  `PipelineImageSchema` state into it, and keep config inheritance
  authoritative;
- collapse two CellProfiler runtime wrappers to one adapter;
- remove hidden contracts, symbol mirrors, generated matching, metadata
  carriers, setup-module registry, callable ownership projection, and manual
  generator source model;
- remove one-field projection carriers whose sole behavior already belongs to
  an existing generic or nominal owner;
- use direct step lists at UI, agent, benchmark, and ZMQ boundaries;
- run aggregate exporters through generic plate-scoped callable execution and
  generic materialization rather than infrastructure flags;
- verify the reconstructed public source through ZMQ compile-then-execute.

The corrected plan is executable phase by phase without an unresolved
architectural decision.

Every replacement is an atomic cutover. All callers, source generation,
serialization, fixtures, and tests migrate in the owning phase, followed by
deletion of the old definitions, files, imports, re-exports, readers, writers,
and fallback branches. No phase retains a forwarding facade, old name,
dual payload shape, state upgrader, or test that exercises the removed
architecture. Existing generated Python, virtual-workspace metadata, and
benchmark outputs are regenerated rather than upgraded at load time.

## Audit Method

The audit traced these paths in the current tree:

1. `FunctionStep` construction and ObjectState resolution.
2. `StepSnapshot` construction and `StepConfigUniverse` inheritance.
3. function-pattern normalization and invocation identity.
4. invocation contract provider selection.
5. reverse and forward artifact planning.
6. source-binding default resolution and step overrides.
7. callable artifact and special-I/O declarations.
8. CellProfiler callable loading and module registry resolution.
9. CellProfiler symbol-table and generated-contract projection.
10. runtime callable construction and module execution.
11. pycodification and FunctionStep transport.
12. benchmark and UI ZMQ compile/execution submission.
13. official benchmark configuration and Napari inheritance.

The audit searched the existing nominal ownership patterns before accepting a
new abstraction:

- `__registry__`
- `AutoRegisterMeta`
- `RegistryConfig`
- `RegistryFamily`
- `MostDerivedContextStrategyMixin`
- `NominalTypeKeyedStrategyMixin`
- `EnumKeyedStrategyMixin`

The repository already uses those patterns for artifact types, module
declarations, runtime value strategies, output recording, measurement behavior,
streaming configuration, and source-domain behavior. The corrected plan follows
the same pattern: leaf behavior stays on a nominal declaration, and generic
consumers resolve through the registered root or an existing strategy mixin.

An AST inventory over `openhcs/**/*.py` confirmed:

- 122 production `special_inputs` or `special_outputs` decorator applications,
  all under CellProfiler backend or interop paths;
- exactly two `RuntimeInvocationOptions` subclasses, namely
  `DefineGridInvocationOptions` and `CalculateMathInvocationOptions`;
- CellProfiler module classes already compose policy mixins through their MRO.
- all 97 Python files under `benchmark/cellprofiler_library` contain only
  imports, docstrings, and `__all__` assignments; none owns an implementation;
- `PipelineStepsCarrier.__registry__` and
  `OpenHCSExecutionConfigCarrier.__registry__` have no consumer;
- `SourceFilterCriteriaParser`, `SourceBindingMatchMetadataParser`,
  `SourceBindingOriginPolicy`, and `CellProfilerDebugView` each have one
  production leaf selected through a constant/default key.
- 102 CellProfiler interop dataclasses declare one local instance field. The
  nominal-type review separates inherited request/strategy/cache identities
  from classes whose only behavior is a scalar projection of that field.

These results support retaining special I/O as callable ABI, deleting its CP
semantic mirror, removing the two CP invocation-option containers, deleting
compatibility facade packages, and replacing one-leaf registries/carriers with
direct calls and existing values. The one-field review removes projection
wrappers while retaining nominal types whose class identity participates in
MRO dispatch, a cache namespace, request composition, validation, or a
multi-operation service boundary.

## Current Generic Authorities

### Artifact Model

Evidence:

- `openhcs/core/artifacts.py::ArtifactType`
- `openhcs/core/artifacts.py::ArtifactSpecRef`
- `openhcs/core/artifacts.py::ArtifactSpecRelation`
- `openhcs/core/artifacts.py::ArtifactSpec`
- `openhcs/core/artifacts.py::ArtifactSpecCollection`
- `openhcs/core/module_artifact_contract.py::ModuleArtifactContractItem`
- `openhcs/core/module_artifact_contract.py::ModuleArtifactContract`

Finding:

These types already represent every field assigned to the proposed port
classes: artifact kind, name, input/output plan, source/runtime/recorded/declared
partition, materialization, relation, requiredness, and collection lookup. A
port hierarchy would duplicate both data and behavior.

Decision:

Keep the existing artifact types. Remove all proposed artifact-port types.

### Compiler Artifact Flow

Evidence:

- `openhcs/core/pipeline/artifact_planning.py::ArtifactGraph`
- `openhcs/core/pipeline/artifact_planning.py::ArtifactSpecAccumulator`
- `openhcs/core/pipeline/artifact_planning.py::extract_artifact_declarations`
- `openhcs/core/pipeline/path_planner.py::PipelinePathPlanner`
- `PipelinePathPlanner.declared`

Finding:

The compiler already owns an ordered artifact graph and the declarations
available at each step. The proposed generic compile-time flow repeated that
state under a different type and required synchronization.

Decision:

Use a transient `ArtifactSpecCollection` during the provider's forward prepass.
Advance it from existing native invocation declarations and CP module contracts,
discard it after exact plans are built, and leave artifact storage planning to
`PipelinePathPlanner.declared`. Remove the proposed flow and
`CellProfilerCompileTimeArtifactFlow`.

### Invocation Contract Extension Point

Evidence:

- `openhcs/core/invocation_artifacts.py::ArtifactDeclarationStepContext`
- `InvocationContractPlan`
- `InvocationContractProviderFactory`
- `PipelineInvocationContractProviderAuthority`
- `openhcs/core/function_patterns.py::FunctionInvocationKey`
- `NormalizedFunctionItem`
- `CompiledFunctionInvocation`

Finding:

The compiler already exposes a registered provider boundary for replacing a
public callable with a compile-only contract. CellProfiler does not need a
hidden FunctionStep field or generated sidecar. The provider must precompute
ordered contracts once because the path planner queries it in reverse and
forward passes. The current composite returns the first claim in registry
order, so the generic authority does not yet enforce unique ownership.

Decision:

Keep the provider factory. Replace its implementation with one exact,
session-scoped immutable provider keyed by step index and invocation key, and
make the generic composite reject multiple claims. Strengthen the generic ABI
to nominal `InvocationContractProvider` instances with concrete normalized
invocation/session types; delete the callable alias and default provider
function, with an empty composite representing no claim.

### Config Inheritance

Evidence:

- `openhcs/core/pipeline/step_config_universe.py::StepConfigUniverse`
- `StepConfigUniverse.runtime_parameter_bindings`
- `openhcs/core/source_bindings.py::resolve_effective_step_source_bindings`
- `StepSourceBindingsConfig.can_inherit_from`
- `openhcs/core/config.py::GlobalPipelineConfig`
- `StreamingConfig.__registry__`
- `NapariStreamingConfig`

Finding:

The existing configuration hierarchy already handles global, pipeline, and
step values. It also exposes callable runtime-config injection. CellProfiler
does not require a new config for invocation identity, source names, or output
names. Napari test configuration enters the benchmark through the existing
`GlobalPipelineConfig` accepted by `run_comparison_suite`. The two listed
source-binding helpers manually enumerate only the current source fields and
therefore bypass that generic inheritance for every added field; their
equivalence test also omits activation.

Decision:

Use existing lazy configs and inheritance. Delete the manual source overlay and
equivalence helpers; consume the resolved `StepSourceBindingsConfig` from
`StepConfigUniverse` and derive only source activation from existing processing
semantics. Add no CP public config and no benchmark viewer flag.

Move the component-key extraction in
`PathPlannerExecutionGroups.source_binding_scope_for_group_by` to a shared
`source_binding_group_keys_for_group_by` function in
`openhcs/core/source_bindings.py`. The path planner and CP provider use that one
generic function. This makes a default callable follow the same source group
routing as native OpenHCS without a generated dict pattern.

Delete `bindings_for_group_key` / `for_group_key`; they erase the grouping
component and return a sole binding as a fallback. Change
`bindings_for_component_group` / `for_component_group` to exact scoped
selection with a no-match error. The unscoped `(None, None)` operation alone
returns the full declaration set, and the shared group-key function determines
whether a source-derived split exists.

### Special I/O

Evidence:

- `openhcs/core/pipeline/function_contracts.py::special_inputs`
- `special_outputs`
- `special_input_names_from_callable`
- `special_output_specs_from_callable`
- `openhcs/core/special_outputs.py::SpecialOutputKindClassifier`
- `openhcs/core/runtime_output_matching.py::RuntimeReturnedOutputMatcher`

Finding:

The special-I/O decorators provide real Python ABI information that the module
artifact contract does not contain: which named parameters receive non-main-flow
runtime artifacts and which returned positions are special outputs. The current
classifier then overreaches by deriving semantic artifact kind from output names
and materializers. CellProfiler also mirrors special outputs in
`artifact_semantics.py`.

Decision:

Keep special-I/O declarations as ABI. Read artifact semantics from the compiled
module contract. Migrate registered CP callables to slot-only
`special_outputs` declarations and delete semantic name/materializer
classification, decorator materialization tuples, and the CP mirror after
compile-time slot validation is implemented. Native callables retain generic
decorator materialization support.

### Public ZMQ Execution

Evidence:

- `openhcs/core/function_step_transport.py::FunctionStepTransportAuthority`
- `openhcs/runtime/zmq_execution_client.py::OpenHCSExecutionSubmission`
- `PycodifiedPipelineCode.from_task`
- `benchmark/adapters/openhcs.py::_execute_pipeline_via_zmq_server`
- `benchmark/cellprofiler_comparison.py::run_comparison_suite`

Finding:

The benchmark already has the correct outer execution shape: normalize public
steps, pycodify, submit compilation, wait for a compile artifact, submit
execution, and validate server observations. Its weakness is upstream: prepared
steps can already contain runtime wrappers and hidden contracts. The UI and
benchmark diverge whenever those hidden values differ despite identical visible
source.

Decision:

Retain the ZMQ sequence. Remove hidden prepared state and compare the submitted
public source between UI-equivalent and benchmark requests. Resolve pipeline
code directly on `OpenHCSExecutionSubmission`; remove the one-field pycodified
source wrappers without changing the wire payload.

### Benchmark Result Caches Bypass Execution

Evidence:

- `benchmark/adapters/openhcs.py::OpenHCSRunRequest.runtime_execution_cache_manifest`
- `runtime_execution_cache_key` and `reuse_runtime_execution_cache`, whose
  default is true
- `RuntimeExecutionCacheWritePolicy`
- `_RuntimeExecutionCacheHit`
- `OpenHCSAdapter._load_runtime_execution_cache`
- the cache-hit branch in `_run_converted_cppipe_pipeline`, which records zero
  seconds for `COMPILE_OPENHCS` and `EXECUTE_OPENHCS`
- `reused_runtime_execution_cache` parity/provenance handling in
  `benchmark/cellprofiler_comparison.py`
- `benchmark/runner.py::_run_or_reuse_cached_openhcs`
- `_cached_benchmark_result`, `_write_benchmark_result_cache`, and the nested
  OpenHCS execution-cache manifest injection
- `reuse_openhcs_cache`, which defaults to true, and the inverse CLI
  `force_openhcs_run` switch
- `reused_cached_output` parity/timing handling
- `CachedNativeReferenceTimingPolicy`, which substitutes
  `cellprofiler_timeout_seconds` for a missing measured native duration

Finding:

Either cache hit bypasses public-source pycodification, ZMQ compilation, server
reconstruction, execution, and current runtime validation. The benchmark can
therefore report a prior OpenHCS result while the UI executes current code and
fails. The zero-duration timing records also make per-pipeline speed statistics
non-execution measurements. Cache-key completeness cannot repair the acceptance
boundary because the accepted event is current compile-then-execute, not reuse
of a prior observation.
Substituting a timeout for missing native timing fabricates a lower bound as an
exact duration and inflates reported speedup. A persisted output is valid parity
evidence but not timing evidence without its measured phase record.

Decision:

Delete both OpenHCS result-cache layers from the benchmark runner, adapter,
comparison DTOs, CLI parameters, manifests, provenance, and tests. Persisted
native CellProfiler references remain the comparison baseline. Every OpenHCS
parity and timing sample starts from deterministic public source, submits ZMQ
compilation, and executes the resulting current compile artifact. Compilation
artifacts remain scoped to that one submission/execution pair and are not a
cross-run result cache.
Delete `CachedNativeReferenceTimingPolicy`. Persist actual native execution
timing beside each reference; report speedup only for references carrying that
measurement. Missing native timing produces a null speedup and fails a speed
gate without affecting parity status.

## Current CellProfiler Parallel Authorities

### Symbol Table

Evidence:

- `openhcs/interop/cellprofiler/symbol_table.py::CellProfilerSymbolKey`
- `CellProfilerSymbol`
- `ModuleArtifactContracts`
- `CellProfilerSymbolTable`
- `_SymbolTableBuilder`

Finding:

The symbol table creates CP-specific identities and availability state for data
already represented by `ArtifactSpec`, `ArtifactSpecRef`,
`ArtifactSpecCollection`, `ModuleArtifactContract`, and `ArtifactGraph`. It also
forces module-number identity into later matching.

Decision:

Remove the file. Use ordered existing contracts and artifact collections.

### Capability Product Hierarchy

Evidence:

- `openhcs/interop/cellprofiler/module_declarations.py::CellProfilerArtifactCapability`
- its image/object/measurement/relationship/grid input and output subclasses
- module declaration `artifact_contract`, `artifact_contract_inputs`, and
  `artifact_contract_outputs` methods

Finding:

The capability classes describe combinations already encoded by artifact type
plus contract partition. The module classes already contain the leaf rules that
produce those combinations. Keeping both creates two declaration surfaces.

Decision:

Remove the capability hierarchy. Module classes directly construct existing
contract items.

### Setup Module Registry Split

Evidence:

- `openhcs/interop/cellprofiler/source_schema.py::SetupModuleCompiler`
- `ImagesModuleCompiler`, `LoadImagesModuleCompiler`,
  `MetadataModuleCompiler`, `NamesAndTypesModuleCompiler`, and
  `GroupsModuleCompiler`
- `NamesAndTypesAssignmentLayout`,
  `NAMES_AND_TYPES_ASSIGNMENT_LAYOUTS`, and
  `NamesAndTypesAssignmentBlockStrategy`
- `SourceImageStackPlanDeclaration`
- `openhcs/core/pipeline_image_schema.py::ImageTypeSourceRole`, its generated
  roots/leaves, and both class-spec factories
- `openhcs/core/source_image_semantics.py::SourceImagePayloadSemantics` and
  `SourceImagePayloadRoleStrategy`
- `SOURCE_IMAGE_TYPE_METADATA_FIELD` / `OpenHCSImageType`
- `openhcs/core/source_projection.py::SourcePlaneProjection.image_type`
- `InfrastructureCellProfilerModule`
- `infrastructure_import_note`, `infrastructure_exports_tables`, and
  `infrastructure_exports_images`
- `SaveImagesModule.infrastructure_retained_artifacts`
- pass-through declarations for `LoadData`, `LabelImages`,
  `CreateBatchFiles`, and `SaveCroppedObjects`
- `openhcs/interop/cellprofiler/module_roles.py::cellprofiler_module_role`
- `openhcs/interop/cellprofiler/module_semantics.py::_declared_semantics`
- `CellProfilerModuleSemantics`, `CellProfilerModuleSemanticTraits`, and
  `CellProfilerModuleSemanticFamily`
- `cellprofiler_module_semantics` and
  `cellprofiler_module_semantics_family`
- `openhcs/interop/cellprofiler/module_declarations.py::CellProfilerModule`

Finding:

Setup modules are CellProfiler modules but live in a second registry.
`module_roles.py` and `module_semantics.py` explicitly union that registry with
`CellProfilerModule`, proving that neither root is authoritative alone.
`SourceImageStackPlanDeclaration` adds a third registry for one
NamesAndTypes-only rule. The setup compiler methods already lower into the
core `PipelineImageSchemaBuilder`; they require no separate contract model.
That schema is itself a parallel public-state problem:
`PipelineImageSchemaSourceBindingsRepresentability` admits that
`SourceBindingsConfig` cannot represent embedded planes, imported metadata,
source-stack components, voxel spacing, grouping, and typed payload loading.
Generated/import results and `SourceBindingContext` therefore retain schema
state outside `PipelineConfig`, so config plus FunctionSteps cannot recreate the
workspace.
The image-type hierarchy also escapes its importer responsibility. Setup
lowering writes a CP image-type label into generic source projection metadata;
generic function runtime, source-bound runtime, the CP adapter, payload
alignment, pure-2D output aggregation, and color conversion resolve that label
back into generated role classes. This is hidden importer vocabulary in runtime
values. The actual generic facts are source loading transforms, artifact kind,
mask creation, and an internal source channel axis. Existing
`NamedSourceBinding` and `ImagePayloadMetadata` are the declaration and runtime
owners of those facts.
`NamesAndTypesAssignmentLayout` adds a four-entry priority table to infer a
repeated-setting layout even though `ModuleBlock` preserves ordered setting
records and the module declaration knows the assignment setting families. The
table is parser control flow disguised as declarations and creates another
ordering authority.
The workspace also writes auxiliary NumPy payloads and then duplicates them in
`_AUXILIARY_PAYLOAD_CACHE`; runtime source loading checks that process-local
cache before a disk-specific `np.load` fallback, bypassing the generic
FileManager path.
The broad infrastructure root independently hides unrelated enabled modules:
official30 contains ExportToSpreadsheet in 21 cases, ExportToDatabase in 3,
and SaveImages in 16, yet the root declares no callable and later validation
reconstructs export expectations from `.cppipe` flags. Four other leaves are
pass-through placeholders rather than equivalent OpenHCS behavior. This makes
pipeline import, rather than public FunctionSteps, the authority for observable
effects. The three ExportToDatabase cases in
`benchmark/manifests/official30_portable_axis1.json` are all `value_only`; the
native snapshot reader in `openhcs/core/equivalence/outputs.py` reads CSV and
image files but not SQLite databases or CPA properties files. Those cases can
therefore report parity without observing the database export at all.

The existing materialization system also proves that the three exporter kinds
cannot be collapsed into one axis-local rule. `SaveImages` materializes one
selected image in the current axis. ExportToSpreadsheet and ExportToDatabase
consume measurement and relationship artifacts from every compiled axis and
emit one plate-level file set. `AnalysisConsolidationPlan` is the existing
post-plate lifecycle owner, but it currently scans CSV directories from config
and cannot execute a FunctionStep over exact contract-selected runtime
artifacts. `CellProfilerAnalystProjectionBuilder` can construct CPA rows from
runtime stores, while no current code writes its projection to SQLite.

Decision:

Register all five setup modules as
`SourceSetupCellProfilerModule` leaves under `CellProfilerModule`. Put each
existing lowering body on its leaf's `contribute_source_bindings` method and
put the 3D stack rule on `NamesAndTypes`. Extend the existing
`SourceBindingsConfig` and `NamedSourceBinding` with the missing generic source
facts, fold setup modules directly into that config, and migrate workspace/UI
consumers to it. Delete `PipelineImageSchema`, its builder, assignment and
one-field wrappers, representability/projection layer, retained result/context
fields, and the setup and stack-plan
registries, and let the module MRO decide executable-step inclusion and manual
semantic category. Delete `InfrastructureCellProfilerModule`. Make
auxiliary workspace files and their persisted mapping authoritative: delete the
process cache and load every backend through `FileManager`. Make
source payload loading resolve the persisted source alias to the submitted
`NamedSourceBinding`, apply its explicit loading fields, and carry only the
resolved source channel axis on `ImagePayloadMetadata`. Delete the image-type
metadata token, runtime role hierarchy, role strategies, and projection field.
Represent external CP image types and source-filter literals with compact
CP-local enums used only during setup lowering. Parse repeated NamesAndTypes
and LoadImages settings as indexed columns with exact cardinality validation;
delete the four-layout priority table. Make
SaveImages an ordinary axis-scoped executable declaration whose contract
materializes its selected image through generic image-file writer options.
Make ExportToSpreadsheet and ExportToDatabase plate-scoped FunctionSteps using
one generic callable execution-scope declaration. Their contracts enumerate
the exact measurement, relationship, and image artifacts they consume; the
post-plate executor builds one generic artifact batch from those plans and
invokes each public callable once. The callables return generic file bundles,
which standard materialization writes. CP leaves own CSV/CPA projection and
SQLite rendering only. Remove unsupported pass-through declarations and
callables so their enabled modules fail import. Delete exporter flags and
derive ZMQ export expectations only from compiled output plans. Add exporter
acceptance that inspects SQLite tables and CPA properties rather than relying
on the current value-only snapshot. Delete
`module_semantics.py`; its only consumer is the compatibility report, which can
read declaration fields directly and does not need a synthetic semantic-family
classification.

Use the existing lazy config MRO for every added source field. Remove the four
manually redeclared step fields, per-field `inherits_*` predicates,
`resolved_against`, `can_inherit_from`, and both manual source-binding overlay
functions. `StepConfigUniverse` supplies the one resolved
`StepSourceBindingsConfig`; compilation derives activation only from its
`enabled` value and `ProcessingConfig.input_source`. Import decides whether to
emit a step override by resolving candidate and inherited-baseline steps under
the existing nested ObjectState config contexts, then comparing complete
resolved source dataclasses and compiled source-binding plans. It maintains no
second field list. Replace untyped group-key scoping with exact component plus
key selection and reject a scoped no-match.

### Source Workspace Selection And Format Mirrors

Evidence:

- `openhcs/core/source_schema_workspace.py::SourceSchemaImageSetSelection`
- `materialize_source_schema_workspace` image-set-count and selection arguments
- `benchmark/adapters/openhcs.py::source_schema_image_set_selection`
- the same field on CellProfiler comparison, CLI, ingestion, adapter, and test
  requests
- `openhcs/core/config.py::WellFilterConfig`
- `openhcs/core/source_schema_workspace.py::SourceSchemaCandidateProvider`,
  `SourceSchemaCandidateDiscovery`, viability/probe records, and provider MRO
  traversal
- `ImagePlaneSourceResolutionStage` and the three ordered resolver leaves
- `SourceSchemaSourcePlaneInventory`, `TiffPageSourcePlaneInventory`,
  `TiffSourcePlaneInventory`, and the suffix/page-count path
- `SourceSchemaAuxiliaryMaterializer`, `NumpyAuxiliaryMaterializer`, and the
  one-leaf target-path policy
- `openhcs/core/source_projection.py::SourcePixelRef`
- `external/PolyStore/src/polystore/virtual_workspace.py::_payload_plane`
- `external/PolyStore/src/polystore/bioformats_storage.py::BioFormatsPlaneRef`
- `BioFormatsStorageBackend._load_mapping`, `_resolve_ref`, and
  `_load_npy_plane`
- `openhcs/microscopes/bioformats.py::BioFormatsHandler.get_primary_backend`
- `SourcePixelRef.to_legacy_workspace_mapping`
- `VirtualWorkspaceSourceRefResolver`, `PathSourceRefResolver`,
  `DiskSourceRefResolver`, `VirtualWorkspaceResolvedRef`, and the resolver
  `accepts` / `priority` traversal
- string workspace mappings written by BBBC, ImageXpress, and Opera Phenix
- `FileManagerLike.load` and `ImageArrayShapeSemantics`
- `openhcs/core/source_bindings_view.py::SourceBindingsViewModel.from_schema_and_bindings`
- `SourceBindingsPreview.from_schema_and_bindings` and
  `SourceInventoryProvider.inventory(schema=...)`
- `SchemaContextSourceInventoryProvider` and its explicit-plane, local,
  filemanager, and existing-workspace provider delegates; the provider registry
  itself has no lookup consumer
- `CellProfilerSourceRootResolver`, `CellProfilerSourceRootCandidate`,
  `CellProfilerSourcePathAdmission`, and the source-path exclusion registry

Finding:

`SourceSchemaImageSetSelection` is a second execution-selection configuration
used before canonical OpenHCS well identities exist. The benchmark transports
it through private adapter fields while UI execution uses `PipelineConfig` and
the compiler's existing `WellFilterConfig`. The two surfaces can therefore
materialize or execute different image universes from the same public source.
This violates the canonical ZMQ parity boundary.

Candidate discovery then retries registered providers, probes whether their
results assemble, and accepts the first usable result. Explicit image-plane
resolution repeats the same shape as a staged local-equivalent/file/remote
chain. Both are hidden source-root decisions outside submitted config. Existing
workspace metadata detection already happens in `SourceBindingsHandler`, and
the plate path plus `SourceBindingsConfig` already identify the source universe.

The ingestion resolver separately searches selected and ancestor roots based
on `.cppipe` placement, recursively buckets child directories, applies a
registered exclusion chain, probes candidate usability, and retains a fallback
candidate. Those are hidden source-universe choices absent from the submitted
config and directly contradict exact-root workspace construction.

Source-plane expansion is selected by TIFF suffix and page count. The actual
semantic fact is a declared source stack axis, not a storage format. The VFS
already loads supported formats and the virtual workspace already projects an
indexed source payload. NPY and future image formats therefore need no parallel
inventory leaf. The NumPy auxiliary materializer compounds this by decoding one
format in workspace code, copying it, and caching it process-locally instead of
retaining the original VFS source ref.

The virtual-workspace boundary also accepts two representations of one source
reference. New projection code serializes a structured `SourcePixelRef`, while
three microscope handlers still write path strings. Resolver selection sorts
registered classes and asks each class to recognize the value shape, although
the structured value already carries its backend key. This is compatibility
dispatch over duplicated representations, not backend polymorphism. Removing
the string representation leaves only the disk resolver, while the existing
FileManager backend registry already owns backend selection.

The current candidate also stores a physical path and plane coordinates, then
`SourceBindingCandidateSourceRef` reconstructs a disk-only source ref. A
context-wide backend cannot represent mixed source refs and makes the
candidate's loading identity dependent on its construction path.

The structured ref still mirrors Bio-Formats fields in generic source
projection. Reader, series/plane indices, and C/Z/T coordinates are not generic
virtual-workspace semantics. `BioFormatsPlaneRef` already owns that backend
address, while `BioFormatsStorageBackend` separately rereads the same workspace
mapping and switches to an NPY loader. The handler then bypasses the shared
virtual-workspace backend. Bio-Formats consequently has two mapping owners and
a fixture-format branch inside the wrong storage backend.

The same file contains request/resolver/projection dataclasses whose sole
operation is immediately invoked by one caller. They do not preserve an
independent invariant and obscure the deterministic candidate-to-projection
flow.
The source editor and preview preserve the split by accepting both schema and
binding values, rendering schema assignments through a second view constructor,
and retaining a payload-type string absent from generic bindings. UI preview is
therefore another authority capable of drifting from workspace materialization.
Every UI caller constructs `SchemaContextSourceInventoryProvider`, which then
chooses another provider from fields already present on the context. The
registry is unused and the provider chain carries no behavior unavailable to
`SourceInventory.from_filemanager`.

Decision:

Delete private source-image-set selection and translate benchmark scoping into
the inherited existing `WellFilterConfig` already carried by
`run_comparison_suite.openhcs_global_config`. Both adapters resolve that same
config, and source-schema-specific well/max CLI and manifest fields disappear.
Materialize the complete config-declared workspace, then let generic
compilation select axes. Consolidate the
two workspace files under the existing `SourceBindingWorkspaceProjector` and
retain only its materialization result plus private validated candidate,
image-set, and imported-metadata index records.

Delete provider/probe and image-plane resolver priority chains. Use the exact
plate/VFS file list and resolved `ImagePlaneSource` paths from config; existing
OpenHCS workspace detection remains in `SourceBindingsHandler`. Resolve URI
schemes during import and fail unsupported schemes there. File URIs resolve
against the exact source root; HTTP(S) responses are atomically
content-addressed below the existing `get_openhcs_cache_dir()` authority's
`source_imports` child and the resolved local path is stored in config.
Add no cache config.

Delete the CellProfiler source-root resolver, candidate/resolved-root records,
path admission/context records, exclusion root, and both leaves. The import
request's source root is exactly `cppipe_path.parent` in `cppipe_backend` and
must exist there; no
ancestor search, child bucket, `.cppipe` placement rule, usability probe, or
fallback root remains.

Drive file-internal source expansion from
`SourceBindingsConfig.source_stack_components`. Load through `FileManager`,
validate shape through `ImageArrayShapeSemantics`, and extend the existing
`SourcePixelRef` with ordered leading source-axis indices. Move that nominal
storage reference to PolyStore and delete every TIFF-specific inventory and
auxiliary format registry.
Require every workspace writer to emit a structured `SourcePixelRef`, rename
its serializer to `to_workspace_mapping`, add the inverse validating parser,
reduce its exact fields to backend, opaque backend address, and ordered source
axis indices, and remove path-string mappings and old generic
reader/path/series/plane/C/Z/T fields in the same phase. Delete both
one-operation workspace-mapping projection helpers and the ref-to-metadata
address copy; the mapping remains the sole storage address. Make retained
`SourceCandidate` own that exact `SourcePixelRef`, relative filter identity,
metadata, source-axis shape, and filter paths; remove candidate path and
duplicate axis-index fields. Discovery constructs or preserves the per-source
ref, axis expansion replaces its index tuple, and workspace writing consumes
it directly.

Make `FileManager` copy and reconstruct an execution-local registry, add its
single `register_backend` mutation operation, and bind that registry to
backends through a polymorphic `BackendBase.bind_registry` hook. Remove global
`storage_registry` replacement from `FileManager.__setstate__`. Give
`VirtualWorkspaceBackend` the bound registry, select `SourcePixelRef.backend`
by one exact lookup, and let `DataSource.resolve_address` dispatch opaque
address interpretation; only the disk leaf resolves a relative address against
the plate root. Apply source-axis projection once. Delete the full source-ref resolver family,
both leaves, resolved-ref wrapper, shape predicates, priority sort, and resolver
batching. Do not retain an alias, compatibility reader, migration facade, or
dual-format interval.
Make `BioFormatsPlaneRef` own only the physical path, series index, and plane
index and provide one canonical backend-address encoding. Make
`BioFormatsStorageBackend` load that address directly. Delete its workspace
mapping/cache/listing role, reader switch, NPY loader, and Bio-Formats-only
handler backend override. Register the direct reader plus the shared
`VirtualWorkspaceBackend`; encode NPY test sources as disk refs with generic
axis indices.
Keep image-set assembly keyed by `SourceBindingMatchMethod` and replace
component projection with one exact
`SourceComponentProjectionStrategy(EnumKeyedStrategyMixin[AllComponents])`
family in `source_binding_workspace.py`. The root exposes only
`project_component` and `metadata_component`; one leaf per current component
owns all aliases, compound parsing, and its deterministic default. Registry
keys must equal `set(AllComponents)`. Remove direct root metadata probing,
multi-leaf priority traversal, and the separate channel projection wrapper.
Make view, preview, and inventory APIs consume the same pipeline
`SourceBindingsConfig` plus resolved step override; render every source through
`NamedSourceBinding` and remove schema-assignment/payload-type projections.
Store the actual FileManager/backend/config on `SourceBindingContext`, call
`SourceInventory.from_filemanager` directly, and delete the entire inventory
provider hierarchy plus its request wrapper.

### Compile-Time Request And Flow

Evidence:

- `CellProfilerCompileTimeArtifactFlow`
- `CellProfilerCompileTimeSettingsRequest`
- `openhcs/interop/cellprofiler/compile_time_contracts.py::_module_items_from_session`
- `_artifact_flow_from_source_bindings`
- `_module_group_keys_for_item`
- `_update_artifact_flow`
- the `compile_time_public_*` and `compile_time_*consumed*` method family

Finding:

The request duplicates invocation, step, source-binding, and artifact-flow
state. The flow tracks image names separately from compiler artifacts. The
provider then compiles a symbol table, calls the generator's runtime projector,
and invokes generated contract matching. The compile-time setting-name methods
also predict which kwargs reconstruction consumes, duplicating the setting
attributes and contract code on the same module leaves. This is the central
indirection chain.

Decision:

Replace the chain with a provider prepass using current snapshots, normalized
items, resolved bindings, transient ModuleBlocks, module declarations, and
existing artifact collections. `module_blocks_for_invocation` returns the exact
sparse identity keys it actually consumed while reconstructing settings. The
existing `SettingToKeywordBinding` sequence is partitioned by canonical raw
signature membership, so the same setting binding owns parser conversion and
compiler reconstruction without a role flag or second list;
`InvocationContractPlan` removes only those keys. Delete the entire
`CellProfilerModule.compile_time_*` MRO API: row reconstruction moves to
`module_blocks_for_invocation`, contract/output decisions move to
`artifact_contract`, flow mutation moves to the provider's artifact cursors,
and leaf compound-setting parsers become private helpers. Derive grouping from
exact reconstructed contract equality.
Remove normalized-name signature inference,
`cellprofiler_source_setting_parameter_mapping`, and
`setting_parameter_aliases`; migrate their genuine mappings to explicit
bindings on the registered module owner. Parsed settings are explicitly bound,
handled by one private compound-row parser, or explicitly ignored. Replace the
coverage enum with exactly `BOUND`, `IGNORED`, and `UNMAPPED`; remove artifact,
caller, infrastructure, and typed-ignore categories. Reconstruct behavior rows
from one raw callable signature with defaults followed by public overrides;
identity rows come only from explicit overrides or module-owned flow. Delete
the cached callable/default lookup methods.

### Generated Contract Matcher

Evidence:

- `openhcs/interop/cellprofiler/runtime/generated_pipeline.py::CellProfilerGeneratedInvocationContractProvider`
- `CellProfilerGeneratedStepContractMatcher`
- `CellProfilerGeneratedStepContract`,
  `CellProfilerGeneratedGroupedStepContract`,
  `CellProfilerGeneratedStepContracts`, and module-number indexes in the same
  file
- `openhcs/interop/cellprofiler/pipeline_generator.py::PipelineGeneratorRuntimeContractProjector`

Finding:

The generator creates a compiled sidecar and runtime code tries to recover the
correct contract by callable name, module number, source alignment, position,
and fallback. That makes imported `.cppipe` state stronger than public
FunctionStep declarations and explains benchmark/UI divergence.

Decision:

Delete generated matching. Implement one frozen, slotted
`CellProfilerInvocationContractProvider` in `compile_time_contracts.py` with an
immutable mapping keyed exactly by `(step_index, FunctionInvocationKey)`. Its
factory precomputes that mapping from public snapshots and rejects duplicate
keys before assignment. The provider returns `None` for non-CP ownership,
validates canonical raw-callable identity for CP ownership, requires the step
index, and performs one fail-loud key lookup.

Change generic `CompositeInvocationContractProvider` from first-non-`None`
registry-order dispatch to unique-claim enforcement. Evaluate every provider,
return zero/one claims as absent/exact, and reject multiple claims with all
claiming provider types in the diagnostic. The compiler owns exact
invocation-key lookup and no registry order owns semantic precedence.

### Source-Binding Contract Guard

Evidence:

- `openhcs/core/artifact_contract_preview.py::SourceBindingRuntimeContractGuard`
- `SourceBindingContractAlignment`
- generated matcher call sites in
  `openhcs/interop/cellprofiler/runtime/generated_pipeline.py`
- UI preview call sites in
  `openhcs/pyqt_gui/widgets/artifact_contract_preview.py`

Finding:

The guard wraps a comparison between source-partition contract specs and
`StepSourceBindingsConfig`, while its diagnostics hardcode CellProfiler. The
expected source specs already belong to `ModuleArtifactContract`. The alignment
result has a distinct UI/compiler reporting role.

Decision:

Move `SourceBindingContractAlignment` to
`openhcs/core/module_artifact_contract.py`, add alignment/projection methods to
`ModuleArtifactContract`, update the UI and compiler to call the contract, and
delete `SourceBindingRuntimeContractGuard`.

### Runtime Policy Registry

Evidence:

- `openhcs/interop/cellprofiler/runtime/policy_registry.py`
- dynamic policy view classes in CP runtime policy modules
- `openhcs/interop/cellprofiler/module_declarations.py::CellProfilerModuleAuthority`
- module declaration MRO mixins for main flow, source behavior, object inputs,
  output behavior, and execution mode

Finding:

Runtime policy registries are keyed by module names and dynamic views copy
attributes from the module declarations. The module MRO already owns the same
behavior.

Decision:

Delete the registry and views. Give `CellProfilerModule` default classmethods
for execution mode, main-flow replacement, image output source/value, object
output context, primary image input, special input binding, dual-scope
measurement, and object input behavior. Existing leaf mixins inherit
`CellProfilerModuleAuthority` and override those methods through normal MRO.
`CellProfilerModuleExecutor` resolves the module type once and dispatches to it
directly.

### Runtime Wrapper Duplication

Evidence:

- `openhcs/interop/cellprofiler/runtime/module_execution.py::CellProfilerRuntimeCallable`
- `CellProfilerGroupedRuntimeCallable`
- `CellProfilerGroupedModuleContracts`
- `CellProfilerModuleContractResolution`
- `_attach_runtime_processing_contract`
- `CellProfilerProcessingContractAuthority` and
  `PROCESSING_CONTRACT_CACHE`
- `cellprofiler_module_callable` and
  `rebuild_cellprofiler_runtime_callable`
- `CellProfilerFunctionReferenceRehydrator`
- `openhcs/core/function_reference_rehydration.py::FunctionReferenceRehydrator`
- `openhcs/core/runtime_adapters.py::RuntimeAdapterSpec`

Finding:

The two callable classes repeat signature projection, metadata attachment,
adapter installation, parameter exclusion, pickling, preparation, and
execution. Grouping changes only contract selection. The one-callable class
then stores declared and nominal processing contracts already owned by the raw
callable's `CallableContract`, while its runtime plan stores `func`,
`function_name`, and `processing_contract` beside the same
`CallableContract`. A CP-only authority adds a second processing-contract cache
and absorbed fallback. `CellProfilerModuleRunRequest` then copies the callable
again and stores `input_image` and `current_image`, even though its sole
construction site assigns the same object to both. The executor caches plans
by callable even though grouped wrappers are removed and one runtime wrapper
has one callable. It also copies canonical module name and primary-image policy
from the contract/module declaration. The compiled-context path must
reconstruct the one runtime callable after generic worker transport resolves a
`FunctionReference`. The compiled `CallableContract` already carries the exact
`RuntimeAdapterSpec` and module contract, so a second supports-based rehydrator
registry duplicates the owner of that reconstruction decision.

Decision:

Use one runtime callable with one aggregate `ModuleArtifactContract`. Combine
transient same-module group contracts before path planning. Generic
`ComponentArtifactPlans.from_step_component` already selects active artifact
plans by execution group before constructing `RuntimeAdapterRequest`, so the CP
executor projects active specs from those plans instead of selecting another
contract. Replace string `artifact_input_keys` / `artifact_output_keys` on
`CompiledFunctionInvocation` with `ArtifactSpecRef` tuples. Join selected plan
specs to contract items through
`ModuleArtifactContract.require_items_for_specs` and require
equality of the complete spec, including partition/plan role, type, name,
relations, and materialization; no name intersection remains. The wrapper
constructs one runtime plan and owns one executor; the
plan stores only resolved `raw_func`, `module_type`, enriched
`callable_contract`; its contract property requires
the module contract from `callable_contract`, and the executor stores only that
plan. Add
`CallableContract.require_processing_contract()` and delete the CP cache,
metadata mutation, old factory, rebuild function, executor plan dictionary,
copied executor fields, `CallableInvocationKwargSpec`, and runtime kwarg
filtering. Add `CallableContract.validate_public_kwargs` and call it from
generic function-pattern compilation after sparse identity keys are consumed;
`CompiledFunctionInvocation` owns the immutable validated kwargs and runtime
forwards them unchanged. Extend the existing `RuntimeAdapterSpec` with one
optional runtime-callable factory and make
`CallableContract.resolve_runtime_callable` invoke it after ordinary direct or
`FunctionReference` resolution. The CellProfiler adapter spec owns the
top-level two-argument wrapper factory. The resulting wrapper is process-local
and never pickled; generic reference plus enriched callable contract are the
only worker identities. Delete
`FunctionReferenceRehydrationRequest`, `FunctionReferenceRehydrator`, and the
CellProfiler leaf; no registry scan or fallback remains. Reduce
`CellProfilerModuleRunRequest` to executor, one image value, adapter, and
kwargs; query plan, callable, and names through the executor.

### Runtime Artifact Binding Projection

Evidence:

- `openhcs/interop/cellprofiler/runtime/artifact_binding.py::RuntimeArtifactBindingScope`
- `RuntimeArtifactInputRequest`
- `RuntimeInputBindingRequestBase`
- `openhcs/core/module_artifact_contract.py::ModuleArtifactContract.external_input_names`
- `ModuleArtifactContract.runtime_input_name_set`

Finding:

`RuntimeArtifactBindingScope` copies source image names, source object names,
and runtime image names from one `ModuleArtifactContract`. It has no independent
state. `RuntimeArtifactInputRequest` then subclasses `ArtifactSpec` and copies
selected fields while omitting relations.

Decision:

Delete the scope. Add a generic partition/ref membership query to the existing
contract. Make the runtime request compose the original `ArtifactSpec`,
aggregate contract, adapter, and current-image context. Keep
`RuntimeImageInputOrigin` as the CellProfiler-specific strategy result, but
derive it from contract partitions instead of a copied name-set object.

### Generator Registry And Stage Mirrors

Evidence:

- `openhcs/interop/cellprofiler/pipeline_generator.py::PipelineGenerator.__init__`
- `PipelineGeneratorRegistryStage`
- `PipelineGeneratorArtifactPruner`
- `PipelineGeneratorRuntimeContractProjector`
- `PipelineGeneratorCodeEmitter`
- `PipelineGeneratorBuildStage`
- `openhcs/interop/cellprofiler/module_declarations.py::CellProfilerModule.__registry__`

Finding:

The generator copies the nominal module registry into `_registry`, supports a
second `contracts.json` authority, and allocates five stage objects whose only
field points back to the generator. Three importer flags prune declared steps or
change artifact materialization before public source reaches the compiler.

Decision:

Resolve module/function ownership through `CellProfilerModule` directly.
Delete the copied registry, absorbed-library fallback, generator-time pruner,
runtime projector, backreference stages, `PipelineGenerator`, and
`GeneratedPipeline`. The direct import operation performs one private lowering
pass, constructs the final import result, and delegates source rendering to
generic pycodify. Generic compilation owns dead-output and materialization
planning; parsed SaveImages becomes an explicit public export step.

### Callable Catalog Ownership Projection

Evidence:

- `openhcs/processing/backends/cellprofiler/__init__.py::CELLPROFILER_MODULE_ATTR`
- `CellProfilerFunctionRuntimeMetadata`
- `CellProfilerFunctionCatalog.runtime_metadata`
- `_make_processing_wrapper`
- `_default_module_names_by_function_name`
- `CellProfilerFunctionReferenceTransportStrategy`
- `FunctionReferenceTransportStrategy`, whose only registered leaf is the CP
  strategy
- `openhcs/processing/backends/cellprofiler/library.py::AbsorbedFunctionMetadata`
- `_absorbed_contracts`, `_absorbed_default_function_contracts`, and
  `_absorbed_function_locations`
- `CellProfilerModule.declared_function_names`

Finding:

The module declaration already owns module name, aliases, primary function,
variants, validation status, and confidence; the raw callable's generic
`CallableContract` already owns processing locality. The catalog
projects those fields into metadata records and dictionaries, then copies
module ownership and processing facts onto an added public wrapper function.
Compiler code reads the copied attribute to recover the declaration it started
from. The wrapper's call body only forwards to the implementation. The CP
FunctionReference strategy then redirects that wrapper through the catalog,
even though each implementation is an importable attribute of its real backend
module and the generic reference authority already supports that shape.

Decision:

Add exact function-name lookup and callable loading to `CellProfilerModule`.
Make the owning declaration import its implementation, construct the
source-declared `CallableContract`, validate generic execution/adapter/special
I/O ABI, and return the same function without mutating its namespace. Delete
the module declaration's duplicate processing-contract field and leaf values.
Make
`CellProfilerInvocationContractProvider` construct one immutable compiled
contract from that source contract, the exact invocation artifact contract,
the declaration's `allowed_group_by`, and the CP `RuntimeAdapterSpec`. Declare
processing mode, special I/O, runtime image execution mode, and plate execution
scope on the raw implementation with generic callable decorators. Delete every
CellProfiler call to `attach_callable_contract_metadata`, every callable
metadata cache, the runtime metadata
DTO, catalog class, CP ownership attribute, wrapper factory, derived
dictionaries, and `library.py`. The backend package lazy attribute hook calls
the declaration and owns no function inventory. Delete the CP FunctionReference
strategy and the resulting empty strategy root. Remove strategy calls from the
generic authority and use its direct registered/importable-callable paths.
Runtime processing-contract lookup consumes the generic `CallableContract`
only.

### Manual Generated Source Model

Evidence:

- `GeneratedStepEmission` and `GeneratedStepEmissionGroup`
- `GeneratedStepSettings`, `GeneratedProcessingConfigShape`, and
  `StepInputSourceLiteral`
- `GeneratedImportCollector`, `python_literal`, and
  `ArtifactContractCommentSection`
- `PipelineGeneratorCodeEmitter`
- `openhcs/serialization/pycodify_formatters.py::FunctionStepFormatter`
- `FunctionStepTransportAuthority`

Finding:

The generator translates parsed modules into a second object model whose only
purpose is to write Python text for values that already have public
FunctionStep, config, enum, dataclass, and pycodify representations. The result
is then imported to recover FunctionSteps. This round trip creates two source
formatters and makes the generated text stronger than the in-memory public
model.

Decision:

Make the direct import pass construct the actual `list[FunctionStep]` and
`PipelineConfig`. Derive source through one method on
`FunctionStepTransportAuthority` using the existing pycodify formatter. Return
the real steps directly from import and use ZMQ as the fresh-source
reconstruction test. Delete all emission, import-collector, literal-renderer,
comment-section, and CP generated-module import types.

### Processing Config Mirror

Evidence:

- `openhcs/interop/cellprofiler/module_processing_components.py::ModuleProcessingComponents`
- `GeneratedGroupByComponentState`
- `GeneratedStepSettings`
- `SourceProcessingAxisRolePolicy`
- `ModuleProcessingScopePolicy`
- `openhcs/interop/cellprofiler/module_declarations.py::CellProfilerModule.resolve_function`
- `CellProfilerModule.resolve_semantic_function`
- `openhcs/processing/backends/cellprofiler/morphology.py::DilateObjectsModule.resolve_semantic_function`
- `RemoveHolesModule.resolve_semantic_function`
- `openhcs/processing/backends/cellprofiler/illumination.py::CorrectIlluminationCalculateModule.processing_components`
- `openhcs/core/config.py::ProcessingConfig`
- `LazyProcessingConfig`

Finding:

The conversion layer represents `variable_components`, `group_by`, and
`input_source` in a custom component object, converts them to Python source
strings, reconstructs enum values from those strings, and then emits the
existing config. The role-policy family has three one-method leaves over three
fields of `SourceProcessingAxisPlan`; the scope-policy family encodes fixed
precedence over request state rather than declaration-owned polymorphism. The
current function selector has two paths: settings-only `resolve_function`
overrides and source-axis-aware `resolve_semantic_function` overrides. Making
the unified selector consume `ProcessingConfig` creates a dependency cycle,
because processing lowering must first read the selected raw callable's
`CallableContract.processing_contract`. The actual source-axis variant owners,
`DilateObjectsModule` and `RemoveHolesModule`, inspect only Z-index presence;
they do not need a completed `ProcessingConfig`. The actual behavior-dependent
processing override, `CorrectIlluminationCalculateModule`, needs typed bound
behavior kwargs after callable selection.

Decision:

Return concrete `ProcessingConfig` from module processing lowering and emit
sparse `LazyProcessingConfig` overrides. Replace role dispatch with named axis
plan methods. Collapse fixed scope precedence into one function over existing
contract/source-binding values. Replace `SourceProcessingAxisPlan.from_schema`
with one constructor over the complete lazy-MRO-resolved
`StepSourceBindingsConfig` and `ModuleArtifactContract`. Compiler snapshots
supply that config through `StepConfigUniverse`; parsed import resolves the
candidate `LazyStepSourceBindingsConfig` directly inside
`objectstate.config_context(pipeline_config)` without a placeholder step.
During parsed import, derive the artifact contract, derive the axis plan, select
the canonical raw callable through one
`resolve_function(module, contract, axis_plan)` MRO method, read its
`CallableContract`, bind typed behavior kwargs, and only then derive
`ProcessingConfig`. Public compilation keeps the callable already named by
`FunctionStep` and never invokes the parsed-module selector. Keep
behavior-specific processing deviations on the owning module MRO. Delete the
request DTO, semantic selector, function-name result DTO, module processing
contract field, and every copied or string-projected processing value in the
same cutover.

### Generated Import And Workspace Wrappers

Evidence:

- `openhcs/interop/cellprofiler/runtime/generated_pipeline.py`
- `GeneratedPipelineModuleIdentity`
- `GeneratedPipelineRuntimeModule`
- `GeneratedPipelineModuleExports`
- `GeneratedFunctionSpec`
- `GeneratedPipelineFunctionRegistration`
- `openhcs/interop/cellprofiler/runtime_pipeline.py`
- `CPPipePipelineArtifact`, `PreparedGeneratedPipeline`, direct execution types
- `openhcs/core/input_workspace.py::InputWorkspacePreparationResult`
- `openhcs/pyqt_gui/widgets/shared/services/cellprofiler_pipeline_rebinding.py`

Finding:

Generated module loading accepts five sidecar arguments and immediately
discards them. It appends import-time function registration even though target
source imports raw declaration-owned callables. A second hand-written parser walks
FunctionStep syntax. Preparation wrappers repeat fields already held by
`GeneratedPipeline`, and `CellProfilerPipelineImportResult` repeats the same
steps/config pair, while the generic workspace result stores another wrapper
as `Any`. The UI runtime-binding service is a no-op compatibility layer. Direct
orchestrator execution remains only as a test path parallel to ZMQ. The import
result also stores generated source text even though public steps plus
`FunctionStepTransportAuthority` determine that text exactly.
`CellProfilerPipelineProvenance` is nested only in that result and has no
independent production consumer after generated/runtime wrappers disappear.

Decision:

Delete CP generated-module import, `PipelineGenerator`, and
`GeneratedPipeline`. The direct import pass constructs the exact public step
list once, saves its generic pycodify projection, and returns those steps on the
sole final import result. Put typed public steps, config, and workspace materialization
on the generic workspace result. Its
`pipeline_steps: list[FunctionStep] | None` and
`pipeline_config: PipelineConfig | None` fields are both present or both
absent, and its materialization field is
`SourceBindingWorkspaceMaterialization | None`. Delete generated identity/loading,
registration/parsing/facade wrappers, retained import-result state, and no-op
rebinding. Store the `.cppipe` path, module-reference tuple, steps, config,
setting-coverage diagnostics, and generated source path directly on the import
result; delete the nested provenance value and expose source text as a derived
property.
Migrate converted integration execution to compile-then-execute over ZMQ and
delete the direct path.

### Compiler And Import Forwarding Stack

Evidence:

- `CellProfilerDialectCompiler`
- `CellProfilerPipelineImporter`
- `CellProfilerGeneratedPipelineDialectCompiler`
- `openhcs/interop/cellprofiler/compiler_registry.py::_REGISTERED_COMPILER`
- `CPPipePipelineGenerationRequest`
- `CPPipePipelinePreparationRequest`
- `CPPipeModulePartition`
- benchmark compiler/importer aliases

Finding:

There is one production `.cppipe` conversion implementation. Two ABCs, a
mutable singleton, aliases, and nested generation/preparation request objects
route to it. Replacing the singleton with an AutoRegisterMeta family would
retain the same one-leaf indirection and add another registry without a
semantic dispatch axis.

Decision:

Keep the validated `CellProfilerPipelineImportRequest` value and replace the
rest with one `import_cellprofiler_pipeline(request)` function in
`import_service.py`. It parses once, resolves every enabled module through the
one module registry, lowers directly to public declarations, writes the derived
source, and returns the sole public import result. Delete `pipeline_compiler.py`,
`compiler_registry.py`, and `runtime_pipeline.py` after callers migrate.

### Generic Pipeline Carrier Stack

Evidence:

- `openhcs/runtime/zmq_pipeline_transport.py::PipelineStepsBoundary`
- `PipelineStepsCarrier`
- `PipelineStepsNamespaceProjection`
- `ZMQPipelineSourcePayload`
- `ZMQPipelineCodeTransport`
- `PycodifiedPipelineStepSource`
- `PycodifiedSource`, `PycodifiedPipelineCode`, and `PycodifiedConfigSource`
- `PycodifyAssignmentSourceRequest`
- `OpenHCSExecutionConfigCarrier`
- `ZMQResolvedConfig`
- agent `PipelineIdBoundary`, `ExecutionPipelinePayload`, and
  `ExecutionPipelineDefinitionProvider` with its two leaves
- direct list inputs already accepted by `OpenHCSExecutionSubmission.__init__`

Finding:

`PipelineStepsBoundary` contains one mutable list and adds no invariant beyond
`list(...)`. The carrier registry has no consumer. Namespace projection only
looks up `pipeline_steps`; the source payload carries a step list that
`ZMQPipelineCodeTransport` ignores while returning its source string.
`OpenHCSExecutionConfigCarrier` repeats the same pattern for an existing config
bundle, and `ZMQResolvedConfig` wraps only that bundle. The four pycodified
source/request values each retain a source string or assignment inputs and
immediately delegate rendering or hashing; the submission, config projection,
and request payload already own those operations. Agent session requests each
allocate a provider whose only method constructs one definition, while the
provider's semantic branch is already represented by the request subtype;
`PipelineIdBoundary` wraps one string without validation.

Decision:

Store direct step lists and config bundles on concrete records. Put the one
pipeline export name, canonical pycodify rendering, and namespace validation
on `FunctionStepTransportAuthority`. Delete the boundary, carrier, projection,
payload, no-op transport, source wrapper, config carrier, and resolved-config
wrapper. Add direct pipeline-code resolution to the submission, pycodify config
values in `ZMQConfigProjection`, and use the request payload's source hash.
Delete every `Pycodified*` class and the assignment-source request without a
replacement carrier. This makes the implementation boundary match the public
API instead of wrapping it at every hop.
Move definition construction onto the existing
`ExecutionPipelineSessionRequest` root and request leaves, retain one concrete
definition with direct ID/steps/source fields, and delete the agent provider,
payload, and ID wrapper families.

### Compatibility Facades And One-Leaf Registries

Evidence:

- all files under `benchmark/cellprofiler_library`
- all files under `benchmark/cellprofiler_compat`
- `benchmark/converter/cppipe_module_roles.py`
- one-time absorber/backfill/fix utilities under `benchmark/converter`
- `openhcs/interop/cellprofiler/thresholding.py`
- `SourceFilterCriteriaParser`, `SourceBindingMatchMetadataParser`, and
  `SourceBindingOriginPolicy`
- `CellProfilerDebugView`
- `CellProfilerSemanticDefaultContract.__registry__`
- `openhcs/interop/cellprofiler/processing_contract_resolution.py`
- `ResolvedProcessingContract` and the one-value
  `ProcessingContractResolutionSource`

Finding:

The two benchmark compatibility packages are import-only mirrors of production
symbols. The absorption utilities target the superseded generated library and
`contracts.json`; no production path invokes them. The threshold compatibility
module forwards to backend functions while injecting benchmark profiling. The
three source-schema registries and debug-view registry each dispatch to one
CellProfiler/default leaf by a constant key. Semantic-default contracts are
already enumerated by each owning module declaration, so their additional root
registry is unused. Processing-contract resolution wraps one generic
`ProcessingContract` with a provenance enum that has only
`CALLABLE_METADATA`; the compatibility report is its only consumer.

Decision:

Delete the compatibility packages, obsolete absorption tools, role alias, and
threshold forwarder; migrate tests and fixture benchmarks to production owner
imports. Replace the single-leaf setup parsing policies with private functions
beside their owning `SourceSetupCellProfilerModule` leaves,
call `DebugViewModel.from_debug_snapshot` directly, and remove AutoRegisterMeta
from semantic-default contracts while preserving module-owned contract tuples.
Delete processing-contract resolution, `CellProfilerModule.contract`, and
every leaf assignment. The loaded callable's generic `CallableContract` is the
sole processing-contract owner; module/compiler/runtime consumers query its
fail-loud `require_processing_contract` method and never compare a second
module value.

### One-Field Projection Wrappers

Evidence:

- `openhcs/interop/cellprofiler/runtime/adapter.py::SourceIdentitySetCardinality`
- `openhcs/interop/cellprofiler/runtime/adapter.py::DeclaredOutputResolution`
- `openhcs/interop/cellprofiler/runtime/binding_authorities.py::CellProfilerOptionalNonemptyString`
- `openhcs/interop/cellprofiler/runtime/source_binding_runtime.py::SourceBindingAxisCardinality`
- `openhcs/interop/cellprofiler/runtime/source_identity.py::SourceImageSetIdentityQuality`
- `openhcs/interop/cellprofiler/runtime/adapter_protocols.py::RequireProcessingContextBoundaryPolicy`
- `openhcs/interop/cellprofiler/runtime/processing_contracts.py::RuntimeShapeInspection` and
  `Pure2DSliceCountCandidate`
- `openhcs/interop/cellprofiler/runtime/execution_mode_policies.py::InvocationSpatialRankCandidates`
- `openhcs/interop/cellprofiler/runtime/runtime_value_authorities.py::DenseLabelShapeSet`,
  `DenseLabelStackRepeatPattern`, `MatlabPayloadEntryName`, and
  `SpatialGridSliceCount`
- `openhcs/interop/cellprofiler/runtime/projection.py::RuntimePlaneSelectedPlaneIndex`
- `openhcs/interop/cellprofiler/runtime/source_candidates.py::CellProfilerImageNumberResolution`
- `openhcs/interop/cellprofiler/runtime/source_binding_runtime.py::SourceBindingPayloadAliasSet` and
  `SourceBindingPayloadComponentMetadata`
- `openhcs/interop/cellprofiler/runtime/runtime_artifact_records.py::RuntimeArtifactRecordDeduplication`
- `openhcs/interop/cellprofiler/runtime/output_value_resolution.py::CellProfilerCallableOutputSpecs`
- `openhcs/interop/cellprofiler/runtime/measurement_rows.py::ObjectLabelFinalLabels`,
  `ObjectLabelSmallRemovedLabels`, and `SparseLabelRowsCoercion`
- `openhcs/interop/cellprofiler/runtime/function_contract_execution.py::Pure2DTraceLabelStats`

Finding:

Each listed class stores one value and exposes only a scalar predicate,
normalization, optional-value projection, or call to an authority that already
exists. No consumer dispatches on the wrapper type. Several duplicate current
owners: `RuntimeShapeInspection` duplicates
`ImageArrayShapeSemantics.shape`; label-variant wrappers duplicate
`ObjectLabelVariantData`; sparse coercion duplicates
`sparse_ijv_rows_from_label_slice`; callable-output projection reclassifies
returned slots already represented by the full ordered declared outputs of
`ModuleArtifactContract`. `CellProfilerPerImageMeasurementPolicy` also
reprojects callable outputs even though its request already contains the
contract output specs.

The AST result does not justify deleting every one-field dataclass. The
following reviewed examples remain:

- `CellProfilerOptionalCurrentImageContext` and
  `CellProfilerRequiredCurrentImageContext` contribute typed inherited fields
  to request records;
- `ProducedArtifactImageMeasurementSource` is a nominal source variant whose
  type selects behavior;
- `StepInputSourcePayloadProcessCache` and
  `PipelineStartSourcePayloadProcessCache` use class identity as independent
  process-cache namespaces;
- `CellProfilerMeasurementImageResolver` owns a cohesive multi-operation
  service over its executor;
- `MeasurementTableCacheMutationPolicy` has two independent concrete policies
  composed by its registry;
- `FilterObjectsMeasurementValuesSource` is an MRO chain with derived,
  table, and relationship leaves;
- `ProjectionStrategy` is an enum-keyed family with distinct projection
  algorithms.

Decision:

Delete the listed projection wrappers. Put fail-loud processing-context access
on `CellProfilerRuntimeAdapter`; put record de-duplication on
`RuntimeArtifactRecordLocationIdentity`; pass the contract's full declared
outputs to the exact matcher; use the existing generic shape, object-label
variant, and sparse-row owners; and inline closed scalar decisions in their
sole consuming method. Move disabled Metadata rule behavior to the Metadata
module declaration. Make per-image measurement policy query its existing
output specs. Retain one-field nominal types only where their class
identity, inherited field contribution, invariant, cache namespace, or
cohesive service surface is itself load-bearing.

### Hidden FunctionStep Contract

Evidence:

- `openhcs/core/function_step_invocation_contracts.py`
- `openhcs/core/steps/function_step.py::FunctionStep.invocation_contracts`
- `openhcs/core/pipeline/step_snapshot.py::StepSnapshot.invocation_contracts`
- pycodify clean-mode handling

Finding:

The field carries compiler output on a public source declaration. It is hidden
in the UI and can differ between code-mode reconstruction and prepared import.

Decision:

Delete the field and file. Derive contracts during compilation.

### Pipeline Metadata And Wrapper

Evidence:

- `openhcs/core/pipeline/compilation_session.py::PipelineMetadataCarrier`
- `PipelineIdentityCarrier`
- `CompilationSession.pipeline_metadata`
- `PIPELINE_SOURCE_SCHEMA_METADATA_KEY`
- `openhcs/core/pipeline/__init__.py::Pipeline`
- `openhcs/pyqt_gui/services/pipeline_object_state_binding.py`

Finding:

Compiler pipeline metadata has no necessary execution consumer. The core
`Pipeline` mixes mutable steps, display text, arbitrary metadata, scope IDs, and
an obsolete config field. The UI needs a reconstructable ObjectState root only
for editor text and child scope IDs; execution uses a list.

Decision:

Delete compiler metadata and both structural carrier protocols. Replace the
core wrapper with plain step lists plus a private, frozen, slotted
`PipelineEditorStateRoot` containing exactly `name`, `description`, and
`step_scope_ids`. `ObjectState` needs that nominal reconstructable parent
because the ordered steps are stored as child states. The root has no steps,
metadata, config, sequence behavior, source, or execution API. Delete the
wrapper's arbitrary metadata, implicit timestamp, embedded config, and
compatibility methods rather than moving them. The compiler's process-local
ObjectState scope uses the existing submitted-list identity branch and does not
participate in artifact or execution identity.

### Hidden Export Modules

Evidence:

- `InfrastructureCellProfilerModule`
- generator handling for skipped SaveImages and other export modules
- `infrastructure_exports_tables` and `infrastructure_exports_images`
- `CPPipeInfrastructureProfile`
- `materialize_skipped_save_images`
- `RuntimeImageExportSpec`, `RuntimeImageExportBitDepth`, and
  `_candidate_image_snapshots_for_equivalence`
- upstream `ArtifactSpec` materialization rewriting
- `materialized_image_artifact_names` and
  `artifact_name_materialized_image_artifact_names` compile-time kwargs and
  metadata keys

Finding:

Skipping observable modules and reconstructing expectations from `.cppipe`
flags hides behavior that cannot be recovered from generated FunctionSteps.
SaveImages additionally mutates an earlier artifact. The hidden materialized
name kwargs then mirror generic artifact materialization in callable metadata.
The current `openhcs/interop/cellprofiler/database_export.py::export_to_database` only
returns `pending_pipeline_export`; `analyst_export.py` builds a render-only CPA
projection and has no SQLite writer. The current
`interop/cellprofiler/spreadsheet_export.py` computes unrelated region
statistics and declares no `export_to_spreadsheet` callable. Calling these
implementations executable would preserve false support.
`RuntimeImageExportSpec.prepare_payload` is a second SaveImages conversion path
used to compare in-memory records when the requested image file was never
written or read. It mirrors bit depth and format settings already owned by the
SaveImages callable/materialization contract.

Decision:

Delete the broad infrastructure root. Generate explicit SaveImages,
ExportToSpreadsheet, and ExportToDatabase FunctionSteps. SaveImages uses the
normal axis-scoped contract and generic image-file materialization. Its
selected image is an existing special input, its converted export copy is an
existing special output, and its first return preserves canonical main flow;
the adapter-free callable therefore runs raw through generic FunctionStep
execution.
ExportToSpreadsheet and ExportToDatabase use a generic plate execution scope,
receive only the artifact records named by their compiled input plans, and
return a generic file bundle for standard materialization. Replace the two
fake export callables with real renderers, add SQLite and CPA-properties
verification, and delete exporter flags, retained-artifact mutation, hidden
materialized-name kwargs, `RuntimeImageExportSpec`, the in-memory candidate
snapshot shortcut, and unsupported pass-through module declarations.

### Plate-Scoped Callable And Writer Gap

Evidence:

- `CallableContract` and `CompiledStepPlan` carry image execution semantics but
  no axis-versus-plate callable lifecycle;
- `worker_execution.py` invokes every FunctionStep once per compiled axis;
- `compiled_plate_execution.py` calls `AnalysisConsolidationPlan` only after all
  worker results return;
- `AnalysisConsolidationPlan` scans materialized CSV directories and has no
  callable or artifact-contract input;
- `RuntimeArtifactExecutionObservation.records_by_axis` proves merged runtime
  records are available in the parent execution boundary;
- `StepInputDependencyKind` represents only pipeline-start and previous-step
  image edges, so it cannot truthfully compile an artifact-only plate callable;
- materialization has registered option-type writers for CSV, JSON, text, ROI,
  and TIFF stacks, but no one-image-path or named-file-bundle writer.
- `MaterializationValue` already admits `bytes`, while
  `DiskStorageBackend.save` rejects every unregistered suffix before examining
  the payload; opaque database bytes therefore cannot traverse the declared
  generic materialization boundary.
- `ImageFileSerializationFormat.for_path` silently falls back to native
  preparation for an unregistered suffix, so it cannot validate SaveImages
  format support; the disk registry has PNG/JPEG/TIFF/NPY writers and no HDF5
  image writer, and no official30 `.cppipe` requests HDF5.

Finding:

Plate-scoped invocation and generic file-bundle persistence are real missing
responsibilities. They are not represented by config inheritance, source
bindings, special I/O, artifact partitions, or the CP module registry. Encoding
them as CP exporter flags, module-name checks, process-global store scans, or
per-axis pass-through calls recreates the boundary violation.

Decision:

Add one generic `FunctionStepExecutionScope` fact to `CallableContract`, one
generic `RuntimeArtifactBatch` containing declared
input specs plus exact contract-selected records, and generic
`ImageFileOptions`/`FileBundleOptions` writer keys in the existing
materialization system. Extend the existing post-plate execution boundary to
invoke plate-scoped plans once; do not add a CP post-run registry or config.
SaveImages remains axis-scoped. Spreadsheet/database callables are
plate-scoped and return file bundles. These four additions each own behavior
not present elsewhere and are the only generic runtime/materialization
additions accepted by this audit; the separate GUI-only
`PipelineEditorStateRoot` remains the sole editor state addition. Add a
fail-loud registered-suffix query to the existing image serialization root;
SaveImages never uses its native fallback.
Declare NPY/TIFF on the native serialization leaf, retain the existing
PNG/JPEG leaves, and reject HDF5 and unknown formats at compilation. Do not add
a CP HDF5 branch without a real generic PolyStore format writer.
Derive a step's one execution scope from the existing compiled invocation
contracts and reject mixed scopes. Do not copy the fact onto
`CompiledStepPlan`.
Place `RuntimeArtifactBatch` in `core/runtime_stores.py`, beside
`StoredRuntimeValue` and `RuntimeArtifactQuery`; placing it in
`runtime_values.py` reverses the current dependency. Its constructor freezes
the input specs and per-axis records, and its queries reject refs absent from
the declared input specs.
Complete the already-declared generic bytes path by making PolyStore's disk
backend write `bytes` directly before extension dispatch. The file-bundle
writer UTF-8 encodes text entries and emits bytes. This adds no carrier or
format table, and keeps every non-byte payload on the existing nominal
`FileFormat` registry.
`RuntimeArtifactBatch` itself implements the existing
`RuntimeParameterDeclaration` protocol and is declared with
`runtime_bound_parameters(RuntimeArtifactBatch)`. The post-plate executor
injects it under its nominal `require_parameter_name()` result; it is neither a
positional main-flow value nor a public kwarg. After each plate callable, the
executor records its output and creates the next immutable records-by-axis map,
so later plate artifacts use normal compiled dependencies without a second
store.
Extend the existing `StepInputDependencyKind` with `NO_MAIN_FLOW` and assign it
to plate plans. This is the compile-time main-input fact owned by the current
dependency model, not another scope descriptor. Path planning produces only an
artifact materialization base for this state and no source or preceding image
load.
Execution scope schedules the callable; it does not duplicate
`ProcessingContract`, which remains the axis-local array contract.

## Proposed Symbol Classification

The table covers every abstraction proposed by the superseded plan and every
current parallel authority it intended to preserve.

`KEEP` denotes either an existing nominal authority retained or extended in
place, or one accepted symbol that owns a demonstrated missing responsibility.
`REPLACE` denotes a complete name, field-shape, or implementation cutover with
no old surface left behind.

| Symbol | Classification | Authoritative replacement |
|---|---|---|
| `ArtifactPort` | REMOVE | `ArtifactSpec` + `ModuleArtifactContractItem` |
| `ArtifactPortDeclaration` | REMOVE | `ArtifactSpec` + `ModuleArtifactContractItem` |
| `ArtifactPortContext` | REMOVE | `ArtifactDeclarationStepContext` |
| `ArtifactNameResolver` | REMOVE | owning `CellProfilerModule` output rule |
| `ArtifactRelationResolver` | REMOVE | `ArtifactSpec.relations` |
| `CompileTimeArtifactFlow` | REMOVE | transient existing `ArtifactSpecCollection` in provider prepass |
| `SourceBoundInputPort` | REMOVE | `SourceArtifactInputPartition` |
| `MainFlowInputPort` | REMOVE | runtime input partition plus module main-flow policy |
| `RuntimeArtifactInputPort` | REMOVE | `RuntimeArtifactInputPartition` |
| `DeclaredArtifactOutputPort` | REMOVE | `DeclaredArtifactOutputPartition` |
| `RecordedArtifactOutputPort` | REMOVE | `RecordedArtifactOutputPartition` |
| `SpecialInputPort` | REMOVE | `special_inputs` ABI + runtime input contract item |
| `SpecialOutputPort` | REMOVE | `special_outputs` ABI + output contract item |
| proposed typed `SpecialInputDeclaration` | REMOVE | existing `special_inputs` |
| proposed typed `SpecialOutputDeclaration` class | REMOVE | existing special-output slot declaration plus contract spec |
| `CellProfilerArtifactSettingDescriptor` | REMOVE | leaf module setting-row conversion |
| `CellProfilerCompileTimeSettingsRequest` | COLLAPSE INTO EXISTING OWNER | normalized item + step context + resolved bindings + artifact collection |
| `ArtifactDeclarationStepContext.source_provenance` | REMOVE | unused never-populated `Any` field |
| `ArtifactDeclarationStepContext.processing_config: Any` | REPLACE | concrete optional `ProcessingConfig` with boundary validation |
| `SettingToKeywordBinding` | KEEP | sole bidirectional setting-to-public-key binding; derive identical keyword names and partition by raw signature membership |
| `CellProfilerModule.module_blocks_for_invocation` / `.artifact_contract` | KEEP | sole module-polymorphic reconstruction and artifact-contract boundary |
| `CellProfilerModule.for_function_name` / `.require_callable` | KEEP | sole callable ownership and exact implementation loading boundary |
| `CellProfilerModule.contract` and leaf assignments | REMOVE | source callable `CallableContract.processing_contract` only |
| `ModuleArtifactContract.combine` and source-binding alignment methods | KEEP | grouped contract union and binding validation on the existing contract owner |
| `ModuleArtifactContract.require_items_for_specs` | KEEP | exact partition/ref/full-spec join on the existing contract owner |
| `CallableContract.require_processing_contract` / `.require_module_artifact_contract` | KEEP | fail-loud queries on the existing compiled callable owner |
| `CallableContract.validate_public_kwargs` | KEEP | one generic compile-time signature validation before immutable invocation construction |
| normalized setting/signature inference and `setting_parameter_aliases` | REMOVE | explicit `SettingToKeywordBinding` values on the owning module declaration |
| compile-time callable/default lookup cache | REMOVE | one raw signature read inside `module_blocks_for_invocation` |
| all `CellProfilerModule.compile_time_*` MRO methods | REMOVE | `module_blocks_for_invocation`, `artifact_contract`, binding partition, and private leaf setting readers |
| `public_artifact_identity_overrides` | REMOVE | exact contract comparison over the module's existing setting bindings |
| `ModuleSettingCoverageStatus` | REPLACE | exact `BOUND`, `IGNORED`, and `UNMAPPED` diagnostic states |
| `ARTIFACT_CONTRACT`, `TYPED_IGNORE`, `CALLER_IGNORE`, and `INFRASTRUCTURE` coverage members | REMOVE | actual binding, module-owned ignore, or fail-loud unmapped result |
| `CellProfilerCompileTimeArtifactFlow` | COLLAPSE INTO EXISTING OWNER | transient existing `ArtifactSpecCollection` in provider prepass |
| `CellProfilerArtifactCapability` hierarchy | COLLAPSE INTO EXISTING OWNER | direct module contract items |
| `CellProfilerInvocationContractProviderFactory` | KEEP | exact session prepass |
| `CellProfilerInvocationContractProvider` | KEEP | frozen immutable exact-key provider in `compile_time_contracts.py` |
| `InvocationContractProvider` / `InvocationContractProviderFactory` | KEEP | nominal typed generic provider and factory boundary |
| `InvocationContractProviderLike` / `public_callable_invocation_contract` | REMOVE | nominal provider instances; empty composite means no claim |
| `CompositeInvocationContractProvider` | REPLACE | unique-claim enforcement independent of registry order |
| `CellProfilerSymbolTable` family | REMOVE | existing artifact graph and contract collections |
| `ModuleArtifactInput` | REMOVE | `ArtifactSpec` input contract item |
| `ArtifactSpecKey` | REMOVE | `ArtifactSpecRef` and collection queries |
| `CellProfilerModuleRole` / `CellProfilerModuleRoleSpec` | REMOVE | raw `ModuleBlock.enabled` provenance plus declaration methods |
| `CellProfilerPipelineProvenance` | REMOVE | direct source path and module-reference fields on final import result |
| module semantics/traits/family DTOs | REMOVE | direct registered declaration fields in compatibility reports |
| `ResolvedProcessingContract` / one-value resolution-source enum | REMOVE | sole source callable `CallableContract.processing_contract` |
| `CellProfilerCallableOutputSpecs` | REMOVE | full ordered declared outputs on compiled `ModuleArtifactContract` |
| `RequireProcessingContextBoundaryPolicy` | REMOVE | `CellProfilerRuntimeAdapter.require_processing_context()` |
| `RuntimeShapeInspection` | REMOVE | generic `ImageArrayShapeSemantics.shape` |
| `RuntimeArtifactRecordDeduplication` | COLLAPSE INTO EXISTING OWNER | `RuntimeArtifactRecordLocationIdentity.unique_records` |
| object-label final/small-removed wrapper classes | REMOVE | `ObjectLabelVariantData` and `LabelPayloadFinalProjection` |
| `SparseLabelRowsCoercion` | REMOVE | `sparse_ijv_rows_from_label_slice` |
| one-field scalar/cardinality/optional projection wrappers | REMOVE | direct decision in the owning consumer |
| one-consumer source-binding alias/component wrappers | REMOVE | `SourceBindingPayloadPlaneResolution` |
| `CellProfilerGeneratedInvocationContractProvider` | REMOVE | exact invocation provider |
| `CellProfilerGeneratedStepContract` / `CellProfilerGeneratedGroupedStepContract` / `CellProfilerGeneratedStepContracts` / `CellProfilerGeneratedStepFunctionSpec` / `CellProfilerGeneratedStepContractMatcher` | REMOVE | immutable exact-key provider plus existing module contract |
| `FunctionStepInvocationContractPayload` / `FunctionStepInvocationContractBinding` | REMOVE | compile provider; no hidden step field |
| `SourceBindingRuntimeContractGuard` | COLLAPSE INTO EXISTING OWNER | alignment methods on `ModuleArtifactContract` |
| `CellProfilerRuntimeCallable` | KEEP | one-contract runtime adapter |
| `CellProfilerModuleRuntimePlan` | REPLACE | exact raw function, module type, and enriched callable contract fields |
| `CallableInvocationKwargSpec` | REMOVE | generic compile-time `CallableContract.validate_public_kwargs` |
| `CompiledFunctionInvocation.artifact_input_keys` / `.artifact_output_keys` | REPLACE | exact `ArtifactSpecRef` tuples and full-spec join validation |
| `CellProfilerModuleExecutor` | REPLACE | one runtime-plan field and declaration-polymorphic behavior |
| `CellProfilerModuleRunRequest` | REPLACE | executor, one image, adapter, and kwargs only |
| `CellProfilerRuntimeAdapter.require_processing_context` | KEEP | fail-loud method on the existing adapter owner |
| `CellProfilerGroupedRuntimeCallable` | REMOVE | generic grouped artifact-plan selection plus single runtime callable |
| `CellProfilerGroupedModuleContracts` | COLLAPSE INTO EXISTING OWNER | `ModuleArtifactContract.combine` |
| `CellProfilerModuleContractResolution` | REMOVE | constructor type validation |
| copied runtime-callable/runtime-plan processing fields | REMOVE | retained `CallableContract` properties |
| executor plan cache and copied contract/name/policy fields | REMOVE | one immutable runtime plan |
| copied run-request plan/callable and duplicate image fields | REMOVE | executor plan plus one invocation image |
| `CellProfilerProcessingContractAuthority` / cache / absorbed fallback | REMOVE | `CallableContract.require_processing_contract()` |
| `cellprofiler_module_callable` / `rebuild_cellprofiler_runtime_callable` | REMOVE | adapter-spec runtime-callable factory plus direct two-argument constructor |
| `CellProfilerRuntimeCallable.__reduce__` / identity equality and hash | REMOVE | process-local wrapper built after generic reference resolution |
| `RuntimeAdapterSpec` | KEEP | exact optional runtime-callable factory after ordinary reference resolution |
| `CellProfilerRuntimeAdapter.runtime_adapter_spec` | KEEP | sole nominal CP adapter declaration consumed by the invocation provider |
| `FunctionReferenceRehydrator` family | REMOVE | runtime-callable factory on the compiled adapter spec |
| `CellProfilerRuntimeCallableFormatter` | REMOVE | raw public callable formatter |
| `RuntimeArtifactBindingScope` | REMOVE | `ModuleArtifactContract` partition/ref query |
| `RuntimeArtifactInputRequest` | REPLACE | compose original spec + contract + runtime context |
| runtime module-name policy registries | COLLAPSE INTO EXISTING OWNER | `CellProfilerModule` MRO |
| `SourceSetupCellProfilerModule` | KEEP | one MRO root on the existing `CellProfilerModule` registry for source-binding-config behavior |
| `InfrastructureCellProfilerModule` | REMOVE | `SourceSetupCellProfilerModule` for setup only; axis- or plate-scoped executable module roots for exporters |
| `RuntimeArtifactLineageScope` | COLLAPSE INTO EXISTING OWNER | `ArtifactSpec.relations` |
| `RuntimeArtifactSourceLineage` | COLLAPSE INTO EXISTING OWNER | relations + source binding |
| `FunctionStepInvocationContracts` | REMOVE | compile provider |
| `PipelineMetadataCarrier` | REMOVE | no replacement |
| `PipelineIdentityCarrier` | REMOVE | no replacement; compiler scope remains process-local |
| core `Pipeline` wrapper | REPLACE | GUI-private `PipelineEditorStateRoot` for editor scopes; list at every execution boundary |
| `PipelineEditorStateRoot` | KEEP | sole GUI-private name/description/step-scope ObjectState root |
| `SpecialOutputKindClassifier` | REMOVE | artifact type on module contract |
| CP `special_outputs` materialization tuples | REMOVE | ordered slot names plus module-contract materialization |
| `RuntimeInvocationOptions` and third-tuple FunctionStep shape | REMOVE | typed public callable kwargs |
| runtime-invocation-options metadata fields on `FunctionContractAttribute`, `CallableMetadata`, `CallableContract`, `NormalizedFunctionItem`, and `RuntimeCallableArgumentPlan` | REMOVE | ordinary typed callable kwargs and existing runtime-bound parameter declarations |
| `CalculateMathInvocationOptions` | REMOVE | public typed `output_name` kwarg plus contract-derived operand object identity |
| `DefineGridInvocationOptions` | REMOVE | typed callable behavior kwarg |
| skipped export-module policy | REPLACE | explicit SaveImages, ExportToSpreadsheet, and ExportToDatabase steps |
| infrastructure exporter flags/profile | REMOVE | compiled generic output plans |
| hidden materialized-image kwargs/metadata | REMOVE | explicit export step + generic materialization config |
| `RuntimeImageExportSpec` / `RuntimeImageExportBitDepth` | REMOVE | SaveImages typed callable settings plus compiled `ImageFileOptions` output |
| `_candidate_image_snapshots_for_equivalence` | REMOVE | actual files in `RuntimeExportObservation.image_outputs` |
| `CellProfilerExecutionExportContext` / `CellProfilerAnalystExportRequest` | REMOVE | exact generic `RuntimeArtifactBatch` plus typed CP settings arguments |
| placeholder `interop/cellprofiler/database_export.py` | REMOVE | plate-scoped callable plus CPA projection and SQLite renderer on the CP leaf |
| placeholder `interop/cellprofiler/spreadsheet_export.py` | REMOVE | plate-scoped callable that renders contract-selected measurement artifacts |
| `FunctionStepExecutionScope` | KEEP | one generic callable-contract fact distinguishing axis execution from post-plate execution |
| `RuntimeArtifactBatch` | KEEP | exact contract-selected runtime records supplied to a plate-scoped callable |
| `StepInputDependencyKind.NO_MAIN_FLOW` | KEEP | artifact-only input state on the existing main-flow dependency owner |
| `ImageFileOptions` / `FileBundleOptions` | KEEP | generic registered materialization writers for one image file and a named file bundle |
| `DiskStorageBackend.save` byte payload path | KEEP | direct `bytes` persistence before the existing non-byte `FileFormat` dispatch |
| `MaterializationFormat.IMAGE_FILE` / `.FILE_BUNDLE` | KEEP | keys on the existing materialization format owner |
| `AnalysisConsolidationPlan` | COLLAPSE INTO EXISTING OWNER | direct post-plate execution and optional consolidation functions in the existing orchestrator module |
| `PipelineGeneratorRegistryStage` | REMOVE | `CellProfilerModule.__registry__` and root lookup |
| `PipelineGeneratorArtifactPruner` | REMOVE | generic compiler/materialization planning |
| `PipelineGeneratorRuntimeContractProjector` | REMOVE | exact compiler provider |
| `PipelineGeneratorCodeEmitter` stage object | REMOVE | generic FunctionStep pycodify authority |
| `PipelineGeneratorBuildStage` | REMOVE | private direct import pass |
| `PipelineGenerator` / `GeneratedPipeline` | REMOVE | one `import_cellprofiler_pipeline` operation and final import result |
| `GeneratedPipelineRequest` | REMOVE | validated import request plus local parsed values |
| `SkippedModuleSelection` | REMOVE | complete ordered parsed module input |
| `GeneratedStepEmission` / `GeneratedStepEmissionGroup` | REMOVE | actual `FunctionStep` values |
| generated import collector / `python_literal` | REMOVE | pycodify formatters |
| `ArtifactContractCommentSection` | REMOVE | no compiled details in public source |
| `GeneratedPipelineConfigDefaults` | REMOVE | existing `PipelineConfig` |
| `GeneratedProcessingConfigShape` | REMOVE | existing `ProcessingConfig` |
| `GeneratedStepSettings` | REMOVE | ordinary ordered kwargs mapping |
| `GeneratedParameterTarget` | REMOVE | owning `SettingToKeywordBinding.require_parameter_name` and `records_from_kwargs` |
| `GeneratedLiteralScalar` / `GeneratedLiteralValue` / `GeneratedStepSettingKey` / `GeneratedParameterName` / `GeneratedGroupByComponent` | REMOVE | concrete typed kwargs and `ProcessingConfig` values |
| `group_by_is_unresolved` / `variable_component_literal` / `all_component_literal` / `coerce_all_component` / `all_component_tuple_literal` / `group_by_literal` / `group_by_component_axis` | REMOVE | concrete config values plus generic pycodify |
| `source_binding_variable_component_literals` / `variable_component_literals` / `generated_function_step_semantic_argument_lines` | REMOVE | concrete `ProcessingConfig` plus generic pycodify |
| `ModuleProcessingComponents` | REMOVE | existing `ProcessingConfig` |
| `ModuleProcessingComponentRequest` | REMOVE | direct module, artifact contract, callable contract, axis plan, and typed behavior arguments |
| `CellProfilerModule.processing_components` and leaf overrides | REPLACE | exact `processing_config(module, contract, callable_contract, axis_plan, behavior_kwargs)` MRO method |
| `CellProfilerModule.with_generated_group_by` / `.generated_group_by` | REMOVE | declaration facts applied to concrete config through `FuncStepContractValidator` |
| `SourceProcessingAxisPlan` | KEEP | sole transient source-axis calculation owner built from resolved bindings and module contract |
| `SourceProcessingAxisPlan.from_schema` | REPLACE | `from_bindings` over one resolved step source config plus the module artifact contract |
| old `SourceProcessingAxisPlan.without_source_set_components` / `.scalar_source_group_component` / `.optional_single_image_set_component` / `.single_component_for_role` | REMOVE | exact four-method projection API in the corrected plan |
| module-level `source_identity_group_by_component` | REPLACE | method on `SourceProcessingAxisPlan` |
| `SourceProcessingComponentSemantics` | REMOVE | `SourceProcessingAxisPlan.from_bindings` |
| `RuntimeArtifactProcessingScope` / `SourceBindingProcessingScope` | REMOVE | one `default_module_processing_config` calculation |
| `default_module_processing_components` | REPLACE | concrete `default_module_processing_config` |
| `default_module_requires_pairwise_object_domain_scope` / `_is_inputless_artifact_only_contract` | REMOVE | direct typed contract predicates inside the default config calculation |
| `SourceProcessingAxisRole` | REMOVE | named `SourceProcessingAxisPlan` methods |
| `SourceProcessingAxisRolePolicy` family | REMOVE | named `SourceProcessingAxisPlan` methods |
| `ModuleProcessingScopePolicy` family | COLLAPSE INTO EXISTING OWNER | deterministic lowering over contract/source values |
| `GeneratedPipelineModuleIdentity` | REMOVE | no CP generated-module import |
| `GeneratedPipelineRuntimeModule` | REMOVE | import returns real steps; ZMQ reconstructs source |
| generated module export/function/registration wrappers | REMOVE | import returns real steps and module declarations own functions |
| `CPPipePipelineArtifact` | REMOVE | final import result used directly |
| `PreparedGeneratedPipeline` | REMOVE | `CellProfilerPipelineImportResult` / generic workspace result |
| `GeneratedCPPipePipeline` | REMOVE | final import result plus local parsed values |
| `CPPipePipelineGenerationRequest` / `CPPipePipelinePreparationRequest` | REMOVE | direct import operation |
| `CPPipeModulePartition` | REMOVE | one local import pass over module declarations |
| CP source-schema result wrappers | REMOVE | `InputWorkspacePreparationResult` |
| `CellProfilerPipelineRuntimeBindingService` | REMOVE | public steps already final |
| stored `CellProfilerPipelineImportResult.generated_source` field | REMOVE | derived `FunctionStepTransportAuthority` property |
| `CellProfilerDialectCompiler` / `CellProfilerPipelineImporter` | REMOVE | `import_cellprofiler_pipeline(request)` |
| `CellProfilerGeneratedPipelineDialectCompiler` / `CellProfilerGeneratedPipelineImporter` / `BenchmarkCellProfilerDialectCompiler` | REMOVE | direct import operation |
| dialect compiler register/get/clear functions and generated/benchmark registration aliases | REMOVE | direct import operation; no mutable compiler singleton |
| `partition_cppipe_modules` / `prepare_generated_pipeline` | REMOVE | one direct ordered import pass |
| `CellProfilerPipelineImportRequest` | REPLACE | exact `.cppipe` and generated-source VFS addresses/backends; no execution or pruning fields |
| `CellProfilerPipelineImportResult` | REPLACE | exact direct source/module/steps/config/coverage/generated-path fields plus derived source property |
| `CellProfilerModuleReference` | REPLACE | raw name, module number, and enabled source facts only |
| `import_cellprofiler_pipeline` | KEEP | one direct validated parser-to-public-state operation |
| mutable compiler singleton registry | REMOVE | direct import operation; no replacement registry |
| `SetupModuleCompiler` | REMOVE | `SourceSetupCellProfilerModule.contribute_source_bindings` |
| `PipelineImageSchema` / builder / assignment wrappers / projection layer | COLLAPSE INTO EXISTING OWNER | complete existing `SourceBindingsConfig`, `NamedSourceBinding`, and source-binding workspace APIs |
| generic-core `ImageTypeSourceRole` and dynamic class specs | REMOVE | CP-local `CellProfilerSourceImageType` enum during setup lowering only |
| `SourceImagePayloadSemantics` / role strategy hierarchy | REMOVE | direct binding payload function plus `ImagePayloadMetadata.source_channel_axis` |
| `SOURCE_IMAGE_TYPE_METADATA_FIELD` / `SourcePlaneProjection.image_type` | REMOVE | persisted source alias resolves the submitted binding; runtime metadata carries generic layout only |
| source literal resolver class hierarchies | REPLACE | three CP-local nominal enums mapping external literals to existing generic enums |
| `CellProfilerSourceImageType` and three CP source-literal enums | KEEP | nominal import-edge translation into existing generic binding enums and fields |
| `NamesAndTypesAssignmentLayout` table and block strategy | REMOVE | indexed setting columns on the NamesAndTypes declaration with exact cardinality validation |
| `SourceBindingsConfig` / `NamedSourceBinding` | KEEP | complete public source declaration carried by `PipelineConfig` |
| `SourceBindingDeclarationsMixin` | KEEP | shared alias lookup, artifact-kind validation, loaded aliases, and measurement source names |
| manual `StepSourceBindingsConfig` inherited-field predicates and overlay functions | REMOVE | existing lazy config MRO plus resolved `StepConfigUniverse` value |
| `CompiledSourceBindingPlan.from_config` | REPLACE | resolved config plus explicit `InputSource` activation, with no second overlay |
| untyped `bindings_for_group_key` / `for_group_key` | REMOVE | exact typed component-group selection |
| `bindings_for_component_group` / `for_component_group` | REPLACE | exact component and value match with fail-loud scoped no-match |
| `source_binding_group_keys_for_group_by` | KEEP | one generic extraction shared by path planner and invocation provider |
| `ImagePayloadMetadata` | KEEP | resolved generic source channel axis and layout queries |
| `SourceBindingContext` | REPLACE | direct FileManager, backend, resolved config, and inventory method; no schema/import result |
| `SourceSchemaImageSetSelection` / private benchmark selection fields | REMOVE | inherited existing `WellFilterConfig` and canonical ZMQ axis selection |
| source candidate provider/discovery/probe family | REMOVE | exact plate/VFS files consumed by `SourceBindingWorkspaceProjector` |
| CellProfiler source-root resolver/candidate/admission/exclusion family | REMOVE | exact import-request root plus `SourceBindingsConfig` filters |
| staged `ImagePlaneSourceResolver` family | REMOVE | import-resolved URI on `ImagePlaneSource` in submitted config |
| TIFF/single-plane source inventory family | REPLACE | declared source stack components + generic VFS load + source-axis indices |
| `SourcePixelRef` | KEEP | exact PolyStore-owned `(backend, backend_address, source_axis_indices)` ref |
| `FileManager` registry construction and pickle state | REPLACE | execution-local mapping reconstructed from nominal backend declarations and injected through `bind_registry` |
| `FileManager.register_backend` | KEEP | sole execution-local backend mutation and registry-binding operation |
| `BackendBase.bind_registry` | KEEP | polymorphic execution-local registry injection; root no-op and virtual-workspace override |
| `DataSource.resolve_address` | KEEP | backend-polymorphic interpretation of opaque addresses; disk owns plate-relative paths |
| old `SourcePixelRef` reader/path/series/plane/C/Z/T fields, `source_metadata`, and workspace projection helpers | REMOVE | opaque backend address plus backend-owned parser; no metadata copy |
| `BioFormatsPlaneRef` | KEEP | exact physical path/series/plane backend address with canonical encoding |
| `BioFormatsStorageBackend` | KEEP | direct `BioFormatsPlaneRef` address loader only; no workspace mapper or NPY branch |
| `BioFormatsHandler.get_primary_backend` override | REMOVE | inherited shared virtual-workspace selection after reader registration |
| string workspace mappings / `PathSourceRefResolver` | REMOVE | one validated structured source-ref representation |
| `VirtualWorkspaceSourceRefResolver` family / resolved-ref wrapper | REMOVE | direct existing-backend-registry dispatch on `VirtualWorkspaceBackend` |
| auxiliary NumPy materializer / target-path policy | REMOVE | persisted original backend/path ref and generic `FileManager` load |
| source candidate/request/projection one-operation wrappers | REMOVE | local functions on `SourceBindingWorkspaceProjector`; validated `SourceCandidate` plus private image-set/index records only |
| `SourceSchemaCandidate` | REPLACE | `SourceCandidate`, shared by workspace projection and `SourceInventory` |
| `SourceCandidate` | KEEP | one validated candidate owning exact source ref, filter identity, metadata, and source-axis shape |
| `ImageSetAssembler` | KEEP | enum-keyed strategy on existing `SourceBindingMatchMethod` |
| ordered `ComponentProjection` registry | REPLACE | one enum-keyed strategy per existing `AllComponents` member |
| `SourceComponentProjectionStrategy` | KEEP | exact enum-keyed root with one leaf per `AllComponents` member and no leaf facts on the root |
| schema-plus-bindings source view/preview API | REPLACE | one `SourceBindingsConfig` plus resolved step override |
| `SourceInventoryProvider` registry and `SourceInventoryBuildRequest` | REMOVE | `SourceBindingContext` + `SourceInventory.from_filemanager` |
| `SourceSchemaWorkspaceMaterialization` | REPLACE | `SourceBindingWorkspaceMaterialization` over config-derived workspace paths and mappings |
| `SourceBindingWorkspaceMaterialization` | KEEP | sole typed workspace materialization result |
| `SourceBindingContext.source_schema` / `.import_result` | REMOVE | resolved `SourceBindingsConfig` only |
| `_AUXILIARY_PAYLOAD_CACHE` and disk `np.load` fallback | REMOVE | persisted workspace mapping plus generic `FileManager` load |
| `SourceImageStackPlanDeclaration` | REMOVE | NamesAndTypes module declaration |
| `ResolvedModuleFunction` | REMOVE | actual callable from `CellProfilerModule.resolve_function` |
| `CellProfilerModule.resolve_function` | REPLACE | canonical callable selector for parsed import over module, artifact contract, and source axis plan only |
| `CellProfilerModule.resolve_semantic_function` | REMOVE | one parsed-import `resolve_function(module, contract, axis_plan)` MRO selector that delegates to `require_callable` |
| `CellProfilerFunctionRuntimeMetadata` / `CELLPROFILER_MODULE_ATTR` | REMOVE | `CellProfilerModule.for_function_name` |
| `AbsorbedFunctionMetadata` and derived catalog maps | REMOVE | `CellProfilerModule` registry |
| public CellProfiler processing wrapper | REMOVE | raw source-declared callable plus immutable compiler-enriched `CallableContract` |
| `CellProfilerFunctionCatalog` and `processing/backends/cellprofiler/library.py` | REMOVE | callable loading on `CellProfilerModule` |
| `FunctionReferenceTransportStrategy` and its CP leaf | REMOVE | direct registered/importable-function paths on `FunctionReferenceTransportAuthority` |
| one-leaf setup parser/policy registries | REMOVE | private functions beside owning setup declaration leaves |
| `CellProfilerDebugView` | REMOVE | `DebugViewModel.from_debug_snapshot` |
| semantic-default-contract registry | REMOVE | module-owned contract type tuples |
| `PipelineStepsBoundary` / `PipelineStepsCarrier` | REMOVE | direct step lists |
| `PycodifiedSource` family / assignment-source request | REMOVE | submission pipeline code, config projection, and request-payload source hash |
| `FunctionStepTransportAuthority.source_from_pipeline` / `.pipeline_steps_from_namespace` | KEEP | sole generic public-step source and namespace boundary |
| `OpenHCSExecutionSubmission.pipeline_code` | KEEP | direct submission-owned source selection without a source wrapper |
| agent pipeline provider/payload/ID wrappers | REMOVE | request-subtype method plus one concrete direct-field definition |
| `ExecutionPipelineSessionRequest.build_pipeline_definition` | KEEP | request-subtype polymorphic construction on the existing request root |
| `ExecutionPipelineDefinition` | REPLACE | one direct pipeline ID, step list, and optional source record |
| `InputWorkspacePreparationResult` | REPLACE | typed optional steps/config pair plus source-binding materialization |
| namespace/source ZMQ pipeline wrappers | REMOVE | `FunctionStepTransportAuthority` methods |
| `OpenHCSExecutionConfigCarrier` / `ZMQResolvedConfig` | REMOVE | direct config bundle fields |
| both OpenHCS benchmark result caches and cache-hit timing/provenance | REMOVE | current public-source ZMQ compile-then-execute; persisted native references only |
| `CachedNativeReferenceTimingPolicy` | REMOVE | measured native reference timing or null speedup |
| benchmark compatibility facade packages | REMOVE | production owner imports |
| obsolete absorber/backfill/fix tools | REMOVE | registered production declarations |
| CP threshold compatibility wrapper | REMOVE | backend threshold functions |
| `TupleMemberTypeValidation` | REMOVE | local import-record validation |
| direct orchestrator execution helpers | REMOVE | canonical ZMQ compile-then-execute |
| `SettingNameFamilySpec` | REMOVE | unused duplicate of `SettingNameFamily` |

## Contradictions Removed From The Previous Plan

### Existing Types Were Declared Load-Bearing While Being Mirrored

The old plan retained `ArtifactSpec` and `ModuleArtifactContractItem`, then added
port classes carrying the same type, partition, name, and relation fields. The
corrected plan uses the existing types directly.

### Leaf Ownership Was Required While Generic Port Types Knew Leaf Roles

The old port roots named source-bound, main-flow, runtime, special, recorded,
and declared cases. Generic code would need to inspect those leaf categories.
The corrected plan lets existing contract partitions and module polymorphism
carry those roles.

### No Mirrored Registry Was Required While Descriptor Registries Were Added

The old CP setting descriptors recreated the module declaration's knowledge of
setting rows and artifact roles. The corrected plan keeps that knowledge on the
registered module class.

### Special I/O Was Kept While A Parallel Typed Special-I/O Taxonomy Was Added

The corrected plan keeps the existing decorators as ABI declarations and uses
the module contract for semantics. It adds no second special-I/O declaration
family.

### Public Source Was Declared Sufficient While Generated Sidecars Remained

The corrected plan removes hidden FunctionStep contracts, generated runtime
contracts, module-number matchers, and pipeline metadata in the same migration
that installs exact compilation.

### Generic Runtime Was Preferred While Export Modules Bypassed It

The corrected plan emits SaveImages, ExportToSpreadsheet, and ExportToDatabase
as explicit steps. SaveImages follows normal axis artifact materialization;
spreadsheet/database export follows one generic plate callable lifecycle and
file-bundle writer. It removes upstream mutation, `.cppipe`-derived exporter
flags, and fake pass-through implementations.

### One Module Registry Was Required While Setup Modules Lived Elsewhere

The corrected plan registers setup modules on the existing module root and
moves source-binding config behavior onto those declarations. Reporting and
import no longer union two registries.

### Public FunctionSteps Were Required While The Generator Emitted A Shadow AST

The corrected plan constructs real FunctionSteps first and uses generic
pycodify. It deletes generated emission DTOs, literal rendering, generated
module import, and compatibility facade packages.

### Plain Lists Were Required While Transport Wrapped Them Repeatedly

The corrected plan removes one-field list/config carriers and puts source
rendering plus namespace validation on the existing FunctionStep transport
authority.

### Registry Discipline Was Required While A One-Leaf Compiler Registry Was Proposed

The corrected plan uses one direct `.cppipe` import function. There is no
semantic compiler implementation axis, so an ABC plus registry would be pure
indirection.

## Phase Dry Run

### Phase 0: Boundary Tests

Input:

- public `PipelineConfig`;
- public FunctionStep list;
- deterministic pycodified source.

Observed current failure:

- prepared CP imports can carry invocation contracts and runtime wrappers not
  visible in source;
- generated contract matching uses module numbers and candidate alignment;
- code-mode reconstruction lacks that authority.

Required test result before implementation:

- tests are red on hidden state and green on native pipelines.

Dependency conclusion:

- no production prerequisite beyond current test fixtures.

### Phase 1: Atomic Runtime-Unification Cutover

The dry run proves nominal ownership, exact provider/runtime reconstruction,
direct import, and public transport are one dependency closure. The current
generator consumes the schema and declaration mirrors removed by nominal
ownership; exact contracts need the new process-local adapter factory; direct
import needs those declarations/contracts; transport collapse needs the direct
public result. These four workstreams therefore share one merge and deletion
gate. No workstream authorizes a releasable intermediate state or transitional
reader, writer, alias, wrapper, or registry.

#### Workstream 1A: Nominal Module Ownership

Input:

- public raw callable identity or parsed `ModuleBlock`;
- `CellProfilerModule.for_module` and `for_function_name`;
- transient setting rows;
- available artifact specs;
- complete lazy-MRO-resolved step source bindings.

Execution trace:

1. Let setup declarations contribute the complete `SourceBindingsConfig`
   through the same module registry before executable parsed blocks are
   lowered.
2. Resolve a public raw callable directly to its nominal module declaration and
   preserve that exact callable through compilation.
3. Resolve a parsed block directly to its nominal module declaration and let
   the declaration MRO derive ordered existing contract items.
4. Resolve the candidate lazy step source config directly under the completed
   pipeline config, build `SourceProcessingAxisPlan` from that concrete config
   and the artifact contract, then select the parsed block's canonical raw
   callable through `resolve_function(module, contract, axis_plan)` without a
   placeholder FunctionStep, catalog, wrapper, function-name DTO, or
   `ProcessingConfig` input.
5. Read the selected raw callable's `CallableContract`, bind typed behavior
   kwargs, and derive concrete `ProcessingConfig`; no module processing field
   participates.
6. Resolve every persisted source alias through
   `SourceBindingDeclarationsMixin`, apply the binding's loading declaration,
   and carry only the resolved channel axis on `ImagePayloadMetadata`.
7. Parse CP source image types and source literals through four CP-local enums;
   parse repeated setup settings as indexed columns with exact cardinality.
8. Materialize the complete config-declared source workspace from exact VFS
   files; expand declared source axes through generic loading and ordered
   `SourcePixelRef` indices. Serialize every mapping as one exact
   backend/address/indices ref, retain that ref on each source candidate, and
   resolve its loader by exact backend key in the execution-local FileManager
   registry. Backend polymorphism owns address interpretation.
   Bio-Formats contributes a backend-owned plane address through the shared
   virtual workspace; NPY fixtures contribute ordinary disk refs.
9. Apply benchmark/UI axis selection only through inherited
   `WellFilterConfig` during canonical compilation.
10. Let SaveImages consume one image and emit an axis-materialized image output.
   Let spreadsheet/database declarations enumerate exact artifact inputs,
   declare generic plate execution scope, and emit one materialized file bundle.
11. Preserve only raw module name, number, and enabled state as direct fields
    on the final import result; query `emits_function_step()` instead of
    storing its result or a provenance wrapper.
12. Validate special-I/O slots against the raw callable ABI and ordered module
    contract.
13. Let the Metadata declaration own disabled-row preservation directly; no
   setup parsing policy or component-set carrier remains.

Dependency conclusion:

- existing module declaration methods and policy mixins contain the required
  leaf behavior;
- capability, setup, callable-catalog, and policy mirrors can be deleted as
  consumers migrate.
- no provenance role remains; `ModuleBlock.enabled` is the raw source fact and
  the module declaration owns behavior.
- no CP image-type token or role survives setup lowering; generic runtime
  alignment reads `ImagePayloadMetadata` and generic source loading reads the
  submitted binding.
- no benchmark-only source cap or TIFF inventory influences workspace identity;
  UI and benchmark selection share the submitted config.
- unsupported pass-through declarations are absent, so enabled use fails at
  nominal module lookup.
- plate-scoped exporters run once from merged contract-selected records; no CP
  exporter flag, store scan, or config participates.

Representative cases:

- CorrectIlluminationCalculate source image to illumination image;
- CorrectIlluminationApply runtime illumination input and main image output;
- Align two image inputs and two ordered image outputs;
- IdentifyPrimaryObjects image input, object output, and measurements;
- RelateObjects object inputs, relationship output, and measurements;
- DefineGrid spatial-grid output;
- CalculateMath measurement input and output feature identity.
- Images/LoadImages/Metadata/NamesAndTypes/Groups source-binding config
  lowering.
- grayscale, color, binary-mask, illumination-function, and object source
  loading without `OpenHCSImageType` metadata.
- equivalent TIFF/NPY source stacks and one undeclared multi-plane file.
- SaveImages, ExportToSpreadsheet, and ExportToDatabase as executable public
  export steps.

#### Workstream 1B: Exact Provider And Runtime Reconstruction

Input:

- ordered snapshots;
- normalized invocation keys;
- ObjectState-resolved source bindings;
- ordered available artifact specs.

Execution trace:

1. Precompute all CP invocation contracts in source order.
2. Derive contract group keys from explicit dict keys or the shared generic
   source-binding/group-by helper.
3. Scope bindings by exact component plus key and derive one module contract
   for each contract group; an invocation without a source-derived split stays
   explicitly unscoped.
4. Combine those declarations into one `ModuleArtifactContract`.
5. Advance only the existing local artifact collection with produced specs.
6. Store exact plans on the frozen
   `CellProfilerInvocationContractProvider` by step index and invocation key.
7. Let generic reverse and forward planner passes query that immutable result;
   reject multiple provider-factory claims independently of registry order.
8. Keep raw callable references in compiled contracts; axis-scoped contracts
   carry the exact value from
   `CellProfilerRuntimeAdapter.runtime_adapter_spec()`, while plate-scoped
   callables remain generic importable functions.
9. Remove import-time artifact pruning; enabled executable modules remain in
   source order and generic compilation owns materialization.
10. Replace compiled string artifact keys with exact `ArtifactSpecRef` values,
    join selected plan specs to contract specs by ref plus complete-spec
    equality, and pass full ordered declared outputs from the compiled contract to the
    generic matcher; do not infer them from the callable a second time.
11. Collapse one-field scalar projections into their existing shape,
    object-label, source-binding, artifact-location, adapter, or consuming
    method owners.
12. Validate remaining public kwargs once through generic `CallableContract`
    compilation, then let the compiled adapter spec's exact runtime-callable
    factory build one three-field immutable runtime plan from raw callable plus
    enriched `CallableContract`, derive module type once, store that plan on one
    executor, and let the runtime adapter query through the plan without a
    signature mirror or kwarg filter.
13. Delete the rehydrator request, registry, and CP leaf; direct, worker, and
    reconstructed-source execution all use the same adapter-spec method.
14. Reduce the run request to executor plus invocation-varying values and query
    plan/callable/name facts through that executor.

Dependency conclusion:

- a prepass is required because the generic planner queries the provider in
  more than source order;
- a second semantic flow type is not required because the prepass only needs
  transient available-artifact and main-flow `ArtifactSpecCollection` cursors;
- a runtime contract map is not required because
  `ComponentArtifactPlans.from_step_component` selects grouped input/output
  plans before adapter construction;
- exact keys remove every generated matcher fallback;
- every `ArtifactSpecKey` consumer and `module_roles.py` are gone after
  Workstream 1A moves diagnostics and exporter behavior to their owning
  declarations;
- no one-field runtime carrier remains when its type is not used for dispatch,
  validation, inherited request composition, a cache namespace, or a cohesive
  multi-operation service.
- no CP processing-contract cache, metadata mutation helper, old rebuild
  function, or duplicate callable/runtime-plan processing field remains.
- no executor plan dictionary or copied contract/name/policy field remains;
- no run-request plan/callable copy or duplicate input/current image field
  remains.

Representative cases:

- a plain callable over all channels uses one public invocation and one
  aggregate contract;
- a dict pattern with differing channel kwargs has one exact invocation per
  dict key;
- a list pattern preserves position in `FunctionInvocationKey`;
- TrackObjects receives timepoint-aligned object artifacts;
- Tile receives aligned named image artifacts and fails compilation for known
  incompatible component identity.

#### Workstream 1C: Public State Only

Input:

- public steps and configs produced by the importer;
- complete source-binding config inside `PipelineConfig`.

Execution trace:

1. Move the complete current module-to-public loop into
   `openhcs/interop/cellprofiler/import_service.py::import_cellprofiler_pipeline`,
   migrate every caller, and
   delete the generator/runtime-pipeline/compiler-forwarding files.
2. Import `.cppipe` into public steps and config through that direct operation.
3. Store those steps directly in UI, agent, and submission records.
4. Pycodify through `FunctionStepTransportAuthority`.
5. Reconstruct a direct step list in the ZMQ server.
6. Compile all CP contracts from the reconstructed values.
7. Execute the compile artifact.

Dependency conclusion:

- hidden FunctionStep contracts and pipeline metadata become unused after
  Workstream 1B;
- the editor still needs a nominal ObjectState root for display/scopes, so
  `PipelineEditorStateRoot` remains local to the UI and never crosses execution
  boundaries;
- pipeline and config carrier registries have no consumer and can be removed;
- export behavior is already public and compiled before transport collapse.
- direct import exists before every forwarding implementation is deleted; the
  same atomic workstream removes both after all callers migrate.

#### Workstream 1D: Direct Sparse Import

Input:

- parsed setup modules;
- parsed processing modules;
- resolved pipeline defaults;
- public typed behavior settings.

Execution trace:

1. Refine the direct import loop established in Workstream 1C.
2. Resolve every module through the one `CellProfilerModule` registry.
3. Setup modules lower directly to `PipelineConfig.source_bindings_config`
   through their MRO.
4. Processing modules lower to real FunctionSteps.
5. Behavior kwargs equal to the canonical raw signature defaults and config or
   source-binding values equal to inherited defaults disappear from the step;
   compiler reconstruction reapplies those same typed signature defaults.
6. Identical all-group invocations collapse to a plain callable.
7. Real subgroup differences remain dict patterns.
8. Compile-only identity overrides remain sparse and are consumed by the
   compiler.
9. Generic pycodify renders the resulting public objects.

Dependency conclusion:

- import sparsity follows contract authority; implementing it before exact
  compilation would erase required information;
- typed callable signatures already provide the UI control types;
- no generated emission or source-import model is required.

Phase 1 exit conclusion:

- every old schema, generator, matcher, callable catalog, symbol table, runtime
  wrapper variant, pipeline carrier, transport carrier, reader, writer, alias,
  and re-export is deleted;
- only public steps/config cross the generic source/ZMQ boundary;
- exact contracts are recreated by the provider and executable CP adaptation
  is constructed process-locally from the enriched compiled contract;
- all Workstream 1A-1D tests pass together with no transitional path.

### Phase 2: Compile-Time Enforcement

Input:

- exact module declaration;
- exact callable ABI;
- resolved config;
- exact source and runtime artifact specs.

Execution trace:

- validate declaration, artifact availability, slot pairing, component
  requirements, output order, relations, and consumed kwargs before producing a
  compiled step plan.

Dependency conclusion:

- all structural failures reported in prior runtime traces have a compile-time
  owner once Phase 1 is complete;
- data-dependent image contents and external I/O remain runtime concerns.

### Phase 3: ZMQ Parity And Napari

Input:

- `benchmark/manifests/official30_portable_axis1.json`;
- persisted native references;
- baseline `GlobalPipelineConfig`;
- viewer run config with inherited
  `LazyNapariStreamingConfig(enabled=True, persistent=False)`.

Execution trace:

1. `run_comparison_suite` passes the real global config into `OpenHCSAdapter`.
2. The adapter rebuilds lazy pipeline config against that global context.
3. `_execute_pipeline_via_zmq_server` pycodifies and submits the public source.
4. The server compiles and executes.
5. SaveImages parity reads emitted files; the three database cases compare
   SQLite tables and CPA properties despite `value_only`.
6. Non-persistent viewer lifecycle resets between cases.

Dependency conclusion:

- the benchmark API already carries global config;
- no viewer flag or config mirror is needed;
- parity timing uses server progress events already captured by
  `_ZMQProgressTimingObserver`.

## Public API Surface After Migration

The public construction surface is small:

```text
GlobalPipelineConfig
PipelineConfig
list[FunctionStep]
FunctionStep.func = callable | (callable, kwargs) | list | dict
registered Lazy step/pipeline configs
```

List and dict patterns contain only those two leaf shapes. The former
three-item leaf is rejected.

The existing `SourceBindingsConfig` gains the source-plane, imported-metadata,
source-stack-component, grouping-field, voxel-spacing, and generic payload-load
fields currently stranded on `PipelineImageSchema`; `NamedSourceBinding` gains
the corresponding payload-load flags, source channel axis, and channel-count
constraint. `ImagePayloadMetadata` gains only the resolved channel axis needed
after loading. This completes existing declaration/runtime owners
rather than adding a config family. `PipelineImageSchema` disappears from the
public and private post-import surface.

CellProfiler-specific public values are only:

- raw registered processing callables;
- typed user-controlled module settings represented directly in callable
  signatures;
- sparse explicit identity overrides for noncanonical flow.

Identity-only settings are absent from raw callable signatures. Their one
existing `SettingToKeywordBinding` references the owning module setting ClassVar
and serves both parser conversion and sparse reconstruction. Signature
membership distinguishes behavior bindings from identity bindings; the
invocation provider consumes present identity kwargs before generic signature
validation. Function panes therefore expose no compiler-only image, object, or
measurement names.

The generic callable/materialization surface gains
`FunctionStepExecutionScope`, `execution_scope`, `RuntimeArtifactBatch`,
`ImageFileOptions`, and `FileBundleOptions`. These values contain no CP module
identity and are not new pipeline configs.

Source/sample selection adds nothing. The benchmark, UI, agent, and headless
paths use the existing inherited `WellFilterConfig`; workspace construction
always projects the complete `SourceBindingsConfig` universe.

The following never appear in public code:

- ModuleBlock;
- module number;
- artifact contract;
- runtime callable wrapper;
- symbol table;
- generated contract matcher;
- sidecar;
- pipeline metadata;
- hidden compiler kwarg payload.

## Compile-Time Versus Runtime Ownership

| Concern | Compile time | Runtime |
|---|---|---|
| resolve raw callable to CP module | yes | no |
| derive artifact names/types/partitions | yes | no |
| resolve source-binding inheritance | yes | consume plan |
| validate special-I/O slots | yes | bind values |
| select invocation contract by exact key | yes | consume generic active group plans |
| validate component requirements | yes for declared identity | enforce data-dependent shape |
| load artifact payloads | no | generic axis runtime or exact plate batch |
| execute raw CP function | no | one CP adapter for axis scope; raw generic call for plate scope |
| match returned slots | plan at compile time | generic matcher |
| materialize artifacts | plan at compile time | generic materialization |
| launch/reuse Napari | configure at compile time | generic streaming lifecycle |

## Static Review Gates

The implementation review runs repository searches proving absence of the old
authorities. The canonical, unique symbol set is in the corrected plan's
`Static Deletion Gates` section; this audit does not duplicate that manually
synchronized list. Implementation AST and `rg` tests consume that one required
set and fail on every production definition, import, call, attribute read, or
re-export.

The review also checks:

- no generic core import of `openhcs.interop.cellprofiler` or a concrete CP
  backend module;
- no CP image-type literal, role, or metadata key in generic source projection,
  runtime values, source loading, alignment, or aggregation;
- every workspace mapping is a validated `SourcePixelRef`; source-reference
  resolution uses one exact existing-backend lookup and has no string-path
  branch, resolver family, shape predicate, priority traversal, old-name alias,
  or dual-format migration path;
- AST inspection requires `SourcePixelRef` to declare exactly `backend`,
  `backend_address`, and `source_axis_indices`; `BioFormatsPlaneRef` declares
  exactly physical path, series index, and plane index; the Bio-Formats backend
  contains no workspace-metadata read or alternate reader branch, and its
  handler contains no primary-backend override;
- AST inspection requires `SourceCandidate` to own exactly one
  `SourcePixelRef` and no physical path/backend/axis-index mirror;
- `FileManager.__setstate__` reconstructs the submitted registry key set
  locally and never imports global `storage_registry`; registered backends
  receive that same map through nominal binding, and address interpretation is
  backend-polymorphic;
- source component projection has one enum-keyed leaf per `AllComponents`
  member and no root-owned alias/default/priority facts;
- every added source config field resolves through the lazy MRO into one
  snapshot value; no manual field overlay/equivalence list or scoped group
  fallback survives;
- source workspace construction uses the exact submitted VFS root and contains
  no parent/child inference, `.cppipe` placement probe, candidate bucket, or
  path-exclusion policy registry;
- no module-name policy dictionary;
- no copied module feature list;
- no separate setup-module registry or setup-module name list;
- no callable ownership attribute, metadata DTO, wrapper call layer, or derived
  function-to-module dictionary;
- parsed import resolves the lazy step source config directly under the
  pipeline config, selects a canonical callable from module/contract/axis plan
  with no `ProcessingConfig` input, and creates no placeholder FunctionStep;
  public compilation contains no parsed-module selector call;
- no CellProfiler import or call of `attach_callable_contract_metadata`, no CP
  callable-metadata cache, and no compiler/import mutation of `vars(func)`;
- no function-reference rehydrator registry; the compiled
  `RuntimeAdapterSpec` performs one exact optional runtime-callable factory
  call after ordinary reference resolution;
- no `RuntimeInvocationOptions`, third-tuple FunctionStep leaf, hidden
  invocation-options parameter, or UI extraction branch;
- no `getattr`-based module capability probe;
- no string classifier for artifact kind;
- no generated runtime wrapper or contract payload;
- exact CP provider shape and immutable key mapping are AST-enforced; generic
  provider composition rejects multiple claims rather than choosing by
  registry order;
- public kwargs are validated once by generic compilation and forwarded
  unchanged; CP runtime contains no signature or kwarg-spec mirror;
- compiled artifact selection carries exact refs and joins complete specs; no
  string key or name-only intersection survives;
- no manual FunctionStep source renderer in import;
- no stored generated-source text beside a public step list;
- AST inspection requires `CellProfilerPipelineImportResult` to declare exactly
  `cppipe_path`, `modules`, `pipeline_steps`, `pipeline_config`,
  `setting_coverage`, and `generated_source_path`; `generated_source` is a
  derived property and never a stored field;
- no pycodified-source/request value; submission, config projection, and
  request payload own direct source rendering and hashing;
- no agent pipeline-definition provider or ID/payload wrapper; request subtype
  polymorphism constructs one concrete direct-field definition;
- no generated-module import step after conversion;
- no callable-output reclassification after the compiled contract supplies
  full ordered declared-output specs;
- AST inspection rejects every materialization-bearing `special_outputs`
  tuple on a registered CellProfiler callable; those decorators declare slot
  names only;
- the axis-scoped CP runtime callable owns one executor, the executor owns one
  three-field immutable plan, and generic `FunctionReference` plus enriched
  `CallableContract` are the only serialized identities;
- AST inspection rejects `__reduce__`, identity `__eq__`, and `__hash__` on
  `CellProfilerRuntimeCallable`;
- the axis-scoped CP runtime plan contains exactly resolved raw callable,
  module type, and enriched callable contract; it has no copied
  module/processing contract, function name, artifact collection,
  signature/kwarg specification, or policy field, and the executor has no plan
  dictionary or copied
  contract/name/policy field;
- the module run request contains one image field and no copied callable or
  plan;
- no one-field CP runtime class whose only behavior is a scalar predicate,
  normalization, optional value, or delegation to an existing authority;
- no pipeline/config carrier registry or one-field boundary wrapper;
- no `cellprofiler_module_callable` code-mode factory allowance or runtime
  wrapper formatter;
- no repeated step binding equal to pipeline defaults;
- no dict pattern whose entries all have identical behavior and cover all
  groups.
- no benchmark-only source selection, candidate-provider retry, URI resolver
  priority, TIFF source inventory, or auxiliary format materializer;
- no CP-specific plate-execution hook, exporter registry, output-format switch,
  or runtime-store scan outside compiled artifact input plans;
- no generic image/file-bundle writer import of a CP module and no CP exporter
  direct filesystem write;
- no production or test import from `benchmark.cellprofiler_library` or
  `benchmark.cellprofiler_compat`.
- no OpenHCS benchmark result cache, cache-reuse switch, cached timing, or
  cached-output provenance; only native CellProfiler references are reused.
- no timeout or configured limit projected as a measured native duration.

## Tests To Remove Or Rewrite

Tests that encode the superseded architecture are not retained as compatibility
requirements:

- rewrite `tests/unit/test_cellprofiler_interop_namespace.py` to assert only
  the direct import operation, nominal module declaration exports, and public
  result fields; delete every compatibility alias, provenance/role,
  `SetupModuleCompiler`, compiler-ABC, and forwarding-export expectation;
- rewrite `external/PolyStore/tests/test_filemanager_extended.py` around exact
  structured `SourcePixelRef` mappings and execution-local registry pickle
  reconstruction; delete every string workspace-mapping fixture and global
  registry replacement expectation;
- delete `tests/unit/test_cellprofiler_symbol_table.py` after moving genuine
  module-contract cases to nominal declaration tests;
- replace `tests/unit/test_cellprofiler_module_processing_components.py` with
  `tests/unit/test_cellprofiler_module_processing_config.py`; test concrete
  `ProcessingConfig`, exact contract/axis/callable selection order, callable
  contract locality, and the illumination behavior override, with no component
  result or request DTO expectation;
- rewrite `tests/unit/test_cellprofiler_module_function_resolution.py` to
  assert canonical callable object identity from the exact artifact contract
  and `SourceProcessingAxisPlan`; remove imports and expectations for
  `module_function_resolution`, source schema, processing request, lineage DTO,
  default function name, and semantic selector;
- delete `tests/unit/test_cellprofiler_module_roles.py`; test raw
  name/module-number/enabled provenance in import-record tests and artifact
  retention in generic contract tests;
- rewrite source-binding config tests so setup modules resolve through
  `CellProfilerModule` declarations and no retained schema participates;
- replace `tests/unit/test_cellprofiler_source_schema.py` with
  `tests/unit/test_cellprofiler_source_bindings.py`; preserve setup and axis
  cases against `SourceBindingsConfig`, module declarations, and
  `SourceProcessingAxisPlan`, and delete every schema DTO expectation;
- delete `tests/unit/test_pipeline_image_schema.py`; move source declaration
  cases to `test_source_bindings.py`, CP source-literal/image-type cases to the
  infrastructure module declaration tests, and runtime layout cases to
  `test_runtime_values.py` and `test_image_stack_layout.py`;
- rewrite source workspace/projection tests around `SourceBindingsConfig` and
  persisted `source_alias`; assert serialized metadata contains no
  `OpenHCSImageType` or `image_type` field;
- rewrite source payload tests around direct `NamedSourceBinding` loading and
  `ImagePayloadMetadata.source_channel_axis`; remove every role instance and
  role-strategy expectation;
- delete all `SourceSchemaImageSetSelection` tests and adapter fields; assert
  integer/list well caps alter only `WellFilterConfig` and produce identical
  UI-equivalent and benchmark ZMQ submissions;
- replace TIFF inventory tests with one format-neutral source-axis contract
  suite covering TIFF and NPY through `FileManager` plus an undeclared
  multi-plane payload that remains one source;
- remove provider fallback, viability probe, image-plane resolver priority,
  auxiliary cache/materializer, and one-operation source projection tests;
  retain behavior assertions on `SourceBindingWorkspaceProjector` and
  `SourcePixelRef` directly;
- rewrite source-ref tests around the exact three-field mapping, direct backend
  registry dispatch, canonical `BioFormatsPlaneRef` address round-trip,
  inherited Bio-Formats virtual-workspace selection, and NPY fixture loading
  through disk source-axis indices;
- rewrite callable-loading tests around direct declaration lookup, assert no
  catalog or public wrapper callable is added, and prove import/compilation
  leave the raw function namespace unchanged;
- rewrite `tests/unit/test_cellprofiler_interop_import_records.py` around one
  final result with direct source path/module references, public steps/config,
  setting coverage, and generated source path; assert no provenance wrapper;
- rewrite `tests/unit/test_pipeline_definition.py` around list execution and the
  GUI-private `PipelineEditorStateRoot`;
- rewrite `tests/unit/test_function_patterns.py` and
  `tests/unit/test_pattern_data_manager.py` to reject third-tuple leaves and
  remove synthetic invocation-options subclasses;
- delete classifier expectations from `tests/unit/test_special_outputs.py` and
  retain callable ABI validation tests;
- delete `tests/unit/test_cellprofiler_processing_contract_resolution.py` and
  replace its genuine cases with raw-callable `CallableContract` ownership
  tests that assert no module-level processing field exists;
- rewrite output-value tests to provide full declared-output specs from the
  compiled module contract, remove `CellProfilerCallableOutputSpecs`
  expectations, and reject every semantic or count fallback;
- rewrite runtime-callable transport tests around one executor owning one
  three-field runtime plan; assert generic `FunctionReference` worker transport
  resolves the canonical callable and the compiled `RuntimeAdapterSpec` builds
  the executable from that callable plus enriched `CallableContract`, with no
  copied contract argument or rehydrator registry;
- rewrite run-request tests around one image field and executor-plan-owned
  callable/name properties;
- rewrite source-binding, image-number, plane-selection, object-label variant,
  sparse-row, and diagnostic tests against their existing owners instead of
  constructing one-field projection wrappers;
- replace `tests/unit/test_cellprofiler_source_schema_ingestion.py` with
  `tests/unit/test_cellprofiler_source_bindings_ingestion.py` around the exact
  submitted source root and config filters; remove ancestor-search,
  bucket-candidate, probe, and exclusion-policy expectations;
- remove hidden-contract expectations from
  `tests/unit/test_function_step_transport.py`,
  `tests/unit/test_cellprofiler_runtime_callable_introspection.py`, and pycodify
  tests;
- replace generated matcher candidate tests with exact provider key tests;
- replace generated-emission tests with assertions over real generated
  FunctionSteps plus exact generic pycodify output;
- replace infrastructure-profile/export-flag tests with axis/plate callable
  scope tests, exact `RuntimeArtifactBatch` selection tests, and real
  image/CSV/SQLite/properties output tests;
- remove `RuntimeImageExportSpec` and candidate in-memory image snapshot tests;
  image parity reads the files named by compiled materialization outputs;
- delete placeholder database/spreadsheet callable tests; compare all three
  official ExportToDatabase SQLite schemas/rows and CPA property keys without
  the manifest's `value_only` shortcut;
- migrate every `benchmark.cellprofiler_library` and
  `benchmark.cellprofiler_compat` import to its production owner;
- rewrite `benchmark/converter/convert.py` over
  `import_cellprofiler_pipeline(CellProfilerPipelineImportRequest)`, rewrite
  `compatibility_matrix.py` over `CellProfilerModule.__registry__`, and remove
  compatibility re-exports from `benchmark/converter/__init__.py`;
- rewrite FunctionStep/ZMQ/agent transport tests around direct lists and delete
  carrier-registry expectations.
- delete both OpenHCS result-cache policy/key/hit tests and add a benchmark
  integration assertion that two runs each submit compile and execution jobs;
  neither run reports zero-duration cached phases or cached provenance.
- replace timeout-derived native timing tests with measured-reference timing and
  missing-timing cases; the latter preserve parity and reject speed claims.

Genuine behavior coverage is preserved and moved to its authoritative owner.

## Resolved Questions

### Does Compilation Need The Original `.cppipe`?

No. The importer uses `.cppipe` to produce public typed steps and config. A
fresh compiler reconstructs all contracts from those declarations.

### Does Compilation Need `pipeline.metadata`?

No. Every source discovery and workspace fact is carried by the existing
`SourceBindingsConfig` inside `PipelineConfig`; no retained source schema or
import result participates. Runtime semantics derive from config, callables,
kwargs, and module declarations.

### Does CellProfiler Need A Symbol Table?

No CP-specific symbol model is required. The compiler needs an ordered derived
index of existing artifact specs while precomputing contracts. The generic
artifact graph remains authoritative.

### Does CellProfiler Need A Separate Setup Registry?

No. Setup modules are CellProfiler modules. Their declarations live in the
existing module registry, fold directly into `SourceBindingsConfig`, and opt
out of executable FunctionStep emission through their MRO.

### Does Callable Ownership Need Wrapper Metadata?

No. `CellProfilerModule.for_function_name` resolves the owner and the owner
loads its implementation function. The implementation's generic decorators
own source callable metadata, and the invocation provider creates the enriched
immutable compiled contract without mutating that function. No catalog, CP
ownership attribute, metadata DTO, callable cache, or forwarding function
remains.

### Does CellProfiler Need Port Classes?

No. Existing artifact specs, partitions, and nominal module methods contain the
same information without a parallel taxonomy.

### Are Export Modules Ordinary Axis Callables?

SaveImages is. ExportToSpreadsheet and ExportToDatabase are not: they aggregate
artifacts across compiled axes. The missing distinction is generic callable
execution scope, not CP infrastructure metadata. The compiler carries that one
scope fact, builds one exact `RuntimeArtifactBatch` from module-contract inputs,
and invokes each plate callable once before generic file-bundle materialization.

### Does CellProfiler Need A Runtime Wrapper?

Axis-scoped CP callables need exactly one. They do not accept the runtime adapter
that loads module-managed artifacts and records outputs. Generic execution
selects grouped artifact plans before that adapter is built, so the adapter
needs neither a grouped wrapper nor a contract map. Plate-scoped callables use
the generic raw-callable path and `RuntimeArtifactBatch`, so they use no CP
wrapper.

### Does CellProfiler Import Need Its Own Source Model?

No. The direct import operation constructs real FunctionSteps and config.
Generic pycodify renders those values, and the ZMQ server is the sole
fresh-source reconstruction path. No generator object or generated-pipeline
result remains.

### Does ZMQ Need A Pipeline List Wrapper?

No. Concrete request/session/context records store direct mutable lists.
`FunctionStepTransportAuthority` owns the export name, source rendering,
normalization, and namespace validation.

### Do Special Inputs And Outputs Remain?

Yes, as callable ABI declarations. They do not own artifact semantics.

### Are Input And Output Names Public Kwargs?

Canonical flow names are compiler-derived. An intentional noncanonical identity
override is public, sparse, compile-only, and consumed before raw invocation.
A user-selected semantic label that also appears in runtime results, such as
CalculateMath `output_name`, remains a normal public callable kwarg.

### Does `RuntimeInvocationOptions` Remain?

No. Its only production subclasses are the two CP option containers being
eliminated. CalculateMath exposes `output_name` as a typed callable kwarg while
deriving operand object identity from contracts, and DefineGrid exposes cycle
scope as a typed callable kwarg. Delete the generic base, third-tuple pattern
shape, hidden runtime parameter, and UI/runtime plumbing; synthetic tests do
not justify a production abstraction.

### Does Napari Need A Benchmark Flag?

No. `run_comparison_suite` already accepts `openhcs_global_config`, and
`GlobalPipelineConfig` already owns inherited Napari configuration and port.

### Is Direct Orchestrator Execution An Acceptance Path?

No. ZMQ compile-then-execute from pycodified public source is the sole parity
acceptance path. It includes the same compiler and worker execution used by
local orchestration while also testing serialization and fresh-process
reconstruction.

## Readiness Decision

The corrected implementation plan is ready to execute from Phase 0 through
Phase 3. Each phase names its prerequisites, exact owners, required deletions,
forbidden replacements, focused tests, and exit criteria. The audit leaves no
new taxonomy to design during implementation and no hidden state authorized as
a temporary compatibility path.
