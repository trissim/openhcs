# Superseded: CellProfiler / OpenHCS Runtime Unification Plan

## Status And Scope

### Sequencing Override For Pending Phase 3 (2026-07-13)

Do not implement further Phase 3 parity fixes against the current runtime
callable, execution-strategy, adapter-cache, or CellProfiler projection layers.
Before continuing the pending official30 ZMQ, export, UI, and Napari work, read
and implement
[the runtime/import/dispatch consolidation plan](cellprofiler_runtime_execution_import_dispatch_consolidation_plan.md)
Phases 0 through 5. Then run that plan's Phase 6 once against the surviving
architecture; its result also discharges this plan's Phase 3 acceptance gate.
This ordering prevents repairing parity on layers that the consolidation
deletes and then repairing it a second time after cutover.

This plan is the implementation contract for removing the parallel CellProfiler
pipeline runtime. It is based on a dry run through the current compiler,
configuration inheritance, function-pattern normalization, artifact planning,
source binding, callable special I/O, CellProfiler module declarations,
pycodification, ZMQ reconstruction, and worker execution paths.

The implementation result is fixed:

- A public pipeline is `PipelineConfig` plus `list[FunctionStep]`.
- The compiler derives CellProfiler invocation contracts from those public
  declarations.
- The generated Python source is sufficient to reproduce compilation in a fresh
  ZMQ server process.
- The generated source contains no compiled contract, module block, module
  number, runtime wrapper, symbol table, sidecar, or pipeline metadata.
- Existing OpenHCS artifact, source-binding, configuration-inheritance,
  function-pattern, special-I/O, and runtime machinery remains load-bearing.
- CellProfiler-specific semantics live on the registered `CellProfilerModule`
  class that owns the module.
- One CellProfiler runtime callable adapter connects one compiled
  `ModuleArtifactContract` to the generic axis-scoped FunctionStep runtime;
  plate-scoped callables use the generic post-plate path.
- Superseded authorities are deleted in the phase that replaces their last
  consumer.

### Atomic Cutover Rule

Every phase performs a complete migration. It updates all production callers,
generated source, serializers, deserializers, tests, fixtures, imports, and
re-exports before deleting the superseded symbols and files in that same
phase. The resulting tree contains no forwarding alias, deprecated facade,
compatibility reader, compatibility writer, dual serialized shape, state
upgrade branch, fallback lookup, dormant registry entry, or retained test for
the removed design. Persisted generated Python, virtual-workspace metadata,
and benchmark output are regenerated from the new public declarations; they
are never adapted at load time. A phase exits only after AST and `rg` deletion
gates prove that the old authority has no production/test import, definition,
call, attribute read, re-export, or surviving file.

This plan introduces one non-public GUI state record:

- `PipelineEditorStateRoot`, in
  `openhcs/pyqt_gui/services/pipeline_object_state_binding.py`, owns only the
  editor display name, description, and step ObjectState scope IDs. ObjectState
  requires a nominal reconstructable root for this UI state. No existing type
  owns those three editor-only values. The type does not contain steps,
  configuration, metadata, runtime contracts, sequence methods, source, or an
  execution API. It is private to `PipelineObjectStateBinding`; no compiler,
  import, transport, orchestrator, agent, or benchmark signature accepts it.

It replaces one over-broad nominal root in the existing `CellProfilerModule`
MRO:

- `SourceSetupCellProfilerModule` owns the behavior shared by `Images`,
  `LoadImages`, `Metadata`, `NamesAndTypes`, and `Groups`: contribute to the
  existing generic `SourceBindingsConfig` and remain absent from executable
  FunctionSteps. These five modules currently
  live in a separate `SetupModuleCompiler` registry, so this root removes a
  registry and gives genuinely shared behavior one declaration owner.
  `InfrastructureCellProfilerModule` is deleted because it currently groups
  source setup, exports, metadata transforms, and unsupported pass-through
  stubs under the unrelated rule “has no callable.” Supported observable
  modules move to axis- or plate-scoped executable declarations; declarations
  that only pretend to support an unavailable effect are removed and fail
  import.

The dry run proves four generic value types are missing from existing
execution/materialization lifecycles:

- `FunctionStepExecutionScope` distinguishes the existing per-axis lifecycle
  from post-plate callable execution.
- `RuntimeArtifactBatch` is the immutable exact artifact input supplied to one
  plate-scoped callable.
- `ImageFileOptions` registers one-image-file materialization through the
  existing image serialization authority.
- `FileBundleOptions` registers safe persistence for a callable-produced
  insertion-ordered `dict` of relative file paths to text or bytes.

All other accepted changes extend or replace behavior on an existing owner:
`CallableContract` validates public kwargs during compilation;
`CompositeInvocationContractProvider` enforces one claimant;
`CompiledFunctionInvocation` carries exact artifact refs; `FileManager`
preserves and mutates its execution-local registry; `BackendBase` receives that
registry; `DataSource` interprets its opaque address; and the source-component
projection family becomes exact enum-keyed dispatch. These changes introduce
no public config, artifact taxonomy, semantic carrier, or parallel registry.

The closed replacement-type list is also explicit:

- `CellProfilerInvocationContractProvider` replaces the generated/matching
  provider with the existing generic provider ABI and one immutable exact-key
  implementation;
- `SourceComponentProjectionStrategy` plus one leaf per component replaces the
  multi-leaf priority `ComponentProjection` family at the same generic source
  projection boundary;
- `CellProfilerSourceImageType`, `CellProfilerSourceFilterSubject`,
  `CellProfilerSourceFilterOperator`, and `CellProfilerSourceMatchMethod`
  replace dynamic role classes and one-attribute literal resolver leaves only
  at the CP import edge;
- `SourceCandidate` is the renamed, narrowed existing validated source
  candidate and owns one `SourcePixelRef` rather than introducing another
  source identity.

`MaterializationValue` already includes `bytes`, but PolyStore currently routes
every disk save through its extension registry and therefore rejects opaque
files such as SQLite databases. This is an implementation defect in the
existing generic byte-payload boundary, not a fifth responsibility. The disk
backend must write a `bytes` payload exactly with `Path.write_bytes` before
consulting format-specific extension handlers. `FileBundleOptions` encodes
text entries as UTF-8 bytes and uses that same generic path. No database,
properties-file, or CellProfiler suffix is registered in PolyStore.

Beyond the records, roots, values, owner methods, and exact replacements listed
above, no new public or private data carrier, strategy family, registry, or
descriptor taxonomy is authorized by this plan.

## Verified Current Path

### Generic Compilation

The generic compiler already has the required owners:

1. `openhcs/core/pipeline/step_snapshot.py::StepSnapshot` captures the resolved
   `FunctionStep` and inherited step configuration.
2. `openhcs/core/function_patterns.py::normalize_function_pattern` assigns a
   `FunctionInvocationKey` to every callable invocation. A plain callable is a
   default invocation applied by the runtime to every discovered group. A dict
   pattern exists only for subgroup omission or group-specific behavior.
3. `openhcs/core/invocation_artifacts.py::PipelineInvocationContractProviderAuthority`
   asks registered `InvocationContractProviderFactory` implementations for a
   compile-only `InvocationContractPlan`.
4. `openhcs/core/pipeline/artifact_planning.py::extract_artifact_declarations`
   projects the resulting `CallableContract` into existing artifact
   declarations.
5. `openhcs/core/pipeline/path_planner.py::PipelinePathPlanner` uses the same
   provider during reverse future-input planning, forward declaration, and
   compiled invocation construction.
   `StepInputDependency` owns the main-flow edge. Extend its existing closed
   enum with `NO_MAIN_FLOW` for artifact-only plate callables; do not represent
   that state as a fake pipeline-start or previous-step image edge.
6. `openhcs/core/function_patterns.py::CompiledFunctionInvocation` carries the
   compiled `CallableContract` into execution.
7. `openhcs/core/steps/function_runtime.py` loads declared artifacts, invokes the
   compiled callable, and delegates returned-value matching to
   `openhcs/core/runtime_output_matching.py::RuntimeReturnedOutputMatcher`.

The CellProfiler compiler provider therefore performs one ordered prepass over
the session and answers every later planner query by exact invocation key. It
does not rebuild or match contracts independently in each planner pass.

### Configuration And Source Binding

The existing hierarchy is authoritative:

- ObjectState's existing lazy MRO resolver resolves pipeline defaults and sparse
  step overrides into a complete resolved `FunctionStep` before compilation.
- `StepSnapshot` stores compiler identity plus that resolved step. The compiler
  and path planner read `snapshot.step.processing_config`,
  `snapshot.step.source_bindings`, and every other resolved config directly.
- `CallableRuntimeConfig` remains the registered callable-configuration
  injection authority. Compilation adds no CellProfiler mirror of config
  inheritance or runtime parameter binding.

`StepSourceBindingsConfig` inherits every source declaration field from
`SourceBindingsConfig`; the existing `global_pipeline_config` / lazy-dataclass
factory projects every inherited dataclass field to the lower scope. Do not
redeclare a manually selected subset of source fields on
`StepSourceBindingsConfig`, and do not copy resolved configs into a second
snapshot carrier.

Delete `StepSourceBindingsConfig.inherits_*`, `resolved_against`,
`can_inherit_from`, `resolve_step_source_bindings`,
`resolve_effective_step_source_bindings`,
`_resolve_step_source_binding_defaults`, and their per-field comparison
helpers. The ObjectState-resolved `FunctionStep` already owns the effective
`StepSourceBindingsConfig`; the compiler and path planner use that exact object.
Change
`CompiledSourceBindingPlan.from_config(config, *, input_source)` to derive
activation from `config.enabled` or
`input_source is InputSource.PIPELINE_START`; the provider, compiler, and path
planner call this same method. Activation does not trigger a second config
overlay. Generated import compares resolved source config values, emits
`LazyStepSourceBindingsConfig` only for a real lower-scope difference, and
never maintains a field-by-field inheritance predicate. Concretely, the import
pass constructs the broad `PipelineConfig`, forms the candidate lazy step
source config from the parsed module contract, and resolves both that candidate
and `LazyStepSourceBindingsConfig()` directly inside the existing
`objectstate.config_context(pipeline_config)` scope with
`resolve_lazy_configurations_for_serialization`. This is the same lazy MRO
resolution used for ordinary step construction and requires no placeholder
FunctionStep or manual overlay. The import pass compares the two complete
resolved source config dataclasses and, after processing config is derived,
their `CompiledSourceBindingPlan.from_config(..., input_source=...)` values.
Equality therefore covers every current and future dataclass field plus
activation without a copied field list.

Remove the untyped `bindings_for_group_key` / `for_group_key` API. Keep one
typed `bindings_for_component_group(component, group_key)` operation and make
it exact: an unscoped `(None, None)` request returns all declarations; a scoped
request returns only bindings whose `ComponentSelector` has that exact
component and value; zero matches raises `ValueError`. The compiler calls the
scoped operation only for keys returned by
`source_binding_group_keys_for_group_by`, so an invocation with no declared
source component split remains explicitly unscoped rather than entering a
fallback branch.

The importer emits lazy configuration overrides only where the
source `.cppipe` differs from the inherited pipeline value.

Pipeline-level ObjectState resolution runs exactly once before axis planning.
`PipelineCompiler.initialize_step_plans_for_context` requires the resolved
steps, their ObjectState map, and their `StepSnapshot` tuple. Delete the
context-local step resolver and the `steps_already_resolved` switch; tests that
exercise a lower compiler layer construct their `CompilationSession` from the
same pipeline-resolved values rather than registering a second scope tree.

`ArtifactDeclarationStepContext` owns only facts required to project artifact
declarations: resolved `source_bindings`, exact `group_by`, exact
`input_source`, available artifacts, available producers, and current main
flow. It does not own `ProcessingConfig`. Delete the unused
`source_provenance` and `processing_config` fields and their propagation.
`CompileFunctionGroupAuthority.from_step_context` copies the exact
`group_by`/`input_source` values needed by the provider rather than a broad
config object.

`PipelineImageSchema` is not retained. It currently exists because
`SourceBindingsConfig` omits source-plane, stack, grouping, voxel-spacing,
imported-metadata, and payload-loading facts. Keeping both would violate the
public boundary. Move the missing generic facts to their existing config owner:

```python
@dataclass(frozen=True)
class SourceBindingsConfig(SourceBindingDeclarationsMixin, _SourceBindingPlanBase):
    source_filters: tuple[SourceFilterClause, ...] = ()
    bindings: tuple[NamedSourceBinding, ...] = ()
    image_plane_sources: tuple[ImagePlaneSource, ...] = ()
    imported_metadata_tables: tuple[ImportedMetadataTable, ...] = ()
    source_stack_components: tuple[AllComponents, ...] = ()
    grouping_metadata_fields: tuple[str, ...] = ()
    source_voxel_spacing: SourceVoxelSpacing = field(
        default_factory=SourceVoxelSpacing
    )


@dataclass(frozen=True, slots=True)
class NamedSourceBinding(SourceAssignmentBase):
    artifact_kind: type[ArtifactType] = ImageArtifactType
    required: bool = True
    participates_in_image_stack: bool = True
    load_as_monochrome: bool = False
    load_as_mask: bool = False
    source_channel_axis: int | None = None
    source_channel_counts: frozenset[int] | None = None
```

`SourceBindingsConfig.__post_init__` normalizes all tuple fields, validates
unique aliases, canonicalizes stack components through `ComponentSet.collect`,
trims grouping fields, and includes every new field in `is_empty`.
`NamedSourceBinding.__post_init__` rejects channel-count constraints without a
declared source channel axis and normalizes a present count set to positive
integers. A negative axis remains relative to the loaded array rank. Source
workspace code reads these declared fields directly; it has no payload-type
string, role lookup, or representability fallback.

Move `binding_for_alias` from `CompiledSourceBindingPlan` to the existing
`SourceBindingDeclarationsMixin`, then use that one implementation from editable,
resolved, and compiled plans. Add `require_binding_for_artifact`,
`loaded_image_aliases`, and `measurement_source_names` to the
same mixin. The first method validates the declared `artifact_kind`; the two
properties derive directly from `binding_declarations` and
`NamedSourceBinding.measurement_source_names`. Compiler, measurement, source
inventory, and runtime consumers call these methods instead of rebuilding
alias maps or source-name sets.

Runtime pixels do not retain a CellProfiler image-type label or role object.
Add `source_channel_axis: int | None` to the existing
`ImagePayloadMetadata`. Its normalization, `has_values`, source-plane
projection, metadata composition, and replacement methods preserve one
compatible axis and reject conflicting non-null axes. Add
`is_declared_source_channel_plane(data)`,
`is_declared_source_channel_stack(data)`, and
`without_source_channel_axis()` on that metadata owner. The first two normalize
the declared axis against the actual rank and distinguish one 2D image with an
internal channel axis from a leading OpenHCS stack of such images.

Replace `SourceImagePayloadSemantics` and its role-strategy registry with one
typed `apply_source_binding_payload(payload, binding, source_context)` function
in `source_image_semantics.py`. It reads the resolved `NamedSourceBinding`,
applies monochrome and mask conversion from that declaration,
normalizes label-representation inputs when
`binding.artifact_kind.uses_label_representation_payload_shape` is true, and
attaches ordinary source provenance plus the resolved source channel axis to
`ImagePayloadMetadata`. It validates `source_channel_counts` at load time. It
contains no CellProfiler label, concrete module name, role registry, or
concrete artifact-type branch. `aligned_image_payload.py` and pure-2D aggregation query
the metadata methods directly. `color_to_gray` calls
`without_source_channel_axis()` after collapsing the channel axis.

Delete `SOURCE_IMAGE_TYPE_METADATA_FIELD`, the `OpenHCSImageType` metadata
token, `SourcePlaneProjection.image_type`, the image-type special case in
`source_matching.py`, `source_image_payload_role`, and all runtime imports of
`ImageTypeSourceRole`. `SourcePlaneProjection.source_alias` remains the typed
link from a persisted source plane to the binding in the submitted config;
source loading resolves that alias through `SourceBindingDeclarationsMixin`
before applying payload semantics. A downstream payload therefore carries only
generic runtime layout and provenance, not an importer vocabulary token.

Move the behavior-bearing `ImagePlaneSource`, `ImportedMetadataJoin`, and
`ImportedMetadataTable` types to `openhcs/core/source_bindings.py`. Collapse
`ImagesRule` into `source_filters`, `SourceImageStackPlan` into
`source_stack_components`, and `GroupingPlan` into
`grouping_metadata_fields`. Collapse `ImageAssignment` and
`SourceArtifactAssignment` into `NamedSourceBinding`; component identity already
represents source-alias identity. Move CellProfiler image-type parsing out of
generic core into `processing/backends/cellprofiler/infrastructure.py`. Replace
the generated role class hierarchy with one `CellProfilerSourceImageType` enum.
Each member owns one parsed CP label and its generic `NamedSourceBinding` field
values, including `source_channel_axis=-1` for CellProfiler color sources. Its
`binding(alias, selector, origin)` method constructs the generic binding
directly. `NamesAndTypes.contribute_source_bindings` resolves each setting
through `CellProfilerSourceImageType`; no role object exists after setup
lowering. Delete both dynamic role-class factories, their class-spec records,
and the role registry; generic source code imports no CP image-type symbol.

Replace `SourceSchemaLiteralResolver` and its subject/operator/match-method leaf
classes with three CP-local enums in `infrastructure.py`:
`CellProfilerSourceFilterSubject`, `CellProfilerSourceFilterOperator`, and
`CellProfilerSourceMatchMethod`. Each member owns one external CellProfiler
literal and one existing generic enum value; the operator member also owns its
negated generic value. Private setup parsing functions resolve those enums
directly. This preserves a nominal translation boundary without fourteen
one-attribute classes or a literal-to-semantic dictionary.

Delete `NamesAndTypesAssignmentLayout`,
`NAMES_AND_TYPES_ASSIGNMENT_LAYOUTS`, and
`NamesAndTypesAssignmentBlockStrategy`. The NamesAndTypes declaration owns
`SettingNameFamily` ClassVars for assignment aliases, image type, rule
criteria, and match metadata. Its private parser reads each repeated setting
column in source order, uses the ordered image/object alias rows as the exact
assignment count, accepts only one shared value or one value per assignment,
and rejects every other cardinality. It pairs the nth repeated value with the
nth alias; no layout probing, priority order, or fallback parser remains.
Apply the same indexed-setting-column rule to repeated LoadImages rows.

Fold setup modules in source order through
`SourceSetupCellProfilerModule.contribute_source_bindings`, which returns
`SourceBindingsConfig`. The immutable config returned by the final setup module
is written directly to `PipelineConfig.source_bindings_config` as the existing
lazy config type. Consolidate `source_schema_workspace.py` into the already
existing `source_binding_workspace.py`, then delete the former file. The
retained public workspace surface is exactly
`SourceBindingWorkspaceProjector` and
`SourceBindingWorkspaceMaterialization`, plus the validated `SourceCandidate`
record shared with inventory; source inventory stays on the
existing `SourceInventory` value in `source_bindings_view.py`. Workspace
materialization, source inventory, UI
source-binding context, compiler source-universe planning, and runtime source
loading accept the resolved `SourceBindingsConfig`. Delete
`PipelineImageSchema`, `PipelineImageSchemaBuilder`, source-schema assignment
types, representability feature registry, projection, and every retained
`source_schema` field. The converted config and FunctionSteps are therefore
sufficient to recreate the workspace in a fresh ZMQ process.

`SourceBindingWorkspaceProjector.materialize(...)` performs one deterministic
flow: discover the files explicitly supplied by the plate/VFS plus resolved
`image_plane_sources`; extract metadata using the config; match
`binding_declarations`; expand declared source-stack axes using each matched
binding's source channel axis; assemble image sets through the existing
`SourceBindingMatchMethod`; project component identities; and write the
ordinary OpenHCS virtual workspace. Existing OpenHCS workspace metadata is
recognized by `SourceBindingsHandler.initialize_workspace` before this method
is called. Delete `SourceSchemaCandidateProvider`, discovery mode, viability
probe/result/request classes, and provider-priority traversal. The projector
never probes alternate source roots or retries a second provider.

Explicit `ImagePlaneSource` resolution is scheme-deterministic. Import resolves
an empty or `file` URI against the declared source root and verifies the exact
path. Import materializes `http` and `https` content atomically under
`get_openhcs_cache_dir() / "source_imports" / sha256(normalized_uri)`, using
`sha256(response_bytes)` plus the declared suffix as the file name, and stores
that resolved local path in the config. The existing XDG cache authority owns
the location; no source-cache config is added. Every other scheme fails
conversion.
Delete `ImagePlaneSourceResolutionStage`, `ImagePlaneSourceResolver`, and its
three priority leaves; basename-equivalent local-file matching and network
fallback at workspace runtime disappear.

Source-file plane expansion is contract-driven and format-neutral. A source
config without `source_stack_components` yields one source candidate regardless
of file suffix. A config with source stack components loads the source once
through the supplied `FileManager` and source backend, uses
`ImageArrayShapeSemantics` plus the binding's `source_channel_axis` to validate
the declared leading source axes, and emits their Cartesian index product.
Replace the existing `SourcePixelRef` field shape with the exact generic form
below, including `source_axis_indices: tuple[int, ...]`;
`VirtualWorkspaceBackend` slices those leading axes in order after generic
backend loading. Candidate metadata assigns
each declared component its one-based semantic value. Delete
`SourceSchemaSourcePlaneInventory`, `TiffPageSourcePlaneInventory`,
`TiffSourcePlaneInventory`, `SinglePlaneSourcePlaneInventory`, and every TIFF
suffix/page-count branch. TIFF, NPY, raster, and later VFS formats therefore use
the same reader and axis path.

Make the workspace mapping representation singular at the same boundary.
Move `SourcePixelRef` from `openhcs/core/source_projection.py` to its storage
owner, `external/PolyStore/src/polystore/virtual_workspace.py`; OpenHCS source
projection imports that nominal type. Its complete field set becomes:

```python
@dataclass(frozen=True, slots=True)
class SourcePixelRef:
    backend: str
    backend_address: str
    source_axis_indices: tuple[int, ...] = ()
```

The type validates nonempty backend/address values and nonnegative indices.
`to_workspace_mapping` emits exactly those three keys, with indices as a JSON
array; `from_workspace_mapping` rejects missing, extra, or wrongly typed keys.
Delete the old generic `reader`, `source_path`, `series_index`, `plane_index`,
`source_channel`, `source_z_index`, and `source_timepoint` fields. Backend-owned
address syntax never enters this generic type. Delete
`SourcePixelRef.source_metadata`; `SourceProjectionSet` writes only the
projection's semantic source metadata, source alias, and canonical component
identity. The workspace mapping remains the sole storage-address record.

Rename `SourcePixelRef.to_legacy_workspace_mapping` to
`to_workspace_mapping` and add the inverse validated constructor
`from_workspace_mapping`. Migrate BBBC, ImageXpress, Opera Phenix,
Bio-Formats, source-binding, and every other workspace writer to emit that
structured value even when it contains only a disk path. Change
all consumers to call `SourcePixelRef.from_workspace_mapping` directly; delete
the one-operation `workspace_mapping_source_ref` and
`workspace_mapping_source_path` helpers. No string mapping value remains.

Keep the existing PolyStore `BioFormatsPlaneRef` as the sole owner of the
physical path, series index, and plane index required by the Bio-Formats
reader. Remove its `reader`, `c`, `z`, and `t` fields. Add deterministic
`to_backend_address()` and
`from_backend_address()` methods using canonical compact JSON and make
`BioFormatsStorageBackend.load` / `load_batch` consume that address directly.
Delete its workspace-metadata cache, virtual-path resolver, metadata-file read,
second virtual listing implementation, reader switch, `_load_npy_plane`, and
`BioFormatsReaderUnavailableError`. `BioFormatsWorkspaceMetadataWriter`
encodes each production plane through `BioFormatsPlaneRef.to_backend_address`;
its NPY test projections use the ordinary disk backend plus generic
`source_axis_indices`.
`BioFormatsHandler` registers the direct Bio-Formats reader, registers
`VirtualWorkspaceBackend`, and uses the inherited primary-backend selection;
delete its Bio-Formats-only `get_primary_backend` override. Bio-Formats thus
uses the same virtual mapping boundary as the other microscope handlers.

Make `FileManager.registry` execution-local across construction and worker
reconstruction. `FileManager.__init__` copies the supplied registry and calls
the polymorphic `BackendBase.bind_registry(registry)` hook on every backend;
the root implementation performs no work. Add
`FileManager.register_backend(name, backend)` as the sole mutation operation,
bind the updated local registry there, and migrate every direct registry write.
`FileManager.__getstate__` stores the exact registered backend keys plus
connection parameters for `PicklableBackend` instances.
`FileManager.__setstate__` rebuilds a fresh local mapping from the existing
`BackendBase.__registry__`, constructs `PicklableBackend` instances through
`from_connection_params`, default-constructs the remaining declared backend
types, and binds that completed mapping. It never reads or mutates the global
`storage_registry`.

Give `VirtualWorkspaceBackend` that execution-local registry through its
`BackendBase.bind_registry` override. Parse each mapping through
`SourcePixelRef.from_workspace_mapping`, reject a `virtual_workspace` source
backend, and resolve `ref.backend` by one exact lookup in the bound registry.
Require the selected value to inherit `DataSource`; a streaming-only sink fails
before loading.
Add `DataSource.resolve_address(backend_address, *, base_path)` with a default
that returns the opaque address unchanged. `DiskStorageBackend` alone resolves
a relative disk address against `base_path`; Bio-Formats and remote backends
interpret their own address unchanged in `load` / `load_batch`.
`VirtualWorkspaceBackend` calls the selected backend's `resolve_address`, then
its ordinary `load` / `load_batch`, and applies ordered
`source_axis_indices` once. `requires_filesystem_validation` remains a
validation capability and is never used as an address-rewrite discriminator.
`from_connection_params` restores only the plate root; the containing
`FileManager` injects the reconstructed execution-local registry after all
backends exist.
Delete `VirtualWorkspaceSourceRefResolver`, `PathSourceRefResolver`,
`DiskSourceRefResolver`, `VirtualWorkspaceResolvedRef`, shape predicates,
priority sorting, and resolver batching. Batch grouping uses local tuples of
backend key and source address. The existing backend registry and backend
instances remain the sole loading authority; no replacement resolver or source
format registry is added.

The retained candidate value is:

```python
@dataclass(frozen=True, slots=True)
class SourceCandidate:
    source_ref: SourcePixelRef
    relative_path: str
    metadata: SourceMetadataMapping
    source_axis_shape: tuple[int, ...] = ()
    source_filter_paths: tuple[str, ...] = ()
```

It validates the nominal source ref, equal
`source_ref.source_axis_indices`/shape cardinality, in-range nonnegative
indices, and normalized relative/filter paths. Its `identity_key()` contains
`source_ref.backend`, `source_ref.backend_address`, ordered source-axis
indices/shape, relative/filter paths, and
`SourceMetadataIdentityProjection(metadata).items()`; no separate identity dataclass
or path/loading field remains. Candidate discovery constructs the source ref
from the supplied `FileManager` backend and canonical address. Explicit source
declarations retain their own refs, so one source universe supports mixed
backends. Source-axis expansion uses `dataclasses.replace` on the candidate and
its `SourcePixelRef`; workspace projection writes `candidate.source_ref`
directly. `SourceBindingContext.source_backend` is only the discovery default
for file paths without an existing ref, never candidate identity.

Delete `SourceSchemaImageSetSelection`, `max_image_set_count`,
`source_schema_image_set_selection`, and all copies on importer, benchmark,
adapter, CLI, request, and materialization records. Workspace construction
materializes the complete source universe declared by `SourceBindingsConfig`.
Pipeline execution selection occurs only through the inherited existing
`WellFilterConfig` / ZMQ axis filter after canonical component identities
exist. `run_comparison_suite` already accepts `openhcs_global_config`; that is
the suite's sole OpenHCS configuration input. The OpenHCS and native adapters
both resolve its existing `well_filter_config`, with the native adapter using
the same `WellFilterProcessor` over projected source wells. The OpenHCS adapter
receives no private source-selection object. Remove the source-schema well/max
CLI flags and manifest fields rather than renaming them to more config-specific
benchmark flags.

Keep `ImageSetAssembler` only as a behavior strategy keyed directly by the
existing `SourceBindingMatchMethod` through `EnumKeyedStrategyMixin`; remove its
manual string key and schema argument. Replace the ten ordered
`ComponentProjection` leaves in the retained
`openhcs/core/source_binding_workspace.py` with this exact nominal family:

```python
class SourceComponentProjectionStrategy(
    EnumKeyedStrategyMixin[AllComponents],
    ABC,
    metaclass=AutoRegisterMeta,
):
    strategy_key: ClassVar[AllComponents | None] = None

    @classmethod
    def project_component(
        cls,
        component: AllComponents,
        metadata: Mapping[str, str],
        image_set_index: int,
    ) -> str:
        ...

    @classmethod
    def metadata_component(
        cls,
        component: AllComponents,
        metadata: Mapping[str, str],
    ) -> str | None:
        ...

    @abstractmethod
    def project(
        self,
        metadata: Mapping[str, str],
        image_set_index: int,
    ) -> str:
        ...

    @abstractmethod
    def metadata_value(self, metadata: Mapping[str, str]) -> str | None:
        ...
```

The two classmethods perform one
`SourceComponentProjectionStrategy.for_enum_member(component)` lookup and
delegate. Declare exactly
`WellSourceComponentProjection`, `SiteSourceComponentProjection`,
`ChannelSourceComponentProjection`, `ZIndexSourceComponentProjection`, and
`TimepointSourceComponentProjection` for the current five `AllComponents`
members. Every leaf owns its canonical metadata key, ordered aliases, compound
projection, and deterministic ordinal/singleton default in its two methods:

- WELL reads `well`, then compounds row aliases `wellrow` / `row` with column
  aliases `wellcolumn` / `wellcol` / `column` / `col`, and defaults to `A01`;
- SITE reads `site`, then `imagenumber`; frame or Z metadata without a site
  projects to `1`, and metadata-free image sets use the one-based image-set
  ordinal;
- CHANNEL reads `channel`, then `channelnumber`, and uses the one-based binding
  ordinal supplied as the zero-based `image_set_index` argument;
- Z_INDEX reads `zindex`, `z`, `zplane`, `zslice`, `plane`, then `slice`, and
  defaults to `1`;
- TIMEPOINT reads `timepoint`, `time`, `framenumber`, then `frame`, and defaults
  to `1`.

The root knows none of those leaf facts. Import-time registry validation
requires the `strategy_key` enum members declared by the registered strategy
types to equal `set(AllComponents)` exactly. The existing
`EnumKeyedStrategyMixin` retains its JSON-safe enum-value registry keys, so a
new configured component requires one nominal projection leaf rather than
entering a generic fallback. Delete direct-metadata probing on the root, two-pass
metadata/default traversal, MRO priority ordering, and the separate channel
projection wrapper.

Make the source view/inventory surface equally singular. In
`source_bindings_view.py`, remove `SourceBindingView.from_schema_assignment`
and its payload-type string. Rename
`PipelineSourceUniverseView.from_schema` to `from_config`,
`SourceBindingsViewModel.from_schema_and_bindings` to
`from_config_and_step_bindings`, and
`SourceBindingsPreview.from_schema_and_bindings` to
`from_config_and_step_bindings`. Change `SourceInventoryBuildRequest.schema`
to `source_bindings: SourceBindingsConfig`, then collapse the request into
`SourceInventory.from_filemanager(...)`. Delete `SourceInventoryProvider` and
all five provider leaves; the registry is never queried and
`SchemaContextSourceInventoryProvider` is another precedence chain.
`SourceBindingContext` stores the resolved `SourceBindingsConfig`, actual
`FileManagerLike`, and source backend beside its existing paths.
`SourceBindingContext.inventory` calls `SourceInventory.from_filemanager`
directly with those values plus the requested step override. Pipeline
and step rows are both rendered from `NamedSourceBinding`; declaration scope
is presentation data only. No view or preview constructs a source assignment,
payload-type token, schema projection, or alternate matching plan.

The resulting context shape is exact:

```python
@dataclass(frozen=True, slots=True)
class SourceBindingContext:
    logical_plate_id: str
    display_plate_root: Path
    execution_plate_path: Path
    source_bindings: SourceBindingsConfig
    filemanager: FileManagerLike
    source_backend: str

    def inventory(
        self,
        step_bindings: StepSourceBindingsConfig = StepSourceBindingsConfig(),
    ) -> SourceInventory:
        ...
```

`cppipe_path`, `import_result`, `source_schema`, and `inventory_provider` are
absent.

Remove projection/request classes whose only operation becomes one local
projector function: `SourceSchemaCandidateIdentity`,
`SourceSchemaCandidateCollection`, `SourceSchemaCandidateMetadataRequest`,
`SourceSchemaCandidateMetadataResolver`, `SourceSchemaCandidateMatches`,
`SourceSchemaImageSetIdentity`,
`SourceSchemaVirtualFilename`, `SourceSchemaFilenameProjection`,
`WorkspaceMappingSink`, `SourceVirtualPathMetadata`,
`VirtualComponentOriginalMetadataProjection`, `SourceMetadataJsonRecord`,
`SourcePlaneGroupSiteAllocator`, `ImageSetMetadataMerge`, and
`SharedCandidateMetadataProjection`. Rename the current validated candidate to
`SourceCandidate` and retain it because workspace projection and public
`SourceInventory` both consume its path, metadata, filter identities, and
source-axis indices. Retain internal `_SourceImageSet` and
`_ImportedMetadataIndex` records only where multiple operations consume their
validated fields. Virtual path formatting delegates
to the existing `OpenHCSPlaneAddress` and
`SourceProjectionMetadataSerializer` authorities.

Setup import resolves every path-bearing `ImagePlaneSource.uri` and
`ImportedMetadataTable.location` against the explicit source root or `.cppipe`
parent before storing the config. Runtime workspace preparation receives the
ordinary plate path plus the resolved config; it never needs parser provenance
or a `.cppipe` path to interpret a relative setup value.

Delete `_AUXILIARY_PAYLOAD_CACHE`,
`cache_source_schema_auxiliary_payload`,
`source_schema_auxiliary_payload`, and its clear/key helpers. The workspace
mapping records each auxiliary source's actual backend and source path. Runtime
source loading reads those refs through the context's generic `FileManager` for
every backend; remove the disk-specific `np.load` branch and all process-local
payload lookup. Delete `SourceSchemaAuxiliaryMaterializer`,
`NumpyAuxiliaryMaterializer`, `SourceSchemaAuxiliaryMaterializationRequest`,
`SourceSchemaAuxiliaryTargetPathPolicy`, and its sole basename leaf. No source
format is copied or decoded by a workspace-only registry; the persisted VFS ref
is the sole source payload state.

### Artifact Authority

The existing artifact model is complete:

- `openhcs/core/artifacts.py::ArtifactType` owns semantic artifact kind.
- `ArtifactSpec` owns artifact name, plan type, artifact type,
  materialization, requiredness, sidecar information, and relations.
- `ArtifactSpecRef` owns relation targets.
- `ArtifactSpecCollection` owns indexed lookup and conflict validation.
- `openhcs/core/module_artifact_contract.py::ModuleArtifactContractItem` binds
  an `ArtifactSpec` to an existing partition type.
- `ModuleArtifactContract` owns one module invocation's ordered artifact ABI.
- `SourceArtifactInputPartition`, `RuntimeArtifactInputPartition`,
  `RecordedArtifactOutputPartition`, and `DeclaredArtifactOutputPartition`
  already distinguish source, runtime, recorded, and declared roles.
- `openhcs/core/pipeline/artifact_planning.py::ArtifactGraph` owns cross-step
  producer and consumer identity.
- `PipelinePathPlanner.declared` owns forward-built artifact storage plans after
  invocation contracts have been resolved.

No port class, capability product type, compile-time flow model, symbol object,
or artifact-key mirror is added.

### Callable Special I/O

The existing special-I/O declarations remain load-bearing at the Python-call
ABI boundary:

- `openhcs/core/pipeline/function_contracts.py::special_inputs` declares the
  non-main-flow callable parameter names supplied by runtime artifact loading.
- `special_outputs` declares returned slots that are separate from the
  canonical main-flow return.
- The compiled `ModuleArtifactContract` supplies semantic artifact type, public
  artifact identity, partition, relation, and materialization.

CellProfiler compilation stops deriving semantic artifact kind from special
output spelling or materializer class. The ordered special-I/O slot declarations
are validated against the ordered runtime input and returned output specs from
the module contract. `SpecialOutputKindClassifier` and CellProfiler's duplicate
special-output projection are deleted after that validation is in place.

Migrate every CellProfiler backend `special_outputs` decorator to ordered slot
names only. Remove `(slot_name, MaterializationSpec)` tuples from CP callables;
their owning module contract already declares the exact output materialization.
The generic decorator retains tuple support for native callables whose callable
contract is the semantic declaration. An AST gate rejects a materialization
tuple in every registered CellProfiler callable. This leaves no artifact
materialization fact on both the raw CP function and its module declaration.

This division preserves both existing systems without parallel ownership:

- callable special I/O owns Python parameter and return-slot ABI;
- `ModuleArtifactContract` owns artifact semantics.

### CellProfiler Module Authority

`openhcs/interop/cellprofiler/module_declarations.py::CellProfilerModule` is the
nominal owner of one CellProfiler module's compile and runtime semantics.
`CellProfilerModule.__registry__` and `CellProfilerModule.for_module` already
resolve the leaf declaration. Module classes and their MRO already provide
artifact-contract methods and policy mixins for main flow, source behavior,
runtime values, object inputs, execution mode, and output recording.

The implementation makes those declarations directly load-bearing. Generic
code resolves a `CellProfilerModule` once and invokes its polymorphic methods.
Generic code never switches on a CellProfiler module name.

The same registry owns setup modules. `Images`, `LoadImages`, `Metadata`,
`NamesAndTypes`, and `Groups` become
`SourceSetupCellProfilerModule` declarations in
`openhcs/processing/backends/cellprofiler/infrastructure.py`. Source-binding
config compilation resolves each enabled `ModuleBlock` through
`CellProfilerModule.require_module` and calls the declaration's
`contribute_source_bindings` method. `SetupModuleCompiler` and
`SourceImageStackPlanDeclaration` disappear; `NamesAndTypes` owns its 3D stack
rule directly.

Non-executable status is not inherited by exporters. `SaveImages`,
`ExportToSpreadsheet`, and `ExportToDatabase`, which occur in the official30
corpus, become executable module declarations and public FunctionSteps.
`SaveImages` is axis-scoped: its contract consumes one named image and declares
one image output with generic `ImageFileOptions` materialization. The selected
image is a contract-loaded special input; the callable returns the unchanged
canonical main-flow image first and the converted export copy in its declared
special-output slot. Saving therefore never replaces downstream main flow.
`ExportToSpreadsheet` and `ExportToDatabase` are plate-scoped: their contracts
enumerate the exact measurement, relationship, and source-image artifacts they
consume, and their callables receive one generic `RuntimeArtifactBatch` after
all axes finish. They return one named file bundle materialized through generic
`FileBundleOptions`. CP code owns only measurement/CPA projection, CSV
rendering, SQLite rendering, and CPA properties content.

This execution-scope distinction is generic callable metadata, not a CP
module-name rule. `FunctionStepExecutionScope.AXIS` remains the default;
`FunctionStepExecutionScope.PLATE` is declared on the two aggregate exporter
callables and carried by their compiled `CallableContract` values. The
compiled function pattern derives and validates one uniform step scope. Worker
lanes execute only axis-scoped plans. Plate execution invokes plate-scoped
plans once, in source order, using the existing metadata-writer context as the
single output owner. No exporter reads a `.cppipe`, import result, process-global
store, or unfiltered runtime store.

The compiler assigns `StepInputDependency.no_main_flow()` to every plate plan.
`PipelinePathPlanner.input_component_scopes` returns the empty scope and
`step_io_dirs` uses the normal compiled output root only as the artifact
materialization base. No source image directory or preceding main-flow output
is loaded. `NoMainFlowOutput` remains the existing runtime result marker; the
new enum member supplies the missing compile-time input fact on its current
owner.

`FunctionStepExecutionScope` does not replace or reinterpret
`ProcessingContract`: execution scope selects when a callable is invoked,
while `ProcessingContract` continues to describe array locality inside one
axis-scoped invocation. Plate callables have no image processing contract.

`SaveCroppedObjects`, `LoadData`, `LabelImages`, and `CreateBatchFiles`
declarations and pass-through stubs are removed until their real semantics are
represented by a source-binding config contribution or an executable contract.
An enabled unsupported module therefore fails at
`CellProfilerModule.require_module`; it is never silently classified as
infrastructure.

The callable direction is equally singular. A public callable is loaded by
`CellProfilerModule.require_callable`; its owning declaration is resolved by
`CellProfilerModule.for_function_name`. The module declaration is the only
function-to-module index. `CELLPROFILER_MODULE_ATTR`,
`CellProfilerFunctionRuntimeMetadata`, `AbsorbedFunctionMetadata`, and the
derived catalog dictionaries disappear. The owning declaration imports the
implementation from its own Python module, validates its name and
source-declared `CallableContract`, and returns that same callable. It never
installs metadata, calls `attach_callable_contract_metadata`, or mutates the
callable namespace. The backend package's lazy attribute hook delegates to this
declaration method; no catalog or public wrapper function remains.

### ZMQ Public-Source Boundary

The canonical parity path already exists in
`benchmark/adapters/openhcs.py::_execute_pipeline_via_zmq_server`:

1. Normalize the public step list with
   `FunctionStepTransportAuthority.normalize_pipeline`.
2. Build `OpenHCSExecutionSubmission` from the public steps,
   `GlobalPipelineConfig`, and `PipelineConfig`.
3. Generate source with
   `FunctionStepTransportAuthority.source_from_pipeline` through the submission's
   direct `pipeline_code()` method.
4. Submit compilation through `ZMQExecutionClient.submit_compile`.
5. Wait for the compile artifact.
6. Submit execution through `ZMQExecutionClient.submit_pipeline` using that
   compile artifact.
7. Wait for execution and validate the server observation.

The UI request path must produce the same `OpenHCSExecutionSubmission` fields.
In-process orchestrator execution is not an acceptance path for this work.
The benchmark performs this sequence on every OpenHCS run. Delete both its
full-result cache and runtime-execution cache plus every cache-hit branch;
persisted native CellProfiler
references are the only reused execution outputs. Compile and execution timing
always measures the current ZMQ submission.

## Authority Map

| Responsibility | Sole owner after migration | Existing symbol |
|---|---|---|
| Public pipeline steps | `FunctionStep` declarations | `openhcs/core/steps/function_step.py::FunctionStep` |
| Pipeline and global options | Config inheritance | `GlobalPipelineConfig`, `PipelineConfig`, lazy registered configs |
| Pattern grouping | Generic pattern normalization | `normalize_function_pattern`, `FunctionInvocationKey` |
| Source universe and alias selection | Source-binding hierarchy | `SourceBindingsConfig`, `NamedSourceBinding`, resolved `StepSourceBindingsConfig` |
| Resolved step configuration | ObjectState-resolved public step | `FunctionStep`, stored directly as `StepSnapshot.step` |
| Execution sample selection | Config inheritance | `WellFilterConfig` |
| Source file loading and leading-axis projection | Generic VFS reference and backend registry | `FileManagerLike`, `SourcePixelRef`, `VirtualWorkspaceBackend` |
| Artifact identity and kind | Generic artifact model | `ArtifactSpec`, `ArtifactType`, `ArtifactSpecRef` |
| Invocation artifact ABI | Existing module contract | `ModuleArtifactContractItem`, `ModuleArtifactContract` |
| Required stack axes and allowed grouping | Generic callable contract | `CallableContract.required_variable_components`, `CallableContract.allowed_group_by` |
| Cross-step availability | Nominal artifact-plan owners and graph | `ArtifactPlanKeySelector`, `CallableContract`, `ModuleArtifactContract`, `ArtifactGraph` |
| CP setting interpretation | Nominal module declaration | `CellProfilerModule` leaf class and MRO |
| CP setup/source lowering | Nominal setup module declaration | `SourceSetupCellProfilerModule.contribute_source_bindings` and `SourceBindingsConfig` |
| CP callable ownership | Nominal module declaration | `CellProfilerModule.for_function_name`, `CellProfilerModule.require_callable` |
| Callable execution scope | Generic callable contract | `FunctionStepExecutionScope`, `CallableContract` |
| Plate-scoped callable input | Generic runtime artifact value | `RuntimeArtifactBatch` built from compiled `ArtifactInputPlan` values |
| Python special parameter ABI | Callable declaration | `special_inputs` |
| Python returned-slot ABI | Callable declaration | `special_outputs` |
| Image and file-bundle persistence | Generic materialization writers | `ImageFileOptions`, `FileBundleOptions`, `MaterializationSpec` |
| CP runtime bridge | One callable adapter over one contract | collapsed `CellProfilerRuntimeCallable` |
| Returned value matching | Generic matcher | `RuntimeReturnedOutputMatcher` |
| Public serialization | Generic FunctionStep pycodify/ZMQ transport | `FunctionStepTransportAuthority`, `OpenHCSExecutionSubmission` |
| Editor-only display/scopes | GUI-local ObjectState root | `PipelineEditorStateRoot` |

## Symbol Disposition

### Keep As Authorities

- `ArtifactType`, `ArtifactSpec`, `ArtifactSpecRef`, `ArtifactPlanKeySelector`,
  `ArtifactSpecCollection`, and existing artifact relations.
- `ModuleArtifactContractItem`, `ModuleArtifactContract`, and all existing
  contract partitions.
- `ArtifactDeclarationStepContext`, `InvocationContractPlan`,
  `InvocationContractProviderFactory`, and
  `PipelineInvocationContractProviderAuthority`.
- `ArtifactGraph`, `ArtifactSpecAccumulator`, and
  `extract_artifact_declarations`.
- `FunctionInvocationKey`, `NormalizedFunctionItem`,
  `NormalizedFunctionPattern`, and `CompiledFunctionInvocation`.
- `FunctionStep`, after removal of its hidden invocation-contract field.
- `StepSnapshot`, after removal of its hidden invocation-contract field.
- ObjectState config inheritance, the resolved `FunctionStep` stored by
  `StepSnapshot`, and the complete `SourceBindingsConfig` source declaration.
- `ImagePayloadMetadata` as the runtime owner of resolved pixel layout,
  provenance, spatial, voxel, dtype, and mask facts.
- `special_inputs` and `special_outputs` as callable ABI declarations.
- `CellProfilerModule`, its registry, its leaf classes, and behavior-bearing MRO
  mixins.
- `ModuleBlock` as a transient CellProfiler setting-row representation used
  only while importing or compiling one invocation.
- `RuntimeReturnedOutputMatcher`, simplified to compiled-contract matching.
- `RuntimeAdapterSpec`, extended with one optional runtime-callable factory so
  the compiled adapter declaration rebuilds its executable without a second
  registry.
- `FunctionStepTransportAuthority`, `OpenHCSExecutionSubmission`, and the ZMQ
  compile artifact protocol, after their pipeline fields become direct mutable
  step lists.
- `FunctionStepExecutionScope` and `RuntimeArtifactBatch`, added as generic
  owners of the previously unrepresented plate-scoped callable lifecycle and
  its exact contract-selected input value.
- `ImageFileOptions` and `FileBundleOptions`, added to the existing
  materialization writer family. They own format-agnostic persistence of one
  image path and a mapping of relative paths to text/bytes; no CP writer is
  added to generic code.
- `SourcePixelRef`, moved to PolyStore and narrowed to generic backend address
  plus leading source-axis indices so the VFS owns format-neutral projection.

### Collapse Into Existing Owners

- `PipelineImageSchema`, its builder, assignment subclasses, one-field stack
  and grouping wrappers, and source-binding representability/projection layer
  collapse into the existing `SourceBindingsConfig`, `NamedSourceBinding`, and
  source workspace machinery.
- `SourceImagePayloadSemantics`, its role-strategy hierarchy, and the runtime
  `ImageTypeSourceRole` lookup collapse into declared `NamedSourceBinding`
  loading fields plus `ImagePayloadMetadata.source_channel_axis`.
- `CellProfilerCompileTimeSettingsRequest` collapses into the existing
  `NormalizedFunctionItem`, `ArtifactDeclarationStepContext`, resolved source
  bindings, and `ArtifactSpecCollection` available at the compiler position.
- `CellProfilerCompileTimeArtifactFlow` collapses into a transient
  `ArtifactSpecCollection` used by the provider's forward prepass. The
  collection is advanced from existing native callable declarations and CP
  module contracts, then discarded after exact plans are built.
- `CellProfilerArtifactCapability` and all input/output capability subclasses
  collapse into direct `ModuleArtifactContractItem` production by the owning
  `CellProfilerModule` class.
- `CellProfilerRuntimeStepBinding` and
  `CellProfilerGroupedRuntimeStepBinding` collapse into direct construction of
  the one compiled runtime callable.
- `CellProfilerGroupedModuleContracts` collapses into
  `ModuleArtifactContract.combine`, which forms one ordered contract from
  transient same-module group contracts before path planning.
- `CellProfilerModuleContractResolution` collapses into constructor type
  validation on `CellProfilerRuntimeCallable`.
- `RuntimeArtifactBindingScope` collapses into partition membership queries on
  the existing `ModuleArtifactContract`. The runtime request carries the
  contract itself instead of copied external/runtime name sets.
- `RuntimeArtifactInputRequest` stops inheriting and reconstructing
  `ArtifactSpec`; it composes the original `ArtifactSpec`, aggregate
  `ModuleArtifactContract`, adapter, and current-image context.
- module-name runtime policy registries and dynamic policy views collapse into
  virtual methods on `CellProfilerModule` and its existing MRO mixins.
- CellProfiler special-output semantic projections collapse into the module's
  existing `ModuleArtifactContract`.
- `RuntimeArtifactLineageScope` and
  `RuntimeArtifactSourceLineage` collapse into `ArtifactSpec.relations`, source
  bindings, and ordered module contracts.
- generated source-binding contract guards collapse into compiler validation
  against effective source bindings and the module contract.
- `SourceBindingRuntimeContractGuard` collapses into source-binding alignment
  methods on `ModuleArtifactContract`; the existing
  `SourceBindingContractAlignment` result remains the UI/compiler report value.
- `GeneratedPipelineModuleIdentity`, `GeneratedPipelineRuntimeModule`, and all
  generated-module import/registration helpers disappear. Conversion already
  has the real FunctionSteps; fresh-source reconstruction belongs to the
  generic ZMQ acceptance path.
- `GeneratedCPPipePipeline`, `GeneratedPipeline`, `PipelineGenerator`, and the
  import request/result carrier hierarchy collapse into one pure translation
  function returning the ordinary public declaration pair
  `tuple[list[FunctionStep], PipelineConfig]`. No CP-specific result object
  survives after translation.
- `SetupModuleCompiler` and `SourceImageStackPlanDeclaration` collapse into
  `SourceSetupCellProfilerModule` declarations on the one existing module
  registry.
- `ResolvedModuleFunction` collapses into the actual callable returned by
  `CellProfilerModule.resolve_function`; no function-name carrier sits between
  module selection and FunctionStep construction.
- `PipelineStepsBoundary`, `PipelineStepsCarrier`,
  `PipelineStepsNamespaceProjection`, `ZMQPipelineSourcePayload`, and
  `ZMQPipelineCodeTransport` collapse into direct `list[FunctionStep]` fields
  plus `FunctionStepTransportAuthority.pipeline_steps_from_namespace`.
- `OpenHCSExecutionConfigCarrier` and `ZMQResolvedConfig` collapse into direct
  `OpenHCSExecutionConfigBundle` fields on their concrete request/context
  records.
- `CellProfilerCallableOutputSpecs` collapses into the ordered returned-output
  specs already present on the compiled `ModuleArtifactContract`.
- `RuntimeArtifactRecordDeduplication` collapses into a classmethod on the
  existing `RuntimeArtifactRecordLocationIdentity` owner.
- `CellProfilerOptionalCurrentImageContext` and
  `CellProfilerRequiredCurrentImageContext` collapse into ordinary typed
  `current_image` fields on the concrete requests that own them; the one
  optional-value requirement is checked at its consumption edge.
- `CurrentSourceIdentityCacheScope`, `RuntimeGroupMatchScope`, and
  `CellProfilerRuntimeScope` collapse into the exact generic query records and
  runtime cache keys that consume their fields.
- `CellProfilerImageMeasurementSource`, its produced-artifact base/carrier,
  and its unqualified leaf collapse into
  `ProducedImageMeasurementRecordMixin`: the module MRO selects the behavior,
  and the mixin resolves the existing output `ArtifactSpec` and payload
  directly.
- `CellProfilerMeasurementFeatureParseCandidate` collapses into
  `CellProfilerMeasurementFeature.parse`; normalization, registered parser
  traversal, and the `OTHER` result now have one owner and no optional-value
  shell.
- `CurrentStepPayloadSelection` collapses into the selector's existing
  `ImagePayloadValue | None` result; callers branch on the value directly
  instead of querying and unwrapping a one-field result object.
- runtime-plane projection collapses its unused registry, marker-only base,
  nested current-image carrier, field-only request bases, and one-call
  plane-index/slice/metadata request objects into
  `RuntimePlaneImagePayloadProjection`. Generic planar shape meaning lives on
  the existing `ImageArrayShapeSemantics`; exact selected planes and their
  source-context result remain first-class values.
- `CurrentRuntimePlaneKwargValue` collapses into classification methods on the
  projector that consumes the result, and `DenseLabelSequenceMemoryBudget`
  collapses into the existing dense-stack byte guard. Neither scalar predicate
  retains a one-field object shell.
- `CellProfilerRadialCVExportValue` collapses into the existing radial-CV
  missing-value authority, and `FirstMeasurementField` collapses into direct
  tuple selection at its sole consumer.
- `PerImageMeasurementProfile` collapses into its sole executor's existing
  `CellProfilerRuntimeProfiler`, while `FilterObjectsKwargSettings` and
  `FilterObjectsBoundMeasurementInputs` collapse into the concrete runtime
  input plan and the one logging edge that consume their values.
- `ModuleRevisionRange` collapses into Watershed's owning module declaration;
  its sole CellProfiler-4 cutoff is now a typed class constant evaluated where
  the runtime family is selected.
- `GranularityImageSeriesCacheEntry` collapses into the cached
  `GranularityImageSeries`, `SourceImagePairCollection` collapses into direct
  exact-cardinality matching, and `FilterObjectsRelationshipEndpointIds`
  collapses into one endpoint projection function shared by its consumers.
- `CellProfilerModuleRuntimePlan.func` is deleted as an exact alias of the
  plan's declared `raw_func` field; runtime consumers now use the sole callable
  identity member directly.
- `RuntimeShapeInspection` collapses into the existing generic
  `ImageArrayShapeSemantics.shape` projection.
- `ObjectLabelFinalLabels` and `ObjectLabelSmallRemovedLabels` collapse into
  the existing `ObjectLabelVariantData` authority and
  `LabelPayloadFinalProjection`.
- `SparseLabelRowsCoercion` collapses into the existing
  `sparse_ijv_rows_from_label_slice` function.
- the one-field `AnalysisConsolidationPlan` collapses into post-plate execution
  and optional consolidation functions in its existing orchestrator module.
- source candidate discovery, matching, address projection, and materialization
  classes collapse into the existing `SourceBindingWorkspaceProjector`; only
  validated internal candidate/image-set/index records remain.

### Replace

- `CellProfilerInvocationContractProviderFactory` remains the registered entry
  point, but its provider is replaced with a session-scoped exact provider that
  precomputes contracts from public snapshots and module declarations.
- Two runtime callable classes are replaced by one
  `CellProfilerRuntimeCallable` that accepts the resolved raw function and its
  enriched compiled `CallableContract`.
- the core `Pipeline` execution/editor wrapper is replaced at the GUI boundary
  by `PipelineEditorStateRoot`; every execution/import/transport API returns a
  plain step list.
- the CP import result, request DTO, compiler ABC, importer forwarding ABC,
  mutable singleton, and one-leaf compiler registry are replaced by
  `pipeline_import.import_cellprofiler_pipeline(cppipe_path, *, filemanager,
  backend, source_root) -> tuple[list[FunctionStep], PipelineConfig]`.
  Path/backend values and the optional CellProfiler default-input-folder root
  are direct import inputs, not fields copied into another carrier.
- skipped export-module behavior is replaced by explicit generated
  FunctionSteps for `SaveImages`, `ExportToSpreadsheet`, and
  `ExportToDatabase`. SaveImages uses axis-scoped image materialization;
  spreadsheet/database exporters use the generic plate execution scope,
  contract-selected `RuntimeArtifactBatch`, and standard file-bundle
  materialization.
- CP `CalculateMathInvocationOptions` is replaced by ordinary typed callable
  parameters. `output_name` remains a public user setting and also drives the
  module's output `ArtifactSpec`; operand object identity is derived from input
  artifact specs and supplied by the module executor.
- CP `DefineGridInvocationOptions.cycle_scope` is replaced by an ordinary typed
  callable kwarg because it changes invocation scheduling semantics.
- generated generic-core `ImageTypeSourceRole` classes are replaced at the
  importer edge by one CP-local `CellProfilerSourceImageType` enum; resolved
  generic loading and layout facts live on `NamedSourceBinding` and
  `ImagePayloadMetadata`.
- source literal resolver class families are replaced by three CP-local enums
  whose members map external syntax to existing generic source-binding enums.
- `SourceSchemaCandidate` is renamed `SourceCandidate` and gains generic
  source-axis indices; it remains the one validated candidate value shared by
  workspace projection and inventory.
- TIFF-specific source-plane inventory is replaced by
  `SourceBindingsConfig.source_stack_components`, generic `FileManager` loading,
  `ImageArrayShapeSemantics`, and `SourcePixelRef.source_axis_indices`.
- the ordered `ComponentProjection` registry is replaced by one enum-keyed
  strategy per existing `AllComponents` member.

### Remove

The following symbols and files have no independent responsibility after the
replacement above:

- proposed `ArtifactPort`, `ArtifactPortDeclaration`, `ArtifactPortContext`,
  `ArtifactNameResolver`, `ArtifactRelationResolver`,
  `CompileTimeArtifactFlow`, `SpecialInputDeclaration`,
  `SpecialOutputDeclaration` replacement classes, and
  `CellProfilerArtifactSettingDescriptor`;
- proposed `SourceBoundInputPort`, `MainFlowInputPort`,
  `RuntimeArtifactInputPort`, `DeclaredArtifactOutputPort`,
  `RecordedArtifactOutputPort`, `SpecialInputPort`, and `SpecialOutputPort`;
- `openhcs/core/function_step_invocation_contracts.py` and every
  `FunctionStep.invocation_contracts` consumer;
- `PipelineMetadataCarrier`, `CompilationSession.pipeline_metadata`,
  `PIPELINE_SOURCE_SCHEMA_METADATA_KEY`, `PipelineIdentityCarrier`, and compiler
  metadata/structural-carrier plumbing;
- `openhcs/interop/cellprofiler/symbol_table.py`;
- `openhcs/interop/cellprofiler/module_artifact_inputs.py`;
- `openhcs/interop/cellprofiler/artifact_semantics.py`;
- `openhcs/interop/cellprofiler/module_roles.py` in full, including
  `CellProfilerModuleRole`, `CellProfilerModuleRoleSpec`, and role lookup;
  import behavior calls module declarations directly;
- `CellProfilerPipelineProvenance`, `CellProfilerPipelineImportRequest`,
  `CellProfilerPipelineImportResult`, and `CellProfilerModuleReference`; no
  import carrier survives the translation boundary;
- `SetupModuleCompiler`, its five compiler leaves, and
  `SourceImageStackPlanDeclaration`;
- `SOURCE_IMAGE_TYPE_METADATA_FIELD`, `OpenHCSImageType`,
  `SourcePlaneProjection.image_type`, `source_image_payload_role`, and every
  runtime image-type-role lookup;
- `SourceImagePayloadRoleStrategy` and its leaves after the direct typed source
  binding payload function owns the same transformations;
- `SourceSchemaImageSetSelection`, `max_image_set_count`, and every
  `source_schema_image_set_selection` transport field; inherited
  `WellFilterConfig` is the only execution-selection owner;
- `SourceSchemaCandidateProvider`, candidate discovery mode/request/probe/
  viability/result classes, and provider leaves;
- `CellProfilerSourceRootResolver`, its candidate/resolved-root/admission/context
  records, and source-path exclusion registry; import uses the exact submitted
  source root and config filters;
- `ImagePlaneSourceResolutionStage`, `ImagePlaneSourceResolver`, and its
  priority leaves;
- `SourceSchemaSourcePlaneInventory`, `TiffPageSourcePlaneInventory`,
  `TiffSourcePlaneInventory`, and `SinglePlaneSourcePlaneInventory`;
- `SourcePixelRef.to_legacy_workspace_mapping`, string `workspace_mapping`
  values, old reader/path/series/plane/C/Z/T fields,
  `SourcePixelRef.source_metadata`,
  `workspace_mapping_source_ref`, `workspace_mapping_source_path`,
  `VirtualWorkspaceSourceRefResolver`, `PathSourceRefResolver`,
  `DiskSourceRefResolver`, `VirtualWorkspaceResolvedRef`, resolver `accepts` /
  `priority` methods, sorting, and resolver batching; all writers use
  `SourcePixelRef.to_workspace_mapping` and all readers validate through
  `SourcePixelRef.from_workspace_mapping` before direct backend-registry
  dispatch;
- the Bio-Formats backend's workspace cache/resolver/listing role, alternate
  reader switch, `_load_npy_plane`, `BioFormatsReaderUnavailableError`, and the
  handler's primary-backend override;
- `SourceSchemaAuxiliaryMaterializer`, `NumpyAuxiliaryMaterializer`, auxiliary
  request/target-policy classes, and the sole basename policy leaf;
- one-operation source workspace projections and accumulators listed in the
  Configuration And Source Binding section;
- `SourceFilterCriteriaParser`, `SourceBindingMatchMetadataParser`, and
  `SourceBindingOriginPolicy` registry families, each of which has one
  CellProfiler leaf selected by a constant key; their methods become private
  functions beside the owning setup module leaves in `infrastructure.py`;
- `CellProfilerDebugView` and `DefaultCellProfilerDebugView`; the only leaf
  delegates to `DebugViewModel.from_debug_snapshot`, so the debug inspector
  calls that generic owner directly;
- the unused `AutoRegisterMeta` registry on
  `CellProfilerSemanticDefaultContract`; module declarations already own the
  exact semantic-default contract types;
- `ResolvedModuleFunction` and
  `openhcs/interop/cellprofiler/module_function_resolution.py`;
- `CellProfilerCompileTimeArtifactFlow`;
- `CellProfilerCompileTimeSettingsRequest`;
- `CellProfilerArtifactCapability` and its product hierarchy;
- `PipelineGeneratorRuntimeContractProjector` and generated runtime-contract
  sidecars;
- `PipelineGeneratorRegistryStage`, its copied `_registry`, the
  `library_root`/`contracts.json` branch, and absorbed-library fallback errors;
- `PipelineGeneratorArtifactPruner` plus the importer/generator options
  `prune_dead_unmaterialized_artifact_steps`,
  `materialize_skipped_save_images`, and `materialize_terminal_images`;
- `PipelineGeneratorBuildStage` and `PipelineGeneratorCodeEmitter` objects that
  store only a backreference to `PipelineGenerator`;
- `PipelineGenerator` and `GeneratedPipeline`; the pure import function owns
  the one lowering pass and returns the public steps/config pair directly;
- `GeneratedPipelineRequest`, `SkippedModuleSelection`,
  `GeneratedStepEmission`, `GeneratedStepEmissionGroup`, generated import
  collectors, `python_literal`, and `ArtifactContractCommentSection`; the
  direct import pass constructs real public objects and delegates source rendering to
  `FunctionStepTransportAuthority`/pycodify;
- `GeneratedPipelineConfigDefaults`; the import pass constructs the existing
  `PipelineConfig` directly from common concrete `ProcessingConfig` and
  `SourceBindingsConfig` values;
- `CellProfilerGeneratedInvocationContractProvider`,
  `CellProfilerGeneratedStepContractMatcher`, generated step contract payloads, module-number
  contract indexes, and candidate/fallback matching;
- `CellProfilerGroupedRuntimeCallable`,
  `CellProfilerGroupedModuleContracts`, and grouped runtime step bindings;
- `CellProfilerProcessingContractAuthority`, its process cache, and absorbed
  contract fallback; runtime code uses
  `CallableContract.require_processing_contract()`;
- `_attach_runtime_processing_contract`, `cellprofiler_module_callable`, and
  `rebuild_cellprofiler_runtime_callable`; the adapter-spec factory builds the
  one process-local runtime wrapper after generic reference resolution, and
  that wrapper is never serialized;
- `openhcs/interop/cellprofiler/runtime/policy_registry.py` and dynamic policy
  view classes that copy module declaration attributes;
- `openhcs/interop/cellprofiler/runtime/current_image_context.py` and its two
  one-field inheritance carriers;
- `openhcs/interop/cellprofiler/runtime/measurement_image_sources.py` and its
  unused registry-shaped source hierarchy;
- `SpecialOutputKindClassifier` and name/materializer-based semantic inference;
- `RuntimeInvocationOptions`, its FunctionStep third-tuple shape, normalization,
  hidden runtime parameter, UI extraction branch, agent guidance, and both
  `CalculateMathInvocationOptions` / `DefineGridInvocationOptions` leaves;
- hidden SaveImages materialization rewriting and
  `materialize_skipped_save_images`;
- `RuntimeImageExportSpec`, `RuntimeImageExportBitDepth`, and the
  `_candidate_image_snapshots_for_equivalence` in-memory comparison shortcut;
- `InfrastructureCellProfilerModule`, `infrastructure_import_note`,
  `infrastructure_exports_tables`, and `infrastructure_exports_images`;
- unsupported pass-through declarations and callables for `LoadData`,
  `LabelImages`, `CreateBatchFiles`, and `SaveCroppedObjects`;
- placeholder `openhcs/interop/cellprofiler/database_export.py`, its
  `pending_pipeline_export` output, and the unrelated region-statistics
  implementations in `openhcs/interop/cellprofiler/spreadsheet_export.py`;
- `CellProfilerExecutionExportContext` and
  `CellProfilerAnalystExportRequest`; generic plate execution supplies exact
  contract-selected `RuntimeArtifactBatch` values and direct typed arguments;
- compile-time materialized-image kwargs and metadata keys, including
  `materialized_image_artifact_names` and
  `artifact_name_materialized_image_artifact_names`;
- `GeneratedPipeline.artifact_contracts`, runtime contract sidecars, and import
  record fields carrying compiled contracts;
- `GeneratedPipelineFunctionRegistration`, `GeneratedPipelineFunction`,
  `GeneratedPipelineModuleExports`, `GeneratedFunctionSpec`, and the appended
  generated-module import-time registration block;
- `GeneratedPipelineModuleIdentity`, `GeneratedPipelineRuntimeModule`, and
  `openhcs/interop/cellprofiler/runtime/generated_pipeline.py` in full after
  exact contract compilation replaces its matcher and source import is no
  longer a CP-specific runtime operation;
- generated-pipeline compatibility facade functions and benchmark aliases in
  `runtime/generated_pipeline.py` and `runtime_pipeline.py`;
- `DirectPipelineExecution`, `DirectExecutionProgressBridge`,
  `DirectExecutionProgressSink`, `execute_pipeline_direct`, and the
  direct-execution branch of `execution_validation.py`;
- `CellProfilerDialectCompiler`, `CellProfilerPipelineImporter`, the
  process-global mutable compiler singleton in `compiler_registry.py`, and
  benchmark-named compiler aliases; no one-leaf replacement registry is added;
- `StepInputSourceLiteral`, `GeneratedProcessingConfigShape`,
  `GeneratedStepSettings`, and `GeneratedParameterTarget`; generated
  intermediates carry existing typed configs and ordinary ordered mappings;
- `GeneratedGroupByComponentState`, `ModuleProcessingComponents`, and their
  string-literal projection helpers; processing-axis lowering returns the
  existing concrete `ProcessingConfig`, and source emission writes sparse
  `LazyProcessingConfig` values;
- `SourceProcessingAxisRole`, `SourceProcessingAxisRolePolicy`, their leaf
  classes, the source-axis summary, and `module_processing_config.py` are
  deleted. Source selection reads the resolved source-binding declaration;
  processing lowering reads the artifact-only module contract, callable-owned
  axis constraints, and inherited `ProcessingConfig` directly.
- `ModuleProcessingScopePolicy` and its precedence leaves are deleted;
  `CellProfilerModule.processing_config` performs the one deterministic lowering
  at the nominal module owner.
- unused `SettingNameFamilySpec`;
- `CELLPROFILER_MODULE_ATTR`, `CellProfilerFunctionRuntimeMetadata`,
  `CellProfilerFunctionCatalog`, `_make_processing_wrapper`, and derived
  function-to-module maps;
- `CellProfilerFunctionReferenceTransportStrategy`, the resulting empty
  `FunctionReferenceTransportStrategy` root, and catalog/module-object
  normalization or compiled-wrapper preservation branches;
- `openhcs/processing/backends/cellprofiler/library.py` after its implementation
  lookup moves onto `CellProfilerModule`;
- `openhcs/processing/backends/cellprofiler/function_documentation.py` and its
  callable-docstring mutation path after the backend package exposes raw
  declaration-owned callables directly;
- the 97 import-only files under `benchmark/cellprofiler_library`, the
  import-only `benchmark/cellprofiler_compat` package, and
  `benchmark/converter/cppipe_module_roles.py`;
- obsolete one-time absorption utilities `benchmark/converter/absorb.py`,
  `library_absorber.py`, `llm_converter.py`, `source_locator.py`,
  `system_prompt.py`, `add_parameter_mappings.py`,
  `backfill_parameter_mappings.py`, and `fix_registry.py`;
- `openhcs/interop/cellprofiler/thresholding.py`; fixture benchmarks and tests
  import the owning backend threshold functions directly;
- `TupleMemberTypeValidation`; import records validate their own members
  directly;
- state wrappers that expose only one predicate or projection over their only
  field: `SourceIdentitySetCardinality`, `DeclaredOutputResolution`,
  `CellProfilerOptionalNonemptyString`, `SourceBindingAxisCardinality`,
  `SourceImageSetIdentityQuality`, `InvocationSpatialRankCandidates`,
  `DenseLabelShapeSet`, `DenseLabelStackRepeatPattern`,
  `MatlabPayloadEntryName`, `SpatialGridSliceCount`,
  `RuntimePlaneSelectedPlaneIndex`, and
  `CellProfilerImageNumberResolution`;
- source-binding wrappers used by one owner only:
  `DisabledPathMetadataRulePolicy`, `DisabledMetadataAxisComponents`,
  `SourceBindingPayloadAliasSet`, and
  `SourceBindingPayloadComponentMetadata`; their behavior moves into the
  owning Metadata module declaration or
  `SourceBindingPayloadPlaneResolution`;
- runtime service wrappers whose work belongs to an existing authority:
  `RequireProcessingContextBoundaryPolicy`, `RuntimeShapeInspection`,
  `RuntimeArtifactRecordDeduplication`, `CellProfilerCallableOutputSpecs`,
  `ObjectLabelFinalLabels`, `ObjectLabelSmallRemovedLabels`,
  `SparseLabelRowsCoercion`, `Pure2DSliceCountCandidate`, and
  `Pure2DTraceLabelStats`;
- `PipelineStepsBoundary`, `PipelineStepsCarrier`,
  `PipelineStepsNamespaceProjection`, `ZMQPipelineSourcePayload`,
  `ZMQPipelineCodeTransport`, `PycodifiedPipelineStepSource`,
  `PycodifiedSource`, `PycodifiedPipelineCode`, `PycodifiedConfigSource`,
  `PycodifyAssignmentSourceRequest`, `OpenHCSExecutionConfigCarrier`, and
  `ZMQResolvedConfig`;
- agent `PipelineIdBoundary`, `ExecutionPipelinePayload`,
  `ExecutionPipelineDefinitionProvider`, `DraftPipelineDefinitionProvider`,
  and `PycodifiedSourcePipelineDefinitionProvider`;
- OpenHCS benchmark runtime result caching, including
  `RuntimeExecutionCacheWritePolicy`, `_RuntimeExecutionCacheHit`, cache
  manifest/key/reuse request properties, read/write methods, cache-hit timing,
  and `reused_runtime_execution_cache` provenance;
- the outer OpenHCS benchmark result cache, including
  `_run_or_reuse_cached_openhcs`, `_cached_benchmark_result`,
  `_write_benchmark_result_cache`, `_cached_metric_values`, both cache-key
  builders/constants, `reuse_openhcs_cache`, `force_openhcs_run`, and
  `reused_cached_output` provenance;
- `CachedNativeReferenceTimingPolicy`; persisted native references without
  measured timing support parity only and never synthesize speed from a timeout;
- core `Pipeline` as a public or execution-facing type;
- pycodify formatting for hidden invocation contracts, runtime wrappers, and
  compiled CellProfiler contracts.

## Target Public API

### Native OpenHCS

Native pipelines remain unchanged:

```python
pipeline_steps = [
    FunctionStep(
        func=(count_cells_single_channel, {
            "return_segmentation_mask": True,
        }),
        name="Cell Counting",
    ),
]
```

### CellProfiler-Backed OpenHCS

CellProfiler-backed pipelines use the same shape:

```python
pipeline_steps = [
    FunctionStep(
        func=correct_illumination_calculate,
        name="CorrectIlluminationCalculate",
        processing_config=LazyProcessingConfig(
            input_source=InputSource.PIPELINE_START,
        ),
    ),
    FunctionStep(
        func=(correct_illumination_apply, {
            "method": IlluminationCorrectionMethod.DIVIDE,
        }),
        name="CorrectIlluminationApply",
    ),
    FunctionStep(
        func=(export_to_database, {
            "sqlite_file": "analysis.db",
            "experiment_name": "Example",
        }),
        name="ExportToDatabase",
    ),
]
```

The callable kwargs contain user-controlled module settings: numerical
behavior, typed execution controls, and user-selected names that are not
canonical derivations. Canonical input image names, output image names, object
names, internal measurement artifact names, module numbers, and compiled
contracts are absent. A sparse explicit identity override appears only where the
user intentionally breaks the module's canonical flow. The compiler consumes
an identity-only override and excludes it from the raw callable call. A setting
that also affects returned runtime data, such as CalculateMath `output_name`,
remains in the raw call after contributing to contract derivation.
`RuntimeArtifactBatch` is supplied by the generic plate executor and never
appears in FunctionStep kwargs or generated source.

Identity-only settings do not appear in raw callable signatures. They appear
once in the owning module's existing `SettingToKeywordBinding` sequence, using
the module's actual setting-name ClassVar; no second compile-identity list
exists. A binding whose keyword is in the raw signature is behavioral. A
binding whose keyword is absent from the raw signature is a sparse identity
binding. The invocation contract provider consumes only present identity-bound
keys before generic callable-signature validation; any other unknown kwarg
remains an error. Function panes derive from the raw signature and therefore
show only behavioral and ordinary runtime controls. Code mode can state an
explicit noncanonical identity override without making that override a runtime
argument.

Add `CallableContract.validate_public_kwargs(kwargs, runtime_parameter_bindings)`
on the existing generic callable owner. It binds
the remaining public kwargs against the raw public signature, excludes only
the contract's main-input, runtime-context, runtime-adapter, and declared
runtime-parameter slots, and rejects unknown or missing required public
parameters. `compile_function_pattern` calls it after
`InvocationContractPlan.consumed_kwarg_names` are removed and before creating
`CompiledFunctionInvocation`. The compiled invocation stores the validated
kwargs as immutable `RuntimeKwargItems`. No CP runtime class inspects a
signature, applies defaults, drops unsupported kwargs, or performs another
binding pass.

### Source Binding

Pipeline-level `LazySourceBindingsConfig` declares pipeline-start sources.
Pipeline-level `LazyStepSourceBindingsConfig` declares shared step defaults.
Generated steps omit `source_bindings` when the ObjectState-resolved candidate
step value equals the value inherited from those two existing config scopes.

Source pixel loading is fully public in the existing binding declaration. A
hand-written pipeline uses `NamedSourceBinding.artifact_kind`,
`load_as_monochrome`, `load_as_mask`, `source_channel_axis`, and
`source_channel_counts`; a converted pipeline receives the same fields from the
CP setup enum. No public API accepts a CellProfiler image-type string, source
schema object, importer result, or runtime role. Once loaded,
`ImagePayloadMetadata.source_channel_axis` is the only extra runtime layout
fact.

A step emits a sparse `LazyStepSourceBindingsConfig` only for one of these
semantic changes:

- selecting a strict subset of pipeline-start sources;
- rebinding a source alias;
- consuming a non-image source artifact;
- returning to `InputSource.PIPELINE_START` after main-flow processing;
- implementing an explicit export step whose input is not the current main
  flow.

`enabled=True` alone is not emitted when inheritance already activates the
same bindings.

### Function Patterns

The import pass emits a plain callable or `(callable, kwargs)` when all
discovered group keys use the same callable and public kwargs.
Generic FunctionStep execution applies that invocation to every array group.

The import pass emits a dict pattern only when at least one group is omitted or
at least two groups differ in callable or public kwargs.
Compile-derived artifact identity does not force a dict pattern.

### Artifact Names

The owning `CellProfilerModule` derives canonical names from:

- effective source-bound `ArtifactSpec` inputs;
- artifacts available at the current compiler position;
- module settings represented by public typed kwargs;
- invocation key and step order;
- the module's nominal output naming rule.

The compiler assigns internal measurement artifact identity from public step
and invocation identity, never from `.cppipe` module number. A `.cppipe`
module number remains parser provenance only and does not enter generated source
or runtime matching.

### Runtime Callable

The compiler keeps each axis-scoped CellProfiler invocation as its raw callable
reference plus enriched `CallableContract`. The compiled contract contains the
exact `ModuleArtifactContract` and the existing `RuntimeAdapterSpec`. Extend
that spec with:

```python
RuntimeCallableFactory = Callable[
    [Callable[..., object], CallableContract],
    Callable[..., object],
]

runtime_callable_factory: RuntimeCallableFactory | None = None

def executable_callable(
    self,
    resolved_callable: Callable[..., object],
    contract: CallableContract,
) -> Callable[..., object]:
    ...
```

`CallableContract.resolve_runtime_callable()` resolves a direct callable or
generic `FunctionReference` once, then delegates to
`RuntimeAdapterSpec.executable_callable`. A spec without a runtime-callable
factory returns the resolved callable; the CellProfiler adapter spec declares
the top-level private
`cellprofiler_runtime_callable_factory(resolved_callable, callable_contract)`
that requires `callable_contract.module_artifact_contract` and constructs
exactly:

```text
CellProfilerRuntimeCallable(
    raw_func,
    callable_contract: CallableContract,
)
```

The constructor resolves `module_type` through
`CellProfilerModule.for_function_name(raw_func.__name__)` and validates its
module name against
`callable_contract.require_module_artifact_contract()`. Those values populate
the single three-field `CellProfilerModuleRuntimePlan`; no signature mirror,
kwarg filter, or module type is serialized beside the compiled contract.

The existing nominal adapter owns that spec:

```python
@classmethod
def runtime_adapter_spec(cls) -> RuntimeAdapterSpec:
    ...
```

`CellProfilerRuntimeAdapter.runtime_adapter_spec()` supplies its existing
parameter name, adapter factory, preparation hook, artifact-input ownership,
and the runtime-callable factory above. The invocation provider places this
exact value on compiled axis-scoped CP contracts. No spec constant, adapter
registry, module-name switch, or per-module copy is added.

The provider combines transient same-module group contracts through
`ModuleArtifactContract.combine` before storing the compiled contract. At runtime,
`ComponentArtifactPlans.from_step_component` has already selected the active
group's input and output plans before `RuntimeAdapterRequest` is built. The
CellProfiler executor projects active specs from that selected plan set; it does
not select a second contract. The wrapper exposes the raw callable's existing
public signature minus runtime-bound parameters.

The wrapper remains necessary because the raw CellProfiler functions do not
accept `CellProfilerRuntimeAdapter`, while generic
`RuntimeAdapterRequest` does not carry a module contract or execute a module.
The source contract remains the sole owner of source-declared processing mode,
declared contract name, memory type, execution mode, and special I/O; the
provider's immutable enriched contract remains the sole owner of the exact
module artifact contract and runtime adapter. The wrapper never recomputes or
attaches callable metadata. It constructs one immutable runtime plan and
retains one `CellProfilerModuleExecutor(plan)`; `raw_func` and
`callable_contract` live only on that plan, not as copied wrapper fields. It is
an execution-process value, not a public or initial submission value. Generic
`FunctionReference` plus `CallableContract` are the only serialized identities.
Delete
`FunctionReferenceRehydrationRequest`, `FunctionReferenceRehydrator`, and
`CellProfilerFunctionReferenceRehydrator`; there is no supports scan, fallback,
or second reconstruction authority. No second wrapper, plan cache, or contract
map represents grouped execution.

Plate-scoped CP callables remain raw generic importable functions. Their
`CallableContract.execution_scope` sends them to the generic post-plate
executor. They declare `RuntimeArtifactBatch` through the existing
`runtime_bound_parameters(...)` ABI, exactly as callable runtime values such as
the dtype config and slice index are declared, and receive it in the
`artifact_batch` keyword-only parameter. They do not construct
`CellProfilerRuntimeCallable`, `CellProfilerModuleRuntimePlan`, or
`CellProfilerModuleExecutor`, and the batch is absent from public FunctionStep
kwargs and function-pane controls.

Add one fail-loud query to the existing generic owner:

```python
def require_processing_contract(self) -> ProcessingContract:
    """Return the nominal processing contract or raise TypeError."""
```

CellProfiler runtime consumers call this method on their existing
`CallableContract`; delete the CP processing-contract cache, absorbed fallback,
and copied `processing_contract` fields.

## Compiler Algorithm

`CellProfilerInvocationContractProviderFactory.provider_for_session` constructs
one provider with the following deterministic prepass:

1. Iterate `CompilationSession.snapshots` in step order.
2. Read the exact ObjectState-resolved `snapshot.source_bindings`; derive only
   activation from that config and `snapshot.input_source`.
3. Normalize `snapshot.func` with `normalize_function_pattern`.
4. Resolve each raw callable directly through
   `CellProfilerModule.for_function_name`; validate that the declaration's
   `require_callable(func.__name__)` is the canonical public callable. Do not
   call `resolve_function` or substitute another declared variant: callable
   selection ended when the public FunctionStep was constructed.
5. Ignore non-CellProfiler callables and leave them to the generic callable
   contract provider.
6. Determine source contract group keys through generic FunctionStep semantics.
   An axis-scoped explicit dict invocation uses its
   `FunctionInvocationKey.group_key`. An axis-scoped default invocation uses
   source-binding component values for the resolved `ProcessingConfig.group_by`
   component. No matching source-binding component produces the single
   `default` contract key. A plate-scoped callable always has the single
   `default` key; dict patterns are rejected and per-axis processing config is
   not consulted.
   Assign `StepInputDependency.no_main_flow()` to the plate plan before path
   planning; axis plans retain the existing pipeline-start/step-output
   dependency resolution.
7. Scope the already-resolved source bindings with exact
   `StepSourceBindingsConfig.for_component_group(group_by.component, group_key)`
   for each source-derived contract group. An invocation without a
   source-derived group remains unscoped and receives all resolved bindings.
8. Ask the module declaration to reconstruct the transient `ModuleBlock` values
   represented by the public invocation. The module uses its declared input
   setting cardinality, scoped source bindings, current main-flow artifact
   collection, and sparse public overrides. A plain invocation therefore
   produces multiple blocks only when one behavior genuinely covers multiple
   source or main-flow groups. Blocks are not stored in the public step or
   generated source. The same call returns the exact sparse identity kwarg
   names read during reconstruction; no separate setting-name projection is
   queried.
9. Query available declared artifacts from the prepass's
   `ArtifactSpecCollection`. Effective source bindings contribute existing
   source `ArtifactSpec` values. Non-CellProfiler invocations contribute outputs
   from `artifact_plan_key_selector_for_contract(item.contract)`: the compiled
   `ModuleArtifactContract` when present, otherwise the `CallableContract`
   itself. The prepass therefore sees native OpenHCS callable declarations
   without copying them into an invocation wrapper.
10. Invoke the resolved `CellProfilerModule.artifact_contract` method. The
   module class directly produces existing `ModuleArtifactContractItem` values.
11. For axis scope, validate callable special-input parameter slots and
   special-output return slots against the ordered contract. For plate scope,
   require `RuntimeArtifactBatch` in the existing
   `CallableContract.runtime_bound_parameter_types`, require its declared
   keyword-only `artifact_batch` parameter, and require one file-bundle output
   spec; contract inputs become batch selectors rather than Python keyword
   slots.
12. Add declared outputs to the local ordered `ArtifactSpecCollection` used by
    the next invocation. Generic main-flow dependency remains owned by the
    existing step dependency and processing configuration machinery.
13. Combine transient same-module group contracts with
    `ModuleArtifactContract.combine`. Build one immutable enriched
    `CallableContract` around the canonical raw callable. Axis-scoped CP module
    execution receives the exact
    `CellProfilerRuntimeAdapter.runtime_adapter_spec()` value; adapter-free
    axis invocations such as SaveImages and every plate invocation receive no
    runtime adapter. The provider never constructs an executable wrapper.
14. Store `InvocationContractPlan` by `(step_index, FunctionInvocationKey)`.
15. Put the reconstruction call's exact consumed-name tuple on
    `InvocationContractPlan`; generic invocation compilation removes only those
    keys before signature validation and runtime kwargs are frozen.

Move the component-value extraction currently embedded in
`PathPlannerExecutionGroups.source_binding_scope_for_group_by` to one generic
function in `openhcs/core/source_bindings.py`:

```python
def source_binding_group_keys_for_group_by(
    source_bindings: StepSourceBindingsConfig,
    group_by: GroupBy,
) -> tuple[str, ...]:
    """Return ordered binding component values for the grouping component."""
```

Both `PathPlannerExecutionGroups` and the CellProfiler provider call that
function. The function knows only `GroupBy`, `ComponentSelector`, and source
bindings. It contains no module name, CellProfiler import, or artifact-setting
rule. A plain callable therefore inherits the same group routing as generic
OpenHCS execution without a generated dict pattern.

Provider lookup performs one exact dictionary lookup. Missing, duplicate, or
ambiguous keys are compile errors. Provider lookup performs no module-number
search, callable-name candidate search, source-binding alignment guess, indexed
fallback, or best-effort match.

The prepass has two local existing-value cursors:

- `available_artifacts: ArtifactSpecCollection` for all declared artifacts;
- `main_flow_artifacts: ArtifactSpecCollection` for the image identities carried
  by the current generic main flow.

Pipeline-start steps seed `main_flow_artifacts` from effective image source
bindings. A CP invocation replaces it with the contract's main-flow image specs
only when the module MRO says the invocation replaces main flow. A native
invocation uses its declared main-flow image spec; a native image callable with
no named artifact receives a deterministic compile-only image spec named from
its step index and `FunctionInvocationKey`. That spec names the main argument
for downstream module setting reconstruction and is not added as a materialized
artifact. Measurement-only invocations preserve the cursor.

Both cursors are discarded after exact plans are built. They introduce no
second graph or symbol model. The generic `ArtifactGraph` remains the compiler
authority after declarations enter path planning.

### Exact Module Method Boundary

Replace request-, builder-, assembler-, and symbol-table-based compilation with
these methods on `CellProfilerModule`:

```python
@classmethod
def module_blocks_for_invocation(
    cls,
    *,
    invocation: NormalizedFunctionItem,
    source_group_keys: tuple[str, ...],
    step_context: ArtifactDeclarationStepContext,
    available_artifacts: ArtifactSpecCollection,
    main_flow_artifacts: ArtifactSpecCollection,
) -> tuple[tuple[ModuleBlock, ...], tuple[str, ...]]:
    """Return transient setting rows and actually consumed identity kwargs."""

@classmethod
def artifact_contract(
    cls,
    *,
    module: ModuleBlock,
    invocation_key: FunctionInvocationKey,
    step_context: ArtifactDeclarationStepContext,
    available_artifacts: ArtifactSpecCollection,
    main_flow_artifacts: ArtifactSpecCollection,
) -> ModuleArtifactContract:
    """Return the exact ordered contract for one public invocation."""
```

Compiler-created transient blocks use `module_num=0` and `enabled=True`;
diagnostics use step index and invocation key instead. Contract names and
lookup keys never use those placeholders. Disabled parsed modules are skipped
before generation and never become public invocations.
`module_blocks_for_invocation` replaces
`CellProfilerCompileTimeSettingsRequest`,
`compile_time_setting_records_for_invocation`, and
`compile_time_module_metadata_for_invocation` as the compiler-facing surface.
Its implementation reuses existing `SettingToKeywordBinding`, setting-name
families, enum coercion, and module-owned setting methods.

Delete `compile_time_public_setting_names`,
`compile_time_public_kwarg_names`, `compile_time_consumed_kwarg_names`,
`compile_time_grouped_public_kwarg_names`, and
`compile_time_coalesced_public_kwarg_names`. During setting-row reconstruction,
partition the owning module's existing `setting_bindings` by membership in the
canonical raw callable signature. For a present sparse kwarg, an
identity-partition binding writes that same setting row and records its
normalized keyword as consumed; `module_blocks_for_invocation` returns those
actual keys beside the blocks. The provider passes that exact tuple to
`InvocationContractPlan.consumed_kwarg_names`. A kwarg absent from the raw
callable signature and absent from that consumed tuple is a compile error.
Nothing stores consumption on `ModuleBlock`, and no declaration list predicts
consumption.

Change `SettingToKeywordBinding.parameter_name` to `str | None = None` and add
`require_parameter_name()`. An omitted name derives once from the first
concrete value of the binding's existing `SettingNameFamily` through
`normalize_cellprofiler_setting_name`; explicit names remain only for genuine
Python/CellProfiler naming differences. Migrate all direct `.parameter_name`
reads to this method and reject duplicate resolved keyword names when the
module class is registered. Identity bindings therefore reference the same
setting ClassVar used by contract reconstruction, never a copied string.
Add `records_from_kwargs(kwargs) -> tuple[ModuleSetting, ...]` to that same
binding. It reads only `require_parameter_name()`, serializes typed scalar or
repeated values through the existing `cellprofiler_setting_literal`, and emits
rows under its own declared setting name. `module_blocks_for_invocation` uses
this method for behavior and identity settings; it marks a returned identity
row consumed only when the key was present in public kwargs. Delete the
parallel `compile_time_public_setting_records_from_kwargs` traversal.

`module_blocks_for_invocation` obtains the canonical raw signature once. For
behavior bindings, it creates one effective value mapping from signature
defaults followed by explicit public kwargs; a bound behavior parameter with
neither a default nor an explicit value is a compile error. For identity
bindings, it reads only explicit public overrides and values derived by the
module's flow method. It never treats an identity setting as a callable
default. This replaces
`compile_time_setting_binding_default_values` and
`compile_time_settings_function`; no cached default map or second callable
lookup remains.

Retain `SettingsBinder` only as the executor of the owning module's explicit
bindings and typed parsers. Delete
`cellprofiler_source_setting_parameter_mapping`, normalized setting-name to
signature inference, and `setting_parameter_aliases`. Mechanically migrate each
current inferred or alias mapping to one `SettingToKeywordBinding` on the
registered module class, referencing its setting-name ClassVar. A parsed row is
accepted only by an explicit binding, a private compound-row parser on that
module, or `ignored_settings_for`; every other row is `UNMAPPED` and fails
conversion. Callable annotations and a binding's explicit parser own value
coercion. Use an AST migration script for this one-time rewrite and do not keep
the script or a generated mapping table as runtime infrastructure.

`artifact_contract` replaces the current `(assembler, builder, module)` method
shape. It returns `ModuleArtifactContract` directly. Input and output helper
methods on module leaf classes are migrated to the same existing-value
arguments; none accepts `_SymbolTableBuilder`, a capability class, or a CP
symbol. `module_blocks_for_invocation` partitions current flow according to the
module's declared setting cardinality; the generic compiler does not know image
or object setting names.

No `CellProfilerModule` MRO class retains a `compile_time_*` method after this
migration. Apply this closed migration:

- setting-row, binding-value/default, selected-function, source-binding-input,
  main-flow-input, canonical-output, and divergent-input hooks move into
  `module_blocks_for_invocation` and the owning `SettingToKeywordBinding`;
- required/public artifact-setting, public-record, public-kwargs, grouped, and
  coalesced hooks disappear in favor of binding partition plus exact contract
  comparison;
- module-metadata hooks disappear in favor of the returned blocks, contract,
  and consumed-name tuple;
- main-flow output-name, image-output-name, replaces-flow, and flow-after hooks
  move into `artifact_contract` plus the provider's two local artifact cursors;
- leaf helpers that parse compound setting rows become private setting readers
  called directly by `module_blocks_for_invocation` or `artifact_contract`.

Do not retain aliases with the old prefix. An AST gate rejects every production
method definition whose name starts with `compile_time_` on a
`CellProfilerModule` subclass.

Add this combining operation to the existing contract owner:

```python
@classmethod
def combine(
    cls,
    contracts: Iterable[ModuleArtifactContract],
) -> ModuleArtifactContract:
    """Combine ordered same-module declarations and reject conflicts."""
```

It validates one module name, preserves first declaration order, de-duplicates
identical `ModuleArtifactContractItem` values, rejects conflicting
`ArtifactSpecRef` declarations, and returns the ordered artifact-only contract.
It contains no group-key, axis, or grouping field; callable constraints remain
on `CallableContract`.

Implement the provider in
`openhcs/interop/cellprofiler/compile_time_contracts.py` with this exact shape:

```python
@dataclass(frozen=True, slots=True)
class CellProfilerInvocationContractProvider(InvocationContractProvider):
    plans: Mapping[
        tuple[int, FunctionInvocationKey],
        InvocationContractPlan,
    ]

    def __post_init__(self) -> None:
        ...

    def __call__(
        self,
        invocation: NormalizedFunctionItem,
        step_context: ArtifactDeclarationStepContext,
    ) -> InvocationContractPlan | None:
        ...
```

Before adding that leaf, strengthen the existing generic provider ABI in
`openhcs/core/invocation_artifacts.py`: `InvocationContractProvider.__call__`
accepts `NormalizedFunctionItem` and `ArtifactDeclarationStepContext`;
`InvocationContractProviderFactory.provider_for_session` accepts
`CompilationSession` and returns `InvocationContractProvider | None`; and
`CompositeInvocationContractProvider.providers` is a tuple of those nominal
instances. Delete `InvocationContractProviderLike` and
`public_callable_invocation_contract`. An empty composite is the exact
no-claim provider. Use `TYPE_CHECKING` imports to preserve the current module
dependency direction; no `Any` or callable-shaped provider remains at this
boundary.

`__post_init__` validates every integer step index,
`FunctionInvocationKey`, and `InvocationContractPlan`, rejects an empty or
malformed mapping, copies it, and stores `MappingProxyType`. `__call__`
resolves ownership through `CellProfilerModule.for_function_name`; a non-CP
invocation returns `None`. A CP invocation requires an integer
`step_context.step_index`, validates object identity against
`module_type.require_callable(invocation.contract.function_name)`, performs one
lookup by `(step_index, invocation.key)`, and raises `ValueError` containing
that exact key when absent. It returns the stored plan unchanged.

`CellProfilerInvocationContractProviderFactory.provider_for_session` performs
the forward prepass above, rejects a duplicate key before mapping assignment,
returns this provider, and returns `None` only when the session contains no CP
invocation. No closure, second provider class, provider payload, contract
wrapper, compilation request, or flow class is introduced.

Change the retained generic `CompositeInvocationContractProvider` from
first-claim dispatch to unique-claim validation. It asks every registered
provider for the same normalized invocation, returns the sole non-`None`
`InvocationContractPlan`, returns `None` for zero claims, and raises a compile
error naming all claiming provider types for multiple claims. Factory registry
order never selects semantic ownership. Tests register two synthetic claiming
factories and prove the duplicate error in both registration orders.

### Callable And Import Boundary

Keep `CellProfilerModule.for_module` as the optional module-name lookup and add
`CellProfilerModule.require_module` as its fail-loud counterpart. Add
`CellProfilerModule.for_function_name` as the optional callable-ownership lookup
on the same existing module root. It iterates
`CellProfilerModule.__registry__.values()`, compares
`declared_function_names()`, and duplicate function ownership fails during
`CellProfilerModule.__init_subclass__`. `for_module`, `require_module`, and
`for_function_name` return the registered declaration class, never an instance
or metadata record. No separate required function-name lookup is added;
`require_callable` owns the fail-loud exact-name operation.
Callable-to-module resolution in the compiler, import pass, backend package,
report, and runtime goes through this root. The generated pipeline path never builds
another module dictionary and never reads `contracts.json`.

Add `CellProfilerModule.require_callable(function_name)`. It resolves the owning
declaration, imports `module_type.__module__`, selects the declared function
object, validates that it is callable, builds its source-declared
`CallableContract`, and validates execution scope, runtime adapter ABI, special
I/O slots, and implementation requirements. It never compares a module-owned
processing value because none remains. It never writes module facts to
`vars(func)`, calls `attach_callable_contract_metadata`, wraps the callable, or
keeps a callable-metadata cache. Processing contract, runtime image execution
mode, special inputs, special outputs, and plate execution scope are declared on
the raw implementation through the existing generic callable decorators. The
plate exporters declare `FunctionStepExecutionScope.PLATE` on their raw
callables and have no image `ProcessingContract`.

`CellProfilerInvocationContractProvider` is the sole composition boundary. It
starts from `CallableContract.from_callable(raw_func)` and constructs one new
immutable compiled contract with `dataclasses.replace`. Its metadata replaces
only compiler-derived fields: the exact invocation's
`ModuleArtifactContract` and the CP `RuntimeAdapterSpec`. Required variable
components and allowed groupings remain the values declared by the raw callable
and read through `CallableContract`; neither the module class nor the artifact
contract copies them. The provider validates every replaced value against the
raw callable and owning declaration before publishing the contract. It never
projects the compiled metadata back onto the function object. The lazy backend
module `__getattr__` and `__dir__` derive their inventory from
`CellProfilerModule.__registry__` and call `require_callable`; they own no
function registry. Delete
`CellProfilerFunctionCatalog`,
`openhcs/processing/backends/cellprofiler/library.py`, the public wrapper
factory, and `CELLPROFILER_MODULE_ATTR`. Compiler code identifies a CP callable
through `CellProfilerModule.for_function_name(func.__name__)`; it never reads a
wrapper attribute.

Delete `CellProfilerFunctionReferenceTransportStrategy` and its now-empty
`FunctionReferenceTransportStrategy` root. Declaration-loaded functions remain
attributes of their real backend modules, so the existing
`FunctionReferenceTransportAuthority.function_reference` importable-callable
branch serializes them by `__module__` and `__name__`; its registered-function
lookup remains the other direct path. Remove strategy calls from
`reference_function_spec`, `function_reference`, and `callable_metadata`. A
module object in a FunctionStep remains a type error. Compiled axis-scoped
`CellProfilerRuntimeCallable`
objects never enter public source or initial submission transport, so no
CP-specific reference or preserve branch remains. Compiled-context worker
transport uses the generic `FunctionReference` plus its existing
`CallableMetadata`. `CallableContract.resolve_runtime_callable` resolves that
reference and invokes the runtime-callable factory on its exact
`RuntimeAdapterSpec`; the CellProfiler factory constructs
`CellProfilerRuntimeCallable(raw_func, callable_contract)` only for a compiled
contract that declares that adapter. Adapter-free axis references and plate-scoped
references return their resolved importable callable directly. No ownership
attribute, copied processing-contract argument, rehydrator request, or
rehydrator registry participates.

`CellProfilerModule.resolve_function(module, contract=..., source_bindings=...)`
returns the actual declaration-loaded callable during parsed `.cppipe` import.
Variant-owning declarations choose one of their own
`declared_function_names()` from parsed settings, the already-derived artifact
contract, and the resolved source bindings, then delegate that exact name to
`require_callable`. The base method selects `function_name`.
`resolve_function` is the sole polymorphic import selector. Delete
`resolve_semantic_function` and migrate its two source-axis overrides and every
settings-only `resolve_function` override into this method; no
`ResolvedModuleFunction`, function-name DTO, second selector, generator
registry stage, or `ProcessingConfig` argument is involved. Public compilation
does not call this selector: a public `FunctionStep` already declares the exact
raw callable, and the compiler validates that callable against
`CellProfilerModule.require_callable` without substituting a variant.

There is one `.cppipe` translation operation:

```python
def import_cellprofiler_pipeline(
    cppipe_path: str | Path,
    *,
    filemanager: FileManagerLike | None = None,
    backend: Backend = Backend.DISK,
    source_root: str | Path | None = None,
) -> tuple[list[FunctionStep], PipelineConfig]:
    ...
```

It lives in `openhcs/interop/cellprofiler/pipeline_import.py`. The UI, workspace
preparer, benchmark, and tests call it directly. Delete `import_service.py`,
`import_records.py`, `pipeline_compiler.py`, `compiler_registry.py`, their
ABCs, DTOs, mutable singleton, aliases, and re-exports.

The parser validates the `.cppipe` address and backend through its existing
typed API. `source_root` is the explicit CellProfiler default input folder used
only to resolve import-time external resources; the workspace/UI/benchmark
caller passes its selected plate root and standalone imports default to the
parent of `cppipe_path`. Translation does not search ancestors or descendants.
Conversion embeds external-resource values into public callable kwargs and
writes resolved `ImagePlaneSource` values and source-binding declarations into
`PipelineConfig`. The `.cppipe` path and import root are never consulted after
the public steps/config pair is returned; execution receives its ordinary plate
path separately through the existing orchestrator request.

`import_cellprofiler_pipeline` owns conversion as one cohesive pure operation.
Its private loop accepts the complete ordered parsed module sequence, resolves
every enabled module through `CellProfilerModule.require_module`, lets setup
modules contribute the existing source-binding config, constructs the actual
public `list[FunctionStep]`, and raises setting-coverage errors at the owning
binding boundary. It does not call or return a generator object, accept a
separate skipped-module list, or retain a copied registry, pruner,
runtime-contract projector, emitter, build stage, diagnostics carrier, or
intermediate pipeline result.

Translation performs no source rendering or filesystem write. A caller that
explicitly requests Python source invokes
`FunctionStepTransportAuthority.source_from_pipeline(steps)` and routes that
string through its existing generic file/VFS boundary. No CP import DTO stores
the source address, source text, module references, or setting coverage. ZMQ
reconstruction of the returned public declarations is the acceptance proof
that the projection is sufficient.

Generator-time artifact pruning and terminal-output materialization disappear.
Dead output selection and runtime artifact materialization remain the generic
compiler's responsibility. A source `.cppipe` `SaveImages` module becomes an
explicit public export step; no importer flag changes upstream contracts or
removes user-declared processing steps.

### Sparse Identity Override Algorithm

After setup modules have contributed the pipeline source config, each ordered
executable source module first follows the non-circular contract, callable,
callable-contract, behavior-binding, and processing-config order
specified below. The private import pass then performs this exact sparse
identity sequence with that selected raw callable:

1. Bind every typed setting through the module's existing
   `SettingToKeywordBinding` declarations. Partition the resulting kwargs by
   the canonical raw callable signature: signature members are behavior kwargs
   and nonmembers are identity candidates. An unbound parsed setting remains
   governed by ordinary bound/ignored/unmapped coverage. Remove each behavior
   kwarg whose typed value equals its raw signature default; retain every
   required or nondefault behavior value. The compiler reconstructs the same
   effective behavior mapping from those signature defaults.
2. Normalize the draft raw callable and behavior kwargs only with
   `normalize_function_pattern`; do not construct a second function-pattern
   parser.
3. Call the same `module_blocks_for_invocation` and `artifact_contract` methods
   used by the compiler with the current effective source bindings,
   `available_artifacts`, and `main_flow_artifacts`.
4. Compare the inferred contract to the contract produced by the original
   parsed `ModuleBlock` by `ArtifactSpecRef`, partition, relation,
   materialization, and order.
5. For a mismatch, reconstruct with every identity candidate and require exact
   contract equality. Failure means the module lacks a correct owning binding
   or contract and is a conversion error.
6. Visit identity candidates in source setting order. Remove one candidate,
   reconstruct, and retain the removal exactly when full contract equality
   remains. The final insertion-ordered mapping is inclusion-minimal under this
   deterministic pass. Require the reconstruction call's consumed-name tuple
   to equal the retained keys in normalized order. The import pass does not emit
   a sidecar or a broader fallback payload.
7. Advance the same two local artifact cursors from the verified source
   contract.

Delete `public_artifact_identity_overrides`. The same binding that reads a
parsed `.cppipe` setting also applies its sparse public override during compiler
reconstruction. Exact contract comparison selects the required subset. No
descriptor table, setting-name registry, leaf override method, or import
switch is added.

### Processing Config Lowering

Delete the source-axis summary and `module_processing_config.py`; neither owns a
fact absent from the public step, callable contract, module artifact contract,
or inherited config. The exact module boundary is:

```python
@classmethod
def processing_config(
    cls,
    *,
    contract: ModuleArtifactContract,
    callable_contract: CallableContract,
    inherited: ProcessingConfig,
) -> ProcessingConfig:
    ...
```

The three arguments are existing nominal authorities. `ModuleArtifactContract`
owns ordered artifact inputs and outputs only. `CallableContract` is the sole
owner of `required_variable_components`, `allowed_group_by`, processing
locality, and execution scope. `inherited` is the concrete `ProcessingConfig`
already resolved by ObjectState or by the importer's pipeline-config context.

The root implementation performs one lowering:

1. choose `InputSource.PIPELINE_START` when the artifact contract declares
   inputs, otherwise choose `InputSource.PREVIOUS_STEP`;
2. for plate scope, set variable components to empty and grouping to
   `GroupBy.NONE`;
3. for axis scope, use callable-required variable components when declared,
   otherwise inherit `inherited.variable_components`;
4. use the module's import `group_by` declaration when present, otherwise
   inherit `inherited.group_by`;
5. validate the resulting axis/group relation through
   `FuncStepContractValidator` and return `dataclasses.replace(inherited, ...)`.

The module import `group_by` value selects the ordinary OpenHCS grouping emitted
for a converted module. It is not an allowed-grouping constraint. Allowed
groupings remain callable metadata. Artifact contracts contain no axis or
grouping fields, and no declaration provider reconstructs them from artifacts.

The parsed-module import order is fixed and non-circular:

1. derive the original `ModuleArtifactContract` from the parsed `ModuleBlock`
   and the current artifact cursors;
2. resolve the candidate `LazyStepSourceBindingsConfig` and inherited
   `ProcessingConfig` under the completed pipeline config;
3. call `module_type.resolve_function` with `module`, `contract`, and the
   resolved source bindings, then require that the result is the exact object
   returned by `module_type.require_callable(result.__name__)`;
4. read `CallableContract.from_callable(raw_func)` and require its declared
   processing contract;
5. bind typed behavior kwargs against that raw callable's public signature;
6. call `module_type.processing_config` with the contract, callable contract,
   and inherited processing config;
7. construct the public `FunctionStep` and perform sparse identity minimization.

No operation in this order reads `ProcessingConfig` to choose the callable,
and no operation derives processing locality from a module class field. The
compiler path starts at step 4 with the exact callable already present in the
public `FunctionStep`; it reconstructs module blocks and contracts for that
callable but never runs the import selector.

The import pass computes the most common concrete config once, constructs
`PipelineConfig(processing_config=LazyProcessingConfig(...))`, and places a
step `LazyProcessingConfig` on each real FunctionStep only for fields that
differ from that inherited value. `input_source` is an enum value in the
concrete config; it is never represented by `StepInputSourceLiteral` or a
Python source string.

### Import And Workspace Boundary

The import boundary has no CP-specific request, result, provenance, module
reference, or diagnostics carrier. `import_cellprofiler_pipeline` returns
exactly the two public values required by ordinary OpenHCS execution:

```python
pipeline_steps: list[FunctionStep]
pipeline_config: PipelineConfig
```

The function validates that it produced at least one executable step and a
concrete `PipelineConfig` before returning. Setting coverage is enforced while
binding each parsed module and is not copied into a post-import record. Source
rendering remains the generic caller-owned projection of `pipeline_steps`; the
import function neither stores nor writes it.

Delete `CellProfilerPipelineProvenance`, `CellProfilerPipelineImportRequest`,
`CellProfilerPipelineImportResult`, `CellProfilerModuleReference`, and every
property, alias, registry entry, test fixture, or UI cache keyed to those
types.

Change `InputWorkspacePreparationResult` in
`openhcs/core/input_workspace.py` to carry typed
`pipeline_steps`, `pipeline_config`, and
`SourceBindingWorkspaceMaterialization` fields instead of
`prepared_pipeline: Any` and `materialization: Any`. It remains the generic
orchestrator workspace result. CellProfiler source ingestion maps its import
result into those public fields before returning to orchestration.

The generic workspace fields are:

```python
pipeline_steps: list[FunctionStep] | None = None
pipeline_config: PipelineConfig | None = None
materialization: SourceBindingWorkspaceMaterialization | None = None
```

`InputWorkspacePreparationResult.__post_init__` requires
`pipeline_steps is None` and `pipeline_config is None` together for a
source-only workspace, or requires both concrete values together for a
workspace containing a pipeline. It copies and validates the step list. There
is no pipeline without config and no config embedded in another pipeline
object.

Delete `CellProfilerSourceSchemaProjection`,
`CellProfilerSourceSchemaWorkspace`, and
`CellProfilerSourceSchemaPreparation`; both source-only and source-plus-pipeline
entry points return `InputWorkspacePreparationResult`. Delete
`CellProfilerPlateWorkspaceRequest` and `CellProfilerPlateWorkspaceResult`;
`CellProfilerPlateWorkspacePreparer` owns its two path fields directly and
returns the generic result.

Collapse `CellProfilerSourceSchemaMaterializer` and
`CellProfilerSourceSchemaMaterializationScope` into one private function that
calls `materialize_source_binding_workspace` with the request's exact declared
source root, resolved config, backend, and file manager. The source root must
exist in the submitted VFS and is never replaced by a parent or child inferred
from `.cppipe` placement. Delete `CellProfilerSourceRootResolver`, candidate and
resolved-root records, path admission/context records, the exclusion-policy
registry, and both leaves. Source inclusion comes only from the submitted
`SourceBindingsConfig` filters over the exact declared source universe; no
recursive bucket inference, candidate usability probe, or fallback root
remains.

Remove `SourceBindingContext.import_result`, GUI maps of retained CP import
results, `CellProfilerImportResultProvider`, and the no-op
`CellProfilerPipelineRuntimeBindingService`. `SourceBindingContext` stores the
resolved `SourceBindingsConfig`, public steps remain in child ObjectStates, and
pipeline config remains on the orchestrator; no later rebinding stage remains.

Delete `pipeline_compiler.py`, `compiler_registry.py`, and
`runtime_pipeline.py` after their one real operation moves to
`import_cellprofiler_pipeline`. Delete `partition_cppipe_modules`; the direct
operation contains one local ordered loop in which each enabled module declaration decides
whether it contributes an executable step; disabled modules are ignored and no
module-reference or diagnostics result carrier survives translation.
There is no partition DTO, generation/preparation request wrapper, compiler
class, registration function, or benchmark alias.

Delete the direct orchestrator execution helper and its result/progress types.
Change integration tests to submit public source through
`ZMQExecutionClient.submit_compile` and `submit_pipeline`. Keep only
`validate_cppipe_runtime_observation`; it consumes the server's generic
observation export.

Extend `ZMQRuntimeExecutionObservationExport` with the existing
`RuntimeArtifactExecutionExpectation` derived from compiled
`CompiledStepPlan.artifact_outputs` and their materialization plans. Benchmark
validation consumes that compiled expectation instead of reparsing skipped
CellProfiler infrastructure modules or reading generated contract sidecars.
Explicit SaveImages steps make image export expectations part of the compiled
generic plans. For `FileBundleOptions`, the recorded file-bundle artifact
supplies the exact relative output paths after the plate callable returns; the
observation validates those paths without a CP exporter flag or filename rule.

### Direct Pipeline List Transport

Delete `openhcs/runtime/zmq_pipeline_transport.py`. Move its single public
export name to `FunctionStepTransportAuthority` and add these generic methods:

```python
PIPELINE_STEPS_EXPORT: ClassVar[str] = "pipeline_steps"

@classmethod
def source_from_pipeline(
    cls,
    pipeline_steps: Sequence[FunctionStep],
    *,
    clean_mode: bool = True,
) -> str:
    """Normalize and pycodify one public FunctionStep list."""

@classmethod
def pipeline_steps_from_namespace(
    cls,
    namespace: Mapping[str, object],
) -> list[FunctionStep]:
    """Require and normalize the public FunctionStep export."""
```

`source_from_pipeline` is the source authority used by UI code mode,
the CellProfiler import operation, agent pipeline rendering, and
`OpenHCSExecutionSubmission.pipeline_code`. `pipeline_steps_from_namespace` is the
single validator used after executing source in the ZMQ server and agent
source-session path. It rejects a missing export, non-list value, and non-step
member before compilation.

Add `OpenHCSExecutionSubmission.pipeline_code()` to return its explicitly
submitted source or call `source_from_pipeline(self.submission_pipeline)`.
Change `ZMQExecutionRequestBuilder.pipeline_transport` to
`pipeline_code: str` and pass that string directly to
`ZMQExecutionRequestPayload`. `ZMQConfigProjection.from_task` pycodifies global
and pipeline configs directly through one private function using the existing
pycodify `Assignment`; the projection stores the resulting source fields and
hashes. Source hash labels use the existing
`ZMQExecutionRequestPayload.pipeline_sha` algorithm rather than a source-value
wrapper.

Delete `PycodifiedSource`, `PycodifiedPipelineCode`,
`PycodifiedPipelineStepSource`, `PycodifiedConfigSource`, and
`PycodifyAssignmentSourceRequest`. They expose one source string plus an
immediately delegated operation and own no validation or dispatch. No alias or
replacement source carrier remains.

`OpenHCSExecutionSubmission`, `ExecutionPipelineDefinition`,
`ExecutionSessionRecord`, and `ZMQExecutionContext` store direct mutable step
lists. `OpenHCSExecutionConfigBundle` is stored directly by submissions and
contexts. No carrier base class or one-field boundary object participates in
dispatch; their `AutoRegisterMeta` registries have no consumer in the current
tree.

Delete agent `PipelineIdBoundary`, `ExecutionPipelinePayload`, and
`ExecutionPipelineDefinitionProvider` with its draft/source leaves. Retain one
concrete `ExecutionPipelineDefinition` with `pipeline_id: str`,
`pipeline_steps: list[FunctionStep]`, and `pipeline_source: str | None`. Add
`build_pipeline_definition(session_id, pipeline_service)` to the existing
`ExecutionPipelineSessionRequest` nominal root and implement it directly on
`DraftPipelineSessionRequest` and `PycodifiedPipelineSessionRequest`.
`ExecutionSessionRecord` stores its direct step list/source rather than
inheriting the payload carrier. Request-type polymorphism is the sole dispatch;
no provider object or one-string identity type remains.

## Module Declaration Contract

`CellProfilerModule` defines default polymorphic behavior for:

- contributing setup semantics to the existing `SourceBindingsConfig`;
- deciding whether an imported module emits an executable FunctionStep;
- reconstructing setting rows from typed public kwargs;
- selecting ordered source and runtime artifact inputs;
- deriving ordered output artifact specs;
- pairing special-input parameters with runtime input specs;
- pairing returned slots with output specs;
- selecting main-flow input and output participation;
- selecting processing and execution mode;
- consuming sparse identity bindings during setting reconstruction;
- validating required settings and artifact cardinality.

Leaf module classes override only behavior that differs from the default. The
implementation reuses `CellProfilerModuleAuthority` and existing MRO policy
mixins already inherited by module classes. It does not add an `issubclass`
chain, module-name table, copied dynamic view, descriptor registry, or product
of independent capability classes.

Add these methods to the existing root:

```python
@classmethod
def contribute_source_bindings(
    cls,
    module: ModuleBlock,
    config: SourceBindingsConfig,
) -> SourceBindingsConfig:
    ...

@classmethod
def emits_function_step(cls) -> bool:
    ...

@classmethod
def resolve_function(
    cls,
    module: ModuleBlock,
    *,
    contract: ModuleArtifactContract,
    source_bindings: StepSourceBindingsConfig,
) -> Callable[..., object]:
    ...
```

The base source contribution is a no-op and the base emission result is true.
The base function selector ignores `contract` and `source_bindings`, loads
`function_name` through `require_callable`, and verifies that the result belongs
to `declared_function_names()`. `ResizeModule`, `ResizeObjectsModule`,
`DefineGridManualModule`, `WatershedModule`, and
`ClassifyObjectsSingleMeasurementModule` select from their typed parsed
settings. `MeasureTextureModule`, `MeasureColocalizationModule`,
`MeasureGranularityModule`, and `IdentifyObjectsInGridModule` select from the
typed artifact input presence already present in `contract`, rather than
reparsing artifact name strings. `DilateObjectsModule` and `RemoveHolesModule` inspect
the resolved `source_bindings.source_stack_components` declaration directly.
No leaf accepts `ProcessingConfig`, a function-name default, or a resolution
request. This method is called only by parsed-module import, never by public
FunctionStep compilation.
Delete `InfrastructureCellProfilerModule`. `SourceSetupCellProfilerModule`
inherits `CellProfilerModule` directly, returns false, and makes source
contribution abstract. Its five leaves contain the
bodies currently held by the five `SetupModuleCompiler` leaves. Import
validation rejects any non-emitting enabled declaration whose source
contribution leaves the config unchanged; “infrastructure” cannot mean
silent omission.

`SaveImagesModule`, `ExportToSpreadsheetModule`, and
`ExportToDatabaseModule` inherit executable module MRO roots. Their public
FunctionStep kwargs contain typed user settings; image/object names remain the
module's existing compile-identity settings and are emitted only as sparse
overrides.

SaveImages uses the existing axis execution path. Move the real callable into
`openhcs/processing/backends/cellprofiler/save_images.py`; delete the interop
pass-through. Its contract consumes the selected `ImageArtifactType` and
declares one materialized `ImageArtifactType` output. The selected image uses
`RuntimeArtifactInputPartition`; the converted copy uses
`DeclaredArtifactOutputPartition` only, because the generic FunctionStep
runtime records it after return. No `RecordedArtifactOutputPartition` pretends
that a CellProfiler adapter wrote it. Its raw callable uses the
existing `special_inputs` declaration for the image-to-save parameter and
one `special_outputs` slot for that materialized output. It returns a two-item
tuple of main image then converted export image, preserving the first output as canonical OpenHCS
main flow. The compile-derived selected-image and materialized-artifact names
never enter public kwargs. Because this callable performs no CellProfiler
workspace operation, its `CallableContract` has no runtime adapter and the
compiler retains the raw generic callable instead of constructing
`CellProfilerRuntimeCallable`. Add generic
`ImageFileOptions` in `openhcs/processing/materialization/options.py` and its
writer in `materialization/core.py`. The options contain a relative suffix/path
only; the writer selects preparation through the existing
`ImageFileSerializationFormat` registry, so PNG, JPEG, TIFF, and NPY support is
not encoded as a CP switch or another format table. The callable performs the
declared bit-depth/image-kind conversion only on the special output. It does
not mutate an upstream `ArtifactSpec` or main-flow payload.

ExportToSpreadsheet and ExportToDatabase use the following closed generic
plate-execution algorithm:

1. Add `FunctionStepExecutionScope` with exactly `AXIS` and `PLATE` to
   `openhcs/core/callable_contract.py`, add its one metadata key to
   `openhcs/core/function_contract_metadata.py::FunctionContractAttribute`,
   expose `execution_scope(...)` from
   `openhcs/core/pipeline/function_contracts.py`, and carry the value through
   `CallableMetadata` into each compiled invocation's existing
   `CallableContract`. `AXIS` is the default. Add a derived
   `CompiledFunctionPattern.execution_scope` property that requires every
   invocation contract in the step to have one identical scope; an empty
   pattern retains the ordinary `AXIS` no-op lifecycle and a mixed-scope
   pattern fails compilation. Do not store scope on `CompiledStepPlan`.
   Add one function in `openhcs/core/function_patterns.py`,
   `resolve_function_pattern_execution_scope(pattern, provider, step_context)`,
   which normalizes the public pattern, applies the same invocation contract
   provider used by compilation, and performs that identical uniform-scope
   check before path planning. Both the pre-plan function and the compiled
   property call one `FunctionStepExecutionScope.require_uniform(contracts)`
   method on the enum owner; no second scope algorithm or cache is added.
2. Add `RuntimeArtifactBatch` to `openhcs/core/runtime_stores.py`, beside
   `StoredRuntimeValue` and `RuntimeArtifactQuery`. It stores an
   immutable tuple of the step's declared input specs plus a mapping from axis
   id to the exact `StoredRuntimeValue` records selected by the current step's
   compiled `ArtifactInputPlan` values. It exposes typed
   `specs_of_type(ArtifactType)`, `records(ArtifactSpecRef)`, and
   `records_of_type(ArtifactType)` queries; it has no CP names, `.cppipe`,
   store, context, or fallback lookup. Source input specs remain available for
   identity/projection, but their image payloads are not loaded into the batch.
   `__post_init__` copies every sequence to tuples and every mapping to
   `MappingProxyType`. `records(ref)` first requires that the exact ref occurs
   in `input_specs`, then returns matching selected records in axis and store
   order; a declared source spec has an empty record mapping. This placement
   adds no reverse `runtime_stores` import to `runtime_values` and preserves the
   existing dependency direction.
3. During worker execution, omit plans whose callable contract declares
   `PLATE`. After all worker results have merged into parent contexts,
   `compiled_plate_execution.py` invokes those plans once in source order.
   The executor starts with a local immutable copy of merged
   `records_by_axis`; after recording each plate output, it creates the next
   immutable map with that output appended under the metadata-writer axis.
   A later plate step therefore consumes earlier plate artifacts through the
   same compiled query path without mutating the runtime observation or adding
   another store.
   Collapse the one-field `AnalysisConsolidationPlan` class into functions in
   `orchestrator/analysis_consolidation.py`; the post-plate executor runs before
   the existing optional CSV consolidation function.
   `validate_compiled_plate_execution` rejects
   `RuntimeObservationMode.OMIT` when any plate-scoped plan exists, before a
   worker starts; the existing default `MERGE_INTO_PARENT` supplies the records.
4. For each plate-scoped plan, require the same invocation contract and kwargs
   in every compiled context, select only records addressed by that context's
   artifact input plans whose module-contract items belong to
   `RuntimeArtifactInputPartition`, through the existing
   `RuntimeArtifactQuery.from_input_plan`, and build one
   `RuntimeArtifactBatch`. Use the context
   whose `CompiledStepPlan.create_openhcs_metadata` is true as the sole output
   owner. Missing, ambiguous, or context-drifting plans fail before invoking
   the callable. `SourceArtifactInputPartition` contributes its `ArtifactSpec`
   to `input_specs` but never a runtime query or image payload; the partition
   type, not an artifact-name or image-type test, owns that distinction.
5. Invoke the compiled raw callable with typed public kwargs plus
   `artifact_batch=batch`, using `RuntimeArtifactBatch.require_parameter_name()`
   rather than a string literal. A plate-scoped callable does not require a
   `ProcessingContract` or the CP image runtime adapter. The compiler rejects a
   plate-scoped callable with an axis-scoped successor, source-bound image
   payload loading, dict-pattern routing, an undeclared batch parameter, or
   undeclared runtime-record access.
6. Add generic `FileBundleOptions` to the existing materialization writer
   family. The callable returns one insertion-ordered built-in `dict` of
   validated relative paths to
   `str | bytes`; the writer rejects absolute paths, parent traversal, duplicate
   normalized paths, and unsupported mapping or payload values, UTF-8 encodes `str`
   entries, and emits only `bytes` outputs beneath the compiled artifact
   analysis directory. Extend
   `external/PolyStore/src/polystore/disk.py::DiskStorageBackend.save` so the
   already-supported generic `bytes` payload is persisted with
   `Path.write_bytes` before extension-based format dispatch. This rule is
   selected by payload type and contains no extension names. The post-plate
   executor requires exactly one declared `SpecialArtifactType` output,
   records its normalized file-bundle value in the metadata-writer context
   using the existing `ArtifactOutputPlan`, and invokes the normal
   materialization entry point. The writer's compile-time candidate path is the
   artifact base path used only for backend routing; its runtime `Output` paths
   are the validated bundle keys and its primary path is the first declared
   bundle entry in dict insertion order. No CP-specific writer or filename
   branch is added to core or PolyStore.
7. `ExportToSpreadsheetModule.artifact_contract` enumerates the available
   `MeasurementsArtifactType` and `RelationshipsArtifactType` specs in source
   order and declares one `SpecialArtifactType` file-bundle output under
   `DeclaredArtifactOutputPartition` only. Replace the
   unrelated functions in `interop/cellprofiler/spreadsheet_export.py` with a
   plate-scoped callable on the backend module that renders those exact records
   according to typed delimiter, prefix, selected-column, aggregate, and NaN
   settings.
8. `ExportToDatabaseModule.artifact_contract` enumerates those table specs plus
   the image specs required by CPA properties and declares the same single
   declared file-bundle output shape. Replace
   `interop/cellprofiler/database_export.py` with a plate-scoped callable in
   `processing/backends/cellprofiler/export_to_database.py`. Refactor
   `CellProfilerAnalystProjectionBuilder` to accept `RuntimeArtifactBatch`,
   `CellProfilerDatabaseExportSettings`, and derived image-channel specs
   directly; remove `CellProfilerExecutionExportContext` and
   `CellProfilerAnalystExportRequest`. Add a real SQLite renderer that writes
   the projection into an in-memory SQLite connection and returns serialized
   database bytes. Return those bytes and the existing CPA properties renders
   in the file bundle.

The generic API added by this phase is exactly:

```python
class FunctionStepExecutionScope(str, Enum):
    AXIS = "axis"
    PLATE = "plate"


def execution_scope(
    scope: FunctionStepExecutionScope,
) -> Callable[[CallableT], CallableT]: ...


@dataclass(frozen=True, slots=True)
class RuntimeArtifactBatch:
    input_specs: tuple[ArtifactSpec, ...]
    records_by_axis: Mapping[str, tuple[StoredRuntimeValue, ...]]

    @classmethod
    def require_parameter_name(cls) -> str:
        return "artifact_batch"

    @classmethod
    def parameter(cls) -> inspect.Parameter: ...

    def specs_of_type(
        self, artifact_type: type[ArtifactType]
    ) -> tuple[ArtifactSpec, ...]: ...

    def records(
        self, ref: ArtifactSpecRef
    ) -> Mapping[str, tuple[StoredRuntimeValue, ...]]: ...

    def records_of_type(
        self, artifact_type: type[ArtifactType]
    ) -> Mapping[str, tuple[StoredRuntimeValue, ...]]: ...


@dataclass(frozen=True)
class ImageFileOptions(FileOutputOptions, SourceOptions):
    relative_path_template: str | None = None


@dataclass(frozen=True)
class FileBundleOptions(FileOutputOptions):
    filename_identity: MaterializedFilenameIdentity = (
        MaterializedFilenameIdentity.ARTIFACT_NAME
    )
```

The existing `StepInputDependencyKind` gains `NO_MAIN_FLOW`, and
`StepInputDependency` gains `no_main_flow()`. This extends the current
main-input authority; it does not add another execution-scope carrier.

`CallableMetadata` and `FunctionContractAttribute` gain one
`execution_scope: FunctionStepExecutionScope` value, and `CallableContract`
exposes it as a property. `CompiledFunctionPattern` derives the uniform value
from its invocation contracts; `CompiledStepPlan` gains no scope field or
exporter-specific state. `MaterializationFormat` gains
`IMAGE_FILE` and `FILE_BUNDLE`; writer dispatch remains keyed by the existing
options-type registry. `ImageFileOptions.relative_path_template` expands named
backreferences only from the output's `SourceImageIdentity.component_metadata`,
then applies the same absolute/parent-traversal validation as file bundles.
Absent templates retain the existing source-identity/artifact-name base-path
authority. The two option classes are behavior-bearing writer dispatch keys,
not duplicated capability descriptors. Add fail-loud
`ImageFileSerializationFormat.require_path`; give the native serialization leaf
explicit NPY/TIFF suffixes and make `ImageFileOptions` call the fail-loud
query. PNG and JPEG retain their existing registered leaves. HDF5 has no
PolyStore image writer and therefore fails SaveImages compilation together
with every unknown suffix; it is not mislabeled as native serialization. A
real HDF5 implementation requires a nominal writer on the generic PolyStore
format registry and is outside this migration. No current SaveImages setting
receives a format-specific branch in the CP compiler.

`RuntimeArtifactBatch.parameter()` returns one required keyword-only parameter
annotated with `RuntimeArtifactBatch`. The two plate callables use the existing
`runtime_bound_parameters(RuntimeArtifactBatch)` decorator. The generic
callable metadata reader, signature projection, ObjectState, pycodification,
and function-pane exclusion paths therefore treat the batch through their
current runtime-bound-parameter machinery; no plate-specific UI filter or
hidden kwarg is added.

Both plate exporters derive their consumed artifact set solely from the module
contract at the public FunctionStep position. They do not scan a
process-global runtime store. Delete the boolean exporter flags and
`CPPipeInfrastructureProfile`; ZMQ observations derive export expectations from
the compiled output plans and emitted bundle paths. Delete
`RuntimeImageExportSpec`, `RuntimeImageExportBitDepth`,
`RuntimeExportExpectation.image_export_specs`, and the corresponding
artifact-record image snapshot parameters. Explicit SaveImages parity reads the
materialized image files produced by `ImageFileOptions`; it never applies a
second conversion to an in-memory record.
`CellProfilerModule.__init_subclass__` permits `function_name=None` only when
`declared_function_names()` is empty and still validates executable
declarations and duplicate function ownership. Delete
`CellProfilerModule.contract` and every leaf assignment to it. The raw
callable's generic `CallableContract.processing_contract` is the sole array
locality declaration. `require_callable` requires an axis-scoped callable to
provide that contract and requires a plate-scoped callable to omit it and its
runtime image execution mode. Module code queries
`CallableContract.from_callable(raw_func).require_processing_contract()`; it
never stores or compares another module-level processing value.

Delete `openhcs/interop/cellprofiler/module_semantics.py` and its category,
dimensionality, trait, module-semantics, and semantic-family DTOs. The only
consumer is `benchmark/converter/compatibility_matrix.py`; rewrite that report
to read `module_name`, `declared_function_names()`, the source callable's
`CallableContract.processing_contract`, `respects_masks`, and MRO identity
directly from each registered declaration.
Delete semantic-family coverage rather than constructing a second family
classification from those fields.

Delete the compile-time public/grouped/coalesced setting-name method family and
replace `ModuleSettingCoverageStatus` with exactly `BOUND`, `IGNORED`, and
`UNMAPPED`. A declared binding or owning compound-row parser produces `BOUND`;
the parser's universal nonexecution rows and the module declaration's explicit
ignore set produce `IGNORED`; every other row produces `UNMAPPED` and fails
conversion. Delete `ARTIFACT_CONTRACT`, `TYPED_IGNORE`, `CALLER_IGNORE`,
`INFRASTRUCTURE`, and the caller-provided ignore set. Coverage does not recreate
a compile-identity category. The import result reports sparse identity
overrides from the actual consumed-name tuple returned by module
reconstruction. Delete
`artifact_setting_symbols` and every materialized-image compile-time kwarg name
or metadata key: explicit export steps plus generic materialization config own
that behavior.
Remove every identity-only parameter from raw CP callable signatures. Retain
one `SettingToKeywordBinding` on the owning module MRO for each such setting,
referencing its existing setting-name ClassVar and deriving the normalized
keyword by default. Keep parameters such as CalculateMath `output_name` in the
signature where runtime values also depend on them; their same binding is
classified as behavioral by signature membership.

Convert these current module-policy mixin families from module-name registry
lookups to classmethod authorities on the module MRO:

- `CellProfilerInvocationExecutionModePolicyMixin`;
- `CellProfilerMainFlowReplacementPolicyMixin`;
- `CellProfilerImageOutputSourcePayloadPolicyMixin`;
- `CellProfilerImageOutputValuePolicyMixin`;
- `CellProfilerObjectLabelOutputSourceContextPolicyMixin`;
- `CellProfilerPrimaryImageInputPolicyMixin`;
- `CellProfilerSpecialInputPolicyMixin`;
- `CellProfilerDualScopeMeasurementPolicyMixin`;
- `CellProfilerObjectInputPolicyMixin`.

`CellProfilerModule` provides each default classmethod. Existing leaf mixins
inherit `CellProfilerModuleAuthority` and override the classmethod. Existing
module declarations continue to compose those mixins through normal Python MRO.
`CellProfilerModuleRuntimePlan.build` receives the resolved raw callable and
compiled `CallableContract`, requires their module declaration through
`CellProfilerModule.for_function_name`, validates it against the compiled
module contract name, and stores that module type once.
`CellProfilerModuleExecutor` stores the plan and runtime code calls classmethods
on `plan.module_type`. Enum-keyed,
nominal-value-keyed, and most-derived-context runtime strategy registries remain
unchanged because they dispatch on their real semantic axis, not on a mirrored
module-name table.

`CellProfilerModuleRuntimePlan` retains only static facts derived from the raw
callable's `CallableContract`, aggregate contract, and resolved module type. It
does not store a copied processing contract, signature/kwarg specification,
module-name policy objects, or assume that every aggregate spec is active in
every execution group. For each `CellProfilerModuleRunRequest`, the executor
forms active `ArtifactSpecCollection` values by exact `ArtifactSpecRef` joins
between the compiled module contract and:

- `RuntimeAdapterRequest.artifact_inputs`;
- `RuntimeAdapterRequest.artifact_outputs`;
- source bindings selected for the request's exact typed component group.

`ModuleArtifactContract.require_items_for_specs(partition_type, specs)` accepts
the selected artifact specs, joins each ordered occurrence by `spec.ref`, and
requires equality of the complete `ArtifactSpec`, including plan role, artifact
type, name, relations, and materialization. Missing refs, conflicting specs,
insufficient occurrence cardinality, partition drift, and spec drift fail
before runtime input loading. `CompiledFunctionInvocation` therefore
replaces `artifact_input_keys` / `artifact_output_keys` with exact
`artifact_input_refs: tuple[ArtifactSpecRef, ...]` and
`artifact_output_refs: tuple[ArtifactSpecRef, ...]`; no runtime component uses
name intersection.

Main-flow replacement, primary-image selection, special-input binding, output
recording, and returned-slot matching use those active collections. A combined
contract output that differs by execution group declares
`GroupLineageSourceRelation`; compilation rejects an ambiguous grouped output
without lineage. This delegates group selection to the generic artifact plans
instead of rebuilding it in CellProfiler runtime code.

The retained `CellProfilerModuleRuntimePlan` fields are exactly:

```python
raw_func: CellProfilerFunction
module_type: type[CellProfilerModule]
callable_contract: CallableContract
```

Add `CallableContract.require_module_artifact_contract()` beside the planned
`require_processing_contract()` method. `CellProfilerModuleRuntimePlan.contract`
is a property that calls the former; it is not a stored field. Delete cached
names, copied module contract, copied processing contract, copied artifact
collections, policy instances, booleans, signature mirrors, and output
projections from the plan.
`func` returns `raw_func`; `function_name`, `contract`, and
`processing_contract` become fail-loud properties over the two existing
authorities. Artifact and policy values are direct queries on these three owners
or active request data. `module_type` is non-optional; runtime-callable factory
construction fails before module execution when the resolved raw function has
no registered module declaration or compiled module contract.

`CellProfilerModuleExecutor` retains exactly one field:

```python
plan: CellProfilerModuleRuntimePlan
```

Delete `contract`, `_canonical_module_name`, `_primary_image_input_policy`, and
`_runtime_plans`. Contract, module name, callable, and module behavior are
properties or classmethod queries through `plan`. `prepare()` and `run()` no
longer accept a callable argument. Delete `CallableInvocationKwargSpec` and
every runtime unsupported-kwarg filter. The executor receives and forwards the
exact kwargs frozen on `CompiledFunctionInvocation`; runtime never reconstructs
or revalidates the public signature.

The retained `CellProfilerModuleRunRequest` fields are exactly:

```python
executor: CellProfilerModuleExecutor
image: CellProfilerRuntimeValue
adapter: CellProfilerRuntimeAdapter
kwargs: CellProfilerKwargDict
```

Delete its copied `func`, `plan`, `input_image`, and `current_image` fields. The
plan and callable come from `executor.plan`; the only construction site
currently assigns the same input object to both image fields, so every consumer
uses `image`. Module and function names come from the plan.

`RuntimeArtifactInputRequest` has this composition boundary:

```python
spec: ArtifactSpec
contract: ModuleArtifactContract
adapter: CellProfilerRuntimeAdapter
current_image: ImagePayloadMetadataInput | None
```

It no longer subclasses `ArtifactSpec`, so it cannot omit relations while
copying spec fields. Add `has_ref_for_partition(partition_type, ref)` to
`ModuleArtifactContract`. CellProfiler's existing
`RuntimeImageInputOrigin` strategy determines runtime, external, or stored
origin by querying `RuntimeArtifactInputPartition` first and
`SourceArtifactInputPartition` second. `RuntimeInputBindingRequestBase` carries
the aggregate contract and active runtime inputs; it does not carry
`module_name` or `RuntimeArtifactBindingScope`.

`CellProfilerOutputValueResolutionRequest` retains exactly:

```python
output_specs: tuple[ArtifactSpec, ...]
main_output: CellProfilerRuntimeValue
artifact_values: CellProfilerRuntimeValues
declared_output_specs: tuple[ArtifactSpec, ...]
```

The executor supplies `output_specs` from the active artifact plans and
`declared_output_specs` from the compiled module contract. The request has no
callable field and performs no callable metadata lookup.

Replace the fallback chain in `RuntimeReturnedOutputMatcher.resolve` with this
closed algorithm:

1. Require every retained spec to have one exact `ArtifactSpecRef` match in
   `declared_output_specs`.
2. Subtract artifact-value count from declared-output-spec count to obtain
   `main_spec_count`, and reject a negative count.
3. For zero, zip all declared specs to artifact values in order and treat
   `main_output` as undeclared passthrough flow.
4. For one, assign `main_output` to the first declared spec for which
   `artifact_spec_participates_in_main_flow` is true, then zip the remaining
   specs and values in declaration order.
5. For a value greater than one, select that many declared image specs with
   main-flow participation in declaration order, require `main_output` to be an
   image stack with exactly that many slices, unstack it onto those specs, and
   zip the remaining specs and artifact values in declaration order.
6. Reject every other count, order, main-flow selection, or stack shape.
7. Select retained outputs from the complete candidate sequence by exact spec
   identity and declaration order.

Delete `single_output_value`, semantic-only matching,
`resolve_positional_outputs`, `resolve_from_returned_specs`, and retained-tail
inference. These paths compensate for incomplete contracts and are forbidden
after Phase 1.

The module contract emits specs in raw callable return order. The first
main-flow-participating output remains the canonical FunctionStep output. Other
declared outputs are passed in that order to the existing returned-output
matcher. The matcher does not inspect the callable or classify names and
materializers a second time. Artifact relations express lineage and group
behavior.

## Migration Work Orders

Each phase ends with one authority for every migrated responsibility. A phase
does not leave a compatibility copy behind.

### Phase 0: Freeze The Boundary With Failing Tests

**Prerequisites**

- Current branch builds and the existing focused CellProfiler tests run.
- The official manifest exists at
  `benchmark/manifests/official30_portable_axis1.json`.

**Changes**

Add boundary tests before production edits:

- `tests/unit/test_cellprofiler_public_boundary_contracts.py`
  - compile a CP pipeline from only `PipelineConfig` and public
    `FunctionStep` declarations;
  - reconstruct and compile the pycodified source in a clean process;
  - assert generated source contains no contract, wrapper, sidecar, module
    number, symbol table, or pipeline metadata;
  - assert compile-derived names are absent from raw callable kwargs;
  - assert source binding inheritance removes redundant step bindings;
  - assert identical all-group patterns collapse to a plain callable.
- `tests/unit/test_function_step_transport.py`
  - assert transport round-trip preserves public steps and typed kwargs only;
  - assert submission, namespace reconstruction, and server context use direct
    lists with no pipeline carrier/boundary object.
- rewrite `tests/unit/test_cellprofiler_interop_namespace.py` so it rejects
  compatibility aliases, provenance/role exports, `SetupModuleCompiler`,
  compiler ABCs, import DTOs, and forwarding imports; retain only the pure
  import function, nominal module declaration, and public steps/config tuple
  assertions.
- rewrite `external/PolyStore/tests/test_filemanager_extended.py` so it rejects
  string workspace mappings and global-registry reconstruction, then validates
  structured refs and execution-local registry pickle round-trip.
- `tests/unit/test_openhcs_adapter.py`
  - assert benchmark submission source and UI-equivalent submission source have
    the same normalized `pipeline_steps`, `global_config`, and
    `pipeline_config` payload.
- module-focused tests in
  `tests/unit/test_cellprofiler_generated_pipeline_execution.py` for
  illumination, Align, MeasureColocalization, object topology,
  measurements, TrackObjects, Tile, multi-output modules, SaveImages,
  ExportToSpreadsheet, and ExportToDatabase.
- callable-signature and function-pane tests assert identity-only setting names
  are absent, sparse recognized identity kwargs compile and disappear before
  invocation, and an unrecognized extra kwarg fails compilation; AST tests
  reject `compile_time_*` methods on every registered module MRO class, and
  binding tests prove signature partition plus exact consumed-name reporting;
  all official module settings resolve through explicit bindings, one owning
  compound-row parser, or explicit ignores, with no normalized-name fallback.
- generic invocation-provider tests reject two claims regardless of factory
  registration order and verify the exact immutable CP provider field/method
  shape; generic callable tests prove compile-time public-kwarg validation and
  exact runtime forwarding with no runtime signature mirror.
- artifact-planning tests replace string invocation keys with exact
  `ArtifactSpecRef` tuples and reject same-name specs that drift in type,
  partition, plan role, relation, or materialization.
- function-pattern and ObjectState tests reject a tuple leaf with any arity
  other than two and contain no synthetic `RuntimeInvocationOptions` subtype;
- generic callable-scope tests assert axis plans run in worker lanes, plate
  plans run exactly once after merged worker observations, and plate batches
  contain only contract-selected specs and records; a second plate plan
  consumes the first plate plan's recorded output through its compiled
  `ArtifactInputPlan`; every plate plan carries
  `StepInputDependencyKind.NO_MAIN_FLOW` and no source/main-flow load plan;
  mixed-scope function patterns fail compilation and `CompiledStepPlan` has no
  stored execution-scope field.
- materialization tests assert `ImageFileOptions` dispatches through
  `ImageFileSerializationFormat` for PNG/TIFF/NPY and `FileBundleOptions`
  rejects unsafe paths while preserving bytes exactly and UTF-8 encoding text;
  PolyStore writes opaque bytes for unregistered suffixes while non-byte
  payloads still require nominal format registration; unknown image suffixes
  fail instead of using native fallback serialization.
- source-binding config tests assert all setup modules resolve through
  `CellProfilerModule.__registry__`, every setup fact survives lazy
  `PipelineConfig` pycodification and fresh-process reconstruction, and no
  `PipelineImageSchema` or setup registry participates; auxiliary source
  payloads reload from persisted workspace paths through `FileManager` after a
  process restart with no cache state; every microscope writer emits a
  structured `SourcePixelRef`, every mapping round-trips through its validating
  parser, and string mappings are rejected; ingestion uses the exact submitted
  root and never selects an ancestor or child from `.cppipe` placement;
  virtual loading selects the declared source backend from the same
  execution-local `FileManager` registry before and after worker
  reconstruction, delegates base-path address resolution to that backend, and
  applies ordered axis indices with no resolver class; mixed-backend
  `SourceCandidate` values retain their own exact refs and component projection
  registry coverage equals `set(AllComponents)`.
- callable-loading tests assert ownership and implementation lookup come only
  from `CellProfilerModule`, the backend package returns the underlying
  implementation callable, importing and compiling leave `vars(func)`
  unchanged, and no CP module attribute, catalog, callable cache, or
  compatibility metadata record exists.
- FunctionReference tests assert CellProfiler backend functions use the generic
  importable-callable reference path and module objects are rejected.
- importer tests assert the result contains real FunctionSteps and its source
  equals `FunctionStepTransportAuthority.source_from_pipeline(...)` exactly.

Change tests that currently require hidden invocation contracts, generated
sidecars, symbol-table lookup, module-number matching, repeated source bindings,
or CP invocation options so they fail on those behaviors.

**Required deletions**

None. This phase only establishes red tests.

**Forbidden additions**

- fixtures that inject hidden contracts;
- direct orchestrator acceptance tests presented as ZMQ parity;
- `.cppipe` state retained outside public source solely to make tests pass.

**Exit criteria**

- New tests fail for the documented current boundary violations.
- Existing unrelated native OpenHCS tests remain green.

### Phase 1: Atomic Runtime-Unification Cutover

Phase 1 is one merge boundary with four implementation workstreams. The
workstreams are not phases, are not independently releasable, and do not
authorize an intermediate compatibility layer. Their changes are developed
against one branch and all required deletions across 1A through 1D occur before
the Phase 1 exit gate. This atomic closure is required because declaration
migration removes inputs used by the current generator, exact compilation
depends on the new runtime-adapter reconstruction, direct import depends on the
new declarations/provider, and transport collapse depends on direct public
import output.

**Live implementation status (2026-07-10)**

- [x] Phase 0 boundary tests are present.
- [x] Workstream 1A source-binding setup declarations and processing-config
  lowering use nominal module owners.
- [x] Workstream 1A module-function ownership and raw-callable selection use
  the `CellProfilerModule` registry.
- [x] Workstream 1A display declarations and implementations are co-located in
  `processing/backends/cellprofiler/display_modules.py`; the five superseded
  interop display implementation files and all imports of them are deleted.
  Every display declaration resolves a raw callable whose `__module__` equals
  the declaration module, and CP `special_outputs` retain slot-name-only ABI
  declarations. Focused evidence: `77 passed`; the AST deletion, ownership,
  and slot-only decorator gates pass.
- [x] Workstream 1A compile-time capability products,
  `artifact_semantics.py`, `module_artifact_inputs.py`, and declaration
  `compile_time_*` hooks are removed.
- [x] Workstream 1A artifact-role bridge collapse is complete:
  `CellProfilerModule` terminates cooperative artifact declaration methods,
  existing image, object, measurement, relationship, and spatial-grid MRO
  owners contribute directly through `super()`, and the superseded artifact
  role root, MRO role scanner, and setting-name projection are deleted. The
  old symbol-table test path is replaced by declaration-owned artifact tests.
  Fresh-process registry discovery succeeds, and focused evidence is
  `73 passed` across contract, topology, cardinality, compiler reconstruction,
  module-role, callable-ownership, and public-API suites.
- [x] Workstream 1A runtime module-name policy registries and the external
  `CellProfilerPerObjectMeasurementPolicy` predicate were migrated to
  classmethod MRO owners; focused nominal-policy tests pass.
- [x] Workstream 1A supported source setup and exporter declarations, generic
  execution scope/artifact batch/file materialization values, and direct byte
  persistence are implemented with focused tests.
- [x] Workstream 1A `ExportToSpreadsheet` ownership slice is complete: the real
  renderer, raw callable, and nominal declaration are co-located in
  `processing/backends/cellprofiler/spreadsheet_export.py`; the superseded
  interop implementation and every direct import of it are deleted. The
  callable retains generic `RuntimeArtifactBatch` input and `FileBundleOptions`
  materialization with no processing contract. Focused evidence: `7 passed`;
  `py_compile`, exact-definition AST, stale-import `rg`, and diff checks pass.
- [x] Workstream 1A settings-source wrapper deletion is complete:
  `ModuleSettingsSourceModule` and `BinderSettingsSourceModule` are deleted;
  compound setting interpretation now lives on the nominal module leaf, while
  ordinary rows use `CellProfilerModule.bind_settings` and declared
  `SettingToKeywordBinding` values directly. The official30 public-import gate
  remains 30/30 after the cutover.
  - **Status, parser-metadata mirror deletion (2026-07-12): complete.**
    `SettingsBinder.SKIP_SETTINGS` is deleted; bind operations require the
    nominal `ModuleBlock` and consume only `ModuleBlock.settings`, while parser
    header metadata remains in `ModuleBlock.metadata`. Declaration coverage
    recognizes only bound rows and declaration-owned ignored rows. Focused
    binder/parser/compatibility/export evidence: `109 passed`.
- [x] Workstream 1A benchmark compatibility-facade deletion slice is complete:
  `benchmark/cellprofiler_library` and `benchmark/cellprofiler_compat` are
  deleted, genuine behavior tests import authoritative backend modules or query
  `CellProfilerModule`, and catalog/facade-only tests are removed. Focused
  evidence: `264 passed` across migrated direct-backend behavior tests,
  `73 passed` across deletion/public-API boundary tests, `7 passed` across nominal
  runtime lookup cases, and the AST import/symbol deletion gates pass.
- [x] Workstream 1A CP debug-view registry deletion slice is complete:
  `CellProfilerDebugView`, `DefaultCellProfilerDebugView`, and
  `openhcs/interop/cellprofiler/debug_views.py` are deleted; the inspector calls
  `DebugViewModel.from_debug_snapshot` directly, and retained generic behavior
  lives in `test_debug_views.py`. Focused evidence: `2 passed` core view tests,
  `4 passed` inspector tests, and zero forbidden symbols/imports in the AST gate.
- [x] Workstream 1A is complete: the old generator, symbol-table,
  runtime-policy, callable-catalog, settings-source wrappers, source-schema
  consumers, compatibility facades, and debug-view registry are deleted. The
  canonical AST/filesystem deletion gate passes.
- [x] Workstream 1B exact provider map, unique-claim provider ABI, generic
  source-group helper, and contract-owned source-binding alignment are
  implemented with focused tests.
- [x] Workstream 1B generic `CallableContract` processing/module-contract
  requirements and runtime-adapter callable factory boundary are implemented;
  generic function-reference rehydration is deleted.
  - **Status, runtime-callable metadata mirror deletion (2026-07-12):
    complete.** `CellProfilerRuntimeCallable` retains only its executor and
    call operation; callable identity, signature, annotations, memory types,
    processing contract, and exclusions remain owned by the canonical raw
    callable and compiled `CallableContract`. Generic invocation preparation
    targets that raw callable directly. Non-adapter inputs and outputs now
    occupy only their generic runtime/declared partitions, composed CP image
    inputs remain ordinary source inputs, and fixed return ordering follows the
    owning module MRO. The collapse removed 205 dead/redefined imports. Focused
    evidence: `50 passed` compiler/callable tests, `56 passed` special-I/O and
    SaveImages tests, and `424 passed` module-execution tests; the touched slice
    is Ruff-clean.
  - **Status, adapter-free SaveImages materialization boundary (2026-07-12):
    complete.** `MaterializationSourceIdentityRelation` declares the exact
    source artifact whose identity names an image export independently from the
    selected image's `GroupLineageSourceRelation`. The former remains a
    `SourceArtifactInputPartition`, the latter is the sole raw-callable
    `RuntimeArtifactInputPartition`, and `save_images` receives only its one
    declared `image_to_save` special input. Generic image contextualization
    projects the declared filename identity before `ImageFileOptions` writes
    variable-component planes; no CP writer branch or hidden callable metadata
    exists. The public compiler derives the selected image from current runtime
    image flow, so generated source retains only a genuinely non-default
    filename-source override. Focused evidence: `206 passed` compiler/import
    tests and `150 passed` planner/runtime/materialization tests. ExampleFly ZMQ
    evidence is parity `1.0`, `3.72x` execution speedup, three exact native
    filenames, and byte-identical decoded TIFF arrays.
- [x] Workstream 1B runtime adapter state now has the exact three-field plan,
  one-field executor, and four-field request; repository AST gates report zero
  old plan-projection consumers.
  - **Status, runtime request/context mirror deletion (2026-07-12): complete.**
    `CellProfilerRuntimeAdapter`, `RuntimeAdapterRequest`,
    `FunctionRuntimeScope`, and `PatternGroupData` compose their authoritative
    request/source context instead of inheriting or copying its fields. Exact
    scoped artifact reads are centralized, component selection uses nominal
    `SourceAxisMetadataScope`, and the hardcoded path-component regex is
    deleted. Focused adapter/source/static evidence: `244 passed`.
- [x] Workstream 1B runtime artifact input requests use spec/contract
  composition with no binding-scope carrier, and returned output matching uses
  exact declared order and identity with no callable or semantic fallback.
  - **Status, output resolution/main-flow partition slice (2026-07-12):
    complete.** Returned values are resolved once; recorded partitions consume
    adapter-enriched artifacts and declared-only image outputs consume the same
    resolved value directly. Canonical single image outputs are unwrapped,
    multi-image outputs retain exact slice contexts, and resolved object outputs
    remain load-bearing for measurement materialization. Full module-execution
    evidence: `424 passed`.
- [x] Workstream 1B compiled invocations carry exact input/output
  `ArtifactSpecRef` tuples, runtime request groups resolve active source,
  runtime-input, declared-output, and recorded-output collections through exact
  contract joins, and string artifact-key selection is deleted.
- [x] Workstream 1B artifact identity mirroring is deleted. The module contract
  canonicalizes ordered inputs before declaring outputs; output and measurement
  relation hooks receive those exact `ArtifactSpec` values instead of rereading
  settings and reconstructing `(plan role, artifact type, name)`. Source
  bindings own their `ArtifactSpec` and source-input-plan projections, and
  `ArtifactSpecRef` construction is statically restricted to `ArtifactSpec.ref`
  and `ArtifactPlan.ref`. Focused artifact/import/deletion evidence is `123
  passed`.
- [x] Workstream 1B `InvocationArtifactDeclarations` is deleted. Artifact
  declaration providers return the existing `ArtifactPlanKeySelector` owner
  directly: `ModuleArtifactContract` for a compiled module contract and
  `CallableContract` otherwise. Generic graph, function-pattern, and prepass
  consumers query that nominal owner without a replacement payload or backend
  branch.
- [x] Workstream 1B processing-contract and runtime-shape authorities,
  callable-output projection, binding-scope carrier, and runtime-record
  deduplication wrapper are deleted; their existing nominal owners now carry
  the behavior and focused deletion gates pass.
- [x] Workstream 1B stale-test migration removes the deleted
  `CellProfilerProcessingContractAuthority`, `RuntimeArtifactBindingScope`,
  and `CallableInvocationKwargSpec` from the focused formatter, callable
  introspection, and module-execution tests; those tests now use compiled
  `CallableContract` requirements, exact contracts/specs, and public-kwarg
  validation.
  - **Status, runtime-adapter executor API stale-test slice only
    (2026-07-10): complete.** The scoped tests now build
    `CellProfilerModuleRuntimePlan` from the exact raw callable and compiled
    `CallableContract`, separate request input/output plans, and call
    `CellProfilerModuleExecutor.run` with only the image payload. AST evidence
    finds zero bare-contract executor constructions and zero deleted
    `run(func, image, ...)` calls. Focused evidence: `15 passed, 6 failed`;
    the retained failures expose production behavior (`5` missing
    `SpecialInputBindingRequest.module_name`, `1` undiscoverable recorded
    CalculateMath table). Full scoped-file evidence: `174 passed, 10 failed`,
    including `4` unrelated source-loader/metadata failures.
- [x] Workstream 1B stale source editor/preview test migration removes the
  deleted `PipelineImageSchema`, `ImageAssignment`, schema providers, and
  schema-assignment constructors from the two focused test files. The retained
  tests construct `SourceBindingsConfig`, `NamedSourceBinding`, and resolved
  `StepSourceBindingsConfig` values and call only the config-based inventory,
  view, preview, and editor APIs; obsolete schema-provider cases are deleted.
  Scoped AST and stale-import scans pass. Focused core evidence:
  `5 passed, 4 failed`; all four failures expose the current production
  `SourceBindingsPreview.image_set_rows` field/method collision. The focused
  editor cases are statically valid, while collection is blocked by the local
  environment's missing `magicgui` dependency.
- [x] Workstream 1B stale symbol-table test migration retains only
  declaration-owned ordered artifact/partition flow, cross-step lineage,
  fail-loud artifact-input validation, and pure-import-to-compiler
  reconstruction through ordinary `FunctionStep` and `PipelineConfig` values.
  The retained tests live in `test_cellprofiler_artifact_declarations.py`; the
  old symbol-table test path is absent, and the focused file imports no symbol
  table, generator, role mirror, or compatibility helper. Focused evidence:
  `3 passed`.
- [x] Workstream 1B is complete: symbol-table and generated-matcher lookup are
  deleted, exact session-scoped invocation contracts compile from public
  declarations, and generic exact artifact/runtime matching is the only active
  path.
- [x] Workstream 1C agent architecture-projection slice exposes the
  `CellProfilerModule` nominal registry, pure
  `import_cellprofiler_pipeline` boundary, and ordinary `FunctionStep` plus
  `PipelineConfig` output. It no longer projects or describes the superseded
  symbol table, `PipelineGenerator`, runtime-preparation boundary, or callable
  catalog. Focused evidence: `5 passed`; `py_compile` and `git diff --check`
  pass.
- [x] Workstream 1C public import-boundary test slice now requires the exact
  pure `(list[FunctionStep], PipelineConfig)` signature, caller-owned generic
  source rendering and reconstruction, and AST/filesystem absence of the
  deleted import carriers, services, compilers, role layer, symbol table, and
  generator/runtime-preparation layer. Focused evidence: `43 passed`; scoped
  `py_compile`, stale-import/name scans, and `git diff --check` pass.
- [x] Workstream 1C integration-test migration slice now imports public
  `(list[FunctionStep], PipelineConfig)` declarations, materializes inputs only
  through generic source-binding machinery, submits compile then execute through
  `ZMQExecutionClient`, and validates the server observation export. The scoped
  file contains no generated/prepared pipeline or direct-orchestrator imports;
  its socket-free boundary/type checks pass (`2 passed`) and all `28` tests
  collect. Live ZMQ completion remains gated by local IPC permission, while the
  retained `LoadData` case exposes the independent missing-declaration failure.
- [x] Workstreams 1C and 1D are complete: the pure importer returns only public
  `list[FunctionStep]` plus `PipelineConfig`, generic transport is the sole
  source renderer, generated state is sparse and typed, and all 30 official
  pipelines import successfully. Current combined evidence is `70 passed` for
  focused import/transport/pycodify boundaries, `1170 passed` for the complete
  CellProfiler/`.cppipe` unit gate, `3 passed` for static deletions, and
  official30 public import `30/30`.
- [x] Phase 2 compile-time failure gates are complete: invalid callable/module
  ownership, exact-key duplication, artifact availability/type/identity,
  special-input/output ABI, canonical output shape, required stack axes,
  grouped module/processing semantics, compile-only/public kwarg ownership,
  artifact relations, and static source cardinality fail at their nominal
  compiler owners. Redundant object/special-input cardinality and measurement
  sink checks are removed from runtime binders. The canonical ZMQ negative
  submits public source and fails during compilation without creating an
  execution observation. Current evidence is `1162 passed` for the complete CP
  unit corpus, `96 passed` for CPPipe/compiler/ZMQ failure gates, `453 passed`
  for the focused nominal runtime-policy slice, and official30 public import
  `30/30`.
- [x] Phase 1B current-image carrier cleanup is complete: the two one-field
  context dataclasses and their module are deleted, concrete request records own
  their typed fields directly, and optional source-bound object resolution
  validates the value only where it is consumed. Focused runtime,
  source-binding, and static evidence: `471 passed`.
- [x] Phase 1B post-cutover shell audit is complete for the current runtime and
  backend surface: AST/call-site analysis removed redundant current-image,
  source-identity, optional-selection, projection-request, parser, profiler,
  cache-entry, revision, and scalar-predicate carriers while retaining
  behavior-bearing views and nominal strategy leaves. The complete
  CellProfiler/CPPipe corpus passes after the collapse: `1232 passed`, with
  only the four pre-existing third-party/parser warnings.
  - **Status, mapping lookup wrapper deletion (2026-07-12): complete.**
    `MappingValueLookup` and its module are deleted. Ordinary mapping defaults
    now remain explicit at the calculation that owns each default instead of
    passing through a CP-specific two-field wrapper.
  - **Status, resolved runtime-input request shell deletion (2026-07-12):
    complete.** `ResolvedRuntimeInputRequest` is deleted; the two concrete
    runtime request records own their `image_count` field directly while the
    shared image execution context retains only genuinely shared provenance
    and execution-mode semantics.
  - **Status, relationship endpoint fallback deletion (2026-07-12): complete.**
    Relationship outputs carry exact nominal parent and child artifact
    relations. Runtime name parsing, two-input fallback, endpoint-match
    carriers, and module-index endpoint reconstruction are deleted; malformed
    relationship topology fails when `ModuleArtifactContract` is constructed.
  - **Status, dead measurement-table axis cache key deletion (2026-07-12):
    complete.** `MeasurementTableAxisProjectionCacheKey` is deleted after AST
    and call-site analysis found no constructor, match, or load; active table
    cache keys retain their source, group, and object semantics.
  - **Status, source-candidate request shell deletion (2026-07-12): complete.**
    `SourceAliasOrderIndexRequest` is collapsed into the existing shared match
    request plus the one candidate argument consumed by order matching, and
    `SourceCandidateMetadataRequest` is collapsed into its sole concrete
    resolver. Neither field-only base retained an invariant or dispatch role.
  - **Status, Align and illumination backend leaf wrapper deletion
    (2026-07-12): complete.** Naming, output-plane collapse, input payload
    validation, additional-mode normalization, crop geometry, and output
    alignment now live on their existing module, execution, strategy, and
    output-request owners. Forwarding request/result shells and single-consumer
    helpers are deleted while retained strategy invariants have direct behavior
    tests.
  - **Status, execution-validation result shell deletion (2026-07-12):
    complete.** `validate_cppipe_runtime_observation` returns the existing
    `RuntimeArtifactExecutionObservation` directly; the two-field
    `CPPipeExecutionValidation` copy is deleted and benchmark consumers retain
    the submitted expectation on the existing ZMQ export owner.
  - **Status, unused intensity result declaration deletion (2026-07-12):
    complete.** `ObjectIntensityResults` is deleted after repository AST
    analysis found no construction or consumption; the existing
    `ObjectIntensityMeasurementRows` remains the runtime result owner.
  - **Status, measurement row wrapper deletion (2026-07-12): complete.**
    Projected rows use standard immutable mappings, projected columnar rows use
    the existing core columnar owners, and concatenation uses
    `ConcatenatedColumnarRows` directly. Descriptor, mapping, columns, and empty
    subtype wrappers are deleted; sparse-cell and declared-domain invariants
    remain on the existing projection/materialization owners.
  - **Status, projection and source-candidate carrier deletion (2026-07-12):
    complete.** Runtime plane projection now retains only the selector,
    projection, stack, source-context, and image/object-label projection
    authorities that enforce independent invariants. Fourteen field-shuttle
    request/result records are deleted. Source-candidate collection,
    image-number shadow resolution, request forwarding, and unused metadata
    helpers are deleted; the existing candidate cache, selector inheritance,
    match-plan context, and registered plane strategies remain authoritative.
  - **Status, CalculateMath field-shuttle deletion (2026-07-12): complete.**
    Indexed operand binding lives on `CalculateMathModule`, repeated settings
    bind through existing `SettingToKeywordBinding`, immutable request changes
    use `dataclasses.replace`, and transform/bounds records own their behavior.
    The three one-use setting carriers are deleted.
  - **Status, duplicate inherited declarations and zero-caller forwards
    (2026-07-12): complete.** CellProfiler measurement markers inherit their
    family qualifier from the generic semantic marker owner, object vector
    bindings inherit the existing feature query, and the unused runtime-scope
    input-plan and source-pair forwarding methods are deleted.
  - **Status, parallel SaveImages interop implementation deletion
    (2026-07-12): complete.** The unused interop implementation and its
    self-referential test are deleted. `SaveImagesModule`, the canonical
    backend callable, generic materialization options, and the existing
    executable SaveImages tests remain the sole public path.
  - **Status, FlagImage forwarding layer deletion (2026-07-12): complete.**
    Flag enums, result construction, and both decorated callables live on the
    backend module beside `FlagImageModule`; the duplicate interop module is
    deleted and no compatibility forwarding surface remains.
  - **Status, imported relationship provenance validation (2026-07-12):
    complete.** Contracts validate relation sources introduced by their own
    outputs. Consumed relationship inputs retain exact upstream endpoint
    provenance without falsely requiring exporters to consume the endpoint
    label payloads; every relationship output still requires exact parent and
    child relations at contract construction.
  - **Status, duplicate-input side predicate deletion (2026-07-12): complete.**
    Cooperative artifact input declarations preserve ordered occurrences
    directly. The root preservation predicate, identical module overrides, and
    deduplication branch are deleted; repeated-role behavior is covered by the
    CorrectIlluminationApply contract itself.
  - **Status, object-domain coverage override deletion (2026-07-12):
    complete.** Shape, intensity, colocalization, and granularity row types
    inherit complete declared-domain coverage from
    `ObjectMeasurementColumnarRows`; their identical literal overrides are
    deleted while custom columns and iteration remain on each leaf.
  - **Status, nominal object-label output cutover (2026-07-13): complete.**
    Object-label-producing callables return `ObjectLabelValue`; the compiler
    rejects raw-array object-label return slots before worker execution. The
    generic raw-output fallback, CellProfiler NumPy and opaque output-context
    strategies, raw adapter entry point, and tests expecting pixel-derived
    output domains are deleted.
- [ ] Phase 3 official30 ZMQ parity, export, UI, and Napari gates are pending.

**Phase prerequisites**

- Phase 0 boundary tests are present and red only for the documented current
  violations.
- The current native OpenHCS compiler, source-binding, ZMQ, and materialization
  tests are green before the cutover branch starts.
- The complete current CellProfiler module registry and official30 source
  corpus are available for declaration and setting-coverage migration.

#### Workstream 1A: Make Module Declarations The Sole CP Semantic Owner

**Prerequisites**

- Phase 0 boundary tests are present and red.

**Changes**

- Update `openhcs/interop/cellprofiler/module_declarations.py` so
  `CellProfilerModule` and existing MRO mixins directly produce ordered
  `ModuleArtifactContractItem` values from existing artifact types and
  partitions.
- Add function-name ownership, callable loading, and duplicate validation to
  `CellProfilerModule`; update backend-package, compiler, import, and runtime
  lookup to call that nominal root.
- Add `SourceSetupCellProfilerModule` and move `Images`, `LoadImages`,
  `Metadata`, `NamesAndTypes`, and `Groups` lowering bodies from
  `source_schema.py` to declarations in `infrastructure.py`. Extend the
  existing `SourceBindingsConfig` and `NamedSourceBinding` with the exact
  generic fields specified above, move the three behavior-bearing source types
  to `source_bindings.py`, and delete `pipeline_image_schema.py` after all
  consumers migrate.
- Move alias lookup and derived loaded-image/measurement-name queries onto
  `SourceBindingDeclarationsMixin`, shared by editable, resolved, and compiled
  plans. Remove the duplicate `CompiledSourceBindingPlan.binding_for_alias`
  implementation and every consumer-built alias map or source-name set.
- Add the resolved generic source-channel-axis field and methods to
  `ImagePayloadMetadata`. Replace role-based source loading with the direct
  binding payload function, remove `OpenHCSImageType` from projections and
  provenance, and migrate alignment, pure-2D aggregation, source candidates,
  source-bound runtime loading, the CP adapter, and color-to-gray metadata
  handling to the binding/metadata owners.
- Replace generated CP source-image role classes and source-literal resolver
  leaves with the four exact CP-local enums specified above. Replace
  NamesAndTypes and LoadImages layout probing with declaration-owned indexed
  setting-column parsing and strict cardinality validation.
- Delete `InfrastructureCellProfilerModule`. Move `SaveImages`,
  `ExportToSpreadsheet`, and `ExportToDatabase` onto executable module roots,
  implement their contracts over existing image, measurement, and relationship
  artifact specs, and emit them as public steps. SaveImages remains axis-scoped;
  the aggregate exporters declare the generic plate execution scope and return
  generic file bundles. Remove unsupported `LoadData`, `LabelImages`,
  `CreateBatchFiles`, and `SaveCroppedObjects` declarations and pass-through
  callables so enabled uses fail import instead of disappearing.
- Add the generic `FunctionStepExecutionScope`, `RuntimeArtifactBatch`,
  `ImageFileOptions`, and `FileBundleOptions` APIs exactly as specified under
  Module Declaration Contract. Carry execution scope through callable metadata
  and compiled invocation contracts; derive it from
  `CompiledFunctionPattern` for worker filtering and post-plate execution. Do
  not add a plan field, exporter registry, or config.
- Extend the existing `StepInputDependencyKind` with `NO_MAIN_FLOW` and use it
  for every plate plan in `path_planner.py`. Make component-scope and directory
  planning handle that state explicitly; do not encode artifact-only execution
  as `PIPELINE_START` or `STEP_OUTPUT`.
- Complete the existing generic byte materialization boundary in
  `external/PolyStore/src/polystore/disk.py`: `DiskStorageBackend.save` writes
  `bytes` directly before extension dispatch. Keep every non-byte payload on
  the current nominal `FileFormat` registry path; do not add exporter suffixes
  or a raw-file extension registry.
- Replace both fake aggregate-export callables. Spreadsheet export renders exact
  contract-selected table records. Database export uses the existing CPA
  projection/dialect/property code plus a real SQLite renderer and returns a
  file bundle. Remove the execution-context/request carriers and direct store
  scans from `analyst_export.py`.
- Remove `CellProfilerModuleRole` from import provenance. Change
  `CellProfilerModuleReference` to retain only the raw source facts
  `name: str`, `module_num: int`, and `enabled: bool`; delete
  `processing_modules`, `infrastructure_modules`, `disabled_modules`, and
  `modules_with_role`. During the import pass, skip a disabled block, require
  the declaration for every enabled block, and call its
  `emits_function_step()` method. No derived import role is stored.
- Add `contribute_source_bindings`, `emits_function_step`, and the exact
  parsed-module `resolve_function(module, contract, source_bindings)` method to
  the existing module MRO. Merge settings-only selectors and the two
  source-axis-aware selectors into that one method, return the canonical raw
  callable through `require_callable`, and delete `resolve_semantic_function`,
  `ResolvedModuleFunction`, and every default-function-name carrier.
- Replace `compile_image_schema` with one fold from `SourceBindingsConfig()`
  that resolves the registered module type and calls
  `contribute_source_bindings`; move the NamesAndTypes 3D stack rule onto its
  declaration. Move disabled Metadata row preservation into the Metadata
  declaration's `contribute_source_bindings` implementation;
  it reads `CellProfilerMetadataPatternBlock` directly and owns the channel
  exclusion rule without a policy carrier or component-set wrapper.
- Consolidate `source_schema_workspace.py` into the existing
  `source_binding_workspace.py` and migrate every workspace, inventory, UI,
  compiler, and runtime call to accept resolved `SourceBindingsConfig`. Delete
  `source_schema_workspace.py`; never create a second file or facade.
  Remove `source_schema` from generated/import results and
  `SourceBindingContext`; workspace preparation reads the config already
  present on `PipelineConfig`.
- Remove private source-image-set selection from every OpenHCS and benchmark
  request. Materialize the complete declared workspace and delete source caps
  from manifests, adapter requests, and CLI arguments. The official-suite
  harness supplies `GlobalPipelineConfig.well_filter_config` directly and
  submits that public inherited config through the canonical ZMQ path.
- Replace format-specific source-plane inventory with config-declared source
  stack axes, generic VFS loading, and `SourcePixelRef.source_axis_indices`.
  Remove source-root/provider probes, image-plane URI priority resolution, and
  one-leaf auxiliary materialization registries; source refs retain their real
  backend.
- Migrate every microscope and source-binding workspace writer to structured
  `SourcePixelRef.to_workspace_mapping` values, require validated structured
  values at every reader, move the type to PolyStore, and make
  `VirtualWorkspaceBackend` dispatch through the execution-local FileManager
  registry. Add backend-polymorphic address resolution, make disk own
  plate-root-relative addresses, migrate direct registry writes to
  `FileManager.register_backend`, and remove the global registry replacement
  from `FileManager.__setstate__`.
  Delete string mappings, the old serializer name, the complete source-ref
  resolver family, resolved-ref wrapper, shape predicates, and priority/batch
  traversal in this same phase.
- Collapse source candidate/request/projection one-operation classes into
  `SourceBindingWorkspaceProjector`; keep the enum-keyed match assembler and
  component strategies as the only workspace behavior families.
- Rewrite source view/inventory/preview APIs to accept one pipeline
  `SourceBindingsConfig` plus one resolved step override, using
  `NamedSourceBinding` for every row. Remove schema-assignment view methods and
  payload-type strings. Remove the unused inventory provider registry and make
  `SourceBindingContext` call `SourceInventory.from_filemanager` with its actual
  filemanager/backend/config.
- Rewrite the backend package's lazy callable lookup to delegate directly to
  the owning module declaration. Validate the implementation's source-declared
  `CallableContract` and leave the function namespace unchanged. Build the
  enriched immutable contract only in
  `CellProfilerInvocationContractProvider`.
- Remove semantic DTO/family fields from the compatibility matrix and read
  declaration fields directly so deleting `module_semantics.py` leaves no
  intermediate report dependency.
- Remove processing-contract resolution DTO/provenance fields from the same
  report and delete `CellProfilerModule.contract` plus every leaf assignment.
  Read the source-declared processing contract exclusively through
  `CallableContract.from_callable(module_type.require_callable(...))`. Compare
  compiler-derived artifact and adapter facts against the provider's compiled
  contract, never against attributes attached to the raw function.
- Move defaults currently represented by module-name policies onto
  `CellProfilerModule` virtual methods.
- Remove identity-only parameters from raw callable signatures. Make their
  existing module-owned `SettingToKeywordBinding` values reference the actual
  setting ClassVars, derive keyword names where identical, and serve both
  parser-to-source conversion and sparse compiler reconstruction. They are
  consumed before generic signature validation and never enter runtime.
- Move leaf differences onto the existing module leaf class or its existing
  nominal mixin in `openhcs/processing/backends/cellprofiler/*.py`.
- Convert the nine module-policy mixin families listed under Module Declaration
  Contract to classmethod MRO authorities and make
  `CellProfilerModuleExecutor` call the resolved module type directly.
- Use `special_input_names_from_callable` and
  `special_output_specs_from_callable` only to bind and validate Python ABI
  slots. Read artifact kind, name, partition, relation, and materialization from
  the module contract.
- Add `output_name: str = "Measurement"` to the public `calculate_math`
  signature and use that same kwarg when deriving the output measurement
  `ArtifactSpec`. Derive operand object names from input artifact specs and
  supply them through the module executor rather than public kwargs.
- Add `cycle_scope: DefineGridCycleScope` to the public
  `define_grid_manual` and `define_grid_automatic` signatures and setting-row
  mappings. The compiler reads it when planning cycle execution. The raw
  functions accept the typed kwarg for direct-call and FunctionStep signature
  consistency while their plane calculation remains unchanged.
- Delete the now-unused `RuntimeInvocationOptions` facility. Restrict every
  FunctionStep pattern leaf to a callable or `(callable, kwargs)`; a tuple of
  any other arity fails validation. Remove invocation-option fields and hidden
  parameters from module binding, function-pattern normalization, callable
  contracts, runtime execution, ObjectState extraction, and agent authoring.
  All behavior uses typed public callable kwargs, and compiler-only identity is
  consumed from sparse ordinary kwargs before runtime.
- Delete `module_processing_components.py`, `module_processing_config.py`, and
  every source-axis summary or processing-scope policy. Make module declarations
  return concrete `ProcessingConfig` from the exact artifact-only
  `ModuleArtifactContract`, the already-selected callable's `CallableContract`,
  and the inherited concrete `ProcessingConfig`. Import obtains the inherited
  value under `objectstate.config_context(pipeline_config)`; compilation reads
  the resolved `FunctionStep` stored in `StepSnapshot.step`. Enforce the
  artifact contract -> raw callable -> callable contract -> inherited config ->
  processing config order during import. Public compilation starts with the
  declared raw callable and never invokes the import selector. No artifact
  contract or module class mirrors callable axis constraints.

**Required deletions**

- `openhcs/interop/cellprofiler/runtime/policy_registry.py`;
- `ModuleSettingsSourceModule` and `BinderSettingsSourceModule`; module leaves
  use `CellProfilerModule.bind_settings` and its existing
  `postprocess_bound_settings` hook directly;
- `openhcs/core/runtime_invocation.py`, `RuntimeInvocationOptions`, the
  FunctionStep third-tuple union, every invocation-options field/parameter, and
  its function-pattern, callable-contract, runtime, UI, and agent branches;
- `ArtifactDeclarationStepContext.source_provenance` and its propagation;
- `openhcs/core/pipeline_image_schema.py`, `PipelineImageSchema`,
  `PipelineImageSchemaBuilder`, `ImagesRule`, `ImageAssignment`,
  `SourceArtifactAssignment`, `SourceImageStackPlan`, `GroupingPlan`, the
  source-binding representability/projection feature registry, dynamic
  image-type role class factories, and every generic/runtime
  `ImageTypeSourceRole` symbol;
- `SourceImagePayloadSemantics`, `SourceImagePayloadRoleStrategy`, all strategy
  leaves, and `source_image_payload_role`;
- `SOURCE_IMAGE_TYPE_METADATA_FIELD`, `OpenHCSImageType`,
  `SourcePlaneProjection.image_type`, and the image-type branch in
  `source_matching.py`;
- `SourceSchemaLiteralResolver`, its subject/operator/match-method subclasses,
  `NamesAndTypesAssignmentLayout`, `NAMES_AND_TYPES_ASSIGNMENT_LAYOUTS`, and
  `NamesAndTypesAssignmentBlockStrategy`;
- `openhcs/core/source_schema_workspace.py` and every `SourceSchema*` public
  symbol after migration to the renamed source-binding workspace API;
- `tests/unit/test_cellprofiler_source_schema.py` after retained setup and axis
  cases move to `test_cellprofiler_source_bindings.py`;
- `tests/unit/test_cellprofiler_source_schema_ingestion.py` after retained
  source-root/config-filter cases move to
  `test_cellprofiler_source_bindings_ingestion.py`;
- `_AUXILIARY_PAYLOAD_CACHE`, `cache_source_schema_auxiliary_payload`,
  `source_schema_auxiliary_payload`, its clear/key helpers, and the runtime
  disk-specific `np.load` path;
- `SourceSchemaImageSetSelection`, `max_image_set_count`,
  `source_schema_image_set_selection`, and all request/adapter/CLI/test fields;
- `SourceSchemaCandidateProvider`, `SourceSchemaCandidateDiscoveryMode`,
  discovery request/probe/viability/result wrappers, and both provider leaves;
- `ImagePlaneSourceResolutionStage`, `ImagePlaneSourceResolver`, and all three
  resolver leaves;
- `SourceSchemaSourcePlaneInventory`, `TiffPageSourcePlaneInventory`,
  `TiffSourcePlaneInventory`, and `SinglePlaneSourcePlaneInventory`;
- `SourcePixelRef.to_legacy_workspace_mapping`, string workspace mapping
  values, its old reader/path/series/plane/C/Z/T fields,
  `workspace_mapping_source_ref`, `workspace_mapping_source_path`,
  `VirtualWorkspaceSourceRefResolver`, `PathSourceRefResolver`,
  `DiskSourceRefResolver`, `VirtualWorkspaceResolvedRef`, and resolver
  selection/batching after `SourcePixelRef` moves to PolyStore and
  `VirtualWorkspaceBackend` uses the execution-local FileManager registry
  directly;
- `BioFormatsStorageBackend` workspace-mapping cache/resolution/listing path,
  its reader switch and `_load_npy_plane`,
  `BioFormatsReaderUnavailableError`, and the BioFormats handler's
  `get_primary_backend` override after that handler registers the shared
  virtual-workspace backend;
- `SourceSchemaAuxiliaryMaterializer`, `NumpyAuxiliaryMaterializer`, auxiliary
  materialization/target request classes, target-path policy, and basename
  policy leaf;
- `SourceSchemaCandidateIdentity`, `SourceSchemaCandidateCollection`,
  `SourceSchemaCandidateMetadataRequest`,
  `SourceSchemaCandidateMetadataResolver`, `SourceSchemaCandidateMatches`,
  `SourceSchemaVirtualFilename`, `SourceSchemaFilenameProjection`,
  `WorkspaceMappingSink`, `SourceVirtualPathMetadata`,
  `VirtualComponentOriginalMetadataProjection`, `SourceMetadataJsonRecord`,
  `SourcePlaneGroupSiteAllocator`, `ImageSetMetadataMerge`, and
  `SharedCandidateMetadataProjection`, plus
  `SourceBindingCandidateSourceRef` after `SourceCandidate.source_ref` becomes
  authoritative;
- `ComponentProjection`, `WellRowColumnMetadataProjection`,
  `SourceSchemaSingletonWellProjection`, `ImageNumberSiteProjection`,
  `FrameNumberSingletonSiteProjection`, `ZIndexSingletonSiteProjection`,
  `MetadataZIndexProjection`, `FrameNumberTimepointProjection`,
  `SourceSchemaSingletonTimepointProjection`,
  `SourceSchemaSingletonZIndexProjection`, and `OrdinalSiteProjection` after
  their behavior moves to enum-keyed component strategies;
- `StepSourceBindingsConfig.inherits_*`, `resolved_against`,
  `can_inherit_from`, `resolve_step_source_bindings`,
  `resolve_effective_step_source_bindings`, per-field overlay/equivalence
  helpers, and untyped group-key scoping methods;
- global-registry replacement in `FileManager.__setstate__`, every direct
  `filemanager.registry[...]` mutation, and generic virtual-workspace address
  rewriting through `requires_filesystem_validation`;
- `SourceBindingView.from_schema_assignment`,
  `SourceBindingView.payload_type_for_assignment`, the `payload_type` view
  field, every `from_schema_and_bindings` constructor, and every
  `SourceInventoryProvider.inventory(schema=...)` signature;
- `SourceInventoryBuildRequest`, `SourceInventoryProvider`,
  `ExplicitImagePlaneSourceInventoryProvider`,
  `LocalDirectorySourceInventoryProvider`,
  `FileManagerSourceInventoryProvider`,
  `OpenHCSWorkspaceSourceInventoryProvider`, and
  `SchemaContextSourceInventoryProvider`; remove redundant
  `SourceInventory.from_directory` / `from_schema_sources` constructors after
  callers use `from_filemanager` / `from_paths`;
- `CellProfilerSourceSchemaProjection`, `CellProfilerSourceSchemaWorkspace`,
  `CellProfilerSourceSchemaPreparation`, `CellProfilerSourceSchemaMaterializer`,
  `CellProfilerSourceSchemaMaterializationScope`, and CP source-schema result
  wrappers;
- dynamic policy view classes in runtime policy modules;
- `CellProfilerArtifactCapability` and its subclasses;
- `CellProfilerCompileTimeSettingsRequest`;
- every `compile_time_*` method on `CellProfilerModule` and its MRO leaves,
  after its behavior is migrated by the closed mapping above;
- `cellprofiler_source_setting_parameter_mapping`, normalized-name signature
  inference, and every `setting_parameter_aliases` mapping after migration to
  owning `SettingToKeywordBinding` values;
- `compile_time_setting_binding_default_values`,
  `compile_time_settings_function`, and their callable/default caches after
  `module_blocks_for_invocation` reads the one canonical raw signature;
- `public_artifact_identity_overrides` overrides;
- `ModuleSettingCoverageStatus.ARTIFACT_CONTRACT`, `.TYPED_IGNORE`,
  `.CALLER_IGNORE`, and `.INFRASTRUCTURE`, plus caller-provided ignored-setting
  arguments;
- `openhcs/interop/cellprofiler/artifact_semantics.py`;
- `SpecialOutputKindClassifier` and its classifier subclasses after all
  semantic consumers use module contracts;
- every materialization-bearing `special_outputs` tuple on a registered
  CellProfiler callable; CP decorators retain slot names only;
- CP `CalculateMathInvocationOptions` and `DefineGridInvocationOptions`;
- `openhcs/interop/cellprofiler/module_processing_components.py` and
  `openhcs/interop/cellprofiler/module_processing_config.py` in full;
- `GeneratedLiteralScalar`, `GeneratedLiteralValue`,
  `GeneratedStepSettingKey`, `GeneratedParameterName`,
  `GeneratedGroupByComponent`, `group_by_is_unresolved`,
  `variable_component_literal`, `all_component_literal`,
  `coerce_all_component`, `all_component_tuple_literal`, `group_by_literal`,
  `group_by_component_axis`, `source_identity_group_by_component`,
  `source_binding_variable_component_literals`, `variable_component_literals`,
  and `generated_function_step_semantic_argument_lines`;
- the source-axis summary and all of its constructors and projection methods;
- `SourceProcessingComponentSemantics`, `RuntimeArtifactProcessingScope`,
  `SourceBindingProcessingScope`, `default_module_processing_components`,
  `default_module_requires_pairwise_object_domain_scope`, and
  `_is_inputless_artifact_only_contract`, and
  `default_module_processing_config`;
- `tests/unit/test_cellprofiler_module_processing_components.py` after its
  retained cases move to `test_cellprofiler_module_processing_config.py`;
- `GeneratedGroupByComponentState`, `ModuleProcessingComponents`,
  `ModuleProcessingComponentRequest`, `GeneratedStepSettings`,
  `SourceProcessingAxisRole`,
  `SourceProcessingAxisRolePolicy`, and `ModuleProcessingScopePolicy` families;
- `CellProfilerModule.processing_components` and every override after their
  behavior moves to the exact `processing_config` method;
- `CellProfilerModule.with_generated_group_by` and `.generated_group_by` after
  `processing_config` applies the declaration facts through
  `FuncStepContractValidator`;
- `SetupModuleCompiler`, `SourceImageStackPlanDeclaration`, and their leaves;
- `InfrastructureCellProfilerModule`, its import-note/export flags,
  `CPPipeInfrastructureProfile`, and hidden SaveImages retained-artifact
  projection, including `materialize_skipped_save_images`;
- `RuntimeImageExportSpec`, `RuntimeImageExportBitDepth`,
  `CellProfilerModule.image_export_specs`, and SaveImages validation that
  synthesizes candidate images from runtime records;
- unsupported pass-through module declarations and callables for `LoadData`,
  `LabelImages`, `CreateBatchFiles`, and `SaveCroppedObjects`;
- `openhcs/interop/cellprofiler/database_export.py`, placeholder export metadata,
  and `pending_pipeline_export`;
- the unrelated region-statistics implementation in
  `openhcs/interop/cellprofiler/spreadsheet_export.py` after its real exporter
  moves to the backend declaration module;
- `CellProfilerExecutionExportContext` and
  `CellProfilerAnalystExportRequest`;
- the one-field `AnalysisConsolidationPlan` class after its existing behavior
  becomes direct orchestrator functions;
- one-leaf source-schema parser/origin policy registries;
- `CellProfilerSourceRootResolver`, `CellProfilerSourceRootCandidate`,
  `CellProfilerResolvedSourceRoot`, `CellProfilerSourcePathAdmission`,
  `CellProfilerSourcePathContext`, `CellProfilerSourcePathExclusion`, and both
  exclusion leaves; the request's exact source root and config filters are the
  sole source-universe authority;
- `openhcs/interop/cellprofiler/module_function_resolution.py` and
  `ResolvedModuleFunction`;
- `CellProfilerModule.resolve_semantic_function` and every override/caller;
- `CellProfilerModuleRole`, `CellProfilerModuleRoleSpec`,
  `cellprofiler_module_role`, and every role-derived provenance projection;
- `CellProfilerFunctionRuntimeMetadata`, `CELLPROFILER_MODULE_ATTR`, the public
  `CellProfilerFunctionCatalog`, function wrapper factory, and
  `processing/backends/cellprofiler/library.py`;
- every CellProfiler call/import of `attach_callable_contract_metadata` and
  every CP callable-metadata cache; the generic helper remains available to
  native decorator construction outside the CellProfiler import/compiler path;
- `CellProfilerFunctionReferenceTransportStrategy` and the resulting empty
  `FunctionReferenceTransportStrategy`; generic importable-callable references
  own transport after wrapper removal;
- the unused semantic-default-contract registry and CP debug-view registry;
- `openhcs/interop/cellprofiler/module_semantics.py`, its public exports, and
  semantic-family compatibility report records;
- `openhcs/interop/cellprofiler/processing_contract_resolution.py`,
  `ResolvedProcessingContract`, and its one-value
  `ProcessingContractResolutionSource`;
- `DisabledPathMetadataRulePolicy` and
  `DisabledMetadataAxisComponents` after their sole behavior moves to the
  Metadata module declaration;

**Forbidden additions**

- module-name keyed dictionaries or sets;
- string normalization used to infer artifact kind;
- `getattr` probes for module policy;
- `issubclass` dispatch chains over capability leaf classes;
- duplicated lists of modules supporting one behavior.
- callable wrappers whose only purpose is copying declaration metadata.
- CP-specific post-execution hooks, exporter registries, output-format switches,
  or config fields; callable scope, artifact batching, and writers remain
  generic.
- runtime-store scans not constrained by the exporter's compiled artifact input
  plans.

**Focused tests**

- module declaration tests assert exact ordered contracts for representative
  image, object, measurement, relationship, grid, source-bound, and multi-output
  modules;
- callable ABI tests assert special-input parameters exist in the raw callable
  signature, CP special-output declarations contain slot names only, and slot
  count/order matches the contract;
- registry tests iterate `CellProfilerModule.__registry__.values()` and compile
  every processing and setup declaration without consulting a parallel
  registry;
- source-binding config tests cover all five setup leaves, NamesAndTypes 3D
  stack contribution through the module MRO, lazy-config pycodification, and
  config-only workspace recreation;
- source workspace tests assert complete-universe materialization, inherited
  `WellFilterConfig` selection during compile/execute, identical UI and
  benchmark submissions, exact source backend retention, and absence of a
  private image-set cap;
- source editor/preview tests construct only `SourceBindingsConfig` and resolved
  step overrides, and assert preview candidates, image sets, and compiled
  workspace mappings agree exactly;
- format-neutral source-axis tests run equivalent TIFF and NPY stacks through
  the same `FileManager`/`SourcePixelRef.source_axis_indices` path, cover a
  channel-axis source, and prove an undeclared stack is not suffix-split;
- backend-address tests assert the exact three-field `SourcePixelRef` shape,
  strict mapping-key rejection, direct disk and Bio-Formats registry dispatch,
  canonical `BioFormatsPlaneRef` address round-trip, inherited BioFormats
  virtual-workspace selection, and NPY fixture loading through disk rather
  than a Bio-Formats reader branch; source metadata contains no duplicate
  backend address;
- source payload tests cover monochrome conversion, explicit mask creation,
  color and object-label source loading, declared source-channel-axis
  preservation across metadata composition, axis removal after color collapse,
  and rejection of conflicting axes or channel counts without any image-type
  metadata token;
- setup parser tests cover every supported CP source image type and source
  literal enum, one and multiple NamesAndTypes/LoadImages assignments, shared
  versus per-assignment repeated columns, and exact rejection of mismatched
  cardinalities;
- declaration callable-loading tests cover primary functions, variants,
  aliases, source-setup
  declarations with no function, supported executable exporters, unsupported
  enabled module rejection, duplicate ownership rejection, identity of the raw
  implementation object, and an unchanged function namespace before and after
  import and compilation;
- callable-selection order tests cover every settings-selected variant plus
  Z-index and non-Z-index `DilateObjectsModule` / `RemoveHolesModule` imports,
  assert `resolve_function` accepts artifact contract and resolved source
  bindings but no `ProcessingConfig`, assert processing lowering requires the
  selected `CallableContract`, and assert public compilation preserves the exact
  FunctionStep callable without invoking the parsed-module selector; parsed
  import source-config resolution constructs no placeholder FunctionStep;
- exporter tests assert `SaveImages`, `ExportToSpreadsheet`, and
  `ExportToDatabase` are present in public source, consume only contract-declared
  artifacts, and expose compiled materialization expectations;
- SaveImages tests cover PNG, TIFF, and NPY path/bit-depth output through
  `ImageFileOptions`, including metadata-derived relative paths; its selected
  image arrives through the compiled special input, its converted image is the
  special output, its first return preserves main-flow value and identity, and
  its compiled invocation uses no CellProfiler runtime adapter;
- plate-scope tests run with at least two axes and two worker lanes, assert each
  aggregate exporter runs once after result merge, and assert records absent
  from its contract never enter `RuntimeArtifactBatch`;
- ObjectState/function-pane tests treat the runtime-bound
  `RuntimeArtifactBatch` parameter as a generic hidden runtime payload and
  expose only typed exporter settings; code mode contains the raw callable and
  public kwargs only;
- spreadsheet tests compare emitted CSV schemas and rows with native references;
  database tests open all three official SQLite references, compare table names,
  schemas, and rows, and compare the semantic CPA properties keys. These tests
  do not use the official30 `value_only` shortcut;
- CalculateMath and DefineGrid code-mode tests assert typed public kwargs and no
  runtime invocation options.
- setting-coverage tests assert the exact three-member status enum, module-owned
  ignores, and fail-loud `UNMAPPED` rows; no status denotes artifact identity or
  caller policy.

**Workstream completion evidence**

- Every concrete CP module contract comes directly from its nominal module
  declaration.
- Processing modules, setup modules, aliases, and function variants resolve
  through one `CellProfilerModule` registry.
- The backend package returns declaration-loaded implementation callables with
  source-declared generic metadata and adds no catalog, callable cache,
  function mutation, wrapper call layer, or CP ownership attribute.
- Axis and plate callable scopes are compiler-enforced generic semantics;
  aggregate exporters run once from exact contract-selected records and emit
  real CSV/SQLite/properties files.
- Runtime policy lookup contains no concrete module name.
- Artifact kind is never inferred from a special-output string.
- Generic source loading, payload alignment, and metadata contain no
  CellProfiler image-type label or role lookup.
- Workspace construction has no benchmark-only selection object, source-root
  probe, provider fallback, TIFF branch, or auxiliary format registry.

#### Workstream 1B: Replace Symbol And Generated Matching With Exact Compilation

**Prerequisites**

- Workstream 1A defines complete ordered contracts for every registered module
  on the same atomic cutover branch.

**Changes**

- Rewrite
  `openhcs/interop/cellprofiler/compile_time_contracts.py::CellProfilerInvocationContractProviderFactory`
  to run the deterministic session prepass specified above.
- Key compiled plans by `(StepSnapshot.index, FunctionInvocationKey)`.
- Extract `source_binding_group_keys_for_group_by` in
  `openhcs/core/source_bindings.py` and make
  `PathPlannerExecutionGroups.source_binding_scope_for_group_by` delegate to
  it.
- Use effective source bindings and an ordered `ArtifactSpecCollection` as the
  prepass input state.
- Return `InvocationContractPlan` with the compiled callable contract and exact
  consumed compile-only kwarg names.
- Keep `openhcs/core/pipeline/path_planner.py` as a consumer of the exact
  provider plans. The provider prepass completes before reverse future-input
  planning, so every reverse and forward query returns the same plan.
- Update `openhcs/interop/cellprofiler/module_processing_config.py` to use
  ordered `ModuleArtifactContract` values, `ArtifactSpec.relations`, and source
  bindings directly.
- Remove `PipelineGeneratorArtifactPruner` and its liveness/materialization
  walk. The import pass retains every enabled executable module in source order;
  generic artifact planning decides storage and materialization after
  compilation.
- Move `SourceBindingContractAlignment` from
  `openhcs/core/artifact_contract_preview.py` to
  `openhcs/core/module_artifact_contract.py`. Add
  `source_binding_alignment` and `projected_source_bindings` methods to
  `ModuleArtifactContract`. Alignment validates that configured bindings select
  declared inputs; it never infers that every non-special input requires a
  source binding because generic main flow owns that distinction. Update the
  preview widget to call the contract directly and remove compiler-side
  contract pruning by scoped source bindings.
- Collapse `CellProfilerRuntimeCallable` and
  `CellProfilerGroupedRuntimeCallable` in
  `openhcs/interop/cellprofiler/runtime/module_execution.py` into one runtime
  adapter with one aggregate contract for axis-scoped callables. Its constructor
  accepts only the raw canonical callable and enriched `CallableContract`,
  builds one immutable runtime plan, and retains one executor over that plan;
  it reads compiled metadata from that contract without attaching or
  recomputing callable metadata. Plate-scoped callables remain generic
  importable functions and never enter this adapter.
- Add `CallableContract.require_processing_contract()` and replace every
  `CellProfilerProcessingContractAuthority.for_callable` call with that
  generic query or the retained runtime plan's property.
- Add `CellProfilerRuntimeAdapter.runtime_adapter_spec()` as the sole nominal
  owner of the CP adapter declaration and make the invocation provider use that
  exact value for axis-scoped CP contracts.
- Extend `RuntimeAdapterSpec` with the optional runtime-callable factory and
  make `CallableContract.resolve_runtime_callable` use it after ordinary
  callable/reference resolution. Register the top-level two-argument
  CellProfiler runtime-callable constructor on the compiled adapter spec.
  Delete `function_reference_rehydration.py` and the CellProfiler rehydrator;
  plate-scoped and adapter-free callables resolve directly.
- Refactor `CellProfilerModuleRuntimePlan` to retain static callable/module
  facts only. Derive active input/output spec collections per
  `CellProfilerModuleRunRequest` from the generic adapter's already-grouped
  artifact plans and source bindings.
  - **Status, declaration-policy consumer slice only (2026-07-10): complete.**
    `CellProfilerModule` runtime-policy classmethods now use only the authorized
    `raw_func`/`contract`/`callable_contract` plan surface, deriving special-input
    parameters through the callable ABI, object inputs through
    `ArtifactSpecCollection`, and bound names through nominal MRO declarations.
    Focused evidence: the policy-boundary plus runtime-plan-shape suites pass
    (`4 passed`), and the declaration/MRO subset passes (`4 passed`).
  - **Status, output/measurement consumer slice only (2026-07-10): complete.**
    `measurement_recording.py`, `output_recording.py`,
    `output_record_request.py`, and `measurement_image_resolver.py` now query the
    three-field plan owners and active adapter artifact plans directly. Focused
    evidence: the runtime-plan shape and consumer suites pass (`7 passed`), with
    zero forbidden plan-projection accesses in the consumer AST gate.
- Refactor `CellProfilerModuleExecutor` to retain one runtime plan; delete its
  derived contract/name/policy fields and per-callable plan dictionary.
- Remove `CellProfilerModuleRunRequest.func`, `plan`, `input_image`, and
  `current_image`; use `executor.plan.func` and one invocation `image` field.
- Replace `RuntimeArtifactInputRequest` inheritance with composition and make
  runtime binding requests query aggregate contract partitions directly.
- Simplify `RuntimeReturnedOutputMatcher` consumption so compiled contract order
  and main-flow participation determine all returned output matching.
- Remove the `func` field from `CellProfilerOutputValueResolutionRequest` and
  `CellProfilerResolvedOutputValues.from_returned_outputs`. Pass the compiled
  contract's full ordered declared-output specs to
  `RuntimeReturnedOutputMatcher` without inspecting wrappers, materializers,
  or output names.
- Change `CellProfilerPerImageMeasurementPolicy.matches` to validate its
  existing `request.outputs` directly; remove its
  `CellProfilerCallableOutputSpecs(request.func)` projection. Callable
  inspection remains only for the Python special-input ABI and generic image
  consumption declaration.
- Add `CellProfilerRuntimeAdapter.require_processing_context()` and replace
  every `RequireProcessingContextBoundaryPolicy(adapter).context` call with
  that fail-loud method.
  - **Status, `RequireProcessingContextBoundaryPolicy` deletion slice only
    (2026-07-10): complete.** The boundary policy is deleted; owned
    source-candidate and source-binding consumers call the adapter authority
    directly.
- Replace one-field runtime projections with their existing owners:
  `ImageArrayShapeSemantics(value).shape` supplies diagnostic shape;
  `RuntimeArtifactRecordLocationIdentity.unique_records(records)` performs
  location de-duplication; `ObjectLabelVariantData` supplies final and
  small-removed variants; and `sparse_ijv_rows_from_label_slice` performs
  sparse-row coercion.
  - **Status, object-label measurement-row wrapper deletion slice only
    (2026-07-10): complete.** `measurement_rows.py` now uses those owners;
    `ObjectLabelFinalLabels`, `ObjectLabelSmallRemovedLabels`, and
    `SparseLabelRowsCoercion` are deleted. Focused evidence:
    `test_cellprofiler_measurement_rows.py` (`3 passed`) and the AST gate.
- Inline closed scalar predicates in the methods that own their decisions:
  optional string normalization in `CellProfilerStringKwargAuthority`, source
  axis cardinality in `SourceBindingMatchedPlaneResolution`, source identity
  quality in `RuntimeRecordSourceImageSetSelector`, invocation rank selection
  in `VolumetricInputExecutionModePolicy`, dense-label memory/repeat checks in
  their current consumers, MATLAB private-name filtering in
  `_matlab_numeric_arrays`, grid slice count in `SpatialGridValueAuthority`,
  and selected plane index in
  `RuntimePlaneImagePayloadPlaneSelectionResult`. Delete the CP image-number
  resolver, reverse map, source-order warmup cache, adapter forwarding methods,
  and TrackObjects start-number injection; runtime rows use canonical local
  `slice_index`, while external CP export/equivalence owns `ImageNumber`
  translation.
- Fold payload alias and component-metadata extraction into
  `SourceBindingPayloadPlaneResolution`, their only consumer.

**Required deletions**

- `openhcs/interop/cellprofiler/symbol_table.py`;
- `tests/unit/test_cellprofiler_symbol_table.py` after its genuine declaration
  and artifact cases move to their owning test modules;
- `openhcs/interop/cellprofiler/module_artifact_inputs.py`;
- every `ArtifactSpecKey` consumer;
- `CellProfilerCompileTimeArtifactFlow`;
- `compile_time_source_binding_group_keys_for_invocation` after both compiler
  and path planner use the generic source-binding helper;
- generated contract provider, matcher, payload, candidate, and fallback code in
  `openhcs/interop/cellprofiler/runtime/generated_pipeline.py`;
- `SourceBindingRuntimeContractGuard` after its alignment behavior moves onto
  `ModuleArtifactContract`;
- `PipelineGeneratorRuntimeContractProjector`;
- `PipelineGeneratorArtifactPruner` and generator-time artifact liveness sets;
- generator/importer options `prune_dead_unmaterialized_artifact_steps` and
  `materialize_terminal_images`;
- `ArtifactSpecKey` and `openhcs/interop/cellprofiler/module_roles.py` after
  Workstream 1A moves diagnostics and exporter behavior to their real owners;
- `CellProfilerGroupedRuntimeCallable`,
  `CellProfilerGroupedModuleContracts`, runtime step binding wrappers, and
  contract resolution wrappers;
- `CellProfilerProcessingContractAuthority`, `PROCESSING_CONTRACT_CACHE`,
  absorbed processing-contract fallback, `_attach_runtime_processing_contract`,
  `cellprofiler_module_callable`, and
  `rebuild_cellprofiler_runtime_callable`;
- `CellProfilerModule.contract`, every leaf assignment,
  `CallableInvocationKwargSpec`, runtime signature reconstruction, and
  unsupported-kwarg filtering;
- `InvocationContractProviderLike`, `public_callable_invocation_contract`, and
  first-claim composite-provider control flow after all factories return
  nominal provider instances;
- `CompiledFunctionInvocation.artifact_input_keys` and
  `.artifact_output_keys` after every planner/runtime consumer uses exact refs;
- `CellProfilerRuntimeCallable.__reduce__`, identity-only `__eq__`, and
  `__hash__`; the wrapper is process-local and does not participate in
  transport or contract-key identity;
- `openhcs/core/function_reference_rehydration.py`,
  `FunctionReferenceRehydrationRequest`, `FunctionReferenceRehydrator`, and
  `CellProfilerFunctionReferenceRehydrator`; `RuntimeAdapterSpec` owns the
  exact runtime-callable factory;
- `FunctionStepTransportAuthority.approved_code_document_factory_names` and
  its UI branch after the sole CP factory is removed;
- `CellProfilerModuleExecutor._runtime_plans`, `_canonical_module_name`, and
  `_primary_image_input_policy`;
- `RuntimeArtifactBindingScope` and every copied binding-scope field in runtime
  plans and requests;
- `RuntimeArtifactLineageScope` and
  `RuntimeArtifactSourceLineage` after their consumers use artifact relations;
- `CellProfilerCallableOutputSpecs`; returned-output specs come from the
  compiled contract;
- `RequireProcessingContextBoundaryPolicy`, `RuntimeShapeInspection`,
  `RuntimeArtifactRecordDeduplication`, `ObjectLabelFinalLabels`,
  `ObjectLabelSmallRemovedLabels`, `SparseLabelRowsCoercion`,
  `Pure2DSliceCountCandidate`, and `Pure2DTraceLabelStats`;
  - **Status, `Pure2DSliceCountCandidate` and `Pure2DTraceLabelStats`
    deletion slice only (2026-07-10): complete.** Slice-count diagnostics now
    use their existing runtime value and `RuntimeSliceProjection` owners, and
    PURE_2D trace recording computes label maxima at its consuming record
    sites. Focused evidence: `6 passed, 419 deselected`.
- `SourceIdentitySetCardinality`, `DeclaredOutputResolution`,
  `CellProfilerOptionalNonemptyString`, `SourceBindingAxisCardinality`,
  `SourceImageSetIdentityQuality`, `InvocationSpatialRankCandidates`,
  `DenseLabelShapeSet`, `DenseLabelStackRepeatPattern`,
  `MatlabPayloadEntryName`, `SpatialGridSliceCount`,
  `RuntimePlaneSelectedPlaneIndex`, and
  `CellProfilerImageNumberResolution`;
  - **Status, scalar projection wrapper deletion slice (2026-07-10):
    complete.** `DenseLabelShapeSet`, `DenseLabelStackRepeatPattern`,
    `MatlabPayloadEntryName`, and `SpatialGridSliceCount` are deleted; their
    predicates execute at the existing dense-label budget,
    measurement-table broadcast, source-payload filtering, and spatial-grid
    alignment owners. Focused behavioral and AST evidence: `5 passed`.
- `SourceBindingPayloadAliasSet` and
  `SourceBindingPayloadComponentMetadata`;
  - **Status, remaining one-field runtime wrapper deletion slice
    (2026-07-10): complete.** `SourceImageSetIdentityQuality`,
    `RuntimePlaneSelectedPlaneIndex`, `CellProfilerImageNumberResolution`,
    `SourceBindingPayloadAliasSet`, and
    `SourceBindingPayloadComponentMetadata` are deleted. Their behavior now
    resides on `SourceImageSetIdentity` or directly in the existing
    plane-selection and source-binding consumers. The subsequently verified
    image-number consumer path was also deleted: TrackObjects uses local
    runtime-slice numbering and external equivalence performs the sole CP
    numbering translation. Focused behavioral and AST evidence: `825 passed`.

**Forbidden additions**

- a renamed symbol table;
- a second compile-time artifact flow;
- candidate matching by callable name, module number, index, or source aliases;
- structural protocols that probe CP objects;
- a CellProfiler-specific execution-group resolver;
- a wrapper whose only role is holding one existing artifact collection or
  contract;
- a one-field class whose only observable behavior is a scalar predicate,
  normalization, optional-value projection, or call to an existing authority;

**Focused tests**

- exact provider lookup for plain, list, and dict patterns;
- AST and object tests require the exact frozen provider shape, immutable
  `(step_index, FunctionInvocationKey)` map, canonical raw-callable identity,
  non-CP `None`, and fail-loud missing CP key;
- two synthetic factories claiming one invocation fail in both registry
  orders; zero and one claim preserve generic behavior;
- source-bound illumination groups;
- source group selection uses exact component plus key, rejects no-match, and
  never returns the sole/all bindings as a scoped fallback;
- Align and MeasureColocalization multi-image contracts;
- object lineage through Identify, Relate, Expand, Filter, and Mask;
- TrackObjects timepoint artifacts;
- Tile with image inputs carrying equal component identity;
- missing, duplicate, and ambiguous invocation keys fail during compilation;
- generic compilation rejects unknown and missing required public kwargs after
  consumed identity keys are removed; CP runtime forwards the exact compiled
  kwargs and contains no signature inspection or unsupported-kwarg filter;
- active artifact selection joins exact refs and rejects same-name/different-
  type, relation, materialization, partition, and plan-role drift;
- one-group and multi-group public invocations compile to the same single
  runtime callable shape;
- runtime callable tests assert the adapter stores only raw callable and
  enriched callable contract as plan-owned properties, the runtime plan stores
  only its three authorized fields, the executor stores only that plan, and generic
  `FunctionReference` worker transport resolves the canonical declaration
  callable and the exact adapter spec builds the executable from that callable
  plus enriched contract; tests assert no rehydrator registry is consulted;
- adapter-spec tests assert every axis-scoped CP compiled contract contains the
  exact value returned by `CellProfilerRuntimeAdapter.runtime_adapter_spec`,
  while SaveImages and plate-scoped exporters contain none;
- run-request tests assert its four authorized fields and prove all image
  execution paths use the one invocation image value;
- active runtime spec projection follows
  `artifact_inputs_by_group`/`artifact_outputs_by_group` and rejects a grouped
  output without declared lineage;
- output-value tests pass full declared-output specs from the compiled contract
  and cover passthrough main output, one declared main output, exact positional
  special outputs, and exact multi-image stack splitting without
  `CellProfilerCallableOutputSpecs` or semantic fallback;
- source-binding, image-number, plane-selection, label-variant, and diagnostic
  tests assert the same values after the one-field projection wrappers are
  removed.

**Workstream completion evidence**

- Provider lookup is exact and deterministic.
- No production import references a deleted symbol or generated matcher.
- The same public source compiles to the same invocation contracts in the UI and
  benchmark server processes.

#### Workstream 1C: Remove Hidden Pipeline State And Collapse Transport

**Prerequisites**

- Workstreams 1A and 1B derive every runtime contract from public declarations
  on the same atomic cutover branch.

**Changes**

- First move the complete ordered module-to-public-declaration loop from
  `runtime_pipeline.py` / `pipeline_generator.py` into the pure
  `import_cellprofiler_pipeline` operation in
  `openhcs/interop/cellprofiler/pipeline_import.py`. That operation parses the
  supplied `.cppipe` address, folds setup modules into `PipelineConfig`, lowers
  every enabled processing/export module to real FunctionSteps through
  `CellProfilerModule`, and returns `(steps, pipeline_config)` without source
  rendering or I/O. Migrate every importer, UI, workspace,
  benchmark, agent, and test caller to this operation. Delete
  `import_service.py`, `import_records.py`, `PipelineGenerator`,
  `GeneratedPipeline`, `pipeline_generator.py`, `runtime_pipeline.py`,
  `pipeline_compiler.py`, `compiler_registry.py`, and all forwarding aliases
  only after those callers compile against the pure operation. This ordered
  move is the first change in Workstream 1C and leaves no interval where
  forwarding files are deleted before their implementation owner exists.
  - [x] Direct-import operation implementation: `import_cellprofiler_pipeline`
    now parses once, folds setup declarations, lowers enabled executable
    declarations to public `FunctionStep` values, and returns the public
    steps/config pair. Focused direct importer coverage passes. Caller
    migration and superseded-file deletion remain pending; Workstream 1C is
    not complete.
  - [x] Benchmark converter caller slice: the CLI now renders the steps returned
    by `import_cellprofiler_pipeline` exclusively through
    `FunctionStepTransportAuthority`; the converter package no longer exports
    production compatibility facades; and setting coverage reads
    `CellProfilerModule.__registry__` plus each declaration's `bind_settings`
    result directly. The focused converter suite passes. Benchmark coverage
    artifact-writer migration remains part of the broader caller cutover.
  - [x] In-tree `.cppipe` corpus-test caller slice: supported cases now consume
    only the public steps/config pair, derive expected executable modules from
    `CellProfilerModule` declarations, and validate canonical generic source
    reconstruction. Unknown-module and empty-pipeline failures plus the AST
    boundary gate pass. Focused evidence is `7 passed, 4 failed`; the four
    retained failures expose production declaration gaps: an unbound
    `ArtifactSpec` reference in `IdentifySecondaryObjects`, undeclared
    `LoadData`, and an undiscoverable retained illumination artifact.
  - [x] Generated-execution test migration slice:
    `test_cellprofiler_generated_pipeline_execution.py` now covers the pure
    importer, ordinary `FunctionStep`/`PipelineConfig` declarations, generic
    FunctionStep transport, compiler-derived invocation contracts, the nominal
    `CellProfilerModuleRuntimePlan`, and declaration-owned module artifact
    contracts. Generated-module loading, registration, pruning options,
    carrier types, and the synthetic direct-runtime harness are removed from
    the file. Focused evidence: `16 passed`; its AST gate rejects imports from
    every deleted generated-pipeline boundary.
- Remove `invocation_contracts` from
  `openhcs/core/steps/function_step.py::FunctionStep` and from
  `StepSnapshot`.
- Remove hidden contract formatting and clean-mode handling from
  `openhcs/serialization/pycodify_formatters.py`.
- Delete `CellProfilerRuntimeCallableFormatter`; compiled wrappers are internal
  execution objects and are never pycodified as public declarations.
- Remove `pipeline_metadata` from `CompilationSession` and all compiler call
  sites.
- Remove `PipelineMetadataCarrier` and `PipelineIdentityCarrier` structural
  protocols. `_compiler_pipeline_scope_id` uses the existing submitted-list
  branch for its process-local ObjectState namespace; that namespace is not
  artifact or execution identity.
- Delete the CP import request/result/provenance records. Return public steps
  and `PipelineConfig` directly. Derive source text from steps only at a generic
  caller that explicitly requests source, and derive every source/workspace
  fact from the config.
- Remove CP generated-module import entirely. The direct import operation returns
  the exact `list[FunctionStep]`; the ZMQ server performs the only required
  fresh-source reconstruction from generic transport source.
- Replace `PipelineStepsBoundary` and `PipelineStepsCarrier` fields throughout
  client, agent, and server records with direct mutable step lists. Route all
  source rendering and namespace validation through
  `FunctionStepTransportAuthority`.
- Add `OpenHCSExecutionSubmission.pipeline_code()` and store its returned string
  directly on `ZMQExecutionRequestBuilder`. Make `ZMQConfigProjection`
  pycodify config assignments directly and use the existing request-payload
  source hash. Delete every `Pycodified*` value and the assignment-source
  request wrapper; retain no replacement source carrier.
- Replace config-carrier mixins and `ZMQResolvedConfig` with direct
  `OpenHCSExecutionConfigBundle` fields.
- Move agent pipeline-definition construction onto
  `ExecutionPipelineSessionRequest.build_pipeline_definition` and its two
  existing request leaves. Replace the agent payload inheritance stack with
  one concrete definition record and direct fields on the session record;
  replace `PipelineIdBoundary` with `str`.
- Replace `InputWorkspacePreparationResult.prepared_pipeline` and untyped
  materialization with typed public steps, pipeline config, and source-workspace
  materialization. Map CP import output into this generic result.
- Remove retained CP import-result state and the no-op UI runtime rebinding
  service.
- Introduce the frozen, slotted, GUI-private `PipelineEditorStateRoot` dataclass
  with exactly `name: str`, `description: str | None`, and
  `step_scope_ids: tuple[str, ...]`. `PipelineObjectStateBinding` uses it only
  as the reconstructable parent `ObjectState`; steps remain exclusively in
  child `ObjectState` instances and every execution/import boundary returns
  `list[FunctionStep]`.
- Delete rather than migrate `Pipeline.metadata`, its implicit `created_at`
  timestamp, `Pipeline.pipeline_config`, `MutableSequence`, `clone`, `to_dict`,
  `add_step`, and sequence compatibility behavior. Display name and description
  move to `PipelineEditorStateRoot`; configuration remains in the existing
  per-plate `PipelineConfig` delegate; execution steps remain a plain list.
- Delete `PipelineObjectStateBinding.pipeline`, `pipeline_for_plate`,
  `pipeline_declaration`, `update_plate_pipeline`, `replace_pipeline`, and
  `registered_plate_pipelines`. Retain `steps_for_plate`,
  `update_plate_steps`, and
  instance `replace_steps(steps) -> None`. Add only
  `editor_state_for_plate(plate_path) -> PipelineEditorStateRoot`,
  `update_editor_text(plate_path, *, name, description) -> None`, and
  `registered_plate_steps() -> dict[str, list[FunctionStep]]` for GUI use.
  `step_scope_ids` is generated inside `replace_steps` and is never supplied by
  import, compiler, or transport callers. Remove every
  `isinstance(..., Pipeline)` compatibility branch from GUI workflows.
- Migrate the remaining production core-`Pipeline` constructors in
  CellProfiler import code, PyQt plate workflows, Textual plate execution, and
  visual-programming services to direct lists. Rewrite integration helpers and
  unit tests that import `Pipeline`; `ResolvedPipelineDefinition` remains the
  compiler-internal coupled record for resolved steps, step states, and
  snapshots, with its metadata field removed.
- Update the textual UI's one core `Pipeline` construction to use the plain step
  list.

**Required deletions**

- `openhcs/core/function_step_invocation_contracts.py`;
- `PipelineMetadataCarrier`, `PipelineIdentityCarrier`,
  `PIPELINE_SOURCE_SCHEMA_METADATA_KEY`, and
  `CompilationSession.pipeline_metadata`;
- generated/import/runtime artifact-contract sidecar fields;
- `CellProfilerPipelineProvenance`, `CellProfilerPipelineImportRequest`,
  `CellProfilerPipelineImportResult`, `CellProfilerModuleReference`, every
  nested/import-result provenance accessor, `import_service.py`, and
  `import_records.py`;
- core `Pipeline` class and public export after GUI and textual consumers are
  migrated.
- `openhcs/interop/cellprofiler/runtime/generated_pipeline.py`;
- `openhcs/runtime/zmq_pipeline_transport.py`, `PipelineStepsBoundary`,
  `PipelineStepsCarrier`, `PycodifiedPipelineStepSource`,
  `PycodifiedSource`, `PycodifiedPipelineCode`, `PycodifiedConfigSource`,
  `PycodifyAssignmentSourceRequest`, `OpenHCSExecutionConfigCarrier`, and
  `ZMQResolvedConfig`;
- `PipelineIdBoundary`, `ExecutionPipelinePayload`,
  `ExecutionPipelineDefinitionProvider`, and both provider leaves;
- `CPPipePipelineArtifact`, `PreparedGeneratedPipeline`, CP plate
  request/result wrappers, and workspace-materializer forwarding wrappers;
- `runtime_pipeline.py`, `pipeline_compiler.py`, `compiler_registry.py`, and
  their request/partition/compiler/benchmark aliases;
- `PipelineGenerator`, `GeneratedPipeline`, `pipeline_generator.py`, every
  generator registry/emitter/build-stage/backreference type,
  `GeneratedPipelineRequest`, `SkippedModuleSelection`,
  `GeneratedStepEmission`, `GeneratedStepEmissionGroup`, generated import
  collectors, `python_literal`, artifact-contract comment emitters,
  `GeneratedPipelineConfigDefaults`, `StepInputSourceLiteral`,
  `GeneratedProcessingConfigShape`, and generated settings/parameter-target
  wrappers;
- direct orchestrator execution helper/result/progress types and direct
  validation entry point.

**Forbidden additions**

- a replacement metadata dictionary on a step, pipeline, callable, or config;
- an opaque serialized contract payload;
- a GUI editor state root accepted by compiler, transport, or orchestrator APIs;
- skipped modules whose effects are recreated by mutating another step.

**Focused tests**

- FunctionStep construction, ObjectState reconstruction, clean pycodification,
  and ZMQ transport without hidden contract fields;
- GUI editor add, remove, reorder, code edit, and plate reload using
  `PipelineEditorStateRoot` plus child step states;
- imported CP pipeline returns a plain step list;
- SaveImages appears in generated public source and materializes through normal
  execution;
- complete lazy `SourceBindingsConfig` round-trips through public source and
  recreates the same workspace in a fresh process with no import result;
- source ingestion returns the generic typed workspace result and contains no
  `Any` pipeline or materialization carrier;
- generated source equals the canonical pycodify source for the returned step
  list and is reconstructed only by the generic ZMQ source path;
- client submission, agent session, and server execution records expose direct
  step lists and no boundary/carrier wrapper;
- draft and source agent request leaves build the one concrete pipeline
  definition directly and allocate no provider or ID wrapper;
- ZMQ request construction carries a direct pipeline-code string and config
  source fields, with no pycodified-source wrapper;
- every converted integration execution uses the two-stage ZMQ submission.

**Workstream completion evidence**

- `PipelineConfig` plus a plain FunctionStep list is sufficient in a fresh
  process.
- No compiled semantics travel through pipeline metadata or import sidecars.
- All observable CP module effects are represented by public steps or config.

#### Workstream 1D: Make Direct Import Sparse And Native-Looking

**Prerequisites**

- Workstream 1C establishes the direct import operation and makes public
  steps/config plus generic source transport the complete execution declaration
  on the same atomic cutover branch.

**Changes**

- Refine the direct import loop established by Workstream 1C so its real
  FunctionSteps contain only user-controlled typed module settings and sparse
  compile-only identity overrides.
- Resolve functions through `CellProfilerModule` directly; remove the copied
  generator registry and `contracts.json` branch.
- Use the compiler-identical reconstruction and exact contract comparison
  algorithm specified above before emitting an identity override.
- Compare ObjectState-resolved source config values before emitting step
  bindings; no source-field equivalence helper survives.
- Promote repeated processing and source-binding values into the generated
  `PipelineConfig`; emit `LazyProcessingConfig` and
  `LazyStepSourceBindingsConfig` only for step differences.
- Collapse adjacent module instances into one public FunctionStep when their
  callable, behavior kwargs, processing semantics, and source selection are
  identical across all groups.
- Construct dict patterns only for omission or differing behavior.
- Remove artifact contract comments and compiled implementation details from
  code mode.
- Keep public enum and dataclass values typed so ObjectState renders comboboxes,
  checkboxes, and numeric controls from callable signatures.
- Construct existing `ProcessingConfig`, `PipelineConfig`, and lazy step
  overrides directly; no generated literal/config-shape carrier remains.
- Keep source generation exclusively on
  `FunctionStepTransportAuthority.source_from_pipeline`; the pure import
  operation contains no source renderer or output address.
- Rewrite `benchmark/converter/convert.py` as a thin CLI: call
  `import_cellprofiler_pipeline` with the `.cppipe` address/backend, render the
  returned steps through the generic transport authority, write that source to
  the CLI output path, and report step count. It does not construct a CP import
  request/result or inspect skipped/converted module carrier fields.
- Rewrite `benchmark/converter/compatibility_matrix.py` to iterate
  `CellProfilerModule.__registry__.values()`, obtain callables from each
  declaration's callable-loading method, and invoke the same direct import operation
  for corpus setting coverage. It does not call `library.list_modules`,
  `library.get_contract`, `partition_cppipe_modules`, or a generator-only
  registry.
- Reduce `benchmark/converter/__init__.py` to corpus/report exports that have
  real external consumers. Parser, schema, settings, module declarations, and
  import operations remain imported from their production packages; the
  benchmark package does not re-export them through a compatibility facade.

**Required deletions**

- `force_grouped_public_function_spec` and equivalent forced-dict branches;
- repeated generated step source-binding blocks equal to pipeline defaults;
- generated `measurement_artifact_name`, canonical image/object names,
  materialized artifact-name lists, module numbers, and `None` runtime payload
  placeholders;
- generated runtime wrapper imports and artifact-contract comments.
- explicit absorbed-library root support and copied module registry;
- CP import request/result types and all derived source text,
  generated-source-path, converted-module, failed-module, module-reference,
  and setting-coverage result fields;
- `benchmark/cellprofiler_library`, `benchmark/cellprofiler_compat`, obsolete
  absorption tools, converter role aliases, and the CP threshold forwarding
  module after tests import authoritative production modules.
- benchmark converter exports of `SourceLocator`, `LLMFunctionConverter`,
  `LibraryAbsorber`, symbol-table types, runtime-pipeline request/result
  wrappers, and production parser/schema/config types.

**Forbidden additions**

- import-only heuristics that the compiler cannot reproduce;
- hardcoded setting-name lists in the import pass;
- stringified enum values;
- eager config classes where a lazy config is required at pipeline or step
  scope.

**Focused tests**

- AST assertions over generated ExampleColocalization, ExampleTrackObjects,
  multi-channel illumination, object pipelines, measurements, grids, and
  SaveImages;
- object assertions over the pre-pycodify generated FunctionSteps and config;
- exact source equality between converter/UI projections and the generic
  FunctionStep transport authority applied to the returned steps;
- converter CLI output equals the canonical generic source for the returned
  steps;
- compatibility reports discover modules exclusively from
  `CellProfilerModule.__registry__` and fail on unsupported enabled modules;
- pycodify round-trip preserves enum, bool, numeric, tuple, list, and dataclass
  types;
- generated behavior kwargs omit values equal to typed raw signature defaults,
  and fresh-process compilation reconstructs the same module contract;
- source bindings appear once at the broadest valid scope;
- all-group identical patterns are not dicts;
- group-specific methods remain dict patterns and route to the correct
  component identity.
- original and reconstructed module contracts compare equal after sparse
  override emission for every converted module.

**Workstream completion evidence**

- Generated CP source has the same declarative shape as a native OpenHCS
  pipeline.
- Re-running the compiler from generated source recreates every required
  contract.

**Phase 1 exit criteria**

- Every required deletion in Workstreams 1A through 1D is complete; no old
  import, generator, compiler, runtime, source-schema, callable-catalog,
  pipeline-wrapper, transport-wrapper, reader, writer, alias, or re-export
  remains.
- `PipelineConfig` plus `list[FunctionStep]` reconstructs the complete source
  workspace and exact invocation contracts in a fresh ZMQ server process.
- One nominal module registry, one exact invocation provider, one process-local
  CP runtime adapter, generic artifact planning/output matching, and generic
  FunctionStep transport are the only active path.
- Generated CP source has native OpenHCS declarative shape, sparse typed kwargs,
  inherited lazy config, and no compiled or importer-only state.
- All Phase 0 boundary tests and every focused Workstream 1A-1D test pass
  together after the old tests and fixtures are migrated or deleted.

### Phase 2: Enforce Invalid Pipelines At Compile Time

**Prerequisites**

- Phase 1 provides complete nominal declarations, exact compilation/runtime,
  direct public import, and exact public source.

**Changes**

Add compiler validation at the owning boundary for:

- unresolved CellProfiler callable ownership;
- missing `CellProfilerModule` declaration;
- duplicate exact invocation key;
- unavailable required input `ArtifactSpec`;
- source binding whose artifact type or identity conflicts with the module
  input contract;
- special-input parameter absent from the raw callable signature;
- special-input count or order inconsistent with runtime input specs;
- special-output slot count or order inconsistent with returned output specs;
- output spec order inconsistent with callable return ABI;
- more than one canonical main-flow output;
- required variable component absent from resolved processing config;
- grouped contracts with incompatible module or processing semantics;
- compile-only kwarg not consumed by the module declaration;
- runtime behavior kwarg excluded from the raw callable without a registered
  config binding;
- artifact relation targeting an unavailable artifact;
- incompatible component cardinality for statically known source-bound inputs.

Error messages identify step index, invocation key, module declaration, input or
output spec, and violated contract. They do not mention generator candidates or
matching fallback.

**Required deletions**

- runtime checks that only compensate for missing compile-time contract
  validation;
- source-binding drift matcher errors based on candidate module contracts.

**Forbidden additions**

- runtime coercion of incompatible artifacts;
- silent omission of an unresolved artifact;
- defaulting to a different module contract after exact lookup fails.

**Focused tests**

- one negative test for every compile failure listed above;
- assertions that worker execution is never entered for each invalid pipeline.

**Exit criteria**

- Structurally invalid CP pipelines fail during ZMQ compilation.
- Runtime errors are reserved for data-dependent failures.

### Phase 3: Canonical ZMQ Parity And Viewer Gates

**Prerequisites**

- All focused unit and integration tests from Phases 0 through 2 pass.

**Changes**

- Delete both OpenHCS benchmark result-cache layers before collecting any
  acceptance result. Each case submits and executes current public source;
  persisted native CellProfiler references remain reusable. Remove the inverse
  `force_openhcs_run` CLI switch because current execution is unconditional.
- Delete timeout-derived native timing. Load measured native phase timing from
  reference provenance; retain parity but report null speedup and fail the
  speed gate when that measurement is absent.
- Run every acceptance case through public-source ZMQ compile-then-execute.
- Export runtime expectations from compiled generic step plans with the ZMQ
  observation and validate that result; do not derive expected artifacts from
  generated CP sidecars or skipped parser modules.
- Compare benchmark-generated source with UI-equivalent pycodified source before
  execution.
- Remove `SourceSchemaImageSetSelection` from both adapters, suite/case context,
  manifests, and CLI. Resolve the existing `well_filter_config` from the
  suite's `openhcs_global_config` for both adapters. Assert the config rendered
  by the benchmark is byte-identical to the UI-equivalent config for the same
  cap.
- Compare explicit SaveImages outputs by reading
  `RuntimeExportObservation.image_outputs`; remove the candidate snapshot path
  that converts in-memory runtime records according to reparsed SaveImages
  settings.
- Run the official 30-case manifest against persisted native CellProfiler
  references.
- For the three ExportToDatabase cases, inspect native and candidate SQLite
  databases with `sqlite3`, compare table names/schemas/rows under the existing
  CellProfiler column dialect, and compare semantic CPA properties keys. The
  `value_only` flag suppresses image comparison only; database export
  validation remains required.
- Run the same 30 cases with a real `GlobalPipelineConfig` containing inherited
  `LazyNapariStreamingConfig(enabled=True, persistent=False)`. Use the existing
  `run_comparison_suite(..., openhcs_global_config=...)` parameter. Add no CLI
  flag and no benchmark-specific viewer option.
- Use the inherited global Napari port. Verify one viewer lifecycle per case,
  non-persistent shutdown between cases, successful layer publication, and no
  viewer-state leakage.
- Record parity status and median OpenHCS execution time per `.cppipe`.

**Commands and harnesses**

Add the official axis-one checked-in integration harness at
`tests/integration/test_cellprofiler_official30_zmq.py`. It loads
`official30_portable_axis1.json` and calls `run_comparison_suite` with the
existing config API:

```python
GlobalPipelineConfig(
    well_filter_config=WellFilterConfig(well_filter=1),
    napari_streaming_config=LazyNapariStreamingConfig(
        enabled=napari_enabled,
        persistent=False,
    ),
)
```

The baseline parameter uses `napari_enabled=False`; the viewer gate uses true.
Run it with:

```bash
OPENHCS_CP_NATIVE_REFERENCE_ROOT=/path/to/native_refs \
pytest -q tests/integration/test_cellprofiler_official30_zmq.py
```

The harness uses a unique output root and the same persisted native reference
root. It has no benchmark-specific source cap, cache switch, viewer flag, or
config mirror.

**Required deletions**

- `RuntimeExecutionCacheWritePolicy`, `_RuntimeExecutionCacheHit`,
  `_runtime_execution_cache_key_matches`, cache request properties,
  `_load_runtime_execution_cache`, `_write_runtime_execution_cache`, cache-hit
  phase records, and `reused_runtime_execution_cache` comparison/provenance;
- `_run_or_reuse_cached_openhcs`, `_cached_benchmark_result`,
  `_write_benchmark_result_cache`, `_cached_metric_values`, OpenHCS cache keys
  and manifests, `reuse_openhcs_cache`, `force_openhcs_run`, and
  `reused_cached_output` comparison/provenance;
- `CachedNativeReferenceTimingPolicy` and timeout-as-execution-time projection;
- benchmark shortcuts that execute prepared runtime wrappers instead of
  pycodified public steps;
- `_candidate_image_snapshots_for_equivalence` and every
  `RuntimeImageExportSpec`-based in-memory image parity branch;
- cache entries created from hidden runtime contracts;
- viewer-specific benchmark flags that duplicate `GlobalPipelineConfig`.

**Forbidden additions**

- direct orchestrator parity acceptance;
- different generated steps or configs for UI and benchmark runs;
- parity relaxations added to hide execution drift;
- per-pipeline viewer process state reused across non-persistent cases.

**Exit criteria**

- All 30 official cases compile, execute, and pass strict parity through ZMQ.
- SaveImages file formats, spreadsheet tables, and all three SQLite/CPA exports
  pass explicit native-output validation rather than parser-derived expectation
  flags.
- All 30 cases complete with Napari enabled and non-persistent without a viewer
  crash, stale layer, or port/config inheritance error.
- UI-equivalent and benchmark submissions have identical public source after
  deterministic formatting.
- The official suite's one-well execution limit is represented only by public
  inherited `WellFilterConfig` in that source; no manifest or adapter-side
  source selection survives.
- Median time is reported per `.cppipe`; performance is not accepted at the
  expense of parity or architecture.
- Every reported OpenHCS compile/execution duration belongs to the current run;
  no cached OpenHCS validation or output contributes to status or timing.
- Every reported speedup uses measured native timing; timeout bounds and missing
  values never become execution durations.

## Static Deletion Gates

The implementation is incomplete while any production or test definition,
import, call, field, or compatibility assertion remains for these names. The
single canonical static-deletion test contains their string literals only to
assert absence:

```text
CellProfilerSymbolTable
_SymbolTableBuilder
CellProfilerSymbol
ModuleArtifactContracts
ModuleArtifactInput
ArtifactSpecKey
CellProfilerModuleRole
CellProfilerModuleRoleSpec
cellprofiler_module_role
ModuleSettingsSourceModule
BinderSettingsSourceModule
CellProfilerObjectInputCountAuthority
CellProfilerPipelineProvenance
PipelineImageSchema
PipelineImageSchemaBuilder
PipelineImageSchemaSourceBindingsRepresentability
materialize_source_mask
ImageTypeSourceRole
ImageTypeSourceRoleClassSpec
ImageTypeSourceRoleSpec
SourceImagePayloadSemantics
SourceImagePayloadRoleStrategy
source_image_payload_role
SOURCE_IMAGE_TYPE_METADATA_FIELD
OpenHCSImageType
SourcePlaneProjection.image_type
SourceSchemaLiteralResolver
SourceFilterSubjectLiteral
SourceFilterOperatorLiteral
SourceBindingMatchMethodLiteral
NamesAndTypesAssignmentLayout
NAMES_AND_TYPES_ASSIGNMENT_LAYOUTS
NamesAndTypesAssignmentBlockStrategy
SourceSchemaImageSetSelection
max_image_set_count
source_schema_image_set_selection
SourceSchemaCandidateProvider
SourceSchemaCandidateDiscoveryMode
SourceSchemaCandidateDiscoveryRequest
SourceSchemaCandidateDiscovery
SourceSchemaCandidateImageSetViability
SourceSchemaImageSetProbe
SourceSchemaImageSetProbeResult
CellProfilerSourceRootResolver
CellProfilerSourceRootCandidate
CellProfilerResolvedSourceRoot
CellProfilerSourcePathAdmission
CellProfilerSourcePathContext
CellProfilerSourcePathExclusion
ImagePlaneSourceResolutionStage
ImagePlaneSourceResolver
SourceSchemaSourcePlaneInventory
TiffPageSourcePlaneInventory
TiffSourcePlaneInventory
SinglePlaneSourcePlaneInventory
SourcePixelRef.to_legacy_workspace_mapping
SourcePixelRef.source_metadata
SourcePixelRef.reader
SourcePixelRef.source_path
SourcePixelRef.series_index
SourcePixelRef.plane_index
SourcePixelRef.source_channel
SourcePixelRef.source_z_index
SourcePixelRef.source_timepoint
BioFormatsPlaneRef.reader
BioFormatsPlaneRef.c
BioFormatsPlaneRef.z
BioFormatsPlaneRef.t
workspace_mapping_source_ref
workspace_mapping_source_path
VirtualWorkspaceSourceRefResolver
PathSourceRefResolver
DiskSourceRefResolver
VirtualWorkspaceResolvedRef
_load_npy_plane
BioFormatsReaderUnavailableError
SourceSchemaAuxiliaryMaterializer
NumpyAuxiliaryMaterializer
SourceSchemaAuxiliaryMaterializationRequest
SourceSchemaAuxiliaryTargetPathRequest
SourceSchemaAuxiliaryTargetPathPolicy
SourceBasenameAuxiliaryTargetPathPolicy
SourceSchemaCandidateIdentity
SourceSchemaCandidate
SourceSchemaCandidateCollection
SourceSchemaCandidateMetadataRequest
SourceSchemaCandidateMetadataResolver
SourceSchemaCandidateMatches
SourceSchemaImageSetIdentity
SourceSchemaVirtualFilename
SourceSchemaFilenameProjection
WorkspaceMappingSink
SourceVirtualPathMetadata
VirtualComponentOriginalMetadataProjection
SourceMetadataJsonRecord
SourcePlaneGroupSiteAllocator
ImageSetMetadataMerge
SharedCandidateMetadataProjection
ComponentProjection
WellRowColumnMetadataProjection
SourceSchemaSingletonWellProjection
ImageNumberSiteProjection
FrameNumberSingletonSiteProjection
ZIndexSingletonSiteProjection
MetadataZIndexProjection
FrameNumberTimepointProjection
SourceSchemaSingletonTimepointProjection
SourceSchemaSingletonZIndexProjection
OrdinalSiteProjection
SourceBindingView.from_schema_assignment
SourceBindingView.payload_type_for_assignment
SourceBindingView.payload_type
StepSourceBindingsConfig.inherits_*
_resolve_step_source_binding_defaults
SourceBindingsViewModel.from_schema_and_bindings
SourceBindingsPreview.from_schema_and_bindings
SourceInventoryBuildRequest
SourceInventoryProvider
ExplicitImagePlaneSourceInventoryProvider
LocalDirectorySourceInventoryProvider
FileManagerSourceInventoryProvider
OpenHCSWorkspaceSourceInventoryProvider
SchemaContextSourceInventoryProvider
SourceSchemaWorkspaceMaterialization
CellProfilerSourceSchemaProjection
CellProfilerSourceSchemaWorkspace
CellProfilerSourceSchemaPreparation
CellProfilerSourceSchemaMaterializer
CellProfilerSourceSchemaMaterializationScope
source_schema
_AUXILIARY_PAYLOAD_CACHE
source_schema_auxiliary_payload
ArtifactDeclarationStepContext.source_provenance
ArtifactDeclarationStepContext.processing_config: Any
FunctionStepInvocationContracts
FunctionStepInvocationContractPayload
FunctionStepInvocationContractBinding
invocation_contracts
InvocationContractProviderLike
public_callable_invocation_contract
CellProfilerGeneratedInvocationContractProvider
CellProfilerGeneratedStepContract
CellProfilerGeneratedGroupedStepContract
CellProfilerGeneratedStepContracts
CellProfilerGeneratedStepFunctionSpec
CellProfilerGeneratedStepContractMatcher
CellProfilerGroupedRuntimeCallable
CellProfilerGroupedModuleContracts
CellProfilerModuleContractResolution
CellProfilerProcessingContractAuthority
PROCESSING_CONTRACT_CACHE
_attach_runtime_processing_contract
cellprofiler_module_callable
rebuild_cellprofiler_runtime_callable
CellProfilerRuntimeCallable.__reduce__
FunctionReferenceRehydrationRequest
FunctionReferenceRehydrator
CellProfilerFunctionReferenceRehydrator
FunctionStepTransportAuthority.approved_code_document_factory_names
CellProfilerModuleExecutor._runtime_plans
CellProfilerModuleExecutor._canonical_module_name
CellProfilerModuleExecutor._primary_image_input_policy
CellProfilerPerObjectMeasurementPolicy
CurrentObjectFeatureVectorAuthority
CellProfilerRuntimeStepBinding
CellProfilerModuleRunRequest.func
CellProfilerModuleRunRequest.plan
CellProfilerModuleRunRequest.input_image
CellProfilerModuleRunRequest.current_image
CellProfilerOutputValueResolutionRequest.func
SourceBindingRuntimeContractGuard
CellProfilerRuntimeCallableFormatter
SpecialOutputKindClassifier
CellProfilerCallableOutputSpecs
RuntimeReturnedOutputMatcher.single_output_value
RuntimeReturnedOutputMatcher.resolve_positional_outputs
RuntimeReturnedOutputMatcher.resolve_from_returned_specs
RuntimeReturnedOutputMatcher.returned_specs_with_retained_tail
RuntimeReturnedOutputMatcher.same_artifact_semantics
CellProfilerCompileTimeArtifactFlow
CellProfilerCompileTimeSettingsRequest
CellProfilerModule.*compile_time_*
CellProfilerModule.image_input_setting_names
CellProfilerModule.object_input_setting_names
CellProfilerModule.image_output_setting_names
CellProfilerModule.object_output_setting_names
CellProfilerModule.spatial_grid_input_setting_names
CellProfilerModule.spatial_grid_output_setting_names
CellProfilerModule.artifact_input_setting_is_repeated
CellProfilerModule.preserve_duplicate_artifact_inputs
CellProfilerModule._active_artifact_context
ModuleArtifactContractItemCollection
ArtifactBindingKeys
InvocationArtifactDeclarationProvider
ObjectMeasurementRowsModule
_partition_public_kwargs
CellProfilerOptionalCurrentImageContext
CellProfilerRequiredCurrentImageContext
CurrentSourceIdentityCacheScope
MappingValueLookup
MeasurementTableAxisProjectionCacheKey
SourceAliasOrderIndexRequest
SourceCandidateMetadataRequest
SourceCandidateRuntimeUniverse
CollectionAttributeProjection
ContextSourceMetadataAuthority.has_metadata_for_any
ParsedSourceCandidateIdentity.from_candidate
CPPipeExecutionValidation
ObjectIntensityResults
ProjectedMeasurementFieldDescriptor
CellProfilerProjectedMeasurementRow
CellProfilerProjectedColumnarRowColumns
ConcatenatedMeasurementColumnarRows
CorrectIlluminationOriginalImageName
CorrectIlluminationCanonicalImageNames
CorrectedImageOutputPlaneStack
AlignCropRequest
AlignGeometryProjection
AlignInputPayloads
AlignAdditionalModePlan
align_offsets
align_offsets_for_cropping
align_offsets_for_padding
alignment_mask
alignment_pixels
crop_mode_outputs
RelationshipEndpointContract
RelationshipEndpointMatches
TwoInputRelationshipEndpointFallback
RelationshipEndpointResolver
RelationshipEndpointResolver.for_request
RelationshipEndpointResolver.endpoint_contract
RelationshipEndpointResolver.artifact_name_matches
RelationshipEndpointResolver.module_relationship_endpoint_contract
RelationshipEndpointResolver.indexed_object_input_contract
RelationshipEndpointResolver.object_input_at
CellProfilerModule.relationship_endpoint_contract
PrimaryObjectInputRelationshipModule
PrimaryObjectInputRelationshipDistanceModule
parent_child_relationship_artifact_endpoints
CellProfilerImageMeasurementSource
ProducedArtifactImageMeasurementSourceBase
ProducedArtifactImageMeasurementSource
UnqualifiedRuntimeImageMeasurementSource
CellProfilerMeasurementFeatureParseCandidate
CurrentStepPayloadSelection
CurrentSourcePlaneProjectionBase
RuntimePlaneCurrentImageContext
RuntimePlaneProjectionContext
RuntimePlaneProjectionRequest
RuntimePlanePayloadProjectionRequest
RuntimePlaneImagePayloadPlaneSelection
RuntimePlaneSelectedImagePayloadPlane
RuntimePlaneSelectedImagePayloadPlanes
RuntimePlaneImagePayloadPlaneSelectionResult
RuntimePlaneImagePayloadSourceContextRequest
RuntimePlaneCurrentImagePayloadPlaneIndex
RuntimePlaneImagePayloadSliceContext
RuntimePlaneImagePayloadProjectedMetadata
RuntimePlaneImagePayloadPlaneIndex
CurrentRuntimePlaneKwargValue
IndexedCalculateMathSettingValue
TypedCalculateMathSettingValue
CalculateMathRepeatedOperandSettings
CellProfilerObjectMeasurementVectorBinding.feature_query
CellProfilerSourceIdentityMixin.primary_source_image_pair
ShapeObjectMeasurementRows.covers_declared_object_measurement_domain
ObjectIntensityMeasurementRows.covers_declared_object_measurement_domain
ObjectColocalizationColumnarMeasurements.covers_declared_object_measurement_domain
ObjectGranularityMeasurementRows.covers_declared_object_measurement_domain
DenseLabelSequenceMemoryBudget
CellProfilerRadialCVExportValue
FirstMeasurementField
PerImageMeasurementProfile
FilterObjectsKwargSettings
FilterObjectsBoundMeasurementInputs
ModuleRevisionRange
GranularityImageSeriesCacheEntry
SourceImagePairCollection
FilterObjectsRelationshipEndpointIds
CellProfilerModuleRuntimePlan.func
RuntimeInvocationOptions
CalculateMathInvocationOptions
DefineGridInvocationOptions
FunctionContractAttribute.runtime_invocation_options_parameter
CallableMetadata.runtime_invocation_options_parameter
CallableContract.runtime_invocation_options_parameter
NormalizedFunctionItem.invocation_options
RuntimeCallableArgumentPlan.invocation_options_parameter_name
_runtime_invocation_options_parameter
CallableInvocationKwargSpec
CellProfilerModule.contract
CellProfilerModule.image_export_specs
CompiledFunctionInvocation.artifact_input_keys
CompiledFunctionInvocation.artifact_output_keys
cellprofiler_source_setting_parameter_mapping
CellProfilerImageNumberCandidateContext
CellProfilerImageNumberMatchedContext
CellProfilerImageNumberMap
CellProfilerImageNumberResolver
CELLPROFILER_IMAGE_NUMBER_MAP_PROCESS_CACHE
cellprofiler_current_step_source_paths
cellprofiler_axis_image_number_start
_cellprofiler_axis_image_number_start
cellprofiler_image_number_start_for_source_paths
cellprofiler_image_number_for_source_paths
cellprofiler_image_number_for_payload
cellprofiler_source_path_for_image_number
cellprofiler_source_paths_for_image_name
cellprofiler_source_order_identity
image_number_start
_source_paths_by_image_name_cache
RuntimeAdapterPrepare
ResolvedRuntimeInputRequest
RuntimeAdapterSpec.prepare
RuntimeAdapterSpec.prepare_request
prepare_compiled_runtime_adapters
compiled_source_binding_context
compile_runtime_adapter_request
prepare_cellprofiler_runtime_adapter
prepare_source_resolution
cellprofiler_ordered_pipeline_image_paths
CELLPROFILER_SOURCE_ORDER_PROCESS_CACHE
SourceOrderCacheValue
setting_parameter_aliases
ModuleSettingCoverageStatus.ARTIFACT_CONTRACT
ModuleSettingCoverageStatus.TYPED_IGNORE
ModuleSettingCoverageStatus.CALLER_IGNORE
ModuleSettingCoverageStatus.INFRASTRUCTURE
ignored_unmapped_settings
artifact_setting_names
typed_ignore_setting_names
compile_time_setting_binding_default_values
compile_time_settings_function
compile_time_public_setting_names
compile_time_public_kwarg_names
compile_time_consumed_kwarg_names
compile_time_grouped_public_kwarg_names
compile_time_coalesced_public_kwarg_names
public_artifact_identity_overrides
CellProfilerArtifactCapability
ArtifactPort
ArtifactPortDeclaration
ArtifactPortContext
ArtifactNameResolver
ArtifactRelationResolver
SourceBoundInputPort
MainFlowInputPort
RuntimeArtifactInputPort
DeclaredArtifactOutputPort
RecordedArtifactOutputPort
SpecialInputPort
SpecialOutputPort
SpecialInputDeclaration
SpecialOutputDeclaration
CellProfilerArtifactSettingDescriptor
CompileTimeArtifactFlow
PipelineMetadataCarrier
PipelineIdentityCarrier
pipeline_metadata
materialize_skipped_save_images
materialize_terminal_images
prune_dead_unmaterialized_artifact_steps
generated_pipeline_path
generated_pipeline_backend
materialized_image_artifact_names
artifact_name_materialized_image_artifact_names
RuntimeArtifactBindingScope
GeneratedPipelineModuleIdentity
GeneratedPipelineFunctionRegistration
GeneratedPipelineModuleExports
GeneratedFunctionSpec
PipelineGeneratorRegistryStage
PipelineGeneratorArtifactPruner
PipelineGeneratorRuntimeContractProjector
PipelineGeneratorBuildStage
PipelineGeneratorCodeEmitter
PipelineGenerator
GeneratedPipeline
GeneratedPipelineRequest
SkippedModuleSelection
GeneratedStepEmission
GeneratedStepEmissionGroup
generated_module_blocks
python_literal
ArtifactContractCommentSection
GeneratedPipelineConfigDefaults
GeneratedProcessingConfigShape
GeneratedStepSettings
GeneratedParameterTarget
GeneratedLiteralScalar
GeneratedLiteralValue
GeneratedStepSettingKey
GeneratedParameterName
GeneratedGroupByComponent
group_by_is_unresolved
variable_component_literal
all_component_literal
coerce_all_component
all_component_tuple_literal
group_by_literal
group_by_component_axis
source_binding_variable_component_literals
variable_component_literals
generated_function_step_semantic_argument_lines
ModuleProcessingComponents
ModuleProcessingComponentRequest
CellProfilerModule.processing_components
CellProfilerModule.with_generated_group_by
CellProfilerModule.generated_group_by
GeneratedGroupByComponentState
SourceProcessingAxisPlan.from_schema
SourceProcessingAxisPlan.without_source_set_components
SourceProcessingAxisPlan.scalar_source_group_component
SourceProcessingAxisPlan.optional_single_image_set_component
SourceProcessingAxisPlan.single_component_for_role
source_identity_group_by_component
_cellprofiler_measurement_target_scope
CellProfilerInvocationOverrideKwarg.measurement_target_scope
runtime_measurement_target_scope
pop_measurement_target_scope
cellprofiler_measurement_scope_selection
SourceProcessingComponentSemantics
RuntimeArtifactProcessingScope
SourceBindingProcessingScope
default_module_processing_components
default_module_requires_pairwise_object_domain_scope
_is_inputless_artifact_only_contract
SourceProcessingAxisRole
SourceProcessingAxisRolePolicy
ModuleProcessingScopePolicy
PreparedGeneratedPipeline
CPPipePipelineArtifact
GeneratedCPPipePipeline
CPPipePipelineGenerationRequest
CPPipePipelinePreparationRequest
CPPipeModulePartition
GeneratedPipelineRuntimeModule
CellProfilerPipelineRuntimeBindingService
CellProfilerImportResultProvider
CellProfilerDialectCompiler
CellProfilerPipelineImporter
CellProfilerGeneratedPipelineDialectCompiler
register_cellprofiler_dialect_compiler
register_generated_cellprofiler_dialect_compiler
register_benchmark_cellprofiler_dialect_compiler
get_cellprofiler_dialect_compiler
clear_cellprofiler_dialect_compiler
BenchmarkCellProfilerDialectCompiler
CellProfilerGeneratedPipelineImporter
partition_cppipe_modules
prepare_generated_pipeline
SetupModuleCompiler
SourceImageStackPlanDeclaration
SourceFilterCriteriaParser
SourceBindingMatchMetadataParser
SourceBindingOriginPolicy
SettingNameFamilySpec
ResolvedModuleFunction
resolve_semantic_function
CellProfilerDebugView
DefaultCellProfilerDebugView
CellProfilerSemanticDefaultContract.__registry__
InfrastructureCellProfilerModule
CellProfilerModuleSemantics
CellProfilerModuleSemanticTraits
CellProfilerModuleSemanticFamily
CellProfilerModuleCategory
CellProfilerModuleDimensionality
cellprofiler_module_semantics
cellprofiler_module_semantics_family
ResolvedProcessingContract
ProcessingContractResolutionSource
resolve_processing_contract
CPPipeInfrastructureProfile
infrastructure_import_note
infrastructure_exports_tables
infrastructure_exports_images
infrastructure_retained_artifacts
CellProfilerExecutionExportContext
CellProfilerAnalystExportRequest
pending_pipeline_export
AnalysisConsolidationPlan
RuntimeImageExportSpec
RuntimeImageExportBitDepth
_candidate_image_snapshots_for_equivalence
LoadDataModule
LabelImagesModule
CreateBatchFilesModule
SaveCroppedObjectsModule
CellProfilerFunctionRuntimeMetadata
CellProfilerFunctionCatalog
CellProfilerFunctionReferenceTransportStrategy
FunctionReferenceTransportStrategy
CELLPROFILER_MODULE_ATTR
AbsorbedFunctionMetadata
_make_processing_wrapper
TupleMemberTypeValidation
DisabledPathMetadataRulePolicy
DisabledMetadataAxisComponents
RequireProcessingContextBoundaryPolicy
RuntimeShapeInspection
RuntimeArtifactRecordDeduplication
SourceIdentitySetCardinality
DeclaredOutputResolution
CellProfilerOptionalNonemptyString
SourceBindingAxisCardinality
SourceImageSetIdentityQuality
InvocationSpatialRankCandidates
DenseLabelShapeSet
DenseLabelStackRepeatPattern
MatlabPayloadEntryName
SpatialGridSliceCount
RuntimePlaneSelectedPlaneIndex
CellProfilerImageNumberResolution
SourceBindingPayloadAliasSet
SourceBindingPayloadComponentMetadata
ObjectLabelFinalLabels
ObjectLabelSmallRemovedLabels
SparseLabelRowsCoercion
Pure2DSliceCountCandidate
Pure2DTraceLabelStats
PipelineStepsBoundary
PipelineStepsCarrier
PipelineStepsNamespaceProjection
PipelineObjectStateBinding.pipeline
PipelineObjectStateBinding.pipeline_for_plate
PipelineObjectStateBinding.pipeline_declaration
PipelineObjectStateBinding.update_plate_pipeline
PipelineObjectStateBinding.replace_pipeline
PipelineObjectStateBinding.registered_plate_pipelines
ZMQPipelineSourcePayload
ZMQPipelineCodeTransport
PycodifiedPipelineStepSource
PycodifiedSource
PycodifiedPipelineCode
PycodifiedConfigSource
PycodifyAssignmentSourceRequest
PipelineIdBoundary
ExecutionPipelinePayload
ExecutionPipelineDefinitionProvider
DraftPipelineDefinitionProvider
PycodifiedSourcePipelineDefinitionProvider
OpenHCSExecutionConfigCarrier
ZMQResolvedConfig
RuntimeExecutionCacheWritePolicy
_RuntimeExecutionCacheHit
_runtime_execution_cache_key_matches
runtime_execution_cache_manifest
runtime_execution_cache_key
reuse_runtime_execution_cache
_load_runtime_execution_cache
_write_runtime_execution_cache
reused_runtime_execution_cache
_run_or_reuse_cached_openhcs
_cached_benchmark_result
_write_benchmark_result_cache
_cached_metric_values
_openhcs_cache_key
_openhcs_execution_cache_key
reuse_openhcs_cache
force_openhcs_run
reused_cached_output
CachedNativeReferenceTimingPolicy
DirectPipelineExecution
DirectExecutionProgressBridge
DirectExecutionProgressSink
execute_pipeline_direct
BioFormatsHandler.get_primary_backend
SourceBindingContext.import_result
StepSourceBindingsConfig.can_inherit_from
StepSourceBindingsConfig.resolved_against
resolve_step_source_bindings
resolve_effective_step_source_bindings
bindings_for_group_key
for_group_key
SourceBindingCandidateSourceRef
ObjectMeasurementVectorDomain
dense_object_label_declared_or_extent_id_domain
dense_object_label_extent_id_domain
collapse_singleton_image_stack
ObjectLabelDataRuntimeSliceStackContract
SparseIJVLabelRowsRuntimeSliceStackContract
DenseArrayLabelRuntimeSliceStackContract
ObjectLabelRuntimeSliceStackContract
ObjectLabelContainerRuntimeSliceStackContract
ObjectLabelPayloadRuntimeSliceStackContract
ObjectLabelSetRuntimeSliceStackContract
ColocalizationMaskRequest
ColocalizationMaskStrategy
SpatialColocalizationMaskStrategy
ImageStackColocalizationMaskStrategy
ChannelLeadingColocalizationMaskStrategy
ReplicatedChannelMonochromeProjection
CurrentPlaneObjectLabelProjection
CurrentImageObjectLabelPlaneAlignment.aligned_dense_value
SpecialInputBindingRequest.object_label_runtime_value
SpecialInputBindingRequest.current_plane_object_label_runtime_value
SpecialInputBindingRequest.object_label_payload
SpecialInputBindingRequest.current_image_aligned_object_label_runtime_value
SourceSpatialAlignedKwargResolutionStrategy
ImageMetadataPayloadAlignedKwargResolutionStrategy
MaskedImagePayloadAlignedKwargResolutionStrategy
ObjectLabelPayloadAlignedKwargResolutionStrategy
ObjectLabelSetAlignedKwargResolutionStrategy
AlwaysMatchingAlignedKwargResolutionMixin
AlignedImageStackKwargResolutionStrategy.for_value
AlignedImageStackKwargResolver.domain_adapter
AlignedImageStackKwargResolver.reference_domain
SpecialInputBindingRequest.parameter_spec_groups
CurrentRuntimePlaneKwargProjectionContract.requires_projection_capability
CurrentRuntimePlaneKwargProjection.required_runtime_slice_projection
RawObjectLabelOutputValueContextStrategy
CellProfilerObjectLabelOutputContextStrategy
ContextualCellProfilerObjectLabelOutputStrategy
NumpyCellProfilerObjectLabelOutputStrategy
OpaqueCellProfilerObjectLabelOutputStrategy
CellProfilerRuntimeAdapter.add_source_image_objects
```

Static import tests also enforce:

- every file named by a phase's **Required deletions** list is absent from the
  filesystem after that phase; no empty module, re-export facade, or dormant
  definition satisfies deletion;
- all gates scan production and tests; only the canonical deletion assertion
  stores forbidden names as data;

- `openhcs/core` imports no concrete CellProfiler backend module;
- generic compiler code contains no CellProfiler module name;
- the import pass and compiler contain no duplicated list of module names;
- setup and processing modules are entries in the same
  `CellProfilerModule.__registry__`;
- callable execution scope is read only from `CallableContract`; worker and
  post-plate executors contain no CP module-name or exporter-name checks;
- `RuntimeArtifactBatch` contains only specs and records selected by the
  compiled plate-scoped step contract and exposes no runtime store;
- generic image/file-bundle writers contain no CP imports, while CP exporter
  modules perform no direct filesystem writes;
- callable ownership is never copied to a callable attribute, metadata DTO, or
  function-to-module dictionary;
- AST inspection requires parsed-module `CellProfilerModule.resolve_function`
  to accept `ModuleBlock`, `ModuleArtifactContract`, and
  `StepSourceBindingsConfig` and forbids a `ProcessingConfig` parameter;
  `CellProfilerModule.processing_config` requires `CallableContract` and the
  inherited concrete `ProcessingConfig`, and the public compiler contains no
  call to `resolve_function`; parsed import resolves candidate
  `LazyStepSourceBindingsConfig` directly under the pipeline config and creates
  no FunctionStep before canonical callable selection;
- AST inspection requires both module-processing helper files and every
  source-axis summary, generated literal, component request/result,
  lineage/scope wrapper, or processing policy class to be absent;
- `ModuleArtifactContract` contains artifact items only and contains no required
  variable components or allowed-grouping fields;
- `CallableContract` is the sole owner queried for required variable components
  and allowed groupings;
- `ArtifactDeclarationStepContext` contains exact `source_bindings`, `group_by`,
  `input_source`, available artifact/producers, and main-flow fields and contains
  no `ProcessingConfig` field;
- `StepSnapshot` contains only compiler identity and the resolved `step`; the
  compiler and path planner consume `snapshot.step` configs directly;
- generated pipeline source is produced only by the generic FunctionStep
  pycodify authority;
- AST inspection requires `import_cellprofiler_pipeline` to return exactly
  `tuple[list[FunctionStep], PipelineConfig]` and forbids
  `CellProfilerPipelineImportRequest`, `CellProfilerPipelineImportResult`,
  `CellProfilerModuleReference`, `CellProfilerPipelineProvenance`,
  `import_service.py`, `import_records.py`, source rendering, and source writes
  anywhere in the import module;
- every workspace mapping value round-trips through `SourcePixelRef`; virtual
  workspace resolution uses one exact existing-backend lookup and contains no
  string-path branch, resolver family, shape predicate, priority traversal,
  old-name alias, or dual-format migration path;
- AST inspection requires `SourcePixelRef` to declare exactly `backend`,
  `backend_address`, and `source_axis_indices`; `BioFormatsPlaneRef` declares
  exactly physical path, series index, and plane index; the Bio-Formats backend
  contains no workspace-metadata read or alternate reader branch, and its
  handler contains no primary-backend override;
- AST inspection requires `SourceCandidate` to declare exactly `source_ref`,
  `relative_path`, `metadata`, `source_axis_shape`, and
  `source_filter_paths`; no candidate path, backend, or duplicate axis-index
  field exists, and workspace projection consumes `source_ref` directly;
- `FileManager.__setstate__` contains no global `storage_registry` reference;
  every direct backend mutation uses `register_backend`, worker reconstruction
  preserves the submitted registry key set, and `VirtualWorkspaceBackend`
  receives that same local mapping through nominal `bind_registry` dispatch;
- `SourceComponentProjectionStrategy` inherits
  `EnumKeyedStrategyMixin[AllComponents]`, its registry keys equal
  `set(AllComponents)`, and its root contains no metadata alias, component
  switch, priority, or leaf default;
- AST inspection requires `CellProfilerInvocationContractProvider` to be
  frozen/slotted with only the immutable `plans` field and the exact typed
  `__call__` signature; generic composite provider code collects all claims and
  contains no early first-claim return, callable provider alias, default
  provider function, or `Any` provider/session parameter;
- generic compiled invocation kwargs pass
  `CallableContract.validate_public_kwargs` once; CP runtime contains no
  `inspect.signature`, unsupported-kwarg filtering, or kwarg-spec type;
- `CompiledFunctionInvocation` carries exact input/output ref tuples and no
  string artifact-key fields; active runtime contract selection uses
  `ModuleArtifactContract.require_items_for_specs` plus complete-spec equality and no name
  intersection;
- source workspace construction uses the exact submitted VFS root and contains
  no parent/child inference, `.cppipe` placement probe, candidate bucket, or
  path-exclusion policy registry;
- every dataclass field on `SourceBindingsConfig` is visible through the
  generated lazy step config and resolves into the snapshot's one
  `StepSourceBindingsConfig`; source-binding code contains no manually mirrored
  inherited-field list, per-field overlay, or equivalence predicate;
- scoped source selection requires exact component plus key and raises on zero
  matches; untyped group-key methods and sole/all-binding scoped fallbacks are
  absent;
- returned-output matching receives full ordered declared specs from the
  compiled module contract and never reclassifies callable special outputs;
- AST inspection rejects every materialization-bearing `special_outputs`
  tuple on a registered CellProfiler callable; those decorators declare slot
  names only;
- the axis-scoped CP runtime callable owns one executor and the executor owns
  one three-field immutable plan; the wrapper constructor receives raw callable
  plus enriched `CallableContract`, while generic `FunctionReference` plus
  `CallableContract` are the only serialized identities;
- AST inspection rejects `__reduce__`, identity `__eq__`, and `__hash__` on
  `CellProfilerRuntimeCallable`;
- runtime callable resolution performs one exact optional-factory call on
  `RuntimeAdapterSpec` and contains no rehydrator registry, supports scan, or
  domain fallback;
- the axis-scoped CP runtime plan contains exactly resolved raw callable,
  module type, and enriched callable contract; it contains no
  copied module/processing contract, function name, artifact collection, or
  signature/kwarg specification or policy field, and the executor has no plan dictionary or copied
  contract/name/policy field;
- the module run request contains one image field, no copied plan, and obtains
  callable/name facts from its executor's runtime plan;
- no one-field CP runtime type exists solely to expose a scalar predicate,
  normalization, optional value, or existing-authority projection;
- ZMQ and agent pipeline records carry direct step lists and no carrier class;
- agent pipeline session request leaves construct the concrete definition
  directly and use plain string identity, with no provider family;
- generated source contains no `ModuleArtifactContract`, runtime callable,
  module number, hidden metadata key, or sidecar payload.
- code-mode contains no CP factory allowance hook.
- production and tests contain no import of `benchmark.cellprofiler_library` or
  `benchmark.cellprofiler_compat`.

## Compile-Time Failure Rules

Compilation fails rather than guessing whenever public declarations do not
determine one exact contract. The compiler never:

- recovers from a missing module declaration by using callable-name rules;
- selects or replaces a public callable from `ProcessingConfig`, or accepts a
  parsed-module selector result that is not the declaration's exact canonical
  callable;
- chooses among candidate contracts;
- infers artifact type from a string;
- reads a `.cppipe` after public source reconstruction;
- reads retained source-schema/import state instead of the submitted
  `SourceBindingsConfig`;
- imports generated sidecar state;
- accepts a runtime wrapper in place of a public callable;
- mutates upstream artifact materialization to emulate an omitted module;
- aligns incompatible component identities by truncating or broadcasting;
- compiles a plate-scoped callable with an axis-scoped successor, dict pattern,
  source image payload load, or missing/ambiguous artifact input;
- allows plate-scoped plans or kwargs to drift between compiled axis contexts.
- accepts an ImageFileOptions suffix not owned by a registered
  `ImageFileSerializationFormat`.
- accepts source channel counts without a declared channel axis, duplicate
  source aliases, or a runtime artifact kind inconsistent with its binding;
- accepts an unresolved/unsupported `ImagePlaneSource` URI in converted public
  config;
- carries any private source-image-set cap beside `WellFilterConfig`.

Workspace initialization, before worker execution, rejects a loaded source
whose declared source-stack axis count, channel axis, or channel count is
incompatible with its actual array rank/shape. This remains data-dependent and
does not become a runtime worker fallback.

## Review Checklist

Implementation review is complete only when every item is true:

- [ ] Public execution input is exactly `PipelineConfig` plus
      `list[FunctionStep]`.
- [ ] Generated source recreates contracts in a fresh ZMQ process.
- [ ] Artifact semantics are represented once by existing artifact and module
      contract types.
- [ ] Callable special I/O owns ABI slots and does not infer artifact kind.
- [ ] Each CP leaf behavior is owned by its nominal module declaration or
      existing MRO mixin.
- [ ] Setup and processing modules share the one `CellProfilerModule` registry.
- [ ] Setup modules fold into the existing lazy `SourceBindingsConfig`; no
      `PipelineImageSchema`, source-schema side state, or importer provenance is
      needed to recreate the workspace.
- [ ] Source workspace construction uses exact submitted bindings, generic VFS
      refs, and complete-universe materialization; only `WellFilterConfig`
      narrows execution.
- [ ] Generic source projection/runtime metadata contains no CP image-type
      token, TIFF branch, provider retry, or auxiliary format registry.
- [ ] The module declaration returns implementation functions without a
      catalog, copied ownership, wrapper call layer, or function-namespace
      mutation.
- [ ] Parsed import derives contract and source axes before callable selection,
      derives processing config from the selected callable contract afterward,
      and public compilation never substitutes the declared callable.
- [ ] CellProfiler import and compilation never call
      `attach_callable_contract_metadata`; immutable compiled contracts carry
      every compiler-derived module fact.
- [ ] Generic core contains no concrete CP module knowledge.
- [ ] The symbol table, generated matcher, hidden FunctionStep contracts,
      policy mirror, grouped wrapper, and pipeline metadata carrier are gone.
- [ ] Source-binding and config inheritance eliminate repeated generated
      declarations.
- [ ] Import output is real FunctionSteps/config and generic pycodify is the
      sole source renderer.
- [ ] Client, agent, and server pipeline boundaries use direct step lists.
- [ ] Plain patterns cover identical all-group behavior; dict patterns encode
      real group differences only.
- [ ] SaveImages is an explicit public step.
- [ ] Spreadsheet and database exporters are explicit plate-scoped public steps
      that consume exact contract-selected batches.
- [ ] ExportToDatabase writes a real SQLite database and CPA properties; no
      value-only parity case can pass without validating those files.
- [ ] Invalid artifact topology fails at compile time.
- [ ] UI-equivalent and benchmark source are identical.
- [ ] Benchmark OpenHCS results and timings never come from a prior runtime
      execution cache.
- [ ] Speedup uses measured native timing and never substitutes a timeout.
- [ ] Official30 passes strict parity through ZMQ.
- [ ] Official30 completes with inherited non-persistent Napari streaming.
- [ ] Every file and symbol under Required deletions is absent from production
      and tests; no alias, facade, old payload reader, state upgrader, dormant
      registry entry, or retained fixture preserves the removed architecture.

## Resulting Data Flow

```text
.cppipe ModuleBlock sequence
        |
        +-> setup declarations -> SourceBindingsConfig
        +-> executable declaration -> ModuleArtifactContract
        +-> resolved source bindings -> resolve_function -> canonical raw callable
        +-> raw callable -> CallableContract axis/group constraints
        +-> artifact contract + callable contract + inherited config
            -> ProcessingConfig
        |
        v
PipelineConfig + list[FunctionStep]
        |
        +-> SourceBindingsConfig -> complete generic source-binding workspace
        +-> WellFilterConfig -> compiled execution-axis selection
        |
        v
ObjectState/config inheritance -> resolved FunctionStep -> StepSnapshot.step
        |
        v
normalize_function_pattern -> FunctionInvocationKey
        |
        v
CellProfilerInvocationContractProviderFactory
        |
        +-> CellProfilerModule.__registry__ -> declaration-loaded callable
        +-> ArtifactDeclarationStepContext
            (source bindings, group_by, input_source, artifacts/producers/main flow)
        +-> transient ModuleBlock setting rows
        |
        v
CallableContract + artifact-only ModuleArtifactContract
        |
        v
generic ArtifactGraph / PipelinePathPlanner
        |
        v
CompiledFunctionInvocation -> derived uniform FunctionStepExecutionScope
        |
        +-> AXIS: generic FunctionStep runtime
        |          -> CellProfilerRuntimeAdapter only when declared
        |          -> raw generic callable when adapter-free
        |          -> RuntimeReturnedOutputMatcher
        |          -> generic artifact persistence/main flow
        |
        +-> PLATE: merged exact RuntimeArtifactBatch
                   -> raw plate callable once
                   -> generic FileBundleOptions materialization
```

The `.cppipe` parser is an importer into this public model. It is not a second
runtime authority. After import, CellProfiler-backed pipelines compile and run
through the same OpenHCS semantics as native pipelines. Axis callables that
operate a CellProfiler workspace use one module-specific adapter; adapter-free
axis callables such as SaveImages remain raw generic functions. Plate-scoped
exporters use the generic plate lifecycle and materialization boundary without
a CP runtime adapter.
