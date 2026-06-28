# Plan: Source Bindings Microscope Handler

## Problem

OpenHCS cannot init plates of arbitrary images that don't match a recognized
microscope format. The init flow requires auto-detection to succeed:

```
orchestrator.initialize()
  → auto-detect microscope handler (needs metadata or recognizable filenames)
  → handler.initialize_workspace() (builds virtual workspace mapping)
  → cache_component_keys() (parses filenames into component indices)
```

Source bindings exist but are runtime-only post-discovery filters. They can't
provide files from scratch.

## Key Insight

The source-binding config family already has the full vocabulary needed for
init-time file discovery:

- `ComponentSelector` — explicit component mapping (well="A01", channel="1")
- `MetadataExtractionRule` — regex extraction from FILE_NAME or FOLDER_NAME
- `SourceFilterClause` — file/directory/extension filtering
- `SourceBindingMatchPlan` — cross-alias image-set matching

No new source-binding vocabulary is needed. Add a regular
`SourceBindingsConfig` for pipeline/plate scope, then make
`StepSourceBindingsConfig(SourceBindingsConfig)` the step specialization, the
same way `StepWellFilterConfig` inherits `WellFilterConfig`:

- pipeline/plate `SourceBindingsConfig` is declaration data only: source
  filters, bindings, metadata rules, and match plans;
- pipeline/plate `microscope=Microscope.SOURCE_BINDINGS` selects the registered
  source-binding microscope handler and uses `SourceBindingsConfig` to initialize
  an arbitrary source folder as an OpenHCS virtual workspace;
- step `StepSourceBindingsConfig` inherits the same typed source-binding payload
  fields. The concrete step class may carry `enabled: bool = False`, but this
  plan does **not** add a compiler `enabled` gate. Runtime source-binding
  semantics remain owned by the resolved typed binding payload and the existing
  `CompiledSourceBindingPlan.from_config()` path.

This intentionally uses the existing `@global_pipeline_config` inheritance
pattern rather than introducing `PlateSourceBindingsConfig` or a separate
handler config.

## Existing Config Architecture

The existing pattern is not a separate runtime lookup table. It is the
ObjectState/global config path:

1. `GlobalPipelineConfig` owns concrete config defaults.
2. `PipelineConfig` is generated as the lazy counterpart of
   `GlobalPipelineConfig`.
3. `@global_pipeline_config` injects each concrete config into
   `GlobalPipelineConfig`, injects the generated lazy config into
   `PipelineConfig`, and creates `Lazy<ClassName>` in the concrete config's
   module.
4. `ObjectState.to_saved_resolved_object()` reconstructs resolved concrete
   dataclasses for saved runtime/compiler snapshots without callers naming
   fields by dotted strings.
5. Step constructor fields must use the generated lazy step config for
   global/pipeline/step inheritance to participate.
6. The compiler registers global, pipeline, orchestrator, and step ObjectState
   scopes once, resolves the pipeline definition once, and then snapshots saved
   resolved values from that hierarchy. New configs must enter through this same
   registration/resolution pass; they should not be remembered as side data by
   generator, UI, MCP, or runtime callers.

`GlobalPipelineConfig`/`PipelineConfig` already carry the `microscope` field.
Source-binding handler activation belongs there, not on the regular
`SourceBindingsConfig`.

For source bindings, the concrete classes live in
`openhcs.core.source_bindings`, but the public config surface should re-export
the generated lazy classes from `openhcs.core.config` alongside the other core
config objects.

### Compiler Step-Config Consumption Pattern

The compiler's current step-config pattern is already the right one for source
bindings:

1. `compile_pipelines()` resolves the submitted pipeline once before per-axis
   context compilation. `_compile_global_config_state()` registers the saved
   `GlobalPipelineConfig`, `_pipeline_config_state()` registers the selected
   plate `PipelineConfig` under the plate scope, the orchestrator state is
   registered under that pipeline config state, and
   `_register_pipeline_step_states()` registers each step under the orchestrator
   state.
2. `_replace_pipeline_with_resolved_steps()` calls `step_state.to_object()` once
   for each submitted step and replaces the pipeline definition with those
   ObjectState-resolved step objects.
3. `_filter_enabled_steps()` builds `StepSnapshot` records from the resolved step
   objects plus their step ObjectStates. `StepSnapshot.from_resolved_step()` reads
   saved resolved values for `enabled`, `processing_config.*`,
   `source_bindings`, `step_materialization_config`, injectable params, and
   every registered `WellFilterConfig` root. This is the compiler-owned bridge
   from ObjectState into typed compile data.
4. Each per-axis context reuses that already-resolved step list, step-state map,
   and snapshot tuple through `CompilationSession.from_context()`.
5. `initialize_step_plans_for_context()` then runs the compile stages in this
   order: ensure a typed `CompiledStepPlan`, configure input conversion if
   needed, path-plan I/O, supplement non-I/O step fields, then collect streaming
   configs.
6. `_supplement_step_plans()` copies `snapshot.variable_components`,
   normalized `snapshot.group_by`, `snapshot.input_source`, and
   `snapshot.processing_config` into the `CompiledStepPlan`, and builds
   `current_plan.source_binding_plan` with
   `CompiledSourceBindingPlan.from_config(snapshot.source_bindings)`.
7. Registry-style step configs follow the same saved-ObjectState rule:
   `_collect_streaming_configs()` iterates `StreamingConfig.__registry__` and
   rebuilds enabled streaming configs from `step_state.get_saved_resolved_value`;
   `_resolve_step_axis_filters()` iterates `snapshot.well_filters`, which were
   discovered from the step ObjectState type map.

Consequences for this plan:

- Do not add a second source-binding compiler pass, generator-side cache, MCP/UI
  side channel, or runtime caller-managed config object.
- `SourceBindingsConfig` enters the pipeline through `@global_pipeline_config`
  and is consumed by orchestrator init through
  `orchestrator.get_effective_config().source_bindings_config`.
- `StepSourceBindingsConfig` enters steps as the generated lazy step config;
  ObjectState resolution must deliver a concrete `StepSourceBindingsConfig` to
  `StepSnapshot.source_bindings`.
- Runtime filtering is compiled only at the existing insertion point:
  `CompiledSourceBindingPlan.from_config(snapshot.source_bindings)`.

## Architecture

### Current

| Phase | Mechanism | Input | Output |
|-------|-----------|-------|--------|
| Init | `microscope_handler` | plate folder | virtual workspace mapping |
| Runtime | `source_bindings` | discovered files | filtered files |

### Proposed

```
SourceBindingWorkspaceProjector (new init-time adapter)
  ├── Uses existing metadata_from_rules/source filter/component projection code
  ├── Emits SourceProjectionSet metadata with SourceSchemaFilenameParser virtual names
  └── SourceBindingsHandler writes that metadata and registers virtual_workspace

Runtime source filtering stays on the existing source-binding runtime path.
Plate-level inheritance is an intended feature: steps default to the generated
lazy `StepSourceBindingsConfig` surface, preserve raw `None` sentinels for
inherited payload fields, and receive resolved concrete `StepSourceBindingsConfig`
values through `StepSnapshot`. Do not add a separate `enabled` workaround in the
compiler, and do not create a parallel runtime side channel.
```

## Dry Run Against Current Code

The concrete init path after the change is:

1. User/agent applies:

   ```python
   PipelineConfig(
       microscope=Microscope.SOURCE_BINDINGS,
       source_bindings_config=SourceBindingsConfig(...),
   )
   ```

2. `PipelineOrchestrator.initialize_microscope_handler()` must read the
   effective global + pipeline config. The current global-only lookup would not
   see `PipelineConfig.microscope`.
3. `create_microscope_handler(..., microscope_type="source_bindings",
   source_bindings_config=...)` constructs `SourceBindingsHandler`.
4. `SourceBindingsHandler.initialize_workspace()` lists disk image files
   recursively, builds a `SourceProjectionSet`, writes OpenHCS metadata under
   `FIELDS.DEFAULT_SUBDIRECTORY`, registers `Backend.VIRTUAL_WORKSPACE`, and
   returns the plate root as `input_dir`.
5. `cache_component_keys()` lists files through the virtual workspace backend.
   This requires a real parser. Use `SourceSchemaFilenameParser`; do not leave
   `parser=None`.
6. The generated virtual filenames parse as canonical OpenHCS component names,
   so pattern discovery and ordinary function steps run against the virtual
   workspace.
7. Runtime source filtering continues through the existing
   `CompiledSourceBindingPlan.from_config(snapshot.source_bindings)` path. Do
   not add an `enabled` guard in that conversion as a workaround for inherited
   pipeline-level source-binding declarations.

Validated with a live dry run using existing classes:

- `metadata_from_rules("img_A01_s1_ch2.tif", ...)` extracted well/site/channel.
- `ComponentProjection.resolve(...)` filled z/timepoint defaults as `1`.
- `SourceProjectionSet.metadata_dict(..., parser=SourceSchemaFilenameParser())`
  emitted virtual paths like `A01_s001_w2_z001_t001.tif`.
- The emitted `workspace_mapping` used structured `SourcePixelRef` payloads.
- `SourceSchemaFilenameParser.parse_filename(...)` parsed those virtual names.
- A missing channel fails loud with
  `Could not project source metadata fields ... onto OpenHCS component 'channel'`.

Validated with a second live dry run through the OpenHCS handler and virtual
workspace path:

- Created a temporary plate with source files under `raw/`.
- Serialized `SourceProjectionSet.metadata_dict(...)` under
  `FIELDS.DEFAULT_SUBDIRECTORY` using `AtomicMetadataWriter`.
- `OpenHCSMicroscopeHandler.initialize_workspace(...)` accepted the metadata,
  selected the plate root as `input_dir`, and registered `Backend.VIRTUAL_WORKSPACE`.
- `OpenHCSMicroscopeHandler.get_primary_backend(...)` returned
  `virtual_workspace`.
- `FileManager.list_image_files(input_dir, "virtual_workspace", ...)` returned
  absolute virtual paths like
  `/tmp/.../A01_s001_w2_z001_t001.tif`.
- The dynamically loaded parser was `SourceSchemaFilenameParser`, and it parsed
  those virtual filenames into well/site/channel/z/timepoint components.
- `VirtualWorkspaceSourceProjectionAuthority(...).projection_or_empty()` mapped
  both relative and absolute virtual paths back to the original disk source
  paths and exposed them through `pipeline_start_files()`.

The dry run also exposed two practical contracts for tests and examples:

- In standalone scripts, import `openhcs` before direct `polystore` imports so
  checkout-local externals are activated before packages like `arraybridge` are
  loaded from `.venv`.
- Regex examples should be tested exactly as written; over-escaping the dot in
  `\.tif` produced empty metadata and the expected fail-loud channel projection
  error.

Validated selector-assigned channels with a third live dry run:

- Used filenames `nuclei_A01_s1.tif` and `membrane_A01_s1.tif` whose metadata
  rule extracts only well/site.
- Existing `SourceFilterClause` + `source_filters_match(...)` selected the
  `nuclei` and `membrane` bindings by filename prefix.
- Existing `ComponentSelector` values supplied channel `1` and channel `2`.
- `ComponentProjection.resolve(...)` produced canonical addresses
  `A01_s001_w1_z001_t001.tif` and `A01_s001_w2_z001_t001.tif`.
- `SourceProjectionSet.metadata_dict(...)` preserved `source_alias` values in
  `source_projection`.
- A metadata/selector conflict such as metadata channel `9` plus selector
  channel `1` failed loud with a targeted conflict error.

Validated the orchestrator config boundary with a fourth live dry run:

- Started with saved global `GlobalPipelineConfig(microscope=Microscope.AUTO)`.
- Constructed a plate-local `PipelineConfig(microscope=Microscope.OPENHCS)`.
- `ObjectState(pipeline_config).to_saved_resolved_object()` returned a concrete
  `GlobalPipelineConfig` with `microscope=OPENHCS` while preserving inherited
  global values such as `num_workers`.
- A monkeypatched `create_microscope_handler(...)` saw
  `microscope_type="openhcsdata"` from `initialize_microscope_handler()`, proving
  the per-pipeline microscope selection reaches the handler factory.
- This replaces the `_create_merged_config(...)` approach; that helper dropped
  scalar lazy overrides in the dry run and should not be extended.

Validated the planned handler-to-viewer path with a fifth live dry run:

- Created a temporary `SimulatedSourceBindingsHandler` whose
  `initialize_workspace()` scans real TIFF files, projects source bindings into
  `SourceProjectionSet`, writes OpenHCS metadata, and registers
  `virtual_workspace`.
- Used real source TIFFs named `nuclei_A01_s1.tif` and
  `membrane_A01_s1.tif`; metadata rules extracted well/site and binding
  selectors assigned channels `1` and `2`.
- `metadata_handler.get_image_files(plate_path, all_subdirs=True)` returned the
  generated virtual names `A01_s001_w1_z001_t001.tif` and
  `A01_s001_w2_z001_t001.tif`, which is exactly the Image Browser listing
  source.
- `PipelineOrchestrator.cache_component_keys()` listed through
  `virtual_workspace` and cached well `A01`, site `1`, channels `1`/`2`,
  `z_index=1`, and `timepoint=1`.
- `ViewerStreamingSource.load_image(..., read_backend="virtual_workspace")`
  loaded the original source TIFF pixel data through the generated
  `workspace_mapping`, proving the viewer streaming read path resolves virtual
  names back to physical source images.
- The browser may still show `size=N/A` for virtual files unless size display is
  taught to query the active backend, because the current UI checks
  `plate_path / virtual_filename` on disk. This is a display artifact, not a
  listing or streaming blocker.

Validated `openhcs_metadata.json` synchronization with a sixth live dry run:

- Seeded a temporary metadata file with source-binding primary metadata for
  `raw/a.tif` and `raw/b.tif`, plus an unrelated `analysis` subdirectory.
- Rewrote `FIELDS.DEFAULT_SUBDIRECTORY` with a new projection set for
  `raw/c.tif`.
- Complete projection fields were replaced for the managed primary subdirectory:
  `image_files`, `workspace_mapping`, `source_metadata`, and
  `source_projection` contained only the new virtual plane and no stale
  `raw/a.tif` or `raw/b.tif` references.
- The unrelated `analysis` subdirectory was preserved.
- A follow-up dry run exposed one important gap:
  `AtomicMetadataWriter.merge_subdirectory_metadata(...)` intentionally merges
  `available_backends`,
  so a stale backend such as `zarr=True` survives if source-binding init uses the
  merge path. Source-binding init therefore needs a replace-subdirectory write
  contract, not additive merge semantics, for the subdirectory it owns.

Validated source-schema parity gaps with a seventh live dry run:

- Existing source-schema workspace materialization correctly order-matched
  `DAPI_001.png` and `Actin_001.png` into one image set:
  `A01_s001_w1_z001_t001.png` and `A01_s001_w2_z001_t001.png`.
- A per-file source-binding projector would not preserve that pairing unless it
  explicitly assembles source aliases into image sets before assigning OpenHCS
  sites/wells/timepoints.
- The current source-schema path also expands source-plane inventories, such as
  multi-page TIFFs, before projection. A one-source-file-to-one-projection
  handler would silently lose those planes.
- `PipelineImageSchema.source_image_stack` is part of `is_empty` and feeds
  CellProfiler axis semantics. Generated pipeline handling must preserve it or
  route through the existing source-schema workspace path.
- `PipelineImageSchema.to_source_bindings_config()` now provides the typed
  representable conversion surface for generator/import work. It preserves
  source filters, metadata rules, match plans, and source assignments as
  `SourceBindingsConfig`.
- `PipelineImageSchemaSourceBindingsRepresentability` now fails loud for schema
  features that still require full source-schema workspace materialization:
  `image_plane_sources`, `imported_metadata_tables`, `source_image_stack`, and
  `grouping`.
- `GeneratedPipelineModuleExports` currently exposes only `pipeline_steps`; a
  generated `pipeline_config` export must be read through that import boundary
  and installed as the pipeline's ObjectState-owned config root, not carried as a
  separate runtime side channel.
- The `WellFilterConfig` / `StepWellFilterConfig` dry run confirmed that
  inherited step-specialized fields become `None` before ObjectState resolution.
  `StepSourceBindingsConfig` normalization must preserve those `None` inheritance
  sentinels until the compiler receives a resolved concrete config.

Validated the compiler step-config consumption pattern with an eighth live
read-through:

- `PipelineCompiler._register_and_resolve_pipeline_once(...)` registers global,
  pipeline-config, orchestrator, and step ObjectStates in one hierarchy, then
  replaces the submitted pipeline definition with `step_state.to_object()`
  results once before per-axis compilation.
- `StepSnapshot.from_resolved_step(...)` is the typed compiler bridge for step
  configs: it uses `step_state.get_saved_resolved_value(...)` for
  `processing_config.*`, `source_bindings`, `step_materialization_config`,
  injectable params, `enabled`, and registered well-filter roots.
- `CompilationSession.from_context(...)` reuses the already-resolved steps,
  `step_state_map`, and snapshots for each axis context.
- `initialize_step_plans_for_context(...)` path-plans I/O, then
  `_supplement_step_plans(...)` writes the non-I/O fields into
  `CompiledStepPlan`, including
  `source_binding_plan = CompiledSourceBindingPlan.from_config(snapshot.source_bindings)`.
- Streaming configs are not a separate side channel either:
  `_collect_streaming_configs(...)` loops over `StreamingConfig.__registry__` and
  rebuilds enabled configs from saved ObjectState values.
- Therefore source bindings should only join the existing ObjectState ->
  `StepSnapshot` -> `CompiledStepPlan` flow. The plan must not require callers to
  remember a source-binding config outside ObjectState or pass one through
  generator/runtime/MCP glue after import.

## Steps

### Step 1: Add `SourceBindingWorkspaceProjector`

**New file: `openhcs/core/source_binding_workspace.py`**

Do not duplicate runtime source-binding matching. Add a small init-time adapter
that projects arbitrary disk files into existing OpenHCS source-projection
metadata:

```python
@dataclass(frozen=True, slots=True)
class SourceBindingWorkspaceProjector:
    """Project arbitrary source-bound image files into an OpenHCS workspace."""

    source_bindings: SourceBindingsConfig
    parser: SourceSchemaFilenameParser = field(default_factory=SourceSchemaFilenameParser)

    def projection_set(
        self,
        plate_path: Path,
        image_files: Sequence[str | Path],
    ) -> SourceProjectionSet:
        ...

    def projections(
        self,
        plate_path: Path,
        image_files: Sequence[str | Path],
    ) -> tuple[SourcePlaneProjection, ...]:
        ...


@dataclass(frozen=True, slots=True)
class SourceBindingProjectionMatcher:
    """Select source files and project binding component assignments at init time."""

    metadata: Mapping[str, str]

    def matches(self, source_path: str, binding: NamedSourceBinding) -> bool:
        ...

    def component_assignments(
        self,
        binding: NamedSourceBinding,
    ) -> Mapping[AllComponents, str]:
        ...
```

Projection rules:

1. Convert every file to a plate-relative POSIX path.
2. Apply `source_bindings.source_filters` with `source_filters_match(...)`
   before projection. These filters represent CP `Images` source-universe
   criteria and other pipeline-level source inclusion rules.
3. Expand source files through the existing source-plane inventory semantics
   before projection. Reuse or extract the typed source-schema candidate path
   (`SourceSchemaSourcePlaneInventory`, `SourceSchemaCandidate`, and related
   source-plane metadata) so multi-plane TIFFs and other indexed sources become
   one projection per logical plane. Do not introduce a parallel "one file equals
   one plane" shortcut.
4. Extract source metadata with existing
   `metadata_from_rules(relative_path, source_bindings.metadata_rules)`.
5. Evaluate optional binding selectors with a small nominal
   `SourceBindingProjectionMatcher` in the same module. It should reuse
   `source_filters_match`, `source_metadata_value`, and
   `source_metadata_values_equal`.
6. If there are multiple image-stack bindings, or a `match_plan` is present,
   assemble selected candidates into image sets before assigning OpenHCS
   addresses. Reuse the existing `SourceBindingMatchPlan` semantics that
   source-schema materialization already uses: metadata matching groups by typed
   match fields, and order matching zips candidates by alias order. Assign
   channel values from the image-set alias order and assign well/site/timepoint
   from the shared image-set metadata. This is required for CellProfiler-style
   `NamesAndTypes` pipelines where separate alias files are channels of the same
   site.
7. Init-time projection semantics differ from runtime matching in one necessary
   way: `ComponentSelector` values can supply canonical address components after
   a binding has selected a candidate by filters or metadata selectors. If the
   source metadata already has that component, require equality. If multiple
   matched bindings assign different values for the same component, fail loud.
8. Resolve canonical OpenHCS address fields with existing
   `ComponentProjection.resolve(component, metadata, image_index)`.
   This gives the source-schema defaults: singleton well `A01`, ordinal site,
   `z_index=1`, `timepoint=1`. There is no channel default, so channel must
   come from metadata rules or a matched binding selector.
9. Emit one `SourcePlaneProjection` per logical source plane with:
   - `address=OpenHCSPlaneAddress(...)`
   - `ref=SourcePixelRef(backend=Backend.DISK.value, source_path=relative_path)`
   - `source_metadata=metadata`
   - `source_alias` only when exactly one image binding matched
10. Let `SourceProjectionSet` validate duplicate virtual addresses. Duplicate
   addresses are not recoverable here; the user must supply more discriminating
   metadata rules/selectors.

### Step 2: Create `SourceBindingsHandler`

**New file: `openhcs/microscopes/source_bindings_handler.py`**

```python
class SourceBindingsHandler(MicroscopeHandler):
    """Handler for arbitrary image folders using source bindings.

    Init flow:
    1. List all image files in plate folder
    2. Use SourceBindingWorkspaceProjector to extract metadata, assemble image sets,
       and project logical source planes
    3. Build standardized virtual filenames from projected components
    4. Save openhcs_metadata.json with workspace_mapping
    5. Register virtual workspace backend
    """

    _microscope_type = Microscope.SOURCE_BINDINGS.value
    _metadata_handler_class = OpenHCSMetadataHandler  # reuse

    def __init__(
        self,
        filemanager,
        source_bindings: SourceBindingsConfig,
        pattern_format: str | None = None,
    ):
        parser = SourceSchemaFilenameParser(filemanager, pattern_format)
        super().__init__(
            parser=parser,
            metadata_handler=OpenHCSMetadataHandler(filemanager),
        )
        if source_bindings.is_empty:
            raise ValueError(
                "SourceBindingsHandler requires SourceBindingsConfig declarations."
            )
        self._source_bindings = source_bindings
        self._projector = SourceBindingWorkspaceProjector(
            source_bindings=source_bindings,
            parser=parser,
        )

    @property
    def root_dir(self) -> str:
        return FIELDS.DEFAULT_SUBDIRECTORY

    @property
    def microscope_type(self) -> str:
        return self._microscope_type

    @property
    def metadata_handler_class(self):
        return OpenHCSMetadataHandler

    @property
    def compatible_backends(self) -> list[Backend]:
        return [Backend.DISK]

    @classmethod
    def detect(cls, plate_folder, filemanager) -> bool:
        """Never auto-detect; select via PipelineConfig.microscope."""
        return False

    def initialize_workspace(self, plate_path, filemanager):
        self.plate_folder = plate_path
        image_files = self._list_image_files(plate_path, filemanager)
        projection_set = self._projector.projection_set(
            Path(plate_path),
            image_files,
        )
        self._save_metadata(Path(plate_path), projection_set)
        self._register_virtual_workspace_backend(plate_path, filemanager)
        return plate_path

    def _list_image_files(self, plate_path, filemanager):
        """List all image files recursively."""
        return filemanager.list_image_files(
            plate_path,
            Backend.DISK.value,
            extensions=LOADABLE_IMAGE_EXTENSIONS,
            recursive=True,
        )

    def _save_metadata(self, plate_path, projection_set):
        metadata = projection_set.metadata_dict(
            parser=self.parser,
            microscope_handler_name=self.microscope_type,
            source_filename_parser_name=SourceSchemaFilenameParser.__name__,
            grid_dimensions=[1, 1],
            pixel_size=1.0,
            available_backends={
                Backend.DISK.value: True,
                Backend.VIRTUAL_WORKSPACE.value: True,
            },
            main=True,
        )
        AtomicMetadataWriter().replace_subdirectory_metadata(
            get_metadata_path(plate_path),
            FIELDS.DEFAULT_SUBDIRECTORY,
            metadata,
        )
```

Metadata synchronization contract:

- `SourceBindingsHandler` owns `FIELDS.DEFAULT_SUBDIRECTORY` when
  `microscope=Microscope.SOURCE_BINDINGS` is selected.
- Each init must regenerate that subdirectory's complete metadata dictionary from
  the current `SourceProjectionSet`.
- The write must atomically replace the managed subdirectory metadata while
  preserving unrelated subdirectories in `openhcs_metadata.json`.
- Do not use `AtomicMetadataWriter.merge_subdirectory_metadata(...)` for this
  managed primary subdirectory. Its additive `available_backends` behavior is
  correct for post-init materialization updates, but it can retain stale backend
  flags during source-binding re-init.
- Add a public writer method such as
  `AtomicMetadataWriter.replace_subdirectory_metadata(...)` rather than
  hand-writing JSON or adding a one-off private helper.

Registration follows the existing microscope-handler architecture: the class
self-registers through `MicroscopeHandler`'s `AutoRegisterMeta` because it
declares `_microscope_type = Microscope.SOURCE_BINDINGS.value`. Place it in an
auto-discovered module under `openhcs/microscopes/`, and add
`Microscope.SOURCE_BINDINGS` so `PipelineConfig.microscope` can select it
explicitly.

Use `SourceSchemaFilenameParser` rather than a custom parser name. The
OpenHCS-metadata reload path currently recognizes this parser through
`OpenHCSMicroscopeHandler._get_available_filename_parsers()`, and the dry run
confirmed it can parse the generated virtual filenames.

### Step 3: Split pipeline and step source-binding configs

**Files: `openhcs/core/source_bindings.py`, `openhcs/core/config.py`**

Follow the same pattern as `WellFilterConfig` and `StepWellFilterConfig`:
pipeline/plate scope gets the regular config, and steps get the step-specialized
subclass.

1. Add `SourceBindingsConfig` to `openhcs.core.source_bindings`.
2. Make `StepSourceBindingsConfig` inherit it.
3. Declare `enabled: bool = False` only on concrete `StepSourceBindingsConfig`
   if the field is kept for the step-local config surface. Do not declare it on
   `SourceBindingsConfig`, and do not use it as a compiler-side workaround.
4. Register both config classes with `@global_pipeline_config`.

```python
@dataclass(frozen=True)
class SourceBindingsConfig(_SourceBindingPlanBase):
    """Pipeline/plate source-binding defaults and init-time discovery config."""

    source_filters: tuple[SourceFilterClause, ...] = ()
    bindings: tuple[NamedSourceBinding, ...] = ()
    metadata_rules: tuple[MetadataExtractionRule, ...] = ()
    match_plan: SourceBindingMatchPlan | None = None

    def __post_init__(self) -> None:
        source_filters = normalize_source_binding_values(
            "SourceBindingsConfig.source_filters",
            self.source_filters,
            SourceFilterClause,
        )
        bindings = normalize_source_binding_values(
            f"{type(self).__name__}.bindings",
            self.bindings,
            NamedSourceBinding,
        )
        seen_aliases: set[str] = set()
        for binding in bindings:
            if binding.alias in seen_aliases:
                raise ValueError(
                    f"{type(self).__name__}.bindings contains duplicate alias "
                    f"{binding.alias!r}."
                )
            seen_aliases.add(binding.alias)
        object.__setattr__(self, "source_filters", source_filters)
        object.__setattr__(self, "bindings", bindings)
        self._normalize_common_fields()

    @property
    def has_primary_content(self) -> bool:
        return bool(self.bindings)

    @property
    def is_empty(self) -> bool:
        return (
            not self.has_primary_content
            and not self.source_filters
            and not self.metadata_rules
            and self.match_plan is None
        )


@dataclass(frozen=True)
class StepSourceBindingsConfig(SourceBindingsConfig):
    """Step-local source-binding config inheriting pipeline/plate defaults."""

    enabled: bool = False
```

```python
from openhcs.core import source_bindings as source_binding_configs

SourceBindingsConfig = global_pipeline_config(
    preview_label="SRC",
    abbreviation="src",
)(source_binding_configs.SourceBindingsConfig)

StepSourceBindingsConfig = global_pipeline_config(
    preview_label="STEP_SRC",
    abbreviation="step_src",
)(source_binding_configs.StepSourceBindingsConfig)

# Public config-module re-exports. ObjectState generates these in the concrete
# config module, but step code should import core config objects from here.
LazySourceBindingsConfig = source_binding_configs.LazySourceBindingsConfig
LazyStepSourceBindingsConfig = source_binding_configs.LazyStepSourceBindingsConfig
```

The exact preview labels/abbreviations can follow the surrounding `config.py`
style. The key is that `SourceBindingsConfig` is the regular pipeline-level config and
`StepSourceBindingsConfig` is the step-level inherited config, just as
`StepWellFilterConfig(WellFilterConfig)` is the step-level specialization. Do
not force `inherit_as_none=False` on `StepSourceBindingsConfig`; the inherited
binding payload fields must remain lazy/inheritable.

5. Update `AbstractStep.source_bindings` to use and accept the generated lazy
   step config:

```python
from openhcs.core.config import LazyStepSourceBindingsConfig

source_bindings: LazyStepSourceBindingsConfig = LazyStepSourceBindingsConfig()
```

`LazyStepSourceBindingsConfig` is autogenerated by the config decorator. Do not
use an explicit `enabled=False` override or a compiler gate as a shortcut. The
raw inherited payload fields must remain `None` on unresolved step configs so
ObjectState can inherit `SourceBindingsConfig` values from plate/pipeline scope.
Downstream compiler/runtime context records continue to receive resolved concrete
`StepSourceBindingsConfig` objects from ObjectState; tests should cover that
boundary.

### Step 4: Wire into orchestrator init

**File: `openhcs/core/orchestrator/orchestrator.py`**

`initialize_microscope_handler()` should keep using the orchestrator's effective
ObjectState-resolved config (`get_effective_config()` /
`ObjectState(self.pipeline_config).to_saved_resolved_object()`) when choosing the
microscope handler. Do not reconstruct the config through ad hoc field sweeps or
`get_saved_resolved_value("...")` string paths. The source-binding handler
factory branch receives the already-resolved
`effective_config.source_bindings_config`; it must not perform a second config
resolution path.

```python
def initialize_microscope_handler(self):
    effective_config = ObjectState(
        self.pipeline_config,
    ).to_saved_resolved_object()
    if not isinstance(effective_config, GlobalPipelineConfig):
        raise TypeError(...)
    microscope_type = (
        effective_config.microscope.value
        if effective_config.microscope != Microscope.AUTO
        else "auto"
    )
    self.microscope_handler = create_microscope_handler(
        plate_folder=str(self.plate_path),
        filemanager=self.filemanager,
        microscope_type=microscope_type,
        source_bindings_config=effective_config.source_bindings_config,
    )
```

Update `create_microscope_handler()` to accept
`source_bindings_config: SourceBindingsConfig | None = None` and pass it through
the registered handler class' polymorphic `create()` hook. `SourceBindingsHandler`
overrides that hook and validates the typed config. Other handlers inherit the
default hook. Do not add a source-bindings-specific microscope-type branch to
the factory; handler selection remains owned by `AutoRegisterMeta`. Auto-detection
should not select `source_bindings`; users select it by setting
`PipelineConfig.microscope`.

### Step 5: Keep Runtime Compilation on the Existing Typed Path

**File: `openhcs/core/source_bindings.py`**

This is the existing compiler insertion point. `PipelineCompiler` should not
learn a new source-binding resolution phase; `_supplement_step_plans()` already
receives `snapshot.source_bindings` from ObjectState and assigns
`current_plan.source_binding_plan =
CompiledSourceBindingPlan.from_config(snapshot.source_bindings)`.

Do not add a separate runtime code path and do not add an `enabled` check here.
The current non-invasive slice leaves runtime compilation governed by the
resolved typed source-binding payload:

```python
@classmethod
def from_config(cls, config: StepSourceBindingsConfig) -> "CompiledSourceBindingPlan":
    if config.is_empty:
        return cls.empty()
    return cls(...)
```

Plate-level inheritance into step source bindings is expected. If a step needs a
different source-binding payload, it should supply a typed step-local override or
subset. Do not hardcode a compiler-side boolean escape hatch.

This lets existing consumers remain no-op:

- `StepAnchorPatternFilter.source_bound_anchor_patterns()` already returns early
  when `not plan.source_binding_plan.has_primary_content`.
- `FunctionStepRuntime._filter_matching_files_for_source_bindings()` already
  returns early when there are no selector-bearing bindings.

### Step 6: Update CellProfiler Pipeline Generation

**Files: `openhcs/interop/cellprofiler/pipeline_generator.py`,
`openhcs/interop/cellprofiler/symbol_table.py`,
`openhcs/interop/cellprofiler/runtime/generated_pipeline.py`,
`openhcs/interop/cellprofiler/runtime_pipeline.py`,
`openhcs/interop/cellprofiler/import_records.py`,
`openhcs/core/pipeline_image_schema.py`**

Current generator dry run:

- `Images`, `Metadata`, `NamesAndTypes`, and `LoadImages` already compile into a
  typed `PipelineImageSchema`.
- Generated code currently exports only `pipeline_steps`; it does not export or
  return a `PipelineConfig`.
- Source-bound module contracts currently emit concrete
  `StepSourceBindingsConfig(...)` on individual `FunctionStep`s, including
  `metadata_rules` and `match_plan`.
- Runtime binding uses the step source bindings to align CellProfiler runtime
  artifact contracts, so step-level bindings still matter for modules that
  explicitly consume named source images.
- A live dry run of `PipelineImageSchema` showed additional setup fields that
  cannot be silently dropped: `image_plane_sources`, `imported_metadata_tables`,
  `source_image_stack`, and `grouping`.
- `PipelineImageSchema.to_source_bindings_config()` now handles the
  representable subset directly. Unsupported schema features are declared as
  nominal `PipelineImageSchemaSourceBindingsFeature` classes registered by
  `AutoRegisterMeta`; feature field names live on those declarations rather than
  in a separate enum or call-site string list.
- A live import-boundary dry run showed
  `GeneratedPipelineModuleExports.pipeline_steps` is currently the only typed
  generated-module export. `pipeline_config` needs a matching optional export
  accessor.

The new behavior should split setup/source-universe configuration from
per-step source access:

1. Convert the compiled `PipelineImageSchema` into pipeline-level
   `SourceBindingsConfig` data through a nominal adapter. Prefer hosting this on
   `PipelineImageSchema` or a dedicated bridge module so `source_bindings.py`
   does not import `pipeline_image_schema.py` and create a circular import. Use
   typed source assignments: `ImageAssignment.to_binding()` and
   `SourceArtifactAssignment.to_binding()`. Do not reconstruct bindings from
   dicts or setting-name strings.
2. Preserve `Metadata`, `NamesAndTypes` match plans, and `LoadImages`
   declarations in that pipeline-level config.
3. Preserve `Images` source-universe filters too. This needs a regular
   `SourceBindingsConfig.source_filters: tuple[SourceFilterClause, ...] = ()`
   field using the existing `SourceFilterClause` vocabulary; otherwise generated
   `.cppipe` imports could admit files that the existing source-schema workspace
   would have filtered out.
4. Do not silently discard broader `PipelineImageSchema` fields. The current
   source-bindings projection fails with a typed representability error for:

   - `image_plane_sources`, which should send the import through the existing
     source-schema workspace materialization path unless explicit handler support
     is added later.
   - `imported_metadata_tables`, which still need full source-schema metadata
     joining.
   - `source_image_stack`, which must preserve CellProfiler source-stack axis
     semantics before it can use source-binding init.
   - `grouping` should remain a typed source-schema/runtime grouping concern; do
     not collapse it into source-binding groups or filename binding data.

5. Generated/imported `.cppipe` pipelines should expose a normal plate-level
   config declaration:

   ```python
   pipeline_config = PipelineConfig(
       microscope=Microscope.SOURCE_BINDINGS,
       source_bindings_config=SourceBindingsConfig(...),
   )
   ```

   The import workflow should install this as the selected pipeline's
   ObjectState-owned `PipelineConfig` root through the existing generic config
   path. From that point on, orchestrator init and compiler resolution see it
   with every other config in one ObjectState pass. This is not an MCP-specific
   change and it must not become a caller-managed side channel.
6. Generated module exports should include the config in a normal, importable
   way (`pipeline_config = PipelineConfig(...)`). Update
   `GeneratedPipelineModuleExports` with an optional typed `pipeline_config`
   accessor so the import workflow can install the config into ObjectState before
   init/compile. Do not require `GeneratedPipeline`, `PreparedGeneratedPipeline`,
   `CellProfilerPipelineImportResult`, UI, MCP, or runtime callers to remember a
   parallel config object after import.
7. Ordinary generated steps should not receive step source bindings just because
   the `.cppipe` declared setup image loading. They should operate on the virtual
   workspace initialized by the pipeline config.
8. Only generated steps whose artifact contract has external source symbols
   should carry a step-local runtime source-binding subset:

   ```python
   FunctionStep(
       ...,
       source_bindings=StepSourceBindingsConfig(
           bindings=(...),  # the exact aliases required by this module contract
       ),
   )
   ```

   The step-level binding list remains a contract-local subset so
   `SourceBindingRuntimeContractGuard` can keep exact drift detection. The step
   should inherit `metadata_rules`, `match_plan`, and source-universe filters
   from the pipeline-level `SourceBindingsConfig`; do not re-emit those fields on
   every generated step.
9. If a later `.cppipe` module explicitly reloads or otherwise references source
   images after the initial setup phase, that module is represented by external
   source symbols and gets the typed step-level subset. Previous-step artifact
   consumers do not.

## File Changes

| File | Change |
|------|--------|
| `openhcs/constants/constants.py` | Add `Microscope.SOURCE_BINDINGS = "source_bindings"` |
| `openhcs/core/source_binding_workspace.py` | **New** — init-time source-binding to `SourceProjectionSet` projector; reuse/extract source-schema source-plane inventory and image-set assembly semantics |
| `openhcs/microscopes/source_bindings_handler.py` | **New** — handler class implementing all `MicroscopeHandler` abstract properties |
| `openhcs/microscopes/openhcs.py` | Add a public atomic replace-subdirectory metadata method for source-binding re-init sync |
| `openhcs/core/source_bindings.py` | Add `SourceBindingsConfig` including inherited `source_filters`, make `StepSourceBindingsConfig(SourceBindingsConfig)`, remove grouped source-binding config, and keep `CompiledSourceBindingPlan` derived from resolved typed bindings without an `enabled` compiler gate |
| `openhcs/core/config.py` | Register both `SourceBindingsConfig` and `StepSourceBindingsConfig` with `global_pipeline_config`, matching the `WellFilterConfig` / `StepWellFilterConfig` pattern |
| `openhcs/core/steps/abstract.py` | Use `LazyStepSourceBindingsConfig()` as the step default so plate-level source-binding payload fields inherit through ObjectState |
| `openhcs/core/orchestrator/orchestrator.py` | Use effective global + pipeline config for microscope selection, then pass `source_bindings_config` into the source-binding handler |
| `openhcs/microscopes/microscope_base.py` | Add a polymorphic handler `create()` hook so `SourceBindingsHandler` receives typed config through AutoRegisterMeta-owned handler construction, without a stringly typed factory branch |
| `openhcs/core/pipeline_image_schema.py` | Host `PipelineImageSchema.to_source_bindings_config()` plus typed representability checks for schema fields that require source-schema materialization |
| `openhcs/interop/cellprofiler/pipeline_generator.py` | Emit `pipeline_config = PipelineConfig(...)` and import config objects from `openhcs.core.config` plus `Microscope` from constants |
| `openhcs/interop/cellprofiler/runtime/generated_pipeline.py` | Add a typed optional generated-module export accessor for `pipeline_config` |
| `openhcs/interop/cellprofiler/runtime_pipeline.py` / `import_records.py` | Install or expose generated `PipelineConfig` only as the ObjectState-owned pipeline config root for the import workflow; do not create a parallel caller-managed runtime config channel |
| `openhcs/interop/cellprofiler/module_processing_components.py` / `symbol_table.py` | Emit step-local source-binding subsets only for external-source module contracts and stop repeating inherited metadata/match fields on those steps |

## What Doesn't Change

- Existing microscope handlers work exactly as before
- Source bindings on FunctionStep continue to work through the existing compiled
  source-binding plan path
- `openhcs_metadata.json` format is unchanged
- Pattern discovery (`auto_detect_patterns`) is unchanged
- No new binding vocabulary — `StepSourceBindingsConfig` inherits the regular `SourceBindingsConfig`
- Runtime filtering remains on the existing source-binding runtime path
- The compiler's config-resolution architecture does not change: source bindings
  use the existing ObjectState-resolved `StepSnapshot` and
  `CompiledStepPlan.source_binding_plan` path
- No MCP-specific tool is needed; existing MCP/UI flows install/update
  `PipelineConfig` through the same ObjectState-backed config workflow and invoke
  plate initialization normally

## Validation Plan

1. Unit-test `SourceBindingWorkspaceProjector` with files like
   `img_A01_s1_ch2.tif`; assert generated virtual paths, structured
   `workspace_mapping`, and parser round-trip through `SourceSchemaFilenameParser`.
2. Unit-test missing channel metadata; assert projection fails loud before
   metadata is written.
3. Unit-test selector assignment: filters select `nuclei_...` and
   `membrane_...`, component selectors assign channels 1 and 2, and conflicting
   assignments raise.
4. Unit-test `create_microscope_handler("source_bindings", ...)` with missing
   or empty `source_bindings_config`; assert a targeted error.
5. Unit-test orchestrator init with `PipelineConfig.microscope` set to
   `Microscope.SOURCE_BINDINGS`; assert it uses
   `ObjectState.to_saved_resolved_object()` and the factory receives the
   pipeline-selected microscope, not only the saved global config.
6. Integration-test handler init against a temporary arbitrary image folder:
   assert `OpenHCSMicroscopeHandler` can reload the generated metadata, the
   virtual workspace backend lists canonical virtual files from the plate root,
   and `VirtualWorkspaceSourceProjectionAuthority` maps virtual paths back to
   source paths.
7. Unit-test source-binding re-init metadata sync: seed stale source-binding
   primary metadata plus an unrelated subdirectory, rerun init with different
   source files, and assert stale `image_files`, `workspace_mapping`,
   `source_metadata`, `source_projection`, and `available_backends` entries are
   gone from `FIELDS.DEFAULT_SUBDIRECTORY` while unrelated subdirectories remain.
8. Unit-test `AbstractStep.source_bindings` accepts
   `LazyStepSourceBindingsConfig` and still rejects unrelated objects.
9. Unit-test unresolved step source-binding inheritance: constructing
   `StepSourceBindingsConfig()` after decorator injection preserves inherited
   `None` sentinels without crashing, and compiler snapshots receive resolved
   concrete tuple-valued configs.
10. Unit-test source-binding image-set assembly parity: order-matched aliases such
    as `DAPI_001.png` and `Actin_001.png` project to the same site with different
    channels, matching existing source-schema materialization. Covered by
    `tests/unit/test_source_binding_workspace.py::test_source_binding_workspace_projector_order_matches_aliases`.
11. Unit-test source-plane inventory parity: multi-plane image sources become one
    virtual projection per logical plane; no silent one-file-to-one-plane
    collapse. Covered by
    `tests/unit/test_source_binding_workspace.py::test_source_binding_workspace_projector_expands_tiff_stack_planes`.
12. Unit-test `.cppipe` generation with setup `Images`/`Metadata`/`NamesAndTypes`:
   assert generated/imported output installs `PipelineConfig(microscope=
   Microscope.SOURCE_BINDINGS, source_bindings_config=SourceBindingsConfig(...))`
   into the same ObjectState-backed config path as other pipeline configs, and
   that the config preserves bindings, metadata rules, match plan, and `Images`
   source filters.
13. Unit-test the generated-module import boundary: optional `pipeline_config`
    export is exposed by `GeneratedPipelineModuleExports` and installed into the
    same ObjectState-owned pipeline config path used by all other configs; no
    caller-managed parallel config object is required after import.
14. Unit-test representability for broader `PipelineImageSchema` fields:
    schemas with `image_plane_sources`, `imported_metadata_tables`,
    `source_image_stack`, or `grouping` fail with a targeted fallback error; they
    must not silently drop those fields. Covered by
    `tests/unit/test_pipeline_image_schema.py::test_pipeline_image_schema_source_bindings_projection_rejects_unrepresented_fields`.
15. Unit-test generated steps: ordinary previous-step consumers do not emit
   duplicate step-local `source_bindings`; modules with external source symbols
   may rely on inherited plate bindings or emit a typed step-local subset without
   repeating inherited metadata rules or match plans.
16. Unit-test source-binding inheritance at the ObjectState boundary. The
   assertion should prove the resolved `StepSnapshot.source_bindings` payload,
   not a compiler-side `enabled` gate. Covered by
   `tests/unit/test_source_bindings.py::test_step_source_bindings_inherit_plate_source_bindings_for_snapshot`.
17. Unit-test the compiler handoff: a `PipelineConfig(source_bindings_config=...)`
    plus a `FunctionStep(source_bindings=LazyStepSourceBindingsConfig(...))`
    resolves through the normal ObjectState hierarchy into a concrete
    `StepSnapshot.source_bindings`, and `_supplement_step_plans()` produces
    `CompiledStepPlan.source_binding_plan` only from that snapshot value. The test
    should not pass any extra source-binding config through generator, MCP, UI,
    orchestrator, or runtime side channels. Covered by
    `tests/unit/test_compilation_session.py::test_compiler_source_binding_plan_comes_from_objectstate_snapshot`.
18. Unit-test incomplete order-matched source-binding image sets: aliases with
    unequal matched candidate counts fail before projection metadata is emitted.
    Covered by
    `tests/unit/test_source_binding_workspace.py::test_source_binding_workspace_projector_rejects_incomplete_order_matches`.

Current focused validation:

- `./.venv/bin/python -m pytest tests/unit/test_pipeline_image_schema.py tests/unit/test_source_binding_workspace.py -q`
  passed with `30 passed`.
- `./.venv/bin/python -m pytest tests/unit/test_source_bindings.py tests/unit/test_step_snapshot.py tests/unit/test_pipeline_image_schema.py tests/unit/test_source_binding_workspace.py -q`
  passed with `45 passed`.
- `./.venv/bin/python -m pytest tests/unit/test_source_schema_workspace.py tests/unit/test_source_binding_workspace.py tests/unit/test_compilation_session.py tests/unit/test_source_bindings.py tests/unit/test_step_snapshot.py -q`
  passed with `89 passed`.

## Example Usage

A biologist has a folder of images named `img_A01_s1_ch2.tif`:

```python
# Select the source-binding microscope handler and configure its declarations.
pipeline_config = PipelineConfig(
    microscope=Microscope.SOURCE_BINDINGS,
    source_bindings_config=SourceBindingsConfig(
        metadata_rules=(
            MetadataExtractionRule(
                source=MetadataSource.FILE_NAME,
                pattern=r"img_(?P<well>[A-Z]\d+)_s(?P<site>\d+)_ch(?P<channel>\d+)",
            ),
        ),
    ),
)
```

Or with filters that identify aliases and component selectors that assign
channels:

```python
pipeline_config = PipelineConfig(
    microscope=Microscope.SOURCE_BINDINGS,
    source_bindings_config=SourceBindingsConfig(
        metadata_rules=(
            MetadataExtractionRule(
                source=MetadataSource.FILE_NAME,
                pattern=r".*_(?P<well>[A-Z]\d+)_s(?P<site>\d+)",
            ),
        ),
        bindings=(
            NamedSourceBinding(
                alias="nuclei",
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            subject=SourceFilterSubject.FILE,
                            match_type=SourceFilterMatchType.STARTS_WITH,
                            value="nuclei_",
                        ),
                    ),
                    components=(
                        ComponentSelector(component=AllComponents.CHANNEL, value="1"),
                    ),
                ),
            ),
            NamedSourceBinding(
                alias="membrane",
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            subject=SourceFilterSubject.FILE,
                            match_type=SourceFilterMatchType.STARTS_WITH,
                            value="membrane_",
                        ),
                    ),
                    components=(
                        ComponentSelector(component=AllComponents.CHANNEL, value="2"),
                    ),
                ),
            ),
        ),
    ),
)
```

At pipeline/plate scope, `microscope=Microscope.SOURCE_BINDINGS` selects
`SourceBindingsHandler` for init and builds the virtual workspace. Steps inherit
these source-binding values through `StepSourceBindingsConfig` by default.
Step-local `source_bindings` declarations are overrides or subsets, not mandatory
duplicates of the plate-level declaration.

## Resolved Decisions

1. `match_plan` is not runtime-only for multi-alias image inputs. Init uses it
   when aliases need to become channels or other components of the same OpenHCS
   image set. Order-matched aliases must now have equal candidate counts; the
   shared `OrderImageSetAssembler` raises a targeted incomplete-image-set error
   before any source-binding workspace metadata is written. Metadata-matched
   aliases continue to use the existing complete-set validation.

## Remaining Decisions

1. `MetadataSource.FOLDER_NAME` is already supported by `metadata_from_rules()`
   and should work because the projector passes plate-relative paths, including
   subdirectories, into metadata extraction.
2. CellProfiler-generated steps with runtime source-binding requirements must
   either rely on inherited plate bindings or emit a typed step-local binding
   subset. They should not require a parallel source-binding config channel
   outside ObjectState.
