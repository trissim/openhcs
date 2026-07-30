# Benchmark Remaining Cleanup Plan

Date: 2026-07-01

Scope: finish the benchmark-package cleanup left after
`docs/plans/benchmark_api_style_refactor_20260630.md`. This plan only covers
benchmark production code and its focused tests. It does not cover MCP work,
OpenHCS CellProfiler compiler/runtime internals, generated CellProfiler source,
or broad report-formatting cleanups unless they cross a benchmark semantic
boundary.

The current edited API slice already passes:

```text
python -m ruff check benchmark/contracts/values.py benchmark/contracts/pipeline.py benchmark/contracts/dataset.py benchmark/contracts/metric.py benchmark/contracts/tool_adapter.py benchmark/pipelines/registry.py benchmark/pipelines/__init__.py benchmark/datasets/registry.py benchmark/datasets/acquire.py benchmark/datasets/__init__.py benchmark/metrics/__init__.py benchmark/adapters/__init__.py benchmark/__init__.py tests/unit/test_dataset_acquisition_infra.py
All checks passed!

python -m pytest tests/unit/test_dataset_registry.py tests/unit/test_dataset_acquisition_infra.py tests/unit/test_benchmark_manifests.py tests/unit/test_comparison_manifest_acquisition.py tests/unit/test_benchmark_timing.py tests/unit/test_metrics_memory.py tests/unit/test_benchmark_progress.py tests/unit/test_runner_cellprofiler_compatibility.py tests/unit/test_openhcs_adapter.py tests/unit/test_cellprofiler_adapter.py -q
76 passed
```

## Current Inventory

AST inventory excluding generated CellProfiler source:

```text
benchmark/contracts/manifest_acquisition.py:26: assign DATASET_REGISTRY enum member
benchmark/contracts/manifest_acquisition.py:227: getattr
benchmark/contracts/manifest_acquisition.py:232: getattr
benchmark/converter/__init__.py:54: getattr
benchmark/converter/__init__.py:59: assign _COMPATIBILITY_EXPORTS
benchmark/converter/__init__.py:74: assign __all__
benchmark/datasets/bioformats_hcs.py:525: annassign BIOFORMATS_HCS_CATALOG
benchmark/datasets/bioformats_hcs.py:529: annassign BIOFORMATS_HCS_REGISTRY
benchmark/runner.py:436: hasattr
benchmark/runner.py:458: getattr
benchmark/runner.py:488: getattr
benchmark/adapters/openhcs.py:86: _RUNTIME_EXECUTION_CACHE_IGNORED_PARAM_KEYS
benchmark/adapters/openhcs.py:97: _RUNTIME_EXECUTION_CACHE_HELPER_KEYS
benchmark/adapters/openhcs.py:100-104: OPENHCS_*_PARAM string constants
benchmark/adapters/cellprofiler.py:56-58: CELLPROFILER_*_PARAM string constants
benchmark/runner.py:41-49: duplicated runtime cache helper/ignored parameter keys
```

Not every hit above is equally bad. The production boundary violations are:

1. Manifest acquisition strategy registration mirrors enum values as strings.
2. Manifest root materialization erases `ManifestPathRoot` to `object`.
3. Converter exports use a manual string set and `getattr`.
4. Tool/metric base contracts document required names but do not declare them.
5. OpenHCS benchmark run options are parsed from raw `pipeline_params`, but
   runtime well selection bypasses `PipelineConfig` by passing
   `execute_pipeline_direct(..., well_filter=selected_axes)`.
6. Bio-Formats HCS duplicates dataset acquisition and metadata-axis authority in
   `benchmark/datasets/bioformats_hcs.py` and
   `benchmark/bioformats_hcs_validation.py`.

## Phase 1: Manifest Acquisition Uses Typed Authorities

### Authority

- `ManifestRootAcquisitionKind` owns acquisition-family identity.
- `ManifestPathRoot` owns resolved root path and acquisition config.
- `ManifestRootAcquisitionStrategy.__registry__` must be keyed by
  `ManifestRootAcquisitionKind`, not `str`.

### Final API

In `benchmark/contracts/manifest_acquisition.py`:

```python
from typing import ClassVar

class ManifestRootAcquisitionStrategy(ABC, metaclass=AutoRegisterMeta):
    __registry_key__ = "kind"
    kind: ClassVar[ManifestRootAcquisitionKind | None] = None

    @classmethod
    def for_spec(
        cls,
        spec: ManifestRootAcquisitionSpec,
    ) -> "ManifestRootAcquisitionStrategy":
        try:
            strategy_type = cls.__registry__[spec.kind]
        except KeyError as exc:
            raise ManifestAcquisitionError(
                f"Unsupported manifest root acquisition kind {spec.kind.name}."
            ) from exc
        return strategy_type()

def materialize_manifest_root(
    *,
    root_name: str,
    root_path: Path,
    acquisition_spec: ManifestRootAcquisitionSpec,
    requirements: tuple[ManifestRootRequirement, ...],
) -> None:
    request = ManifestRootAcquisitionRequest(
        root_name=root_name,
        root_path=root_path,
        spec=acquisition_spec,
        requirements=requirements,
    )
    if not request.missing_requirements():
        return
    ManifestRootAcquisitionStrategy.for_spec(acquisition_spec).materialize(request)
    missing_after = request.missing_requirements()
    if missing_after:
        missing_lines = ", ".join(
            f"{item.case_name}:{item.path_key}={item.relative_path}"
            for item in missing_after[:5]
        )
        raise ManifestAcquisitionError(
            f"Manifest root {root_name!r} materialized but still misses "
            f"{len(missing_after)} required paths: {missing_lines}"
        )
```

Leaves:

```python
class DatasetRegistryRootAcquisitionStrategy(ManifestRootAcquisitionStrategy):
    kind = ManifestRootAcquisitionKind.DATASET_REGISTRY

class GitSparseRootAcquisitionStrategy(ManifestRootAcquisitionStrategy):
    kind = ManifestRootAcquisitionKind.GIT_SPARSE
```

In `benchmark/contracts/comparison_manifest.py`:

```python
from benchmark.contracts.manifest_acquisition import (
    ManifestRootRequirementMap,
    manifest_root_requirements_by_root,
    materialize_manifest_root,
)

def materialize_manifest_path_roots(
    roots: Mapping[str, ManifestPathRoot],
    raw_cases: object,
) -> None:
    requirements_by_root = manifest_root_requirements_by_root(raw_cases)
    for root in roots.values():
        if root.acquisition is None:
            continue
        materialize_manifest_root(
            root_name=root.name,
            root_path=root.path,
            acquisition_spec=root.acquisition,
            requirements=tuple(requirements_by_root.get(root.name, ())),
        )
```

Then `ComparisonManifest.load()` calls
`materialize_manifest_path_roots(path_resolver.roots, payload.get("cases"))`.

### Required Deletes/Renames

- Delete `materialize_manifest_roots(roots: Mapping[str, object], ...)` from
  `manifest_acquisition.py`.
- Rename `_requirements_by_root()` to
  `manifest_root_requirements_by_root()` and import that function from
  `comparison_manifest.py`.
- Remove all `.value` uses from `ManifestRootAcquisitionStrategy.kind` and
  strategy lookup.

### AST Bulk Edit

Use AST for the exact manifest acquisition rewrites before manual cleanup:

```python
# scripts/scratch_manifest_acquisition_ast.py
import ast
from pathlib import Path

path = Path("benchmark/contracts/manifest_acquisition.py")
tree = ast.parse(path.read_text())

for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "kind":
                print(path, node.lineno, "class kind assignment", ast.unparse(node.value))
    if isinstance(node, ast.Attribute) and node.attr == "value":
        # Confirm parents are ManifestRootAcquisitionKind or spec.kind.
        print(path, node.lineno, ast.unparse(node))
```

Manual edit only the reported class-body `kind = ...` assignments after the AST
report. Do not broad-replace
`.value`, because manifest JSON parsing still uses string values through
`ManifestRootAcquisitionKind(str(raw_kind))`.

### Tests

Add to `tests/unit/test_comparison_manifest_acquisition.py`:

```python
def test_manifest_root_acquisition_strategies_are_registered_by_enum() -> None:
    assert (
        ManifestRootAcquisitionStrategy.for_spec(
            ManifestRootAcquisitionSpec(kind=ManifestRootAcquisitionKind.GIT_SPARSE)
        ).kind
        is ManifestRootAcquisitionKind.GIT_SPARSE
    )
```

Run:

```text
python -m pytest tests/unit/test_comparison_manifest_acquisition.py tests/unit/test_benchmark_manifests.py -q
```

### Dry-Run Review

- No circular import is introduced because `comparison_manifest.py` imports
  acquisition helpers; `manifest_acquisition.py` does not import
  `comparison_manifest.py`.
- Do not add a protocol or wrapper class. Keep the root loop in
  `comparison_manifest.py`, where `ManifestPathRoot` is already visible.
- The only string boundary remains manifest JSON parsing, which is appropriate:
  external JSON strings are converted once into `ManifestRootAcquisitionKind`.

## Phase 2: Converter Public API Has Object-Owned Exports

### Authority

The imported objects themselves own public identity. The converter package must
not maintain a parallel `_COMPATIBILITY_EXPORTS` string registry.

### Final API

In `benchmark/converter/__init__.py`:

```python
from openhcs.core.public_api import public_names_from_objects

__all__ = public_names_from_objects(
    SourceLocator,
    LLMFunctionConverter,
    LibraryAbsorber,
    ContractInference,
    infer_contract,
    CPPipeModulePartition,
    CPPipePipelineGenerationRequest,
    CPPipePipelinePreparationRequest,
    DirectPipelineExecution,
    GeneratedCPPipePipeline,
    PreparedGeneratedPipeline,
    execute_pipeline_direct,
    prepare_generated_pipeline,
    CellProfilerSymbol,
    CellProfilerSymbolKind,
    CellProfilerSymbolTable,
    ModuleArtifactContracts,
    CPPipeParser,
    ModuleBlock,
    PipelineGenerator,
    SettingsBinder,
    compile_image_schema,
    GroupingPlan,
    ImageAssignment,
    ImagesRule,
    PipelineImageSchema,
    MetadataExtractionRule,
    MetadataSource,
)
```

### Required Deletes

- Delete `_is_public_api_export`.
- Delete `_COMPATIBILITY_EXPORTS`.
- Delete the dynamic `globals().items()` scan and `getattr(value, "__module__")`.

### AST Bulk Edit

Use AST to verify the final object list before editing:

```python
import ast
from pathlib import Path

path = Path("benchmark/converter/__init__.py")
tree = ast.parse(path.read_text())
imports = []
for node in tree.body:
    if isinstance(node, ast.ImportFrom):
        for alias in node.names:
            if alias.name != "*":
                imports.append(alias.asname or alias.name)
print(tuple(imports))
```

The edit itself is small enough to do manually with `apply_patch`.

### Tests

Run:

```text
python -m pytest tests/unit/test_cellprofiler_interop_namespace.py tests/unit/test_cellprofiler_source_schema.py -q
```

After the tests pass, measure converter import cost:

```text
python -X importtime -c "import benchmark.converter" 2> /tmp/benchmark_converter_importtime.txt
```

### Dry-Run Review

- The compatibility symbols are still exported, but by object identity rather
  than string membership.
- `public_names_from_objects()` already exists in core and is used by the
  benchmark dataset/pipeline/metric modules.
- No lazy import behavior is introduced here; converter already eagerly imports
  these objects today.

## Phase 3: Tool and Metric Base Contracts Declare Required Names

### Authority

- `ToolAdapter` owns adapter-level fields: `name`, `version`, and `run()`.
- `MetricCollector` owns metric-level field: `name`.
- Concrete adapters and tests must not need `getattr` or `hasattr` checks for
  these fields.

### Final API

In `benchmark/contracts/tool_adapter.py`:

```python
from typing import ClassVar

class ToolAdapter(ABC):
    name: ClassVar[str]
    version: str = "unknown"

    @abstractmethod
    def validate_installation(self) -> None:
        pass

    @abstractmethod
    def run(
        self,
        dataset_path: Path,
        pipeline_name: str,
        pipeline_params: BenchmarkParameterMap,
        metrics: Sequence[MetricCollector],
        output_dir: Path,
    ) -> BenchmarkResult:
        pass
```

In `benchmark/contracts/metric.py`:

```python
from typing import ClassVar

class MetricCollector(ABC):
    name: ClassVar[str]

    @abstractmethod
    def __enter__(self) -> Self:
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    @abstractmethod
    def get_result(self) -> BenchmarkMetricValue:
        pass
```

Then in `benchmark/runner.py`:

```python
def _cached_metric_values(
    cached_metrics: object,
    *,
    requested_metrics: Sequence[MetricCollector],
) -> BenchmarkMetricMap:
    if not isinstance(cached_metrics, Mapping):
        return {}
    requested_names = tuple(metric.name for metric in requested_metrics)
    return {
        name: cached_metrics[name]
        for name in requested_names
        if name in cached_metrics
    }
```

And:

```python
"tool_version": adapter.version
```

In `OpenHCSAdapter.run()` and `CellProfilerAdapter.run()`:

```python
def run(
    self,
    dataset_path: Path,
    pipeline_name: str,
    pipeline_params: BenchmarkParameterMap,
    metrics: Sequence[MetricCollector],
    output_dir: Path,
) -> BenchmarkResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    request = OpenHCSRunRequest(
        dataset_path=dataset_path,
        pipeline_name=pipeline_name,
        pipeline_params=dict(pipeline_params),
        metrics=self._validated_metric_collectors(metrics),
        output_dir=output_dir,
    )
    return self._run_converted_cppipe_pipeline(request)
```

Internal request dataclasses normalize to a concrete dict:

```python
pipeline_params=dict(pipeline_params)
```

### AST Bulk Edit

Use AST to locate only contract-bound `getattr`/`hasattr`:

```python
import ast
from pathlib import Path

for path in (Path("benchmark/runner.py"),):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = (
                node.func.id if isinstance(node.func, ast.Name)
                else node.func.attr if isinstance(node.func, ast.Attribute)
                else None
            )
            if name in {"getattr", "hasattr"}:
                print(path, node.lineno, ast.unparse(node))
```

Expected rewrites:

```text
hasattr(metric, "name") -> delete the filter
getattr(adapter, "version", "unknown") -> adapter.version
```

Do not touch report/dataclass formatting `getattr` calls in this phase.

### Tests

Run:

```text
python -m pytest tests/unit/test_runner_cellprofiler_compatibility.py tests/unit/test_openhcs_adapter.py tests/unit/test_cellprofiler_adapter.py -q
```

### Dry-Run Review

- This phase does not introduce a registry.
- `version = "unknown"` on the base class preserves current behavior for test
  adapters that do not define `version`.
- `MetricCollector.name` remains a class-var contract. Do not add
  `__init_subclass__` validation in this batch; existing tests and concrete
  classes already exercise the contract through direct `metric.name` access.

## Phase 4: Benchmark Runs OpenHCS Plates With OpenHCS Config

### Authority

Benchmark code must not own OpenHCS execution semantics. It prepares the plate,
installs OpenHCS config, calls the existing OpenHCS compile/execute path, and
records timings. The authorities are:

- `GlobalPipelineConfig` for worker count, threading, multiprocessing start
  method, materialization policy, microscope selection, output paths, and VFS.
- `PipelineConfig` for per-pipeline settings. Well selection is
  `PipelineConfig.well_filter_config`, not benchmark axis selection.
- `PipelineOrchestrator.initialize()`, `compile_pipelines()`, and
  `execute_compiled_plate()` for runtime work. Benchmark timing wraps these
  existing calls only.
- `PreparedGeneratedPipeline.generated_pipeline.pipeline_config` for
  generated pipeline config. The adapter passes this config into
  `PipelineOrchestrator` without reconstructing selection semantics.
- Any benchmark code that computes selected OpenHCS wells, axes, or axis counts
  is deleted. Well selection stays in `PipelineConfig.well_filter_config`.
- Generated CellProfiler pipelines with source schemas emit the current
  product-owned `PipelineConfig` from
  `PipelineGeneratorBuildStage._pipeline_config()`: `source_bindings_config`
  is always derived from `PipelineImageSchema.to_runtime_source_bindings_config()`
  when non-empty, and `microscope=Microscope.SOURCE_BINDINGS` is set only when
  `PipelineImageSchemaSourceBindingsRepresentability(...).unsupported_fields()`
  is empty.
- Direct runtime execution receives an initialized orchestrator and a pipeline,
  then calls `compile_pipelines()` and `execute_compiled_plate()` without a
  runtime `well_filter` argument.

### Required Deletes

Delete OpenHCS selection and execution-setting strings from
`benchmark/adapters/openhcs.py`:

```text
OPENHCS_AXIS_FILTER_PARAM
OPENHCS_MAX_AXIS_COUNT_PARAM
OPENHCS_NUM_WORKERS_PARAM
OPENHCS_START_METHOD_PARAM
OPENHCS_USE_THREADING_PARAM
OpenHCSBenchmarkExecutionConfig
OpenHCSAxisSelection
OpenHCSRunRequest.axis_selection
_normalized_axis_filter
```

Delete comparison-run OpenHCS execution-setting fields from
`benchmark/cellprofiler_comparison.py`:

```text
ComparisonSuiteRunContext.openhcs_axis_filter
ComparisonSuiteRunContext.openhcs_max_axis_count
ComparisonSuiteRunContext.openhcs_num_workers
ComparisonSuiteRunContext.openhcs_start_method
ComparisonSuiteRunContext.openhcs_use_threading
```

Delete CLI arguments from `benchmark/cellprofiler_benchmark_cli.py`:

```text
--openhcs-axis
--openhcs-max-axis-count
--openhcs-start-method
--openhcs-num-workers
--openhcs-use-threading
```

Delete every `_run_comparison_case()` write that injects `OPENHCS_*` values into
`pipeline_params`.

Replace OpenHCS-named selection coupling in `benchmark/adapters/cellprofiler.py`
with a neutral source-schema selection projection:

```text
from benchmark.adapters.openhcs import OpenHCSAxisSelection
CellProfilerRunRequest.openhcs_axis_selection
_request_has_openhcs_axis_selection()
```

Do not delete the selected-source native CellProfiler behavior. It is needed to
run native CP against the same selected source-schema samples as OpenHCS. Keep
these declarations, but make them depend on the neutral selection projection
instead of `OpenHCSAxisSelection`:

```text
NativeCellProfilerInputDomainStrategyKey.SELECTED_SOURCE_SCHEMA_WELLS
NativeCellProfilerSelectedSourceMode
NativeCellProfilerSelectedSourceUniverse
NativeCellProfilerImportedMetadataPlacementPlan
SelectedWellSourceSchemaNativeCellProfilerInputDomainStrategy
native_cellprofiler_sample_scope_slug()
```

`SelectedWellSourceSchemaNativeCellProfilerInputDomainStrategy.accepts()` must
ask for a `SourceSchemaImageSetSelection | None`, not for OpenHCS axis fields.
`EmbeddedImagePlaneNativeCellProfilerInputDomainStrategy.accepts()` must guard
on that same neutral selection projection.

Delete `OpenHCSAxisSelection` from `benchmark/__init__.py::__all__`.

Replace OpenHCS-named subset settings in benchmark manifests and tests. Remove
`openhcs_max_axis_count`, `openhcs_axis_filter`, `openhcs_num_workers`,
`openhcs_start_method`, and `openhcs_use_threading` from:

```text
benchmark/manifests/*.json
benchmark/manifests/README.md
tests/unit/test_openhcs_adapter.py
tests/unit/test_cellprofiler_adapter.py
tests/unit/test_cellprofiler_comparison.py
tests/unit/test_runner_cellprofiler_compatibility.py
tests/unit/test_well_throughput_scaling.py
tests/unit/test_benchmark_manifests.py
tests/unit/agent/test_knowledge_base_service.py
```

For manifests that intentionally run a source subset, move the subset out of
`pipeline_params` and into a neutral source-schema selection payload:

```json
{
  "source_schema_image_set_selection": {
    "max_image_set_count": 1
  }
}
```

Case-level `source_schema_image_set_selection` overrides the manifest-level
value. The manifest loader parses that JSON object into
`openhcs.core.source_schema_workspace.SourceSchemaImageSetSelection` and stores
the typed object on the `CellProfilerComparisonCase`. Do not pass that selection
through `pipeline_params`.

Tests that only need a merged-parameter example use a neutral benchmark-owned
parameter such as `compare_image_outputs`; they do not invent a replacement
OpenHCS subset parameter.

### Product Generator Contract

Do not change `openhcs/interop/cellprofiler/pipeline_generator.py` as part of
the benchmark cleanup. The current generator contract is already explicit and
is covered by `tests/unit/test_cppipe_corpus.py`.

`PipelineGeneratorBuildStage.generate()` parses the `.cppipe`, partitions
processing and infrastructure modules, compiles a `CellProfilerSymbolTable`, and
uses `symbol_table.source_schema` in two places:

- `PipelineGeneratorBuildStage._pipeline_config(symbol_table.source_schema)`
  builds the generated `PipelineConfig`.
- `PipelineGeneratorCodeEmitter.generate_steps_from_registry(...,
  symbol_table.source_schema)` passes the schema into module processing
  component resolution.

The current `_pipeline_config()` implementation is:

```python
@staticmethod
def _pipeline_config(source_schema: PipelineImageSchema) -> PipelineConfig | None:
    """Return ObjectState-owned pipeline config derived from source schema."""
    if source_schema.is_empty:
        return None
    source_bindings_config = source_schema.to_runtime_source_bindings_config()
    if source_bindings_config.is_empty:
        return None
    if PipelineImageSchemaSourceBindingsRepresentability(
        source_schema
    ).unsupported_fields():
        return PipelineConfig(source_bindings_config=source_bindings_config)
    return PipelineConfig(
        microscope=Microscope.SOURCE_BINDINGS,
        source_bindings_config=source_bindings_config,
    )
```

This is a deliberate two-mode contract:

- `PipelineConfig.microscope` selects the microscope handler.
- `PipelineConfig.source_bindings_config` supplies typed source-binding
  declarations.
- Source schemas representable as source-bindings workspaces set both fields.
- Source schemas with unsupported init-time fields still emit
  `source_bindings_config` but leave `microscope` unset; product source-schema
  preparation owns any workspace materialization needed for those schemas.

`ModuleArtifactContracts.source_bindings` is a separate step-level authority.
It is built in `openhcs/interop/cellprofiler/symbol_table.py` from external
source symbols and rendered into generated `FunctionStep(source_bindings=...)`
by
`openhcs/interop/cellprofiler/module_processing_components.py::generated_function_step_semantic_argument_lines()`.
Do not infer step source bindings from the top-level pipeline config.

The runtime call chain is:

```text
PipelineOrchestrator.initialize_microscope_handler()
  -> shared_context = orchestrator.get_effective_config()
  -> microscope_type = shared_context.microscope.value
  -> create_microscope_handler(
         microscope_type=microscope_type,
         source_bindings_config=shared_context.source_bindings_config,
     )
  -> SourceBindingsHandler.create(..., source_bindings_config=...)
```

Do not describe this as `microscope` taking source bindings. The enum selects a
handler; `source_bindings_config` is the separate typed payload.

Keep the current `tests/unit/test_cppipe_corpus.py` expectations: non-empty
schemas must have `pipeline_config.source_bindings_config ==
source_schema.to_runtime_source_bindings_config()`, and only schemas with no
`PipelineImageSchemaSourceBindingsRepresentability(...).unsupported_fields()`
must set `pipeline_config.microscope is Microscope.SOURCE_BINDINGS`.

Do not move currently supported corpus cases to `KNOWN_INVALID` in this
benchmark cleanup.

In `openhcs/interop/cellprofiler/runtime_pipeline.py`, delete the `well_filter`
argument from `execute_pipeline_direct()`:

```python
def execute_pipeline_direct(
    orchestrator: Any,
    pipeline: Pipeline,
    *,
    phase_timing: Any | None = None,
    compile_phase: Any | None = None,
    execute_phase: Any | None = None,
) -> DirectPipelineExecution:
    progress_bridge = DirectExecutionProgressBridge(
        queue=DirectExecutionProgressSink(),
    )
    try:
        set_progress_queue(progress_bridge.queue)
        with _optional_phase(phase_timing, compile_phase):
            compilation_result = orchestrator.compile_pipelines(
                pipeline_definition=pipeline,
            )
        execution_bundle = compilation_result["execution_bundle"]
        ...
```

The compiler reads `PipelineConfig.well_filter_config`. Runtime direct execution
does not perform its own axis discovery, empty-axis validation, or well-filter
intersection.

### OpenHCS Adapter Edit Recipe

In `benchmark/adapters/openhcs.py`, add `replace` to the dataclass imports and
make the adapter receive the existing config object:

```python
from dataclasses import dataclass, fields, is_dataclass, replace
from objectstate.lazy_factory import (
    ensure_global_config_context,
    rebuild_lazy_config_with_new_global_reference,
)
from openhcs.core.config import GlobalPipelineConfig, LazyWellFilterConfig
from openhcs.core.source_schema_workspace import SourceSchemaImageSetSelection


class OpenHCSAdapter(ToolAdapter):
    name = "OpenHCS"

    def __init__(
        self,
        *,
        global_config: GlobalPipelineConfig | None = None,
        source_schema_image_set_selection: SourceSchemaImageSetSelection | None = None,
    ) -> None:
        import openhcs
        from polystore.base import ensure_storage_registry, storage_registry
        from polystore.filemanager import FileManager

        self.version = openhcs.__version__
        ensure_storage_registry()
        self._filemanager = FileManager(storage_registry)
        self.global_config = global_config or GlobalPipelineConfig()
        self.source_schema_image_set_selection = source_schema_image_set_selection
```

Do not add an adapter-owned execution-config dataclass. `GlobalPipelineConfig`
is the config authority.

Remove `OpenHCSRunRequest.axis_selection`. Keep request properties that are
benchmark concerns, such as timeout, cache, comparison, materialization, and
reference-output controls.

Add `source_schema_image_set_selection:
SourceSchemaImageSetSelection | None` to `OpenHCSRunRequest`; `OpenHCSAdapter.run()`
sets it from `self.source_schema_image_set_selection`.

Update the `OpenHCSRunRequest(...)` construction in `OpenHCSAdapter.run()` to:

```python
request = OpenHCSRunRequest(
    dataset_path=dataset_path,
    pipeline_name=pipeline_name,
    pipeline_params=pipeline_params,
    metrics=self._validated_metric_collectors(metrics),
    output_dir=output_dir,
    source_schema_image_set_selection=self.source_schema_image_set_selection,
)
```

Remove imports that only supported benchmark-owned OpenHCS execution parsing:

```text
openhcs.constants.MULTIPROCESSING_AXIS
openhcs.core.config.MultiprocessingStartMethod
```

Keep these product CellProfiler ingestion imports. They are the correct
authority for source-root admission, source-schema workspace preparation, and
the execution plate path:

```text
openhcs.core.source_schema_workspace.SourceSchemaImageSetSelection
openhcs.interop.cellprofiler.source_schema_ingestion.CellProfilerSourceSchemaWorkspaceRequest
openhcs.interop.cellprofiler.source_schema_ingestion.prepare_cellprofiler_source_schema_workspace
openhcs.interop.cellprofiler.source_schema_ingestion.CellProfilerPipelinePreparationError
openhcs.interop.cellprofiler.source_schema_ingestion.CellProfilerSourceWorkspaceMaterializationError
openhcs.interop.cellprofiler.source_schema_ingestion.CellProfilerSourceSchemaWorkspace
```

The benchmark adapter must not call `materialize_source_schema_workspace()`
directly. It does choose the product ingestion output location, then passes that
path to the product request as `workspace_root`.

In `_run_converted_cppipe_pipeline()`, keep:

```python
ingestion = prepare_cellprofiler_source_schema_workspace(
    CellProfilerSourceSchemaWorkspaceRequest(
        source_root=request.dataset_path,
        cppipe_path=cppipe_path,
        workspace_root=source_workspace_path,
        generated_pipeline_path=generated_module_path,
        filemanager=self._filemanager,
        image_set_selection=request.source_schema_image_set_selection,
        prune_dead_unmaterialized_artifact_steps=(
            generation_policy.prune_dead_unmaterialized_artifact_steps
        ),
        materialize_skipped_save_images=(
            generation_policy.materialize_skipped_save_images
        ),
        materialize_terminal_images=(
            generation_policy.materialize_terminal_images
        ),
    )
)
prepared = ingestion.prepared_pipeline
execution_plate_path = ingestion.execution_plate_path
source_workspace_path = ingestion.source_workspace_path
```

After `prepared = ...`, keep the generated pipeline config, combine the neutral
source-schema selection into `PipelineConfig.well_filter_config`, and build the
global config by replacing benchmark-run fields on the injected
`self.global_config`:

```python
pipeline_config = prepared.generated_pipeline.pipeline_config or PipelineConfig()
selection = request.source_schema_image_set_selection
if selection is not None and selection.well_filter:
    pipeline_config = replace(
        pipeline_config,
        well_filter_config=LazyWellFilterConfig(
            well_filter=list(selection.well_filter),
        ),
    )
elif selection is not None and selection.max_image_set_count is not None:
    pipeline_config = replace(
        pipeline_config,
        well_filter_config=LazyWellFilterConfig(
            well_filter=selection.max_image_set_count,
        ),
    )
global_config = replace(
    self.global_config,
    analysis_consolidation_config=AnalysisConsolidationConfig(enabled=False),
    path_planning_config=PathPlanningConfig(
        global_output_folder=request.output_dir,
        output_dir_suffix=output_suffix,
    ),
    vfs_config=VFSConfig(materialization_backend=MaterializationBackend.DISK),
    materialize_runtime_artifacts=request.materialize_runtime_artifacts,
    materialization_results_path=output_plate_root / "results",
)
ensure_global_config_context(GlobalPipelineConfig, global_config)
pipeline_config = rebuild_lazy_config_with_new_global_reference(
    pipeline_config,
    global_config,
    GlobalPipelineConfig,
)
orchestrator = PipelineOrchestrator(
    execution_plate_path,
    pipeline_config=pipeline_config,
)
with phase_timing.phase(BenchmarkPhase.INITIALIZE_RUNTIME):
    orchestrator.initialize()
```

Do not copy `pipeline_config.microscope` into `GlobalPipelineConfig`. ObjectState
resolution already makes `PipelineConfig(microscope=Microscope.SOURCE_BINDINGS)`
resolve to `GlobalPipelineConfig.microscope=Microscope.SOURCE_BINDINGS`.

This inline merge uses the lazy config type generated from `WellFilterConfig`.
The value comes from the neutral source-schema selection projection:

```text
selection is None -> leave pipeline_config unchanged
selection.well_filter is non-empty -> LazyWellFilterConfig(well_filter=list(selection.well_filter))
selection.well_filter is empty and selection.max_image_set_count is not None -> LazyWellFilterConfig(well_filter=selection.max_image_set_count)
selection has neither field set -> leave pipeline_config unchanged
```

This projection does not query available OpenHCS component keys. The compiler's
existing `WellFilterProcessor` resolves the typed `WellFilterConfig` against the
initialized plate. Product source-schema ingestion still applies
`SourceSchemaImageSetSelection` before materialization when materialization is
needed; the well-filter config is the compile-time OpenHCS projection of the
same typed selection for representable source-bindings workspaces.

`execute_pipeline_direct()` owns compile and execute timing through the existing
OpenHCS runtime facade.

Delete the post-initialize selected-axis computation:

```python
selected_axes = request.axis_selection.resolve(
    tuple(orchestrator.get_component_keys(MULTIPROCESSING_AXIS))
)
```

Delete the `well_filter=` keyword at both `execute_pipeline_direct()` callsites:

```python
execution = execute_pipeline_direct(
    orchestrator,
    prepared.pipeline,
    phase_timing=phase_timing,
)
```

Update `_run_table_only_streaming_equivalence()` by removing its
`selected_axes` parameter and making its internal `execute_pipeline_direct()`
call use the same no-`well_filter` form.

In result provenance, delete the nested `axis_selection` payload. Keep
`axis_count` and `executed_axes` because they are runtime observations:

```python
"axis_count": axis_count,
"executed_axes": executed_axes,
```

Delete `source_workspace_path` from `_RuntimeExecutionCacheHit`, runtime cache
manifest writes, runtime cache manifest reads, and result provenance. The
benchmark runtime cache records OpenHCS outputs and timings, not internal
workspace topology.

Partial source-schema execution is a comparison-suite input, not an OpenHCS-only
axis input. The comparison suite carries it as
`SourceSchemaImageSetSelection | None`, passes it to both adapters, and the
OpenHCS adapter projects it into `PipelineConfig.well_filter_config` before
orchestrator compilation.

### Suite Runner

Change `run_comparison_suite()` to accept a real config object:

```python
def run_comparison_suite(
    cases: Iterable[CellProfilerComparisonCase],
    *,
    output_root: Path,
    suite_id: str,
    repeats: int = 1,
    reuse_openhcs_cache: bool = True,
    speedup_target: float = DEFAULT_SPEEDUP_TARGET,
    native_reference_root: Path | None = None,
    require_native_reference: bool = False,
    discard_openhcs_outputs: bool = False,
    continue_on_error: bool = False,
    metric_policy: ComparisonMetricPolicy = ComparisonMetricPolicy(),
    coverage_manifest_path: Path | None = None,
    openhcs_global_config: GlobalPipelineConfig | None = None,
    source_schema_image_set_selection: SourceSchemaImageSetSelection | None = None,
) -> tuple[CellProfilerComparisonObservation, ...]:
    context = ComparisonSuiteRunContext(
        suite_id=suite_id,
        speedup_target=speedup_target,
        reuse_openhcs_cache=reuse_openhcs_cache,
        native_reference_root=native_reference_root,
        require_native_reference=require_native_reference,
        discard_openhcs_outputs=discard_openhcs_outputs,
        continue_on_error=continue_on_error,
        metric_policy=metric_policy,
        openhcs_global_config=openhcs_global_config or GlobalPipelineConfig(),
        source_schema_image_set_selection=source_schema_image_set_selection,
    )
```

Change `ComparisonSuiteRunContext`:

```python
@dataclass(frozen=True, slots=True)
class ComparisonSuiteRunContext:
    suite_id: str
    speedup_target: float
    reuse_openhcs_cache: bool
    native_reference_root: Path | None
    require_native_reference: bool
    discard_openhcs_outputs: bool
    continue_on_error: bool
    metric_policy: ComparisonMetricPolicy
    openhcs_global_config: GlobalPipelineConfig
    source_schema_image_set_selection: SourceSchemaImageSetSelection | None
```

In `_run_comparison_case()`, pass a configured adapter into the existing runner
hook:

```python
result = run_cellprofiler_cppipe_parity(
    case.dataset_path,
    case.cppipe_path,
    metrics=context.metric_policy.collectors(),
    dataset_id=case.dataset_id,
    pipeline_name=case.name,
    microscope_type=case.microscope_type,
    pipeline_params=pipeline_params,
    output_root=output_root / "tool_outputs",
    equivalence_reference_output_dir=native_reference.reference_output_dir,
    native_cellprofiler_output_dir=native_reference.output_dir,
    reuse_openhcs_cache=context.reuse_openhcs_cache,
    cellprofiler_adapter=CellProfilerAdapter(
        source_schema_image_set_selection=(
            context.source_schema_image_set_selection
        ),
    ),
    openhcs_adapter=OpenHCSAdapter(
        global_config=context.openhcs_global_config,
        source_schema_image_set_selection=(
            context.source_schema_image_set_selection
        ),
    ),
)
```

Do not pass OpenHCS execution settings through `pipeline_params`.

In `_run_comparison_case()`, delete the block that writes
`OPENHCS_AXIS_FILTER_PARAM`, `OPENHCS_MAX_AXIS_COUNT_PARAM`,
`OPENHCS_NUM_WORKERS_PARAM`, `OPENHCS_START_METHOD_PARAM`, and
`OPENHCS_USE_THREADING_PARAM`. Build `pipeline_params` only from case data,
native CellProfiler image-set controls, timeout, comparison, cache, and
reference-output controls.

### CLI

The comparison CLI no longer accepts OpenHCS axis or worker flags. The CLI calls
`run_comparison_suite()` without OpenHCS config overrides. Programmatic callers
that need worker settings pass:

```python
run_comparison_suite(
    cases,
    openhcs_global_config=GlobalPipelineConfig(
        num_workers=4,
        use_threading=False,
        multiprocessing_start_method=MultiprocessingStartMethod.FORK,
    ),
)
```

Direct OpenHCS execution tests that cover well filtering construct a
`PipelineOrchestrator` with a typed pipeline config:

```python
PipelineConfig(
    well_filter_config=LazyWellFilterConfig(well_filter=1)
)
```

They do not route that need through `pipeline_params`.

The comparison CLI removes OpenHCS axis flags and, where source subsetting is
needed, exposes neutral source-schema flags:

```text
--source-schema-well
--source-schema-max-image-set-count
```

The CLI constructs `SourceSchemaImageSetSelection` and passes it to
`run_comparison_suite(source_schema_image_set_selection=...)`. It does not
construct an OpenHCS selector.

### Throughput Scaling

In `openhcs/interop/cellprofiler/source_schema_ingestion.py`, add a typed request
field:

```python
force_materialization: bool = False
```

Then change `prepare_cellprofiler_source_schema_workspace()` so
`force_materialization=True` bypasses the early return for
`pipeline_config.microscope is Microscope.SOURCE_BINDINGS` and still uses
`CellProfilerSourceSchemaMaterializer`.

In `benchmark/well_throughput_scaling.py`, remove the
`OpenHCSAxisSelection` import and stop reading `pipeline_params` for OpenHCS
axis selection. This benchmark's synthetic plate size is already owned by
`WellThroughputMode.well_count`.

Do not call `materialize_source_schema_workspace()` directly from throughput
scaling. Runtime preparation is:

```python
ingestion = prepare_cellprofiler_source_schema_workspace(
    CellProfilerSourceSchemaWorkspaceRequest(
        source_root=dataset_path,
        cppipe_path=cppipe_path,
        workspace_root=(
            output_root / f"{dataset_path.name}_{cppipe_path.stem}_source_workspace"
        ),
        generated_pipeline_path=generated_module_path,
        image_set_selection=source_schema_image_set_selection,
        prune_dead_unmaterialized_artifact_steps=True,
        materialize_skipped_save_images=False,
        materialize_terminal_images=False,
        force_materialization=True,
    )
)
prepared = ingestion.prepared_pipeline
source_workspace_path = ingestion.source_workspace_path
if source_workspace_path is None:
    raise RuntimeError("Forced source-schema materialization returned no workspace.")
well_ids = expand_source_schema_workspace_wells(
    source_workspace_path / "openhcs_metadata.json",
    _synthetic_well_ids(mode.well_count),
)
pipeline_config = replace(
    prepared.generated_pipeline.pipeline_config or PipelineConfig(),
    well_filter_config=LazyWellFilterConfig(well_filter=list(well_ids)),
)
orchestrator = PipelineOrchestrator(
    source_workspace_path,
    pipeline_config=pipeline_config,
)
orchestrator.initialize()
execution = execute_pipeline_direct(orchestrator, prepared.pipeline, ...)
```

The only benchmark-specific operation left here is
`expand_source_schema_workspace_wells()`, because synthetic throughput scaling
intentionally duplicates virtual wells without copying source images.

### Cache Cleanup

Do not replace ignored-key sets with a parameter registry. Delete the old
runtime cache helper path and compare the canonical execution cache key exactly.

In `benchmark/adapters/openhcs.py`, delete:

```text
_RUNTIME_EXECUTION_CACHE_IGNORED_PARAM_KEYS
_RUNTIME_EXECUTION_CACHE_HELPER_KEYS
_runtime_execution_cache_identity()
_legacy_runtime_execution_cache_identity()
_runtime_execution_pipeline_params()
```

Replace `_runtime_execution_cache_key_matches()` with:

```python
def _runtime_execution_cache_key_matches(
    cached_key: object,
    expected_key: object,
) -> bool:
    return cached_key == expected_key
```

Replace `_runtime_execution_cache_key_for_snapshot()` with:

```python
def _runtime_execution_cache_key_for_snapshot(cache_key: object) -> object:
    return cache_key
```

In `benchmark/runner.py`, delete:

```text
_LEGACY_SOURCE_TREE_CACHE_KEY
_EXECUTION_CACHE_IGNORED_PARAM_KEYS
_execution_cache_pipeline_params()
```

Then change `_openhcs_execution_cache_key()` so `pipeline_params` are built from
the already canonical params passed before runtime cache controls are injected:

```python
"pipeline_params": _json_ready(dict(pipeline_params)),
"execution_source_tree": _source_tree_fingerprint(
    excluded_cache_domains=_EXECUTION_SOURCE_CACHE_EXCLUDED_DOMAINS,
),
```

Do not include `legacy_source_tree` in new cache keys.

Delete the `well_filter` argument from the benchmark facade
`benchmark/converter/runtime_pipeline.py::execute_pipeline_direct()`. The facade
calls product runtime execution with only:

```python
return execute_pipeline_direct_runtime(
    orchestrator,
    pipeline,
    phase_timing=phase_timing,
    compile_phase=BenchmarkPhase.COMPILE_OPENHCS,
    execute_phase=BenchmarkPhase.EXECUTE_OPENHCS,
)
```

Edit `openhcs/interop/cellprofiler/runtime_pipeline.py` in the product contract
cleanup above so the product direct-execution ABI also has no `well_filter`
argument.

### AST Inventory

Run this inventory before editing. The only production hits are the deletion
targets listed below the script:

```python
import ast
from pathlib import Path

names = {
    "OPENHCS_AXIS_FILTER_PARAM",
    "OPENHCS_MAX_AXIS_COUNT_PARAM",
    "OPENHCS_NUM_WORKERS_PARAM",
    "OPENHCS_START_METHOD_PARAM",
    "OPENHCS_USE_THREADING_PARAM",
    "OpenHCSAxisSelection",
    "OpenHCSBenchmarkExecutionConfig",
    "openhcs_axis_selection",
    "native_cellprofiler_sample_scope_slug",
    "_request_has_openhcs_axis_selection",
    "materialize_source_schema_workspace",
    "source_workspace_path",
    "well_filter",
}

for path in sorted(Path("benchmark").rglob("*.py")):
    if "cellprofiler_source" in path.parts or "cellprofiler_library" in path.parts:
        continue
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in names:
            print(path, node.lineno, node.id)
        if isinstance(node, ast.keyword) and node.arg == "well_filter":
            print(path, node.lineno, "keyword well_filter")
```

Expected production hits to fix:

```text
benchmark/adapters/openhcs.py: delete OpenHCSAxisSelection declaration/uses
benchmark/adapters/openhcs.py: delete callsite passing execute_pipeline_direct(..., well_filter=selected_axes)
benchmark/converter/runtime_pipeline.py: delete facade well_filter argument
benchmark/adapters/cellprofiler.py: replace OpenHCSAxisSelection import/property/strategy helpers with SourceSchemaImageSetSelection
benchmark/well_throughput_scaling.py: replace OpenHCSAxisSelection and direct source workspace materialization with CellProfilerSourceSchemaWorkspaceRequest(force_materialization=True)
benchmark/__init__.py: delete OpenHCSAxisSelection export
benchmark/cellprofiler_comparison.py: delete openhcs_* context fields and pipeline_params injection
benchmark/cellprofiler_benchmark_cli.py: delete openhcs_* CLI arguments
openhcs/interop/cellprofiler/runtime_pipeline.py: delete execute_pipeline_direct well_filter parameter and compile-time axis lookup
```

Then run a manifest grep, not only AST, because JSON manifests are part of the
benchmark API:

```text
rg -n "openhcs_(axis_filter|max_axis_count|num_workers|start_method|use_threading)" benchmark tests
```

After the edit, production hits are allowed only for unrelated environment
variables such as `OPENHCS_BENCHMARK_DATASET_CACHE_ROOT` or test names that no
longer carry OpenHCS execution semantics.

### Tests

Run:

```text
source .venv/bin/activate
python -m pytest tests/unit/test_openhcs_adapter.py tests/unit/test_cellprofiler_adapter.py tests/unit/test_runner_cellprofiler_compatibility.py tests/unit/test_cellprofiler_comparison.py -q
```

Then focused benchmark suite:

```text
python -m pytest tests/unit/test_dataset_registry.py tests/unit/test_dataset_acquisition_infra.py tests/unit/test_benchmark_manifests.py tests/unit/test_comparison_manifest_acquisition.py tests/unit/test_benchmark_timing.py tests/unit/test_metrics_memory.py tests/unit/test_benchmark_progress.py tests/unit/test_runner_cellprofiler_compatibility.py tests/unit/test_openhcs_adapter.py tests/unit/test_cellprofiler_adapter.py -q
```

### Dry-Run Review

- OpenHCS worker settings enter the benchmark as `GlobalPipelineConfig`, not as
  `pipeline_params`.
- OpenHCS well filtering enters through `GeneratedCPPipePipeline.pipeline_config`
  and `PipelineConfig.well_filter_config`, not as benchmark axis-selection code.
- The benchmark adapter calls existing OpenHCS initialization, compilation, and
  execution APIs; timing wraps those APIs only.
- No `BenchmarkPipelineParameter`, `BenchmarkParameterCacheRole`,
  `BenchmarkCacheKeyField`, `OpenHCSBenchmarkImageSetSelection`,
  `OpenHCSAxisSelection`, or replacement selection class is added.

## Phase 5: Bio-Formats HCS Validation Uses Microscope Handler Authority

### Authority

Do not treat the benchmark Bio-Formats catalog as a plate-format registry or
axis authority. Product authority already exists:

- `BioFormatsHandler` is the microscope handler registered under
  `_microscope_type = "bioformats"`.
- `BioFormatsMetadataHandler.find_metadata_file()` detects Bio-Formats-readable
  folders by calling `BioFormatsDatasetAuthority().project(plate_path)`.
- `BioFormatsDatasetAuthority` reads through `BioFormatsCompositeAdapter` and
  projects through `BioFormatsSPWProjector` or `BioFormatsLayoutProjector`.
- `BioFormatsCompositeAdapter` tries registered metadata adapters: manifest,
  Java Bio-Formats, then filename-layout discovery.
- `BioFormatsFilenameLayoutParser` subclasses are AutoRegisterMeta-owned layout
  parsers for vendor filename layouts.
- `BioFormatsWorkspaceMetadataWriter.write()` emits `openhcs_metadata.json`
  from the projected source planes.

The benchmark file `benchmark/datasets/bioformats_hcs.py` declares only public
sample acquisition sources. It does not declare expected wells, sites, channels,
Z, or timepoints.

### Final API

Do not add a Bio-Formats-specific sample registry, catalog accessor, or sample
declaration tuple. Reuse the existing dataset declaration infrastructure.

First add generic dataset tags in `benchmark/contracts/dataset.py`:

```python
class BenchmarkDatasetTag(Enum):
    BIOFORMATS_HCS_VALIDATION = "bioformats_hcs_validation"


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    tags: frozenset[BenchmarkDatasetTag] = frozenset()
```

Then add the matching declaration field in `benchmark/datasets/registry.py` and
pass it through `BenchmarkDatasetDeclaration.to_spec()`:

```python
class BenchmarkDatasetDeclaration(ABC, metaclass=AutoRegisterMeta):
    tags: ClassVar[frozenset[BenchmarkDatasetTag]] = frozenset()

    @classmethod
    def to_spec(cls) -> DatasetSpec:
        return DatasetSpec(
            tags=cls.tags,
        )
```

Keep `benchmark/datasets/bioformats_hcs.py`, but replace its current dataclasses,
axis expectations, catalog, and registry with registered dataset declaration
classes. The first class must be:

```python
class OmeTiffHcsCompanionDataset(
    ImageCountValidatedDatasetMixin,
    BenchmarkDatasetDeclaration,
):
    id = "ome_tiff_hcs_companion"
    public_alias = "OME_TIFF_HCS_COMPANION"
    size_bytes = 64_000
    microscope_type = "bioformats"
    expected_count = 5
    tags = frozenset({BenchmarkDatasetTag.BIOFORMATS_HCS_VALIDATION})
    source = DatasetSourceSpec(
        kind=DatasetSourceKind.URL_FILES,
        urls=(
            f"{OME_DOWNLOADS_ROOT}/OME-TIFF/2016-06/plate-companion/hcs.companion.ome",
            f"{OME_DOWNLOADS_ROOT}/OME-TIFF/2016-06/plate-companion/well-A2.ome.tiff",
            f"{OME_DOWNLOADS_ROOT}/OME-TIFF/2016-06/plate-companion/well-B1.ome.tiff",
            f"{OME_DOWNLOADS_ROOT}/OME-TIFF/2016-06/plate-companion/well-B3.ome.tiff",
            f"{OME_DOWNLOADS_ROOT}/OME-TIFF/2016-06/plate-companion/well-C2.ome.tiff",
            f"{OME_DOWNLOADS_ROOT}/OME-TIFF/2016-06/plate-companion/well-C2-2.ome.tiff",
        ),
    )
```

Convert every remaining row in `BIOFORMATS_HCS_DECLARATIONS` to a
`BenchmarkDatasetDeclaration` subclass in the same module. Preserve these fields
exactly:

```text
dataset_id -> id
display_name -> class docstring
source_page + files -> DatasetSourceSpec.urls
size_bytes -> size_bytes
expected_count -> expected_count
format/vendor/notes -> remove from production declarations
axes -> remove from production declarations
```

Import the module in `benchmark/datasets/registry.py` after
`BenchmarkDatasetDeclaration`, `ImageCountValidatedDatasetMixin`, and
`_git_sparse_with_archives()` are defined and before `dataset_declarations()`:

```python
from benchmark.datasets import bioformats_hcs as _bioformats_hcs_declarations
```

The SSOT for acquisition is then:

```text
BenchmarkDatasetDeclaration.__registry__
  -> dataset_specs()
  -> DATASET_REGISTRY
  -> get_dataset_spec()
  -> acquire_dataset()
```

The Bio-Formats validation CLI selects dataset ids from `DATASET_REGISTRY` with
this predicate:

```python
def _bioformats_hcs_validation_specs() -> tuple[DatasetSpec, ...]:
    return tuple(
        spec
        for spec in DATASET_REGISTRY.values()
        if spec.microscope_type == "bioformats"
        and BenchmarkDatasetTag.BIOFORMATS_HCS_VALIDATION in spec.tags
    )
```

Delete from benchmark production:

```text
BioFormatsHcsAxisExpectation
BioFormatsHcsCatalogRow.axes
BioFormatsHcsDatasetDeclaration.axes
BioFormatsHcsCatalogRow
BioFormatsHcsDatasetDeclaration
BIOFORMATS_HCS_CATALOG
BIOFORMATS_HCS_REGISTRY
BIOFORMATS_HCS_SAMPLE_DECLARATIONS
```

### Validation Rewrite

Change `benchmark/bioformats_hcs_validation.py` to accept
`Iterable[DatasetSpec]` instead of Bio-Formats catalog rows, and change it from
expected-vs-observed axis checking to handler-derived evidence reporting.

Use microscope auto-detection to prove the product path:

```python
from openhcs.microscopes import create_microscope_handler

filemanager = _bioformats_filemanager()
handler = create_microscope_handler(
    "auto",
    plate_folder=acquired.path,
    filemanager=filemanager,
)
if not isinstance(handler, BioFormatsHandler):
    raise ValueError(
        "Bio-Formats HCS validation expected auto-detection to select "
        f"BioFormatsHandler, got {type(handler).__name__}."
    )
handler.initialize_workspace(acquired.path, filemanager)
```

Then report dimensions from the handler metadata, not the benchmark catalog:

```python
metadata_handler = handler.metadata_handler
wells = metadata_handler.get_well_values(acquired.path) or {}
sites = metadata_handler.get_site_values(acquired.path) or {}
channels = metadata_handler.get_channel_values(acquired.path) or {}
z_indexes = metadata_handler.get_z_index_values(acquired.path) or {}
timepoints = metadata_handler.get_timepoint_values(acquired.path) or {}
```

The CSV/JSON contains observed values only:

```text
wells
sites
channels
z_indexes
timepoints
well_count
site_count
channel_count
z_count
timepoint_count
virtual_file_count
loaded_plane_count
load_shapes
load_dtypes
```

Move the existing exact expected-axis checks into unit tests for
`BioFormatsDatasetAuthority`, `BioFormatsSPWProjector`, or the filename-layout
parser classes. Delete expected axes from benchmark production code.

### Grid-Dimension Instruction

The current usage inventory shows `grid_dimensions` feeds stitching and
position-generation artifact inputs (`mist`, `ashlar`, and source projection).
Treat it as logical tile-grid dimensions, not plate well layout. Keep
`BioFormatsMetadataHandler.get_grid_dimensions()` and
`BioFormatsWorkspaceMetadataWriter.write()` at `(1, 1)` for projected
Bio-Formats source planes. Report plate layout through component values from
`get_well_values()`, `get_site_values()`, `get_channel_values()`,
`get_z_index_values()`, and `get_timepoint_values()`.

Add a focused Bio-Formats test:

```python
def test_bioformats_metadata_reports_single_tile_grid_and_axis_components(
    tmp_path: Path,
) -> None:
    write_bioformats_manifest_fixture(tmp_path)
    filemanager = _bioformats_filemanager()
    handler = BioFormatsHandler(filemanager)
    handler.initialize_workspace(tmp_path, filemanager)

    metadata_handler = handler.metadata_handler
    assert metadata_handler.get_grid_dimensions(tmp_path) == (1, 1)
    assert tuple((metadata_handler.get_well_values(tmp_path) or {}).keys()) == ("A01",)
    assert tuple((metadata_handler.get_channel_values(tmp_path) or {}).keys()) == (
        "1",
        "2",
    )
```

### AST Inventory

Run:

```python
import ast
from pathlib import Path

names = {
    "BIOFORMATS_HCS_CATALOG",
    "BIOFORMATS_HCS_REGISTRY",
    "BioFormatsHcsAxisExpectation",
    "BioFormatsHcsCatalogRow",
    "BioFormatsHcsDatasetDeclaration",
    "axes",
    "axis_projection",
}

for path in (
    Path("benchmark/datasets/bioformats_hcs.py"),
    Path("benchmark/datasets/registry.py"),
    Path("benchmark/bioformats_hcs_validation.py"),
    Path("benchmark/cellprofiler_benchmark_cli.py"),
    Path("tests/unit/test_bioformats_hcs_validation.py"),
):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in names:
            print(path, node.lineno, node.id)
        if isinstance(node, ast.Attribute) and node.attr in names:
            print(path, node.lineno, f".{node.attr}")
```

Expected production rewrites:

```text
benchmark/datasets/bioformats_hcs.py: replace data catalog with BenchmarkDatasetDeclaration subclasses
benchmark/datasets/registry.py: import bioformats_hcs before DATASET_REGISTRY is built
benchmark/bioformats_hcs_validation.py: remove expected-vs-observed projection
benchmark/cellprofiler_benchmark_cli.py: select DatasetSpec entries from DATASET_REGISTRY/get_dataset_spec
tests/unit/test_bioformats_hcs_validation.py: assert handler-derived observed axes
```

### Tests

Run:

```text
source .venv/bin/activate
python -m pytest tests/unit/test_bioformats_hcs_validation.py tests/unit/test_bioformats_microscope_handler.py -q
```

Then run the integration evidence for real filename-layout projection:

```text
python -m pytest tests/integration/test_bioformats_imagexpress_synthetic.py -q
```

Accepted outcomes for the integration command:

```text
passed
skipped because the Bio-Formats Java dependency is unavailable
```

Any failed test blocks the phase.

### Dry-Run Review

- The benchmark no longer owns Bio-Formats axes.
- The benchmark no longer owns a Bio-Formats sample registry; it reuses
  `BenchmarkDatasetDeclaration` / `DATASET_REGISTRY`.
- Bio-Formats folder autodetection remains owned by the microscope handler and
  metadata adapters.
- The benchmark validates that the handler can autodetect, initialize
  `openhcs_metadata.json`, list virtual source planes, and load sample planes.
- The only curated benchmark authority left is which public sample files to
  download.

## Phase 6: Residual `getattr`/`Any` Audit Classification

After phases 1-5, rerun:

```text
python - <<'PY'
import ast
from pathlib import Path
for path in sorted(Path("benchmark").rglob("*.py")):
    if any(part in {"cellprofiler_library", "cellprofiler_source", "results"} for part in path.parts):
        continue
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
            if name in {"getattr", "hasattr"}:
                print(f"{path}:{node.lineno}:{name}")
PY
```

Allowed remaining categories:

- Dataclass/report field projection:
  - `benchmark/reports/cppipe_figures.py`
  - `benchmark/well_throughput_scaling.py` CSV/table helpers
  - `benchmark/bioformats_hcs_validation.py` axis CSV/JSON flattening
- Dynamic loader code:
  - `benchmark/converter/library_absorber.py`
  - `benchmark/runtime_env.py` logging level lookup
- Compatibility/probing code:
  - `benchmark/converter/contract_inference.py`

For every remaining hit outside the listed categories, add an exact phase entry
with owner, file, edit, and test command before implementation proceeds.

## Full Verification

Run in the repo virtualenv:

```text
source .venv/bin/activate
python -m ruff check \
  benchmark/contracts/manifest_acquisition.py \
  benchmark/contracts/comparison_manifest.py \
  benchmark/converter/__init__.py \
  benchmark/contracts/tool_adapter.py \
  benchmark/contracts/metric.py \
  benchmark/adapters/openhcs.py \
  benchmark/adapters/cellprofiler.py \
  benchmark/runner.py \
  benchmark/cellprofiler_comparison.py \
  benchmark/datasets/bioformats_hcs.py \
  benchmark/bioformats_hcs_validation.py \
  benchmark/cellprofiler_benchmark_cli.py \
  tests/unit/test_comparison_manifest_acquisition.py \
  tests/unit/test_benchmark_manifests.py \
  tests/unit/test_openhcs_adapter.py \
  tests/unit/test_cellprofiler_adapter.py \
  tests/unit/test_runner_cellprofiler_compatibility.py \
  tests/unit/test_cellprofiler_comparison.py \
  tests/unit/test_bioformats_hcs_validation.py

python -m pytest \
  tests/unit/test_comparison_manifest_acquisition.py \
  tests/unit/test_benchmark_manifests.py \
  tests/unit/test_openhcs_adapter.py \
  tests/unit/test_cellprofiler_adapter.py \
  tests/unit/test_runner_cellprofiler_compatibility.py \
  tests/unit/test_cellprofiler_comparison.py \
  tests/unit/test_bioformats_hcs_validation.py \
  -q
```

Then rerun the previous focused suite:

```text
python -m pytest tests/unit/test_dataset_registry.py tests/unit/test_dataset_acquisition_infra.py tests/unit/test_benchmark_manifests.py tests/unit/test_comparison_manifest_acquisition.py tests/unit/test_benchmark_timing.py tests/unit/test_metrics_memory.py tests/unit/test_benchmark_progress.py tests/unit/test_runner_cellprofiler_compatibility.py tests/unit/test_openhcs_adapter.py tests/unit/test_cellprofiler_adapter.py -q
```

## Handwaving Review

This section is the dry-run review of the plan itself.

### Resolved Details

- Manifest acquisition has exact owner classes, function signatures, and a
  circular-import-free placement.
- Converter export cleanup has an exact `__all__` object list.
- Tool and metric base-contract cleanup has exact class attributes and exact
  `runner.py` rewrites.
- OpenHCS cleanup has an exact owner chain: benchmark config input is
  `GlobalPipelineConfig`, generated/selected pipeline settings are
  `PipelineConfig`, and runtime work is OpenHCS orchestrator compile/execute.
- Bio-Formats cleanup has an exact owner chain: sample acquisition reuses
  `BenchmarkDatasetDeclaration` / `DatasetSpec`, while format detection and axis
  projection stay on the Bio-Formats microscope handler stack.
- Verification commands are file-scoped and test-scoped.

### Required Test Updates

- Replace tests importing `OPENHCS_*_PARAM` with assertions on
  `OpenHCSAdapter.global_config`, `GlobalPipelineConfig`, and generated
  `PipelineConfig` behavior.
- Replace cache-key tests that expected ignored-key subtraction with exact
  canonical cache-key equality.
- Keep `CELLPROFILER_*_PARAM` tests only where they exercise native
  CellProfiler image-set boundary parsing.

### Explicit Non-Goals

- Do not remove all `getattr` from report rendering in this batch.
- Do not refactor generated or absorbed CellProfiler source.
- Do not replace JSON manifest field names with classes inside serialized JSON.
  External JSON remains stringly at the boundary and is converted at parse time.
- Do not add compatibility shims for removed parameter constants.
