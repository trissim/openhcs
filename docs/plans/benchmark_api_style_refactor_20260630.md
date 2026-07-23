# Benchmark API Style Refactor Plan

Date: 2026-06-30

## Purpose

Clean up the `benchmark` public API so it follows the style of the recent
OpenHCS declaration and capability refactors.

The goal is not cosmetic import churn. The target is to remove hand-maintained
mirrors of public symbols, registry keys, and semantic ownership while
preserving the existing benchmark import surface.

## Recent Style Baseline

Use these current OpenHCS patterns as the target:

- declaration classes own semantic identity;
- `AutoRegisterMeta` is used for open-ended families;
- frozen slotted dataclasses are used for public records and projections;
- final string names exist only as ABI/export names emitted from typed owners;
- `__all__` surfaces are derived from object identity with
  `public_names_from_objects`, `exported_public_names`, or
  `declared_public_names`;
- generated views consume declarations instead of re-declaring parallel tables.

Concrete source examples:

- `openhcs.agent.capabilities.AgentCapabilityDeclaration`
- `openhcs.agent.capabilities.AgentCapabilitySpec`
- `openhcs.agent.capabilities.AgentCapabilityNamespace`
- `openhcs.mcp.dev_client_commanding.GeneratedMcpDevCommandProfile`
- `openhcs.core.public_api`

## Current Benchmark Evidence

AST and source inspection show these priority smells:

- `benchmark/__init__.py` and `benchmark/adapters/__init__.py` maintain
  `_PUBLIC_EXPORTS: dict[str, tuple[str, str]]` and load attributes with
  `__getattr__`. This manually mirrors module and object ownership.
- `benchmark/pipelines/registry.py` uses mutable `PipelineSpec`, one manual
  `PIPELINE_REGISTRY` dict, and extension comments instead of nominal pipeline
  declarations.
- `benchmark/datasets/registry.py` has a useful `DatasetCatalogRow`, but then
  aliases public constants by string lookup into `DATASET_REGISTRY`.
- `benchmark/contracts/tool_adapter.py`, `benchmark/contracts/dataset.py`, and
  `benchmark/contracts/metric.py` still expose loose `Any`, mutable dict/list
  contracts, and non-slotted dataclasses.
- `benchmark/datasets/acquire.py` already uses `AutoRegisterMeta` for
  validation/source handlers, but registry keys are stored as enum `.value`
  strings instead of the enum authorities.

## Non-Goals

- Do not change benchmark behavior or dataset IDs in the first pass.
- Do not rewrite generated CellProfiler libraries as part of this plan.
- Do not remove public aliases such as `BBBC021_SINGLE_PLATE` or
  `NUCLEI_SEGMENTATION`; preserve them while changing their authority.
- Do not replace string tables with constants that still mirror the same table.
- Do not introduce compatibility shims that become permanent alternate APIs.

## Final API Contract

The final public benchmark API after this plan is explicit and compatibility
preserving. Public imports keep working, but each exported object comes from a
nominal owner instead of a string export table.

Root package:

```python
from benchmark import (
    AcquiredDataset,
    BBBC021_SINGLE_PLATE,
    BenchmarkCaseProgress,
    BenchmarkProgressEvent,
    BenchmarkProgressEventKind,
    BenchmarkProgressSnapshot,
    BenchmarkResult,
    CellProfilerAdapter,
    CellProfilerCompatibilityResult,
    DATASET_REGISTRY,
    DatasetAcquisitionError,
    DatasetSpec,
    MemoryMetric,
    MetricCollector,
    NUCLEI_SEGMENTATION,
    OpenHCSAdapter,
    OpenHCSAxisSelection,
    PIPELINE_REGISTRY,
    PipelineSpec,
    TimeMetric,
    ToolAdapter,
    ToolAdapterError,
    ToolExecutionError,
    ToolNotInstalledError,
    ToolVersionError,
    acquire_dataset,
    get_dataset_spec,
    get_pipeline_spec,
    iter_progress_events,
    run_benchmark,
    run_cellprofiler_compatibility_benchmark,
    summarize_progress,
)
```

Direct owner imports:

```python
from benchmark.contracts.dataset import DatasetSpec, AcquiredDataset
from benchmark.contracts.metric import MetricCollector
from benchmark.contracts.pipeline import PipelineSpec
from benchmark.contracts.tool_adapter import BenchmarkResult, ToolAdapter
from benchmark.contracts.values import (
    BenchmarkMetricMap,
    BenchmarkMetricValue,
    BenchmarkParameterMap,
    BenchmarkParameterValue,
    BenchmarkProvenanceMap,
)
from benchmark.datasets.registry import BBBC021_SINGLE_PLATE, DATASET_REGISTRY
from benchmark.pipelines.registry import NUCLEI_SEGMENTATION, PIPELINE_REGISTRY
```

The root `benchmark` package is a lazy projection only. It is not an authority
for symbol ownership, dataset identity, pipeline identity, adapter behavior, or
metric semantics.

## Nominal Authority Model

The final SSOT ownership graph is:

```text
benchmark.contracts.dataset
  DatasetSpec, DatasetSourceSpec, DatasetValidationRule, DatasetSourceKind,
  BenchmarkCategory, CellProfilerBenchmarkCaseSpec, AcquiredDataset

benchmark.contracts.pipeline
  PipelineSpec

benchmark.contracts.metric
  MetricCollector

benchmark.contracts.tool_adapter
  BenchmarkResult, ToolAdapter, ToolAdapterError family

benchmark.contracts.values
  BenchmarkScalarValue, BenchmarkParameterValue, BenchmarkParameterMap,
  BenchmarkMetricValue, BenchmarkMetricMap, BenchmarkProvenanceValue,
  BenchmarkProvenanceMap

benchmark.datasets.registry
  BenchmarkDatasetDeclaration -> DATASET_REGISTRY and public dataset aliases

benchmark.datasets.acquire
  DatasetValidationStrategy and DatasetSourceHandler registered by enum keys

benchmark.pipelines.registry
  BenchmarkPipelineDeclaration -> PIPELINE_REGISTRY and public pipeline aliases

benchmark.__init__ and benchmark.adapters.__init__
  lazy public API projections only
```

Boundary rule:

- Contract modules own typed public records and type aliases.
- Registry modules own declaration classes registered by `AutoRegisterMeta`.
- Package `__init__.py` modules own only export projection.
- Runner/adapters consume `PipelineSpec`, `DatasetSpec`, `MetricCollector`, and
  `BenchmarkResult`; they do not define those contracts.
- No package export surface manually maps `public_name -> "module.attr"`.

Final type registration authorities:

- `BenchmarkPipelineDeclaration.__registry__`
- `BenchmarkDatasetDeclaration.__registry__`
- `DatasetValidationStrategy.__registry__`
- `DatasetSourceHandler.__registry__`

Every type family added or migrated by this plan registers through
`AutoRegisterMeta`. Compatibility dicts such as `PIPELINE_REGISTRY` and
`DATASET_REGISTRY` contain materialized specs only; they are not type
registries.

This mirrors the current OpenHCS pattern where declarations are the authority,
DTO/spec records are projections, and MCP/dev-client/package surfaces are
generated views.

## Architecture Integration

Use existing infrastructure, not benchmark-local substitutes:

- Public export derivation uses `openhcs.core.public_api`.
  - Cheap package `__all__` values use `public_names_from_objects`.
  - Root and adapter package laziness follows the existing
    `openhcs.agent.dto` source-owner resolver pattern.
- Type registries use `metaclass_registry.AutoRegisterMeta`.
  - `BenchmarkPipelineDeclaration` is the pipeline registry authority.
  - `BenchmarkDatasetDeclaration` is the dataset registry authority.
  - `DatasetValidationStrategy` and `DatasetSourceHandler` use enum members as
    registry keys.
- Compatibility registries are projections.
  - `PIPELINE_REGISTRY` is generated from `BenchmarkPipelineDeclaration`.
  - `DATASET_REGISTRY` is generated from `BenchmarkDatasetDeclaration`.
  - Public dataset aliases are generated from
    `BenchmarkDatasetDeclaration.public_alias`.
- Runtime and adapter code consume contracts.
  - `benchmark.runner` consumes `PipelineSpec`, `DatasetSpec`,
    `MetricCollector`, `ToolAdapter`, and `BenchmarkResult`.
  - `benchmark.adapters.openhcs` and `benchmark.adapters.cellprofiler` return
    `BenchmarkResult` and accept the `ToolAdapter.run` contract.
  - Tool-specific parameter semantics stay in adapter request/config classes;
    the benchmark registry only declares benchmark-level parameter values.
- Agent/MCP code stays out of scope.
  - `openhcs.agent.services.knowledge_base_service` imports
    `ComparisonManifest`; this plan does not change that manifest contract.
  - No MCP tool, UI bridge, or OpenHCS compiler changes are required for this
    benchmark API cleanup.

Final invariant: every public benchmark symbol has exactly one of these
authorities: contract type, `AutoRegisterMeta` declaration class, adapter class,
metric class, runner function, or progress DTO. Package exports and
compatibility registries are projections only.

## Target Shape

### Public API

Lightweight package `__init__.py` files must import public objects directly
and declare their public names from object identity:

```python
from openhcs.core.public_api import public_names_from_objects

from benchmark.contracts.dataset import DatasetSpec
from benchmark.contracts.tool_adapter import BenchmarkResult

__all__ = public_names_from_objects(DatasetSpec, BenchmarkResult)
```

The root `benchmark` package and `benchmark.adapters` package must keep lazy
resolution. Dry runs show direct access to adapter/runner exports is expensive.
The replacement follows the `openhcs.agent.dto` package pattern:

- keep a plain `__all__` tuple of public ABI names;
- resolve names on demand from module/package owners;
- cache resolved objects in package globals after lookup;
- use module objects and module `__all__`/namespace ownership, not a
  per-symbol `_PUBLIC_EXPORTS` dict of module strings.

The root resolver checks these source modules in this order:

1. cheap contract modules:
   - `benchmark.contracts.dataset`
   - `benchmark.contracts.metric`
   - `benchmark.contracts.tool_adapter`
2. cheap benchmark packages/modules:
   - `benchmark.datasets`
   - `benchmark.pipelines`
   - `benchmark.metrics`
   - `benchmark.progress`
3. lazy/heavy owners only when needed:
   - `benchmark.adapters`
   - `benchmark.runner`

This preserves cheap `import benchmark` while removing the hand-authored
symbol-to-module mirror.

`benchmark.adapters` uses the same shape with this source order:

1. `benchmark.adapters.openhcs`
2. `benchmark.adapters.cellprofiler`

The source-order resolver yields modules one at a time. It stops after the
requested name is found, so cheap root imports do not import adapters or the
runner.

### Pipeline Contracts And Registry

Create `benchmark/contracts/values.py` as the shared value contract owner:

```python
from collections.abc import Mapping
from pathlib import Path
from typing import TypeAlias

BenchmarkScalarValue: TypeAlias = str | int | float | bool | None
BenchmarkParameterValue: TypeAlias = (
    BenchmarkScalarValue
    | Path
    | tuple["BenchmarkParameterValue", ...]
    | Mapping[str, "BenchmarkParameterValue"]
)
BenchmarkParameterMap: TypeAlias = Mapping[str, BenchmarkParameterValue]
BenchmarkMetricValue: TypeAlias = BenchmarkScalarValue
BenchmarkMetricMap: TypeAlias = Mapping[str, BenchmarkMetricValue]
BenchmarkProvenanceValue: TypeAlias = BenchmarkParameterValue
BenchmarkProvenanceMap: TypeAlias = Mapping[str, BenchmarkProvenanceValue]
```

This reflects the dry run:

- registry-level defaults are currently scalar;
- runner-enriched params add strings and sometimes `Path`;
- adapter-specific params include tuples such as OpenHCS axis filters;
- cache/provenance paths can carry nested mapping-like payloads.

Create `benchmark/contracts/pipeline.py` as the pipeline spec contract owner:

```python
from dataclasses import dataclass, field
from types import MappingProxyType

from benchmark.contracts.values import BenchmarkParameterMap


def immutable_benchmark_parameters(
    values: BenchmarkParameterMap | None = None,
) -> BenchmarkParameterMap:
    return MappingProxyType(dict(values or ()))

@dataclass(frozen=True, slots=True)
class PipelineSpec:
    name: str
    description: str
    parameters: BenchmarkParameterMap = field(
        default_factory=immutable_benchmark_parameters
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parameters",
            immutable_benchmark_parameters(self.parameters),
        )
```

Use this nominal declaration root in `benchmark/pipelines/registry.py`:

```python
from abc import ABC
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

from benchmark.contracts.pipeline import PipelineSpec
from benchmark.contracts.values import BenchmarkParameterMap

class BenchmarkPipelineDeclaration(ABC, metaclass=AutoRegisterMeta):
    __registry__: ClassVar[dict[str, type["BenchmarkPipelineDeclaration"]]] = {}
    __registry_key__ = "name"
    __skip_if_no_key__ = True

    name: ClassVar[str | None] = None
    description: ClassVar[str]
    parameters: ClassVar[BenchmarkParameterMap] = {}

    @classmethod
    def to_spec(cls) -> PipelineSpec:
        if cls.name is None:
            raise ValueError(f"{cls.__name__} must declare a pipeline name.")
        return PipelineSpec(
            name=cls.name,
            description=cls.description,
            parameters=cls.parameters,
        )
```

`PIPELINE_REGISTRY` remains as a compatibility projection, and it is built from
`BenchmarkPipelineDeclaration.__registry__`.

The current concrete declaration becomes:

```python
class NucleiSegmentationPipeline(BenchmarkPipelineDeclaration):
    name = "nuclei_segmentation"
    description = "BBBC021 nuclei segmentation (CellProfiler-equivalent)"
    parameters = {"cppipe_reference_index": 0}


def pipeline_specs() -> tuple[PipelineSpec, ...]:
    return tuple(
        declaration.to_spec()
        for declaration in BenchmarkPipelineDeclaration.__registry__.values()
    )


PIPELINE_REGISTRY = {spec.name: spec for spec in pipeline_specs()}
NUCLEI_SEGMENTATION = PIPELINE_REGISTRY["nuclei_segmentation"]
```

### Dataset Registry

Use AutoRegisterMeta for dataset registry authority. `DatasetCatalogRow` and
`DATASET_CATALOG` are removed in this plan.

Final declaration root in `benchmark/datasets/registry.py`:

```python
from abc import ABC
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta


class BenchmarkDatasetDeclaration(ABC, metaclass=AutoRegisterMeta):
    __registry__: ClassVar[dict[str, type["BenchmarkDatasetDeclaration"]]] = {}
    __registry_key__ = "id"
    __skip_if_no_key__ = True

    id: ClassVar[str | None] = None
    public_alias: ClassVar[str | None] = None
    urls: ClassVar[tuple[str, ...]] = ()
    size_bytes: ClassVar[int]
    archive_format: ClassVar[ArchiveFormat] = ArchiveFormat.ZIP
    microscope_type: ClassVar[str]
    validation_rule: ClassVar[DatasetValidationRule] = DatasetValidationRule.NON_EMPTY
    reference_cppipe_urls: ClassVar[tuple[str, ...]] = ()
    expected_count: ClassVar[int | None] = None
    manifest_path: ClassVar[Path | None] = None
    source: ClassVar[DatasetSourceSpec | None] = None
    benchmark_cases: ClassVar[tuple[CellProfilerBenchmarkCaseSpec, ...]] = ()

    @classmethod
    def to_spec(cls) -> DatasetSpec:
        if cls.id is None:
            raise ValueError(f"{cls.__name__} must declare a dataset id.")
        return DatasetSpec(
            id=cls.id,
            urls=list(cls.urls),
            size_bytes=cls.size_bytes,
            archive_format=cls.archive_format,
            microscope_type=cls.microscope_type,
            validation_rule=cls.validation_rule,
            reference_cppipe_urls=cls.reference_cppipe_urls,
            expected_count=cls.expected_count,
            manifest_path=cls.manifest_path,
            source=cls.source,
            benchmark_cases=cls.benchmark_cases,
        )
```

Type-specific declaration mixins used in this migration:

```python
class PublishedPipelineDatasetMixin:
    microscope_type: ClassVar[str] = PUBLISHED_PIPELINE


class ImageCountValidatedDatasetMixin:
    validation_rule: ClassVar[DatasetValidationRule] = DatasetValidationRule.IMAGE_COUNT
    expected_count: ClassVar[int]
```

Use `PublishedPipelineDatasetMixin` for declarations whose current row has
`microscope_type=PUBLISHED_PIPELINE`. Use `ImageCountValidatedDatasetMixin` for
declarations whose current row has
`validation_rule=DatasetValidationRule.IMAGE_COUNT`. Do not create a mixin for a
single dataset.

Final dataset projections:

```python
def dataset_declarations() -> tuple[type[BenchmarkDatasetDeclaration], ...]:
    return tuple(BenchmarkDatasetDeclaration.__registry__.values())


def dataset_specs() -> tuple[DatasetSpec, ...]:
    return tuple(declaration.to_spec() for declaration in dataset_declarations())


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    spec.id: spec for spec in dataset_specs()
}


def _dataset_public_aliases() -> dict[str, DatasetSpec]:
    aliases: dict[str, DatasetSpec] = {}
    for declaration in dataset_declarations():
        dataset_id = declaration.id
        public_alias = declaration.public_alias
        if dataset_id is not None and public_alias is not None:
            aliases[public_alias] = DATASET_REGISTRY[dataset_id]
    return aliases


globals().update(_dataset_public_aliases())
```

### Contracts

Public contracts must be immutable and typed:

- add `slots=True` to already-frozen dataset dataclasses;
- convert `BenchmarkResult` to `@dataclass(frozen=True, slots=True)`;
- convert `PipelineSpec` to `@dataclass(frozen=True, slots=True)`;
- store `PipelineSpec.parameters` as an immutable `MappingProxyType`
  projection;
- leave `AcquiredDataset` mutable in this plan;
- leave `DatasetSpec.urls` as `list[str]` in this plan;
- replace `dict[str, Any]` / `list[Any]` in public contract signatures with
  `Mapping`, `tuple`, and named type aliases.
- Keep adapter request objects as the authority for tool-specific parameter
  semantics.

The contract dry run found no benchmark-side mutations of
`DatasetSpec.urls`, `BenchmarkResult.metrics`, or `BenchmarkResult.provenance`.
It did find `benchmark.progress.BenchmarkProgressSnapshotBuilder` assigning its
own `.metrics`, which is unrelated and does not block freezing
`BenchmarkResult`.

## AST Bulk Refactor Instructions

Run all scripts from the repository root with the project environment active:

```bash
. .venv/bin/activate
```

### 1. Inventory Benchmark Public Imports

Use this before and after each phase to ensure public imports remain visible:

```bash
python - <<'PY'
import ast
from pathlib import Path

targets = {
    "benchmark",
    "benchmark.adapters",
    "benchmark.datasets",
    "benchmark.metrics",
    "benchmark.pipelines",
}

for path in sorted(Path(".").rglob("*.py")):
    if path.parts[0] in {".git", ".mypy_cache", ".pytest_cache", ".venv", "results"}:
        continue
    if any(part in {"cellprofiler_library", "cellprofiler_source"} for part in path.parts):
        continue
    tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in targets:
            names = ", ".join(alias.name for alias in node.names)
            print(f"{path}:{node.lineno}: from {node.module} import {names}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in targets:
                    print(f"{path}:{node.lineno}: import {alias.name}")
PY
```

### 2. Inventory Mirrored Export Tables And Registries

Use this to find modules that encode public/API mirrors by dict/list assignment:

```bash
python - <<'PY'
import ast
from pathlib import Path

for path in sorted(Path("benchmark").rglob("*.py")):
    if any(part in {"cellprofiler_library", "cellprofiler_source", "results"} for part in path.parts):
        continue
    tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
        else:
            continue
        if not targets:
            continue
        flagged = [
            name for name in targets
            if name == "__all__"
            or name.endswith("_REGISTRY")
            or name.endswith("_EXPORTS")
            or name.endswith("_CATALOG")
        ]
        if flagged:
            print(f"{path}:{node.lineno}: {', '.join(flagged)}")
PY
```

### 3. Bulk Convert Simple `__all__` Lists

Only use this for package `__init__.py` files where every exported name is
already imported in that file. Review the generated diff before keeping it.

Algorithm:

1. Parse each target `__init__.py`.
2. Collect imported public symbols from `ImportFrom` nodes.
3. Replace a literal `__all__ = [...]` assignment with
   `__all__ = public_names_from_objects(...)`.
4. Add `from openhcs.core.public_api import public_names_from_objects` if
   missing.

Suggested target list:

```text
benchmark/metrics/__init__.py
benchmark/pipelines/__init__.py
benchmark/datasets/__init__.py
```

`benchmark/datasets/__init__.py` has now been characterized: every public name
comes from explicit imports in that file, so replace `sorted(globals())` with an
explicit `public_names_from_objects(...)` projection.

### 4. Bulk Replace Lazy String Export Maps With Source Resolvers

Target files:

```text
benchmark/__init__.py
benchmark/adapters/__init__.py
```

AST rewrite outline:

1. Remove the `import importlib` import if no longer used.
2. Remove `_PUBLIC_EXPORTS` assignment.
3. Remove `__getattr__`.
4. Build a literal `__all__` tuple from the previous export table keys.
5. Add `_EXPORT_NAMES`, `_MISSING_EXPORT`, `__dir__`, and the exact
   source-owner resolver skeleton from Phase 4.
6. Resolve from module/package owners in the source orders listed in the Public
   API target section.

This rewrite intentionally uses the old string map only as a one-time migration
seed for `__all__`. The resulting source must not keep any module/attribute
string map.

### 5. Pipeline Registry Declaration Migration

Use AST to preserve existing `PipelineSpec(...)` values while moving authority
to declaration classes.

Algorithm:

1. Parse `benchmark/pipelines/registry.py`.
2. Find module-level assignments where the value is `PipelineSpec(...)`.
3. For each assignment, extract the assigned symbol, `name`, `description`, and
   `parameters`.
4. Generate a `BenchmarkPipelineDeclaration` subclass whose class name is
   derived from the assigned symbol.
5. Generate the public alias as:

   ```python
   NUCLEI_SEGMENTATION = NucleiSegmentationPipeline.to_spec()
   ```

6. Generate `PIPELINE_REGISTRY` from `pipeline_specs()`.
7. Preserve `get_pipeline_spec(name)` as a compatibility lookup over the
   generated registry.

Expected first migration inventory:

```text
NUCLEI_SEGMENTATION -> name="nuclei_segmentation"
```

### 6. Dataset Declaration Migration

Do this after pipeline registry migration. Use AST to convert the current
`DATASET_CATALOG` rows into `BenchmarkDatasetDeclaration` subclasses.

Algorithm:

1. Parse `benchmark/datasets/registry.py`.
2. Find every `DatasetCatalogRow(...)` call inside `DATASET_CATALOG`.
3. For each row, extract keyword arguments and the matching public alias from
   the old `ALIAS = DATASET_REGISTRY["dataset_id"]` block.
4. Generate one declaration class per row.
5. The generated class bases are:
   - `ImageCountValidatedDatasetMixin, BenchmarkDatasetDeclaration` when the
     row declares `validation_rule=DatasetValidationRule.IMAGE_COUNT`;
   - `PublishedPipelineDatasetMixin, BenchmarkDatasetDeclaration` when the row
     declares `microscope_type=PUBLISHED_PIPELINE`;
   - both mixins, in that order, if both conditions are true;
   - only `BenchmarkDatasetDeclaration` otherwise.
6. Copy row keywords to class attributes, except:
   - `id` stays `id`;
   - the matching alias assignment becomes `public_alias`;
   - omit `validation_rule` when inherited from
     `ImageCountValidatedDatasetMixin`;
   - omit `microscope_type` when inherited from
     `PublishedPipelineDatasetMixin`.
7. Delete `DatasetCatalogRow`, `DATASET_CATALOG`, and all direct
   `ALIAS = DATASET_REGISTRY["dataset_id"]` assignments.
8. Generate `DATASET_REGISTRY` and public aliases from
   `BenchmarkDatasetDeclaration.__registry__` using the exact projection code
   in Phase 5.

## No-Invention Dry Run Gaps

The dry run found these places where the previous plan still required an
implementing agent to design while editing:

- phase order referenced `benchmark.contracts.pipeline` before it existed;
- cheap package `__all__` cleanup did not provide the exact final export
  expressions;
- root/adapters lazy export cleanup did not include the resolver skeleton;
- the benchmark value contract did not explicitly cover runner-enriched params
  and adapter-specific tuple/path params;
- pipeline parameter immutability was not load-bearing until
  `MappingProxyType` normalization was specified.
- dataset alias cleanup still treated `DATASET_CATALOG` as a registry authority
  instead of migrating datasets to `AutoRegisterMeta` declaration classes.

The implementation sequence below resolves those gaps. Do not reorder it unless
a focused test fails and the plan is updated with the reason.

## Implementation Phases

### Phase 1: Contract Foundation

Concrete edits:

- add `benchmark/contracts/values.py` exactly as shown in the shared value
  contract section;
- add `benchmark/contracts/pipeline.py` exactly as shown in the Pipeline
  Contracts And Registry section;
  - `PipelineSpec.parameters` must normalize to immutable
    `MappingProxyType`, not a mutable dict;
- `benchmark/contracts/dataset.py`
  - add `from __future__ import annotations`;
  - add `slots=True` to `DatasetSourceSpec`, `BenchmarkCategory`,
    `CellProfilerBenchmarkCaseSpec`, and `DatasetSpec`;
  - do not change `DatasetSpec.urls` from `list[str]` in this plan.
- `benchmark/contracts/tool_adapter.py`
  - add `from __future__ import annotations`;
  - convert `BenchmarkResult` to `@dataclass(frozen=True, slots=True)`;
  - import `BenchmarkMetricMap`, `BenchmarkProvenanceMap`, and
    `BenchmarkParameterMap` from `benchmark.contracts.values`;
  - update `ToolAdapter.run(...)` to take `BenchmarkParameterMap` and
    `Sequence[MetricCollector]`.
- `benchmark/contracts/metric.py`
  - add `from __future__ import annotations`;
  - use `Self` for `__enter__`;
  - import `BenchmarkMetricValue` from `benchmark.contracts.values`;
  - replace `Any` result with `BenchmarkMetricValue`.
- `benchmark/contracts/__init__.py` is absent today; do not add it in this
  plan. Owner modules stay explicit.

Verification:

```bash
. .venv/bin/activate
python -m pytest \
  tests/unit/test_openhcs_adapter.py \
  tests/unit/test_cellprofiler_adapter.py \
  tests/unit/test_runner_cellprofiler_compatibility.py \
  tests/unit/test_benchmark_progress.py \
  -q
```

### Phase 2: Pipeline Registry Declaration Ownership

Concrete edits:

- delete the local `PipelineSpec` dataclass from
  `benchmark/pipelines/registry.py`;
- import `PipelineSpec` from `benchmark.contracts.pipeline`;
- import `BenchmarkParameterMap` from `benchmark.contracts.values`;
- implement the exact `BenchmarkPipelineDeclaration` shape shown above;
- replace the direct `NUCLEI_SEGMENTATION = PipelineSpec(...)` construction with
  `NucleiSegmentationPipeline`;
- generate `PIPELINE_REGISTRY` from `pipeline_specs()`;
- bind `NUCLEI_SEGMENTATION` from the generated registry to preserve object
  identity expectations.

Final registry block:

```python
def pipeline_specs() -> tuple[PipelineSpec, ...]:
    return tuple(
        declaration.to_spec()
        for declaration in BenchmarkPipelineDeclaration.__registry__.values()
    )


PIPELINE_REGISTRY = {spec.name: spec for spec in pipeline_specs()}
NUCLEI_SEGMENTATION = PIPELINE_REGISTRY["nuclei_segmentation"]
```

The Phase 2 diff is wrong if it adds any of these shapes:

- `PIPELINE_REGISTRY = {"nuclei_segmentation": ...}`;
- a second dict mapping public alias names to pipeline classes;
- `NUCLEI_SEGMENTATION = PipelineSpec(...)`;
- a constant whose only purpose is to repeat
  `NucleiSegmentationPipeline.name`.

Verification:

```bash
. .venv/bin/activate
python - <<'PY'
from benchmark.pipelines.registry import NUCLEI_SEGMENTATION, PIPELINE_REGISTRY, get_pipeline_spec
assert NUCLEI_SEGMENTATION is PIPELINE_REGISTRY["nuclei_segmentation"]
assert get_pipeline_spec("nuclei_segmentation") is PIPELINE_REGISTRY["nuclei_segmentation"]
PY
python - <<'PY'
import ast
from pathlib import Path

tree = ast.parse(Path("benchmark/pipelines/registry.py").read_text())
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id == "NUCLEI_SEGMENTATION"
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "PipelineSpec"
            ):
                raise SystemExit("NUCLEI_SEGMENTATION still constructs PipelineSpec directly")
print("pipeline declaration ownership ok")
PY
python -m pytest tests/unit/test_runner_cellprofiler_compatibility.py tests/unit/test_dataset_registry.py -q
```

### Phase 3: Cheap Package Export Cleanup

Concrete edits:

- `benchmark/metrics/__init__.py`

  ```python
  """Metric collectors."""

  from openhcs.core.public_api import public_names_from_objects

  from benchmark.metrics.memory import MemoryMetric
  from benchmark.metrics.time import TimeMetric

  __all__ = public_names_from_objects(TimeMetric, MemoryMetric)
  ```

- `benchmark/pipelines/__init__.py`

  ```python
  """Pipeline registry."""

  from openhcs.core.public_api import public_names_from_objects

  from benchmark.contracts.pipeline import PipelineSpec
  from benchmark.pipelines.registry import (
      NUCLEI_SEGMENTATION,
      PIPELINE_REGISTRY,
      get_pipeline_spec,
  )

  __all__ = public_names_from_objects(
      PipelineSpec,
      get_pipeline_spec,
      extra_names=("NUCLEI_SEGMENTATION", "PIPELINE_REGISTRY"),
  )
  ```

- `benchmark/datasets/__init__.py`

  ```python
  """Dataset utilities and registry."""

  from openhcs.core.public_api import public_names_from_objects

  from benchmark.contracts.dataset import (
      ArchiveFormat,
      CellProfilerBenchmarkCaseSpec,
      DatasetSourceKind,
      DatasetSourceSpec,
      DatasetValidationRule,
  )
  from benchmark.datasets.acquire import DatasetAcquisitionError, acquire_dataset
  from benchmark.datasets.manifest import (
      comparison_manifest_cases,
      comparison_manifest_payload,
      write_comparison_manifest,
  )
  from benchmark.datasets.registry import (
      BBBC021_SINGLE_PLATE,
      CELLPROFILER4_BENCHMARK_SUPPLEMENT,
      CELLPROFILER_TUTORIALS,
      DATASET_REGISTRY,
      get_dataset_spec,
  )
  from benchmark.datasets.visible_source import resolve_visible_source_path

  __all__ = public_names_from_objects(
      ArchiveFormat,
      CellProfilerBenchmarkCaseSpec,
      DatasetSourceKind,
      DatasetSourceSpec,
      DatasetValidationRule,
      DatasetAcquisitionError,
      acquire_dataset,
      comparison_manifest_cases,
      comparison_manifest_payload,
      write_comparison_manifest,
      get_dataset_spec,
      resolve_visible_source_path,
      extra_names=(
          "BBBC021_SINGLE_PLATE",
          "CELLPROFILER4_BENCHMARK_SUPPLEMENT",
          "CELLPROFILER_TUTORIALS",
          "DATASET_REGISTRY",
      ),
  )
  ```

The only new public-name machinery in Phase 3 is the three
`public_names_from_objects(...)` calls above. The `extra_names` tuples contain
exactly the constants shown in the snippets; do not add a second list, dict, or
helper mapping for these package exports.

Verification:

```bash
. .venv/bin/activate
python - <<'PY'
from benchmark.datasets import BBBC021_SINGLE_PLATE, DATASET_REGISTRY
from benchmark.metrics import TimeMetric, MemoryMetric
from benchmark.pipelines import NUCLEI_SEGMENTATION, PipelineSpec

assert BBBC021_SINGLE_PLATE is DATASET_REGISTRY["BBBC021_Week1_22123"]
assert NUCLEI_SEGMENTATION.name == "nuclei_segmentation"
assert TimeMetric.name == "execution_time_seconds"
assert MemoryMetric.name == "peak_memory_mb"
print("cheap package exports ok")
PY
python -m pytest tests/unit/test_dataset_registry.py tests/unit/test_benchmark_timing.py -q
```

### Phase 4: Lazy Root And Adapter Exports

Concrete edits:

- Replace `benchmark/adapters/__init__.py` with this source-owner resolver.

  ```python
  """Tool adapters."""

  from __future__ import annotations

  __all__ = ("CellProfilerAdapter", "OpenHCSAdapter")

  _EXPORT_NAMES = frozenset(__all__)
  _MISSING_EXPORT = object()


  def _adapter_export_modules():
      import benchmark.adapters.openhcs as openhcs_adapter

      yield openhcs_adapter

      import benchmark.adapters.cellprofiler as cellprofiler_adapter

      yield cellprofiler_adapter


  def resolve_adapter_export(name: str):
      if name not in _EXPORT_NAMES:
          raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
      existing = globals().get(name, _MISSING_EXPORT)
      if existing is not _MISSING_EXPORT:
          return existing
      for module in _adapter_export_modules():
          namespace = vars(module)
          if name in namespace:
              value = namespace[name]
              globals()[name] = value
              return value
      raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


  def __getattr__(name: str):
      return resolve_adapter_export(name)


  def __dir__() -> list[str]:
      return sorted(set(globals()) | _EXPORT_NAMES)
  ```

- Replace `benchmark/__init__.py` with the same source-owner pattern. The
  module-level `__getattr__` hook is allowed here because it is the Python
  package lazy-export protocol; do not use `getattr(...)` on owner modules.

  ```python
  """Public API for the benchmark platform."""

  from __future__ import annotations

  import openhcs as _openhcs_dependency_bootstrap  # noqa: F401

  __all__ = (
      "DatasetSpec",
      "AcquiredDataset",
      "MetricCollector",
      "BenchmarkResult",
      "ToolAdapter",
      "ToolAdapterError",
      "ToolExecutionError",
      "ToolNotInstalledError",
      "ToolVersionError",
      "DatasetAcquisitionError",
      "acquire_dataset",
      "BBBC021_SINGLE_PLATE",
      "DATASET_REGISTRY",
      "get_dataset_spec",
      "PipelineSpec",
      "NUCLEI_SEGMENTATION",
      "PIPELINE_REGISTRY",
      "get_pipeline_spec",
      "TimeMetric",
      "MemoryMetric",
      "OpenHCSAxisSelection",
      "BenchmarkCaseProgress",
      "BenchmarkProgressEvent",
      "BenchmarkProgressEventKind",
      "BenchmarkProgressSnapshot",
      "iter_progress_events",
      "summarize_progress",
      "CellProfilerAdapter",
      "OpenHCSAdapter",
      "CellProfilerCompatibilityResult",
      "run_benchmark",
      "run_cellprofiler_compatibility_benchmark",
  )

  _EXPORT_NAMES = frozenset(__all__)
  _MISSING_EXPORT = object()


  def _benchmark_export_modules():
      import benchmark.contracts.dataset as dataset_contracts

      yield dataset_contracts

      import benchmark.contracts.metric as metric_contracts

      yield metric_contracts

      import benchmark.contracts.pipeline as pipeline_contracts

      yield pipeline_contracts

      import benchmark.contracts.tool_adapter as tool_adapter_contracts

      yield tool_adapter_contracts

      import benchmark.datasets as dataset_exports

      yield dataset_exports

      import benchmark.pipelines as pipeline_exports

      yield pipeline_exports

      import benchmark.metrics as metric_exports

      yield metric_exports

      import benchmark.progress as progress_exports

      yield progress_exports

      import benchmark.adapters.openhcs as openhcs_adapter

      yield openhcs_adapter

      import benchmark.adapters.cellprofiler as cellprofiler_adapter

      yield cellprofiler_adapter

      import benchmark.runner as runner_exports

      yield runner_exports


  def resolve_benchmark_export(name: str):
      if name not in _EXPORT_NAMES:
          raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
      existing = globals().get(name, _MISSING_EXPORT)
      if existing is not _MISSING_EXPORT:
          return existing
      for module in _benchmark_export_modules():
          namespace = vars(module)
          if name in namespace:
              value = namespace[name]
              globals()[name] = value
              return value
      raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


  def __getattr__(name: str):
      return resolve_benchmark_export(name)


  def __dir__() -> list[str]:
      return sorted(set(globals()) | _EXPORT_NAMES)
  ```

The only name containers left in `benchmark/__init__.py` and
`benchmark/adapters/__init__.py` after Phase 4 are `__all__` and
`_EXPORT_NAMES`. `_EXPORT_NAMES` is derived from `__all__`; it is used only for
membership checks. The source modules are listed once in
`_benchmark_export_modules()` and `_adapter_export_modules()`. There is no dict
whose keys are public names, no module-string table, and no call to
`getattr(...)`.

Required behavior gates:

- `import benchmark` remains under `0.2s` on this checkout.
- `from benchmark import OpenHCSAdapter` may stay around `4s`, but must not
  become materially slower than current timing.
- `from benchmark import well_throughput_scaling` must still import the
  submodule even though it is not in root `__all__`.

Verification:

```bash
. .venv/bin/activate
/usr/bin/time -f 'import benchmark %e' python - <<'PY'
import benchmark
PY
/usr/bin/time -f 'root OpenHCSAdapter %e' python - <<'PY'
from benchmark import OpenHCSAdapter
PY
python - <<'PY'
from benchmark import well_throughput_scaling
print(well_throughput_scaling.__name__)
PY
if rg -n '_PUBLIC_EXPORTS|importlib|getattr\(' benchmark/__init__.py benchmark/adapters/__init__.py; then
  echo "lazy export mirror remains"
  exit 1
fi
python -m pytest tests/unit/test_metrics_memory.py tests/unit/test_dataset_registry.py -q
```

### Phase 5: Dataset Declaration Registry Ownership

Concrete edits:

- delete `DatasetCatalogRow`;
- delete `DATASET_CATALOG`;
- import `ABC`, `ClassVar`, and `AutoRegisterMeta`;
- add `BenchmarkDatasetDeclaration`, `PublishedPipelineDatasetMixin`, and
  `ImageCountValidatedDatasetMixin` exactly as shown in the Dataset Registry
  target section;
- convert every current `DatasetCatalogRow(...)` row into one
  `BenchmarkDatasetDeclaration` subclass;
- generate `DATASET_REGISTRY` and public aliases from
  `BenchmarkDatasetDeclaration.__registry__` using the exact projection code in
  the Dataset Registry target section.

Concrete declaration inventory:

```text
Bbbc021Week122123Dataset -> id="BBBC021_Week1_22123", public_alias="BBBC021_SINGLE_PLATE", bases=(ImageCountValidatedDatasetMixin, BenchmarkDatasetDeclaration)
Bbbc02220585W1Dataset -> id="BBBC022_20585_w1", public_alias="BBBC022_SINGLE_PLATE_DNA", bases=(ImageCountValidatedDatasetMixin, BenchmarkDatasetDeclaration)
Bbbc010WormsDataset -> id="BBBC010_worms", public_alias="BBBC010_WORMS", bases=(BenchmarkDatasetDeclaration,)
Bbbc011WormsMetabolismDataset -> id="BBBC011_worms_metabolism", public_alias="BBBC011_WORMS_METABOLISM", bases=(BenchmarkDatasetDeclaration,)
Bbbc012WormsInfectionMarkerDataset -> id="BBBC012_worms_infection_marker", public_alias="BBBC012_WORMS_INFECTION_MARKER", bases=(BenchmarkDatasetDeclaration,)
Bbbc013U2osTranslocationDataset -> id="BBBC013_u2os_translocation_bmp", public_alias="BBBC013_U2OS_TRANSLOCATION", bases=(BenchmarkDatasetDeclaration,)
Bbbc038FullDataset -> id="BBBC038_full", public_alias="BBBC038_FULL", bases=(ImageCountValidatedDatasetMixin, BenchmarkDatasetDeclaration)
Bbbc039NucleiSegmentationDataset -> id="BBBC039_nuclei_segmentation", public_alias="BBBC039_NUCLEI_SEGMENTATION", bases=(ImageCountValidatedDatasetMixin, BenchmarkDatasetDeclaration)
Singh2014IlluminationCorrectionDataset -> id="Singh_2014_illumination_correction", public_alias="SINGH_2014_ILLUMINATION_CORRECTION", bases=(PublishedPipelineDatasetMixin, BenchmarkDatasetDeclaration)
Sanz2019HistologyDataset -> id="Sanz_2019_histology", public_alias="SANZ_2019_HISTOLOGY", bases=(PublishedPipelineDatasetMixin, BenchmarkDatasetDeclaration)
Tian2019NeuronsDataset -> id="Tian_2019_neurons", public_alias="TIAN_2019_NEURONS", bases=(PublishedPipelineDatasetMixin, BenchmarkDatasetDeclaration)
Sokolov2023NeuronsDataset -> id="Sokolov_2023_neurons", public_alias="SOKOLOV_2023_NEURONS", bases=(PublishedPipelineDatasetMixin, BenchmarkDatasetDeclaration)
CellOrientationWoundHealingDataset -> id="CellOrientation_wound_healing", public_alias="CELL_ORIENTATION_WOUND_HEALING", bases=(PublishedPipelineDatasetMixin, BenchmarkDatasetDeclaration)
ChromTrans3dFishDataset -> id="ChromTrans_3d_fish", public_alias="CHROMTRANS_3D_FISH", bases=(PublishedPipelineDatasetMixin, BenchmarkDatasetDeclaration)
CellProfilerTutorialsDataset -> id="CellProfiler_tutorials", public_alias="CELLPROFILER_TUTORIALS", bases=(BenchmarkDatasetDeclaration,)
CellProfiler4BenchmarkSupplementDataset -> id="CellProfiler4_benchmark_supplement", public_alias="CELLPROFILER4_BENCHMARK_SUPPLEMENT", bases=(BenchmarkDatasetDeclaration,)
```

For each generated declaration, copy the remaining row fields as class
attributes. The AST conversion preserves calls to `_case(...)`,
`_git_sparse(...)`, and `_git_sparse_with_archives(...)` in the declaration
class body.

Final projection block:

```python
def dataset_declarations() -> tuple[type[BenchmarkDatasetDeclaration], ...]:
    return tuple(BenchmarkDatasetDeclaration.__registry__.values())


def dataset_specs() -> tuple[DatasetSpec, ...]:
    return tuple(declaration.to_spec() for declaration in dataset_declarations())


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    spec.id: spec for spec in dataset_specs()
}


def _dataset_public_aliases() -> dict[str, DatasetSpec]:
    aliases: dict[str, DatasetSpec] = {}
    for declaration in dataset_declarations():
        dataset_id = declaration.id
        public_alias = declaration.public_alias
        if dataset_id is not None and public_alias is not None:
            aliases[public_alias] = DATASET_REGISTRY[dataset_id]
    return aliases


globals().update(_dataset_public_aliases())
```

There are no remaining `DatasetCatalogRow`, `DATASET_CATALOG`, or top-level
assignments of the form `PUBLIC_ALIAS = DATASET_REGISTRY["dataset_id"]`.

Verification:

```bash
. .venv/bin/activate
python - <<'PY'
from benchmark.datasets.registry import (
    BBBC021_SINGLE_PLATE,
    DATASET_REGISTRY,
    BenchmarkDatasetDeclaration,
)

assert len(BenchmarkDatasetDeclaration.__registry__) == 16
assert set(BenchmarkDatasetDeclaration.__registry__) == set(DATASET_REGISTRY)
assert BBBC021_SINGLE_PLATE is DATASET_REGISTRY["BBBC021_Week1_22123"]
PY
python - <<'PY'
import ast
from pathlib import Path

tree = ast.parse(Path("benchmark/datasets/registry.py").read_text())
for node in ast.walk(tree):
    if isinstance(node, ast.Name) and node.id in {"DatasetCatalogRow", "DATASET_CATALOG"}:
        raise SystemExit(f"{node.id} remains after dataset declaration migration")
    if not isinstance(node, ast.Assign):
        continue
    if not any(isinstance(target, ast.Name) and target.id.isupper() for target in node.targets):
        continue
    if (
        isinstance(node.value, ast.Subscript)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "DATASET_REGISTRY"
    ):
        raise SystemExit("dataset public alias still mirrors DATASET_REGISTRY by id")
print("dataset declaration ownership ok")
PY
python -m pytest \
  tests/unit/test_dataset_registry.py \
  tests/unit/test_dataset_acquisition_infra.py \
  tests/unit/test_benchmark_manifests.py \
  tests/unit/test_comparison_manifest_acquisition.py \
  -q
```

### Phase 6: Dataset Handler Registry Tightening

Concrete edits:

- `DatasetValidationStrategy.validation_rule` becomes
  `DatasetValidationRule | None`;
- validation leaves declare these enum members directly:

  ```python
  ImageCountValidationStrategy.validation_rule = DatasetValidationRule.IMAGE_COUNT
  ManifestValidationStrategy.validation_rule = DatasetValidationRule.MANIFEST
  NonEmptyValidationStrategy.validation_rule = DatasetValidationRule.NON_EMPTY
  ```

- `DatasetValidationStrategy.for_rule(...)` looks up `cls.__registry__[rule]`;
- `DatasetSourceHandler.source_kind` becomes `DatasetSourceKind | None`;
- source leaves declare these enum members directly:

  ```python
  ArchiveUrlSourceHandler.source_kind = DatasetSourceKind.ARCHIVE_URLS
  UrlFilesSourceHandler.source_kind = DatasetSourceKind.URL_FILES
  GitSparseWithArchiveUrlsSourceHandler.source_kind = DatasetSourceKind.GIT_SPARSE_WITH_ARCHIVES
  GitSparseSourceHandler.source_kind = DatasetSourceKind.GIT_SPARSE
  ```

- `DatasetSourceHandler.for_source(...)` looks up
  `cls.__registry__[source.kind]`.

Final lookup methods:

```python
@classmethod
def for_rule(cls, rule: DatasetValidationRule) -> "DatasetValidationStrategy":
    try:
        strategy_type = cls.__registry__[rule]
    except KeyError as exc:
        raise DatasetAcquisitionError(f"Unknown validation rule '{rule.name}'") from exc
    return strategy_type()


@classmethod
def for_source(cls, source: DatasetSourceSpec) -> "DatasetSourceHandler":
    try:
        handler_type = cls.__registry__[source.kind]
    except KeyError as exc:
        raise DatasetAcquisitionError(
            f"Unsupported dataset source: {source.kind.name}"
        ) from exc
    return handler_type()
```

After Phase 6 there is no `.value` access in `DatasetValidationStrategy`,
`ImageCountValidationStrategy`, `ManifestValidationStrategy`,
`NonEmptyValidationStrategy`, `DatasetSourceHandler`, `ArchiveUrlSourceHandler`,
`UrlFilesSourceHandler`, `GitSparseWithArchiveUrlsSourceHandler`, or
`GitSparseSourceHandler`.

Dry-run proof: an isolated `AutoRegisterMeta` family registered enum keys
directly and did not create string-key entries.

Verification:

```bash
. .venv/bin/activate
python - <<'PY'
from benchmark.contracts.dataset import DatasetSourceKind, DatasetValidationRule
from benchmark.datasets.acquire import DatasetSourceHandler, DatasetValidationStrategy

assert set(DatasetValidationStrategy.__registry__) == {
    DatasetValidationRule.IMAGE_COUNT,
    DatasetValidationRule.MANIFEST,
    DatasetValidationRule.NON_EMPTY,
}
assert set(DatasetSourceHandler.__registry__) == {
    DatasetSourceKind.ARCHIVE_URLS,
    DatasetSourceKind.URL_FILES,
    DatasetSourceKind.GIT_SPARSE_WITH_ARCHIVES,
    DatasetSourceKind.GIT_SPARSE,
}
print("dataset handler enum-key registration ok")
PY
if rg -n 'validation_rule = .*\.value|source_kind = .*\.value|__registry__\.(get|\[).*\.value' benchmark/datasets/acquire.py; then
  echo "dataset handler registry still uses enum .value keys"
  exit 1
fi
python -m pytest tests/unit/test_dataset_acquisition_infra.py -q
```

## Dry-Run Checklist

Before implementation, run:

```bash
. .venv/bin/activate
python - <<'PY'
from benchmark import BBBC021_SINGLE_PLATE, OpenHCSAdapter, TimeMetric, MemoryMetric, run_benchmark
from benchmark.adapters import CellProfilerAdapter, OpenHCSAdapter as OpenHCSAdapterFromAdapters
from benchmark.datasets.registry import DATASET_REGISTRY
from benchmark.pipelines.registry import PIPELINE_REGISTRY, NUCLEI_SEGMENTATION

print("dataset count", len(DATASET_REGISTRY))
print("pipeline count", len(PIPELINE_REGISTRY))
print("root dataset", BBBC021_SINGLE_PLATE.id)
print("root pipeline", NUCLEI_SEGMENTATION.name)
print("adapter identity", OpenHCSAdapter is OpenHCSAdapterFromAdapters)
PY
```

Then run the AST inventories in this plan and confirm:

- only two `_PUBLIC_EXPORTS` maps exist;
- pipeline registry currently has one concrete pipeline spec;
- dataset alias assignments are all in one block;
- simple package `__all__` rewrites are limited to metrics/pipelines first.

## Dry-Run Results From 2026-06-30

The dry run was executed against the current checkout with `. .venv/bin/activate`.

Public import smoke:

```text
dataset count 16
pipeline count 1
root dataset BBBC021_Week1_22123
root pipeline nuclei_segmentation
adapter identity True
```

Import timing distinction:

```text
import benchmark: 0.04 seconds
from benchmark import BBBC021_SINGLE_PLATE, OpenHCSAdapter, TimeMetric,
MemoryMetric, run_benchmark: about 5.2 seconds and OpenHCS/CellProfiler
registry discovery logs
```

This means Phase 1 must not blindly replace root lazy exports with broad direct
imports unless that import cost is accepted. If laziness is preserved, it must
be generated from declarations or object owners rather than `_PUBLIC_EXPORTS`.

Per-module import timing:

```text
benchmark.contracts.dataset: 0.052s
benchmark.contracts.metric: 0.045s
benchmark.contracts.tool_adapter: 0.047s
benchmark.datasets: 0.323s
benchmark.datasets.registry: 0.306s
benchmark.metrics: 0.058s
benchmark.pipelines: 0.050s
benchmark.progress: 0.537s
benchmark.adapters: 0.042s
benchmark.adapters.openhcs: 4.298s
benchmark.adapters.cellprofiler: 4.151s
benchmark.runner: 4.397s
```

Per-export import timing:

```text
import benchmark: 0.065s
from benchmark import BBBC021_SINGLE_PLATE: 0.422s
from benchmark import TimeMetric, MemoryMetric: 0.064s
from benchmark import OpenHCSAdapter: 3.920s
from benchmark import CellProfilerAdapter: 4.100s
from benchmark import run_benchmark: 4.358s
from benchmark.adapters import OpenHCSAdapter, CellProfilerAdapter: 4.331s
from benchmark.datasets.registry import BBBC021_SINGLE_PLATE: 0.321s
from benchmark.metrics import TimeMetric, MemoryMetric: 0.063s
from benchmark.pipelines import NUCLEI_SEGMENTATION: 0.061s
```

Conclusion: root and adapter laziness is load-bearing, but only because of
heavy owner modules. Cheap packages can use direct re-exports.

Public root import users found by AST:

```text
examples/benchmark_example.py: from benchmark import BBBC021_SINGLE_PLATE, OpenHCSAdapter, TimeMetric, MemoryMetric, run_benchmark
tests/unit/test_well_throughput_scaling.py: from benchmark import well_throughput_scaling
```

`from benchmark import well_throughput_scaling` currently succeeds by normal
submodule import mechanics and triggers the heavy benchmark/OpenHCS import path.
The root lazy-export cleanup must not add `well_throughput_scaling` to
`__all__`; it only needs to avoid breaking normal submodule import.

Planned lazy-export parity:

```text
current root count 32
planned root count 32
root missing from plan []
root added by plan []
current adapters ('CellProfilerAdapter', 'OpenHCSAdapter')
planned adapters ('CellProfilerAdapter', 'OpenHCSAdapter')
adapter missing from plan []
adapter added by plan []
```

Mirrored export and registry inventory after fixing the AST script to include
annotated assignments:

```text
benchmark/__init__.py: _PUBLIC_EXPORTS, __all__
benchmark/adapters/__init__.py: _PUBLIC_EXPORTS, __all__
benchmark/datasets/registry.py: DATASET_CATALOG, DATASET_REGISTRY
benchmark/datasets/bioformats_hcs.py: BIOFORMATS_HCS_CATALOG, BIOFORMATS_HCS_REGISTRY
benchmark/pipelines/registry.py: PIPELINE_REGISTRY
benchmark/converter/__init__.py: _COMPATIBILITY_EXPORTS, __all__
```

The enum member `ManifestRootAcquisitionKind.DATASET_REGISTRY` in
`benchmark/contracts/manifest_acquisition.py` is a false positive for this
refactor. It is a domain value, not a registry mirror.

Pipeline declaration migration inventory:

```text
NUCLEI_SEGMENTATION -> name="nuclei_segmentation"
```

Pre-migration dataset alias inventory:

```text
BBBC021_SINGLE_PLATE -> "BBBC021_Week1_22123" row line 116
BBBC022_SINGLE_PLATE_DNA -> "BBBC022_20585_w1" row line 128
BBBC010_WORMS -> "BBBC010_worms" row line 136
BBBC011_WORMS_METABOLISM -> "BBBC011_worms_metabolism" row line 146
BBBC012_WORMS_INFECTION_MARKER -> "BBBC012_worms_infection_marker" row line 152
BBBC013_U2OS_TRANSLOCATION -> "BBBC013_u2os_translocation_bmp" row line 158
BBBC038_FULL -> "BBBC038_full" row line 170
BBBC039_NUCLEI_SEGMENTATION -> "BBBC039_nuclei_segmentation" row line 182
SINGH_2014_ILLUMINATION_CORRECTION -> "Singh_2014_illumination_correction" row line 194
SANZ_2019_HISTOLOGY -> "Sanz_2019_histology" row line 203
TIAN_2019_NEURONS -> "Tian_2019_neurons" row line 209
SOKOLOV_2023_NEURONS -> "Sokolov_2023_neurons" row line 215
CELL_ORIENTATION_WOUND_HEALING -> "CellOrientation_wound_healing" row line 224
CHROMTRANS_3D_FISH -> "ChromTrans_3d_fish" row line 238
CELLPROFILER_TUTORIALS -> "CellProfiler_tutorials" row line 252
CELLPROFILER4_BENCHMARK_SUPPLEMENT -> "CellProfiler4_benchmark_supplement" row line 318
```

Pre-migration dataset row projection simulation:

```text
catalog rows 16
unique ids 16
registry values match row.materialize order True
projected registry equals current keys True
projected alias count 16
all aliases resolve True
```

Contract mutation inventory, limited to `benchmark`, `tests`, `scripts`, and
`examples`:

```text
benchmark/progress.py: assignment to .metrics
tests/unit/pyqt_gui/test_dual_editor_window_artifact_refresh.py: assignment to .parameters
tests/unit/pyqt_gui/test_ui_agent_bridge.py: item assignment to .metadata[...]
tests/unit/test_cellprofiler_source_schema.py: item assignment to .metadata[...]
scripts/prototype_flatten_object_state.py: assignment/item assignment to .parameters
```

None of these are mutations of `DatasetSpec.urls`, `BenchmarkResult.metrics`,
or `BenchmarkResult.provenance`. The progress assignment is an unrelated
builder/state field.

AutoRegisterMeta enum-key dry run:

```text
registry keys (<Kind.A: 'a'>,)
enum lookup works True
string lookup present False
```

Current dataset handler registries:

```text
validation registry keys ('image_count', 'manifest', 'non_empty')
source registry keys ('archive_urls', 'url_files', 'git_sparse_with_archives', 'git_sparse')
```

Conclusion: converting dataset handler registry keys from enum `.value` strings
to enum members is a concrete safe target, with tests as the final gate.

## Risks

- Root `benchmark` imports stay lazy in this plan. Measure import time before
  and after Phase 4; if it is materially worse, fix the chosen source-owner
  resolver rather than restoring the string map.
- `DatasetSpec.urls` is currently a list even though most sources are tuples.
  This plan leaves it as `list[str]`.
- `AcquiredDataset.metadata` may be mutated by acquisition or reporting code.
  This plan leaves `AcquiredDataset` mutable.
- Public aliases are compatibility API. Preserve names and object values.

## Completion Criteria

- No `_PUBLIC_EXPORTS` table remains in benchmark API modules.
- Pipeline public aliases and registry are projected from nominal declarations.
- Pipeline and dataset declaration families are registered through
  `AutoRegisterMeta`; compatibility dicts contain materialized specs, not type
  registration authority.
- Dataset public aliases no longer duplicate dataset ID strings outside the
  dataset authority.
- `BenchmarkResult`, `PipelineSpec`, `DatasetSourceSpec`, `BenchmarkCategory`,
  `CellProfilerBenchmarkCaseSpec`, and `DatasetSpec` are frozen/slotted typed
  records.
- Public benchmark parameter, metric, and provenance values flow through
  `benchmark.contracts.values` aliases rather than `Any`.
- Focused benchmark tests pass.
- Any remaining string ABI names are owned by typed declarations or documented
  as external API identifiers.
