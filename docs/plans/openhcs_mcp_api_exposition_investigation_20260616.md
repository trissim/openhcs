# OpenHCS MCP API Exposition Investigation

## Status

Drafted 2026-06-16 as a focused addendum to
`docs/plans/openhcs_mcp_server_plan_20260616.md`.

This document investigates how OpenHCS should expose its capabilities to agents
through MCP. It is not an implementation.

## Question

What API should OpenHCS expose to an agent through MCP, and how do we expose it
without leaking internal architecture or creating a second runtime surface?

## External MCP Constraints

Current MCP documentation distinguishes three server primitives with different
control models:

- Prompts are user-controlled workflow templates.
- Resources are application-controlled context/data.
- Tools are model-controlled executable actions.

References:

- https://modelcontextprotocol.io/specification/2025-11-25/server
- https://modelcontextprotocol.io/specification/2025-11-25/server/resources
- https://modelcontextprotocol.io/specification/2025-11-25/server/tools
- https://modelcontextprotocol.io/specification/2025-11-25/server/prompts

This matters for OpenHCS because tools are the dangerous surface: an agent can
discover and invoke them automatically. Anything that writes files, downloads
data, launches a runtime, compiles a pipeline, or submits execution must be a
tool with explicit side-effect metadata and path/network policy.

The official Python SDK also supports structured tool output from typed return
annotations, including dataclasses and typed dictionaries. That means OpenHCS
can keep using typed DTOs without forcing the core package to become a Pydantic
application. Reference:

- https://py.sdk.modelcontextprotocol.io/protocol/

The 2025-11-25 spec also includes experimental tasks. They are a good future fit
for long-running benchmarks, but the current plan should not depend on them as
the primary job abstraction because the spec labels them experimental:

- https://modelcontextprotocol.io/specification/2025-11-25/basic/utilities/tasks

Roots are useful when clients provide them, but they are client capabilities and
not guaranteed. OpenHCS still needs its own server-side path policy:

- https://modelcontextprotocol.io/specification/2025-11-25/client/roots

Elicitation can ask users for additional information, but it should not be the
first-line safety mechanism. For OpenHCS v1, plan/apply tools are clearer,
more testable, and work across more clients:

- https://modelcontextprotocol.io/specification/2025-11-25/client/elicitation

## Repo Findings

### Existing Load-Bearing Authorities

OpenHCS already has several good authorities that MCP should project from:

- Dataset catalog/acquisition:
  - `benchmark/contracts/dataset.py`
  - `benchmark/datasets/registry.py`
  - `benchmark/datasets/acquire.py`
  - `benchmark/datasets/manifest.py`
- Benchmark execution/artifacts:
  - `benchmark/cellprofiler_benchmark_cli.py`
  - `benchmark/cellprofiler_comparison.py`
  - `benchmark/contracts/comparison_manifest.py`
- CellProfiler source and import:
  - `openhcs/core/input_workspace.py`
  - `openhcs/interop/cellprofiler/parser.py`
  - `openhcs/interop/cellprofiler/plate_workspace.py`
  - `openhcs/interop/cellprofiler/runtime_pipeline.py`
- Runtime execution/progress:
  - `openhcs/core/orchestrator/orchestrator.py`
  - `openhcs/runtime/zmq_execution_client.py`
  - `openhcs/core/progress/projection.py`
- Function discovery:
  - `openhcs/processing/func_registry.py`
- Runtime export/result observation:
  - `openhcs/core/runtime_exports.py`

These are not all public agent APIs. They are semantic sources that the MCP
layer should project into stable, smaller agent-facing DTOs.

### Existing Public API Plans Matter

The existing public API plans already warn about accidental public surface:

- `docs/plans/public_api_export_surface_authority_20260518.md`
- `docs/plans/public_internal_boundary_audit_20260518.md`

MCP exposition should follow the same rule: public surfaces derive from explicit
authorities. Do not let every importable dataclass, registry object, or helper
method become an agent API just because it is easy to serialize.

## Exposition Rule

OpenHCS should expose capabilities, not modules.

Bad shape:

```text
Agent sees PipelineOrchestrator, FileManager, GlobalPipelineConfig,
SourceSchemaWorkspaceMaterialization, ProcessingContext, raw FunctionStep
Python source, and arbitrary pipeline_code execution.
```

Good shape:

```text
Agent sees datasets, cases, manifests, pipelines, workspaces, jobs, executions,
artifacts, functions, diagnostics, and plans.
```

The MCP API should be a semantic projection:

- internal authority owns behavior;
- `openhcs.agent` service owns policy and DTO projection;
- `openhcs.mcp` owns protocol registration and transport details.

## Proposed Exposition Architecture

```text
OpenHCS internals
  DatasetSpec, CPPipeParser, InputWorkspacePreparationResult,
  OpenHCSExecutionSubmission, ExecutionRuntimeProjection, RuntimeExportObservation

        |
        v

openhcs.agent services
  DatasetService, BenchmarkService, PipelineService,
  RuntimeService, FunctionCatalogService, ResultService

        |
        v

openhcs.agent DTOs
  stable JSON-schema-compatible request/result types

        |
        v

openhcs.mcp
  resources, tools, prompts, completions, transport
```

The boundary is `openhcs.agent`, not `openhcs.mcp`. This keeps MCP from becoming
load-bearing business logic and lets CLI/tests/reviewer scripts reuse the same
API.

## DTO Policy

Use frozen dataclasses or typed dictionaries for the headless `openhcs.agent`
layer. Convert paths, enums, exceptions, and complex objects into explicit JSON
fields before returning them through MCP.

### Common DTOs

```python
@dataclass(frozen=True, slots=True)
class AgentResourceRef:
    uri: str
    title: str
    mime_type: str = "application/json"
    size_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class AgentWarning:
    code: str
    message: str
    hint: str | None = None


@dataclass(frozen=True, slots=True)
class AgentError:
    code: str
    message: str
    hint: str | None = None
    exception_type: str | None = None
    path: str | None = None


@dataclass(frozen=True, slots=True)
class AgentActionPlan:
    action: str
    summary: str
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    network: bool = False
    estimated_download_bytes: int | None = None
    required_extras: tuple[str, ...] = ()
    warnings: tuple[AgentWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class AgentOperationReceipt:
    action: str
    status: str
    outputs: tuple[AgentResourceRef, ...] = ()
    warnings: tuple[AgentWarning, ...] = ()
```

### Page DTO

Any potentially large list should be paginated or bounded:

```python
@dataclass(frozen=True, slots=True)
class AgentPage:
    items: tuple[object, ...]
    next_cursor: str | None = None
    total_count: int | None = None
```

For the actual implementation, avoid `object` in exposed schemas. Each endpoint
should use a concrete `DatasetCatalogPage`, `FunctionCatalogPage`, etc. The
generic sketch is only conceptual.

### Job DTO

```python
@dataclass(frozen=True, slots=True)
class AgentJobStatus:
    job_id: str
    action: str
    state: str
    percent: float | None = None
    started_at: str | None = None
    finished_at: str | None = None
    latest_message: str | None = None
    outputs: tuple[AgentResourceRef, ...] = ()
    error: AgentError | None = None
```

This should be the v1 job surface even if MCP tasks are later supported. If MCP
tasks become stable and widely supported, they can wrap or mirror this job
model.

## Resource Design

Resources should be GET-like. They can read state and summarize known things,
but should not do meaningful computation, network, runtime launch, writes, or
pipeline compilation.

### URI Policy

Use custom `openhcs://` URIs for semantic resources:

```text
openhcs://about
openhcs://capabilities
openhcs://datasets/catalog
openhcs://datasets/runnable-cases
openhcs://datasets/{dataset_id}
openhcs://benchmarks/presets
openhcs://benchmarks/results/{result_id}/summary
openhcs://pipelines/cellprofiler/{pipeline_id}/modules
openhcs://runtime/executions/{execution_id}/status
openhcs://functions/catalog
openhcs://functions/{function_id}
```

Do not put raw file paths directly into resource URIs except where the path is
already a user-visible local artifact and is safely encoded. Prefer opaque IDs
for path-backed objects:

- `pipeline_id`: stable hash/handle for a `.cppipe` path registered by a tool.
- `result_id`: stable hash/handle for a result directory.
- `manifest_id`: stable hash/handle for a manifest path.

This avoids URI escaping bugs, leaking usernames or temp directory structure into
model-visible resource names, and accidental reliance on raw paths as public API.

### Resource Payload Policy

Resources should return:

- concise JSON summaries by default;
- resource links for large artifacts;
- counts and hashes instead of full raw tables unless requested;
- warnings when a resource is incomplete or stale.

Examples:

- result summary resource returns aggregate pass/fail/speedup/module coverage and
  links to CSV/JSON artifacts;
- function catalog resource returns names/signatures only, while
  `openhcs://functions/{function_id}` returns full help;
- `.cppipe` module resource returns ordered module summaries and setting counts,
  not every long setting value by default.

## Tool Design

Tools are model-controlled, so every tool must have:

- a narrow verb;
- a typed input schema;
- a typed output schema;
- declared side effects;
- path policy;
- bounded runtime expectations;
- structured errors.

### Tool Families

#### Discovery

Read-only or near-read-only tools:

```text
openhcs_health_check
openhcs_list_capabilities
openhcs_list_functions
openhcs_get_function_help
```

These should be safe to call automatically.

#### Dataset

```text
openhcs_list_benchmark_datasets
openhcs_describe_benchmark_dataset
openhcs_plan_dataset_acquisition
openhcs_validate_dataset_cache
openhcs_acquire_dataset
```

`openhcs_acquire_dataset` is the only mutating/network-capable dataset tool. It
must require either explicit `allow_network=True` or a server config that enables
network by default.

#### Benchmark

```text
openhcs_build_benchmark_manifest
openhcs_plan_benchmark_run
openhcs_run_benchmark
openhcs_get_job_status
openhcs_cancel_job
openhcs_summarize_benchmark_results
```

The benchmark family should use plan/apply:

1. plan what will run;
2. write manifest only when requested;
3. run benchmark as a job;
4. summarize artifacts through result service.

#### Pipeline

```text
openhcs_find_cppipes
openhcs_describe_cppipe
openhcs_prepare_input_workspace
openhcs_generate_pipeline_from_cppipe
```

These tools should expose CellProfiler semantics without exposing the full
converter implementation. Generated Python pipeline source should be a resource
artifact, not a model-editable code execution channel by default.

#### Runtime

```text
openhcs_probe_plate
openhcs_compile_pipeline
openhcs_submit_execution
openhcs_get_execution_status
openhcs_get_execution_progress
```

Runtime tools should use ZMQ when possible. They should not accept arbitrary
Python source text in v1. They should accept handles or paths produced by prior
tools: manifest case IDs, generated pipeline IDs, or explicitly selected
pipeline artifact refs.

## Prompt Design

Prompts should encode safe workflows. They are not a replacement for tool
validation.

Initial prompts:

```text
openhcs/reviewer_smoke_benchmark
openhcs/cellprofiler_pipeline_triage
openhcs/benchmark_result_review
openhcs/openhcs_bug_report_context
```

Each prompt should start by calling discovery/capability tools, then planning
tools, then mutating tools only after the user asks to proceed.

Prompt text should explicitly tell agents:

- do not run large downloads without user consent;
- prefer runnable benchmark cases;
- summarize result artifacts rather than pasting full CSVs;
- report structured errors and artifact links;
- do not edit or execute arbitrary pipeline Python unless the user explicitly
  asks and the server exposes that capability.

## Capability Registry

Add an explicit registry for the exposed agent API.

Sketch:

```python
@dataclass(frozen=True, slots=True)
class AgentCapabilitySpec:
    name: str
    kind: Literal["resource", "tool", "prompt"]
    title: str
    description: str
    service: str
    side_effects: tuple[str, ...] = ()
    requires_network: bool = False
    required_extras: tuple[str, ...] = ()
    input_type: str | None = None
    output_type: str | None = None
```

This registry should drive:

- `openhcs://capabilities`;
- tool/resource/prompt registration smoke tests;
- documentation tables;
- advisor-like checks that every mutating tool has side-effect metadata;
- future compatibility/version reporting.

This mirrors the existing repo direction from public API authority plans: public
surfaces should derive from a nominal authority rather than scattered manual
lists.

## Path And Permission Exposition

MCP clients may expose roots, but OpenHCS should still have its own path policy.

Recommended policy:

1. Server starts with configured roots:
   - repo root;
   - `/tmp`;
   - configured cache root;
   - configured output root;
   - `OPENHCS_MCP_ALLOWED_ROOTS`.
2. If the client supports MCP roots, include those roots as allowed read roots.
3. Writes are allowed only to configured output/cache roots unless explicitly
   enabled.
4. All paths are resolved before policy checks.
5. Symlink escapes are denied.
6. Tools return `path_denied` with the denied path and allowed roots summary.

The path policy must live under `openhcs.agent`, not inside individual MCP tool
decorators.

## What Not To Expose

Do not expose these as first-class MCP API:

- `PipelineOrchestrator` object methods;
- `FileManager` or `StorageRegistry`;
- `ProcessingContext`;
- raw `GlobalPipelineConfig` mutation;
- arbitrary `pipeline_code` execution;
- raw `FunctionStep` Python source as executable input;
- raw source-schema materialization internals;
- internal CellProfiler symbol-table/build strategy classes;
- PyQt plate manager row state;
- ObjectState widget scope IDs;
- full benchmark observations as unbounded default payloads;
- image arrays or large tables inline unless explicitly requested as resources.

Expose projections instead:

- plate probe summary;
- workspace preparation summary;
- generated pipeline artifact ref;
- benchmark case summary;
- execution status/progress;
- artifact manifest/resource links.

## Schema Versioning

Add a small schema version field to all major response families:

```python
schema_version: str = "openhcs.agent.v1"
```

Do not promise that internal OpenHCS dataclass field names are stable. Promise
that the agent DTOs are stable within a schema version.

Breaking changes should:

- add `openhcs.agent.v2` DTOs;
- keep v1 tools/resources for at least one release window if practical;
- expose supported schema versions in `openhcs://capabilities`.

## Result Shape Examples

### Dataset Catalog Entry

```python
@dataclass(frozen=True, slots=True)
class DatasetCatalogEntry:
    dataset_id: str
    title: str
    microscope_type: str
    source_kind: str
    size_bytes: int
    validation_rule: str
    benchmark_case_count: int
    cached_state: str
    runnable: bool
    warnings: tuple[AgentWarning, ...] = ()
```

### Benchmark Run Plan

```python
@dataclass(frozen=True, slots=True)
class BenchmarkRunPlan:
    manifest_path: str
    output_dir: str
    case_count: int
    cases: tuple[str, ...]
    required_extras: tuple[str, ...]
    expected_artifacts: tuple[str, ...]
    estimated_writes: tuple[str, ...]
    warnings: tuple[AgentWarning, ...]
```

### CPPipe Summary

```python
@dataclass(frozen=True, slots=True)
class CPPipeSummary:
    pipeline_id: str
    path: str
    module_count: int
    modules: tuple[CPPipeModuleSummary, ...]
    image_plane_source_count: int
    generated_pipeline: AgentResourceRef | None = None
    diagnostics: tuple[AgentWarning, ...] = ()
```

### Execution Projection

```python
@dataclass(frozen=True, slots=True)
class ExecutionStatusSummary:
    execution_id: str
    state: str
    overall_percent: float
    plate_count: int
    failed_count: int
    active_axes: tuple[str, ...]
    latest_message: str | None = None
```

## Implementation Guidance

Use the official Python SDK/FastMCP in the MCP adapter layer:

- use typed return annotations for structured output;
- use `Context.report_progress(...)` for short/medium tool progress;
- use the OpenHCS job model for long benchmark runs;
- use direct `CallToolResult` only when hidden `_meta` or custom error content
  is genuinely needed;
- keep server lifespan context for services, path roots, and job store.

Example shape:

```python
@dataclass(frozen=True, slots=True)
class OpenHCSMCPContext:
    path_policy: AgentPathPolicy
    dataset_service: DatasetService
    benchmark_service: BenchmarkService
    pipeline_service: PipelineService
    runtime_service: RuntimeService
    result_service: ResultService
    job_store: AgentJobStore
```

MCP tool functions should be one or two lines after validation:

```python
@mcp.tool()
def openhcs_plan_dataset_acquisition(dataset_id: str) -> DatasetAcquisitionPlan:
    return ctx.dataset_service.plan_acquisition(dataset_id)
```

If a tool becomes a workflow implementation, move that workflow into
`openhcs.agent.services`.

## Testing Requirements

Add tests that assert the exposition boundary, not just happy-path behavior:

- every MCP tool maps to an `AgentCapabilitySpec`;
- every mutating tool declares side effects;
- every network-capable tool declares network;
- resources have no side effects;
- tool schemas do not expose internal class names such as
  `PipelineOrchestrator`, `ProcessingContext`, or `FileManager`;
- path escapes are denied before lower-level services run;
- large summaries are paginated or bounded;
- errors return stable `AgentError.code`;
- result tools return resource refs instead of embedding full large artifacts;
- `openhcs://capabilities` matches registered MCP handlers;
- read-only stage works without CellProfiler, PyQt, Napari, or GPU extras.

## Recommended First Exposition Slice

The first slice should not include benchmark execution. It should prove the API
shape without high-risk side effects:

1. `openhcs.agent.contracts`
2. `AgentCapabilitySpec`
3. `AgentPathPolicy`
4. `DatasetService.list_catalog(...)`
5. `DatasetService.plan_acquisition(...)`
6. `PipelineService.find_cppipes(...)`
7. `PipelineService.describe_cppipe(...)`
8. `ResultService.summarize_benchmark_results(...)`
9. MCP `about`, `capabilities`, dataset catalog, pipeline summary, result
   summary resources/tools
10. schema and boundary tests

Then add controlled writes/downloads. Then add benchmark jobs. Then add ZMQ
runtime bridging.

## Conclusion

The correct exposition is a stable, capability-oriented agent API that projects
OpenHCS semantics into small typed DTOs. MCP should expose those DTOs as
resources/tools/prompts, but should not expose OpenHCS objects or Python
execution directly.

The key design move is `openhcs.agent`: it becomes the public operational API
for agents and reviewers. `openhcs.mcp` is only one adapter over it.
