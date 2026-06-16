# OpenHCS MCP Server Plan

## Status

Drafted 2026-06-16 after inspecting the current benchmark harness, dataset
registry, CellProfiler import path, orchestrator input-workspace contract, ZMQ
execution client, and progress projection code.

This is a planning document. It does not implement an MCP server yet.

Detailed API exposition guidance lives in:

- `docs/plans/openhcs_mcp_api_exposition_investigation_20260616.md`

The current worktree is actively being refactored in parallel, so this plan
anchors to the stable headless contracts rather than PyQt workflow internals.

## MCP Assumptions

MCP servers expose three main server-side concepts:

- Resources: URI-addressed context/data.
- Tools: model-callable actions with schemas.
- Prompts: user-selected workflow templates.

The current specification also describes client-side features such as roots,
sampling, and elicitation. OpenHCS should not depend on advanced client features
for the first version. A local stdio server with explicit filesystem roots is
the safest starting point.

Reference docs:

- https://modelcontextprotocol.io/specification/2025-11-25
- https://modelcontextprotocol.io/specification/2025-11-25/server/resources
- https://modelcontextprotocol.io/specification/2025-11-25/server/tools
- https://modelcontextprotocol.io/specification/2025-11-25/server/prompts

## Goal

Build an OpenHCS MCP server that lets a completely blind agent, or a reviewer
unfamiliar with the repository, discover OpenHCS capabilities and run meaningful
benchmark/reproducibility workflows without reverse-engineering the codebase.

The MCP server should answer practical questions:

1. What can this checkout do?
2. What benchmark datasets and cases exist?
3. Which datasets are already cached, which require download, and how large are
   they?
4. Which cases are runnable because they include both data and pipeline
   declarations?
5. What does this `.cppipe` contain?
6. Can OpenHCS prepare this input workspace?
7. Can OpenHCS compile or execute this pipeline?
8. Where did results go, and how should they be summarized?
9. What exact artifacts should a reviewer inspect?

## Non-Goals

- Do not expose arbitrary Python execution as an MCP tool.
- Do not require PyQt or a GUI event loop.
- Do not make MCP the new owner of benchmark, CellProfiler, dataset, or runtime
  semantics.
- Do not silently download large datasets.
- Do not silently launch long-running benchmark suites in a caller's working
  directory.
- Do not make the first version a remote multi-user service.

## Current Codebase Anchors

### Benchmark CLI And Artifacts

Relevant files:

- `scripts/benchmark_cellprofiler_vs_openhcs.py`
- `benchmark/cellprofiler_benchmark_cli.py`
- `benchmark/cellprofiler_comparison.py`
- `benchmark/contracts/comparison_manifest.py`

The benchmark CLI already has a nominal command family:

- `run`
- `official-cp3-manifest`
- plot/validation commands

The `run` command already writes structured artifacts including:

- `observations.jsonl`
- `observations.csv`
- `phase_timing.csv`
- `summary.csv`
- `suite_metadata.json`
- `module_coverage_summary.json`
- module coverage CSV files

MCP should call the same Python functions as the CLI where possible. If CLI
methods contain too much orchestration logic, split shared service functions
first, then bind both CLI and MCP to those services.

### Dataset Registry And Acquisition

Relevant files:

- `benchmark/contracts/dataset.py`
- `benchmark/datasets/registry.py`
- `benchmark/datasets/acquire.py`
- `benchmark/datasets/manifest.py`

The dataset layer already has the right nominal building blocks:

- `DatasetSpec`
- `DatasetSourceSpec`
- `DatasetSourceKind`
- `DatasetValidationRule`
- `CellProfilerBenchmarkCaseSpec`
- `DatasetSourceHandler`
- `DatasetValidationStrategy`
- `acquire_dataset(...)`
- `comparison_manifest_payload(...)`

This is the natural basis for MCP dataset tools. The MCP server should expose
the catalog, cache state, acquisition plans, and runnable benchmark cases, not
raw downloader internals.

Important product rule: default MCP dataset listing should separate "dataset has
files" from "dataset has benchmark cases." A reviewer trying to test parity
should see runnable case-bearing entries first, not pipeline-only or data-only
entries that cannot prove anything end to end.

### CellProfiler Import And Workspace Preparation

Relevant files:

- `openhcs/core/input_workspace.py`
- `openhcs/interop/cellprofiler/parser.py`
- `openhcs/interop/cellprofiler/plate_workspace.py`
- `openhcs/interop/cellprofiler/runtime_pipeline.py`
- `openhcs/interop/cellprofiler/import_service.py`

OpenHCS already has a generic input preparation contract:

- `InputWorkspacePreparationRequest`
- `InputWorkspacePreparationResult`
- `PipelineImportDiagnostic`

The CellProfiler bridge already exposes:

- visible `.cppipe` discovery;
- selected pipeline handling;
- generated OpenHCS pipeline path;
- source schema materialization;
- non-fatal import diagnostics when pipeline import fails.

MCP should reuse this path for `.cppipe` inspection and dry-run preparation.
Agents should not be forced to discover through the PyQt plate manager.

### Orchestrator And Runtime Execution

Relevant files:

- `openhcs/core/orchestrator/orchestrator.py`
- `openhcs/runtime/zmq_execution_client.py`
- `openhcs/runtime/zmq_execution_server.py`
- `openhcs/core/progress/types.py`
- `openhcs/core/progress/projection.py`

The orchestrator owns initialization, input workspace rebinding, compilation,
and execution. The ZMQ client already has an explicit submission payload:

- `OpenHCSExecutionSubmission`
- `submit_compile(...)`
- `submit_pipeline(...)`
- `get_status(...)`

The progress layer already projects raw events into:

- `PlateRuntimeState`
- `AxisRuntimeProjection`
- `PlateRuntimeProjection`
- `ExecutionRuntimeProjection`

MCP runtime tools should call ZMQ client APIs or orchestrator services. They
should not reconstruct compile/execution semantics from GUI row state.

### Function Registry

Relevant file:

- `openhcs/processing/func_registry.py`

The function registry is the right source for a discoverable catalog of OpenHCS
processing functions. MCP should expose filtered discovery and help, but should
avoid dumping the full registry into context unless asked.

## Load-Bearing Abstraction

Do not make `openhcs.mcp` a direct collection of ad hoc wrappers. Add a small
headless agent API layer, then make MCP one transport for that layer.

Proposed package:

```text
openhcs/agent/
  __init__.py
  contracts.py
  path_policy.py
  capability_registry.py
  services/
    benchmark_service.py
    dataset_service.py
    pipeline_service.py
    runtime_service.py
    function_catalog_service.py
    result_service.py

openhcs/mcp/
  __init__.py
  server.py
  resources.py
  tools.py
  prompts.py
  serializers.py
```

The `openhcs.agent` layer should be reusable by:

- MCP server;
- future CLI cleanup;
- reviewer smoke-test scripts;
- tests;
- possibly PyQt, but only where it helps remove duplicated logic.

### Core Agent Contracts

Sketch:

```python
@dataclass(frozen=True, slots=True)
class AgentPathRoots:
    allowed_roots: tuple[Path, ...]
    cache_root: Path
    output_root: Path

@dataclass(frozen=True, slots=True)
class AgentActionDescriptor:
    name: str
    purpose: str
    side_effects: tuple[str, ...]
    required_extras: tuple[str, ...]
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    network: bool

@dataclass(frozen=True, slots=True)
class AgentActionPlan:
    action: str
    summary: str
    estimated_download_bytes: int | None = None
    inputs: Mapping[str, object] = field(default_factory=dict)
    outputs: Mapping[str, object] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
```

The plan/apply split matters for blind agents. Mutating operations should have a
dry-run or plan form that reports paths, size, required extras, expected
artifacts, and likely runtime before writing or downloading.

## MCP Server Shape

### Entry Points And Dependency

Add an optional dependency group:

```toml
[project.optional-dependencies]
mcp = [
    "mcp>=1.0",
]
```

Add a console entry point:

```toml
[project.scripts]
openhcs-mcp = "openhcs.mcp.server:main"
```

Implementation should use the official Python MCP SDK if it is acceptable at
implementation time. Pin or constrain after checking current SDK/API stability.

Default transport:

- `stdio`: first release, local trusted users, easiest for agent clients.

Later transport:

- HTTP/streaming HTTP with authentication, root restrictions, and audit logs.

### Server Configuration

Environment variables:

- `OPENHCS_MCP_ALLOWED_ROOTS`: path-separated roots agents may read/write.
- `OPENHCS_MCP_CACHE_ROOT`: default benchmark dataset cache root.
- `OPENHCS_MCP_OUTPUT_ROOT`: default benchmark/result output root.
- `OPENHCS_MCP_ALLOW_NETWORK`: default `0`; tools can still require explicit
  `allow_network=True`.
- `OPENHCS_MCP_ZMQ_HOST`: default `localhost`.
- `OPENHCS_MCP_ZMQ_PORT`: default from `OPENHCS_ZMQ_CONFIG`.

CLI flags should mirror the env vars.

## Resources

Resources should be mostly read-only and cheap enough to call freely.

Initial static resources:

- `openhcs://about`
  - package version, git commit if available, Python version, install extras
    detected, key paths.
- `openhcs://capabilities`
  - tool/resource/prompt list with side-effect labels and required extras.
- `openhcs://getting-started`
  - concise blind-agent workflow: inspect catalog, validate cache, run smoke
    benchmark, summarize.

Dataset resources:

- `openhcs://datasets/catalog`
  - all dataset specs, source kinds, sizes, validation rules, microscope types,
    benchmark-case counts.
- `openhcs://datasets/runnable-cases`
  - only cases with declared data and `.cppipe` paths.
- `openhcs://datasets/{dataset_id}`
  - detailed spec and acquisition source.
- `openhcs://datasets/{dataset_id}/cache`
  - cache existence, validation state, image count if cheaply available.

Benchmark resources:

- `openhcs://benchmarks/presets`
  - reviewer presets and smoke suites.
- `openhcs://benchmarks/manifest/{manifest_id_or_hash}`
  - manifest summary if registered by an MCP tool.
- `openhcs://benchmarks/results/{run_id}/summary`
  - parsed summary artifacts.
- `openhcs://benchmarks/results/{run_id}/artifacts`
  - artifact manifest with paths, sizes, and types.

Pipeline resources:

- `openhcs://pipelines/cellprofiler/{pipeline_id}/modules`
  - parsed modules and settings for a `.cppipe` registered by a tool call.
- `openhcs://pipelines/cellprofiler/{pipeline_id}/import-diagnostics`
  - import or source-schema preparation diagnostics.

Runtime resources:

- `openhcs://runtime/executions/{execution_id}/status`
  - current ZMQ status payload.
- `openhcs://runtime/executions/{execution_id}/progress`
  - `ExecutionRuntimeProjection` payload when progress events are available.

Function resources:

- `openhcs://functions/catalog`
  - filtered function catalog metadata.
- `openhcs://functions/{function_id}`
  - one function signature/help record.

## Tools

Tool names should be explicit and boring. Inputs and outputs should be stable
JSON payloads. Every tool that reads or writes paths must pass through
`AgentPathPolicy`.

### Discovery Tools

`openhcs_health_check()`

- Imports core modules.
- Reports optional extras availability.
- Reports whether a ZMQ execution server is reachable.
- Reports path root configuration.

`openhcs_list_capabilities()`

- Returns all MCP tools/resources/prompts with side-effect labels.

`openhcs_list_functions(query=None, backend=None, limit=50)`

- Initializes the function registry if needed.
- Returns names, modules, backends, signatures, and short help.

`openhcs_get_function_help(function_id)`

- Returns full help for one registered function.

### Dataset Tools

`openhcs_list_benchmark_datasets(include_non_runnable=False)`

- Returns dataset catalog rows.
- Defaults to runnable/testable datasets first.
- Includes benchmark case counts and estimated download size.

`openhcs_describe_benchmark_dataset(dataset_id)`

- Returns full `DatasetSpec` projection, including source kind and declared
  benchmark cases.

`openhcs_plan_dataset_acquisition(dataset_id, cache_root=None)`

- No network.
- Reports expected size, URLs or git source, validation rule, target paths, and
  whether cache appears present.

`openhcs_acquire_dataset(dataset_id, cache_root=None, allow_network=False)`

- Requires explicit `allow_network=True` unless server config permits network.
- Calls `acquire_dataset(...)`.
- Returns `AcquiredDataset` projection and validation metadata.

`openhcs_validate_dataset_cache(dataset_id, cache_root=None)`

- No network.
- Uses existing validation strategies where possible.
- Returns missing/present/invalid with actionable diagnostics.

### Manifest And Benchmark Tools

`openhcs_build_benchmark_manifest(...)`

Inputs:

- dataset IDs or case names;
- cache root;
- output path;
- include/exclude value-only cases;
- optional path roots.

Behavior:

- Reuses `comparison_manifest_payload(...)`.
- Refuses missing cached datasets unless `auto_acquire=True`.
- Writes only to allowed output roots.

`openhcs_plan_benchmark_run(manifest_path, output_dir, ...)`

- No benchmark execution.
- Loads `ComparisonManifest`.
- Reports case count, dataset paths, required tools, estimated writes, expected
  artifacts, and likely native CellProfiler requirements.

`openhcs_run_benchmark(manifest_path, output_dir, ..., async=True)`

- Calls `run_comparison_suite(...)` through a service, not by shelling out.
- Defaults should be reviewer-safe:
  - `repeats=1`
  - `continue_on_error=True`
  - `no_memory_metric=True`
  - small axis limits for smoke presets
  - explicit output directory
- Returns a job id immediately for long runs.

`openhcs_get_job_status(job_id)`

- Returns state, timestamps, current command/service, latest log lines, and known
  output paths.

`openhcs_cancel_job(job_id)`

- Best-effort cancellation for MCP-managed jobs.

`openhcs_summarize_benchmark_results(output_dir)`

- Parses `summary.csv`, `suite_metadata.json`,
  `module_coverage_summary.json`, and known CSV artifacts.
- Returns a reviewer-facing summary with failures first.

### CellProfiler And Pipeline Tools

`openhcs_find_cppipes(folder)`

- Lists direct visible `.cppipe` files using the same semantics as
  `CellProfilerPlateWorkspacePreparer`.
- Reports preferred pipeline and start/final/unlabeled stage.

`openhcs_describe_cppipe(cppipe_path)`

- Uses `CPPipeParser`.
- Returns header, ordered modules, setting counts, source declarations, and
  unsupported/import-risk hints where available.

`openhcs_prepare_input_workspace(selected_path, selected_pipeline_path=None, dry_run=True)`

- Uses `InputWorkspacePreparationRequest` and
  `prepare_cellprofiler_input_workspace(...)`.
- In `dry_run=True`, should avoid durable writes where practical. If current
  implementation cannot avoid writes, the service should use a temporary
  workspace root and report that limitation.
- Returns original source root, execution plate path, generated pipeline path,
  source schema summary, materialization summary, and import diagnostics.

`openhcs_generate_pipeline_from_cppipe(cppipe_path, output_path, dry_run=True)`

- Uses existing CellProfiler import/generation path.
- Reports diagnostics and generated step summary.
- Writes only with explicit `dry_run=False`.

### Runtime Tools

`openhcs_probe_plate(plate_path, selected_pipeline_path=None, microscope_type="auto")`

- Initializes enough headless state to report detected microscope, axes,
  channels, sites, z/time if available.
- Uses orchestrator/input-workspace semantics, not GUI row state.

`openhcs_compile_pipeline(...)`

- Prefer ZMQ compile path through `OpenHCSExecutionSubmission.compile_request()`
  when a server is available.
- Otherwise expose a direct local compile service only if it can be made
  deterministic and headless.
- Returns compile artifact id or structured failure.

`openhcs_submit_execution(...)`

- Uses `ZMQExecutionClient.submit_pipeline(...)`.
- Requires explicit pipeline source from a file, generated pipeline, or a
  registered manifest/case. It should not accept arbitrary Python text from the
  agent in v1.

`openhcs_get_execution_status(execution_id=None)`

- Uses `ZMQExecutionClient.get_status(...)`.

`openhcs_get_execution_progress(execution_id=None)`

- Returns progress projection where event history is available.

## Prompts

Prompts should teach unfamiliar users and agents how to use the tools in the
right order.

`openhcs/reviewer_smoke_benchmark`

- Inspect health.
- List runnable cases.
- Validate or acquire the chosen small dataset.
- Build a manifest.
- Run the benchmark.
- Summarize artifacts and failures.

`openhcs/cellprofiler_pipeline_triage`

- Find `.cppipe` files.
- Parse modules.
- Prepare workspace in dry-run mode.
- Report import diagnostics and next likely fix.

`openhcs/benchmark_result_review`

- Load a benchmark output directory.
- Summarize pass/fail, speedups, value-only caveats, module coverage, and
  artifact locations.

`openhcs/openhcs_bug_report_context`

- Collect health, git revision, package extras, selected dataset/case info,
  manifest summary, recent job status, and result artifact manifest.

## Reviewer Presets

Add a benchmark preset layer rather than hard-coding "small benchmark" inside
MCP.

Proposed file:

```text
benchmark/presets.py
```

Sketch:

```python
@dataclass(frozen=True, slots=True)
class BenchmarkPreset:
    id: str
    purpose: str
    dataset_ids: tuple[str, ...]
    case_names: tuple[str, ...] = ()
    max_axis_count: int | None = None
    expected_download_bytes: int | None = None
    required_extras: tuple[str, ...] = ()
```

Initial presets:

- `reviewer_smoke`
  - smallest runnable CellProfiler tutorial case with data and `.cppipe`.
  - strict goal: proves end-to-end path, not full coverage.
- `reviewer_cellprofiler_tutorials`
  - selected CellProfiler tutorial cases.
- `paper_labmeeting_snapshot`
  - points to known local/manuscript benchmark result roots when available.

The preset layer should be consumed by MCP, CLI, and documentation. This avoids
another hidden place where "official benchmark" semantics drift.

## Job Model

Long-running operations need a simple job service.

Proposed package:

```text
openhcs/agent/job_store.py
```

Job states:

- `queued`
- `running`
- `succeeded`
- `failed`
- `cancelled`

Job payload:

- job id;
- action name;
- request summary;
- start/end timestamps;
- output paths;
- latest structured status;
- log path;
- exception type/message/traceback path.

The first implementation can be in-process for stdio MCP. If remote serving is
added later, the job store should move to SQLite or a small file-backed store.

Benchmark jobs and runtime execution jobs are different:

- Benchmark jobs are owned by the MCP service and wrap benchmark harness calls.
- Runtime execution jobs may be owned by the ZMQ server. MCP should proxy status
  rather than pretending to own them.

## Path And Side-Effect Policy

Path policy must be part of the agent layer, not scattered through MCP handlers.

Rules:

- Resolve all user paths before use.
- Require every read/write path to be inside an allowed root.
- Default allowed roots:
  - repository root;
  - `/tmp`;
  - configured benchmark cache root;
  - configured output root.
- Never follow a user-provided path into a denied root through symlinks.
- No destructive delete tool in v1.
- No arbitrary shell command tool.
- No arbitrary Python source execution tool.
- Network operations require an acquisition plan and explicit approval flag.
- Tools must report writes before performing them.

## Error Model

Every MCP tool should return structured errors rather than only free-form text.

Sketch:

```python
@dataclass(frozen=True, slots=True)
class AgentError:
    code: str
    message: str
    hint: str | None = None
    path: str | None = None
    exception_type: str | None = None
```

Examples:

- `path_denied`
- `dataset_not_found`
- `dataset_cache_missing`
- `network_not_allowed`
- `cppipe_not_found`
- `cppipe_import_failed`
- `orchestrator_init_failed`
- `zmq_unavailable`
- `benchmark_failed`

This is important for agents because they can recover from explicit error codes
without parsing logs.

## Implementation Plan

### Stage 1: Read-Only Discovery

Implement `openhcs.agent` DTOs, path policy, and read-only services:

- health check;
- capability listing;
- dataset catalog listing;
- runnable case listing;
- cached dataset status;
- benchmark result summary;
- `.cppipe` parsing;
- function catalog filtering.

Add MCP resources/tools for those services.

Tests:

- unit tests for path policy;
- dataset catalog projection test;
- runnable case projection test;
- `.cppipe` parser tool fixture;
- summary parser fixture with tiny fake result directory.

### Stage 2: Planning Tools For Mutating Work

Add plan-only tools:

- dataset acquisition plan;
- benchmark manifest plan;
- benchmark run plan;
- input workspace preparation dry-run;
- generated pipeline dry-run.

No network or long-running execution yet.

Tests:

- plan payloads are deterministic;
- missing cache is reported without download;
- output path outside allowed roots is rejected;
- dry-run workspace preparation writes only to a temp root, or reports that a
  durable write would be required by existing lower-level code.

### Stage 3: Controlled Writes And Downloads

Enable:

- `openhcs_acquire_dataset(..., allow_network=True)`;
- `openhcs_build_benchmark_manifest(...)`;
- `openhcs_generate_pipeline_from_cppipe(..., dry_run=False)`;
- `openhcs_prepare_input_workspace(..., dry_run=False)`.

Tests:

- network tests marked and skipped by default;
- fixture archive acquisition through a local HTTP/file server if practical;
- path policy rejects writes outside roots;
- generated pipeline path is explicit and deterministic.

### Stage 4: Benchmark Execution Jobs

Add in-process job store and benchmark execution service.

Enable:

- `openhcs_run_benchmark(..., async=True)`;
- `openhcs_get_job_status(job_id)`;
- `openhcs_cancel_job(job_id)`;
- `openhcs_summarize_benchmark_results(output_dir)`.

Tests:

- mocked benchmark service success/failure;
- real tiny manifest integration if a fixture exists;
- summary artifacts are parsed and exposed after completion;
- failed cases are visible without reading raw logs.

### Stage 5: ZMQ Runtime Bridge

Enable:

- execution server health;
- compile submission;
- execution submission;
- status polling;
- progress projection;
- cancellation if supported by the runtime server/client path.

Tests:

- ZMQ client mocked at unit level;
- optional integration marker for a real execution server;
- compile failure status maps to structured `AgentError`;
- progress events map through `ExecutionRuntimeProjection`.

### Stage 6: Reviewer Packaging

Add reviewer-facing preset resources and prompts:

- `reviewer_smoke`;
- `reviewer_cellprofiler_tutorials`;
- result review prompt;
- bug report context prompt.

Add documentation:

- `docs/mcp_server.md`;
- install command;
- example client config;
- example reviewer workflow;
- expected artifacts.

## Testing Matrix

Fast unit tests:

- `tests/unit/agent/test_path_policy.py`
- `tests/unit/agent/test_dataset_service.py`
- `tests/unit/agent/test_pipeline_service.py`
- `tests/unit/agent/test_result_service.py`
- `tests/unit/mcp/test_resources.py`
- `tests/unit/mcp/test_tools.py`

Integration tests:

- `tests/integration/mcp/test_stdio_server.py`
- `tests/integration/benchmark/test_mcp_smoke_benchmark.py`
- `tests/integration/runtime/test_mcp_zmq_bridge.py`

Optional tests:

- network acquisition;
- native CellProfiler parity;
- full reviewer preset.

CI default should run read-only and mocked execution tests. Network/native
CellProfiler tests should remain opt-in.

## Architecture Risks

### Risk: MCP Becomes A Second CLI

Mitigation: MCP handlers should be thin transport adapters over
`openhcs.agent.services`. CLI commands should migrate toward those services when
they currently contain reusable logic.

### Risk: Agents Trigger Huge Downloads

Mitigation: plan/apply split, explicit `allow_network`, size reporting, and
runnable-case-first discovery.

### Risk: Agents Run Arbitrary Code

Mitigation: no arbitrary Python source tool in v1. Runtime submissions must
refer to existing files, generated pipeline products, or manifest cases.

### Risk: GUI State Leaks Into Headless API

Mitigation: MCP uses dataset, input workspace, orchestrator, ZMQ, and progress
contracts. It does not depend on PyQt plate rows, ObjectState widget scope, or
dual editor state.

### Risk: Result Summaries Drift From Benchmark Artifacts

Mitigation: result service parses existing artifact names from
`benchmark/cellprofiler_comparison.py`; do not invent parallel artifact names.

### Risk: Tool Payloads Become Too Large

Mitigation: list tools return summaries and IDs. Detailed module/function/result
payloads require separate calls.

## Open Questions

1. Which cases should be in `reviewer_smoke`?
   - Choose after measuring current dataset sizes and runtimes.
2. Should the MCP server ever launch the ZMQ execution server?
   - Recommendation: v1 only connects to an existing server. Managed launch can
     be a later explicit tool.
3. Should MCP expose CP Analyst export workflows?
   - Recommendation: wait until the CP Analyst export plan is implemented. Then
     expose it through result/materialization services, not a standalone MCP
     shortcut.
4. Should the benchmark CLI be refactored before MCP implementation?
   - Recommendation: only split reusable service functions as needed. Do not do
     a broad CLI rewrite before a read-only MCP prototype.
5. Should remote HTTP MCP be supported immediately?
   - Recommendation: no. Start with stdio. Add HTTP only after local semantics,
     auth, roots, and audit logging are stable.

## Example Blind-Agent Flow

1. Read `openhcs://about`.
2. Call `openhcs_health_check()`.
3. Read `openhcs://getting-started`.
4. Call `openhcs_list_benchmark_datasets(include_non_runnable=False)`.
5. Call `openhcs_plan_dataset_acquisition(dataset_id)`.
6. Ask the user before download if cache is missing.
7. Call `openhcs_acquire_dataset(dataset_id, allow_network=True)`.
8. Call `openhcs_build_benchmark_manifest(...)`.
9. Call `openhcs_plan_benchmark_run(...)`.
10. Call `openhcs_run_benchmark(..., async=True)`.
11. Poll `openhcs_get_job_status(job_id)`.
12. Call `openhcs_summarize_benchmark_results(output_dir)`.
13. Read `openhcs://benchmarks/results/{run_id}/artifacts`.

That workflow should be possible without opening source files or knowing
CellProfiler/OpenHCS internals.

## First Implementation Slice

The first useful PR should be intentionally small:

1. Add `openhcs.agent.path_policy`.
2. Add `openhcs.agent.services.dataset_service`.
3. Add `openhcs.agent.services.pipeline_service` for `.cppipe` parsing only.
4. Add `openhcs.agent.services.result_service` for existing benchmark result
   summary parsing.
5. Add `openhcs.mcp.server` with read-only resources/tools.
6. Add unit tests for the above.
7. Add `docs/mcp_server.md` with a read-only demo.

This gives collaborators immediate value for repository orientation and result
review, while leaving downloads and execution behind explicit follow-up stages.
