# OpenHCS MCP Implementation Blueprint

## Status

Drafted 2026-06-16 after tightening the MCP plan around core OpenHCS pipeline
authoring, config, and orchestrator APIs.

This is still a plan, but it is intended to be concrete enough to implement
from.

Related plans:

- `docs/plans/openhcs_mcp_server_plan_20260616.md`
- `docs/plans/openhcs_mcp_api_exposition_investigation_20260616.md`

## Design Correction

The MCP server should expose OpenHCS core semantics:

- `FunctionStep` pipeline authoring;
- function registry discovery and signatures;
- `GlobalPipelineConfig` and `PipelineConfig` schema/patch/serialization;
- orchestrator lifecycle operations;
- compile and execute operations;
- source-code rendering for review.

But it should not expose these as raw live Python objects or arbitrary method
calls.

The proper public surface is:

```text
OpenHCS object/model            Agent-facing MCP shape
----------------------------    ---------------------------------------------
FunctionStep                    FunctionStepSpec + pipeline authoring tools
GlobalPipelineConfig            ConfigSchema + ConfigPatch + ConfigRef
PipelineConfig                  ConfigSchema + ConfigPatch + ConfigRef
PipelineOrchestrator            OrchestratorSessionRef + lifecycle tools
Function registry callables     FunctionCatalogEntry + FunctionSpecRef
pycodify source                 rendered artifact/resource, not canonical state
ZMQ execution payload           compile/execute tools with typed submissions
```

This keeps the MCP API useful for agents while preserving OpenHCS authorities.

## Package Layout

Add a reusable headless API layer:

```text
openhcs/agent/
  __init__.py
  contracts.py
  capabilities.py
  path_policy.py
  ids.py
  job_store.py
  dto/
    __init__.py
    common.py
    config.py
    functions.py
    pipeline.py
    orchestrator.py
    benchmark.py
    results.py
  services/
    __init__.py
    config_service.py
    function_catalog_service.py
    pipeline_authoring_service.py
    pipeline_serialization_service.py
    orchestrator_service.py
    llm_context_service.py
    dataset_service.py
    benchmark_service.py
    result_service.py

openhcs/mcp/
  __init__.py
  server.py
  context.py
  resources.py
  tools.py
  prompts.py
  serializers.py
```

Rules:

- `openhcs.agent` owns policy, DTOs, validation, and service logic.
- `openhcs.mcp` owns MCP registration/transport only.
- Existing CLI/PyQt code can later import `openhcs.agent` services, but MCP
  should not import PyQt.

## Reuse Existing LLM Support

Relevant existing file:

- `openhcs/pyqt_gui/services/llm_pipeline_service.py`

Reusable pieces:

- `LLMParameterDocumentationPolicy`
- `LLMParameterFormatter`
- `LLMFunctionDocumentationBuilder`
- `LLMPromptResourceCatalog`
- `LLMPromptBuilder`

Do not reuse as-is:

- `LLMPipelineService.generate_code(...)`
- endpoint/model/Ollama HTTP logic;
- GUI-facing assumptions;
- docs that recommend `exec(pipeline_code)` as the normal workflow.

Refactor target:

```text
openhcs/agent/services/llm_context_service.py
```

Move or wrap the reusable prompt/catalog classes under neutral names:

```python
AgentParameterDocumentationPolicy
AgentParameterFormatter
AgentFunctionDocumentationBuilder
AgentPromptResourceCatalog
AgentPromptBuilder
```

Then update the PyQt LLM service to depend on `AgentPromptBuilder` instead of
owning a duplicate function/context catalog.

MCP use:

- expose prompt resources and authoring context;
- do not call LLM providers by default;
- do not make model-generated source executable input in v1.

Potential MCP resources:

```text
openhcs://authoring/prompt-context/pipeline
openhcs://authoring/prompt-context/custom-function
openhcs://authoring/examples/pipeline-basic
openhcs://authoring/enums
```

Potential MCP tools:

```text
openhcs_get_authoring_context(kind="pipeline" | "custom_function")
```

## Reuse Existing Serialization And Transport

Relevant files:

- `openhcs/serialization/pycodify_formatters.py`
- `openhcs/core/function_step_transport.py`
- `openhcs/runtime/zmq_pipeline_transport.py`
- `openhcs/runtime/zmq_execution_client.py`
- `openhcs/runtime/zmq_execution_signature.py`
- `openhcs/runtime/zmq_compilation.py`

Reuse rules:

- Use `FunctionStepTransportAuthority.normalize_pipeline(...)` before compile,
  execution, and source rendering.
- Use `pycodify`/`OpenHCSCallableFormatter`/`FunctionStepFormatter` for source
  rendering.
- Use `LazyDataclassFormatter` for config source rendering.
- Use `ZMQExecutionClient` and `OpenHCSExecutionSubmission` for runtime compile
  and execute.
- Treat `ZMQPipelineCodeTransport` as compatibility with existing ZMQ server
  behavior, not as the public authoring model.

Do not add another serializer for `FunctionStep`.

## Common DTOs

Create in `openhcs/agent/dto/common.py`:

```python
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
class AgentResourceRef:
    uri: str
    title: str
    mime_type: str = "application/json"
    path: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
```

Every public response should include:

```python
schema_version: str = "openhcs.agent.v1"
```

## Capability Registry

Create in `openhcs/agent/capabilities.py`:

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

The registry must drive:

- `openhcs://capabilities`;
- MCP registration tests;
- documentation generation;
- side-effect policy tests.

Tests should fail if a mutating tool is registered without side-effect metadata.

## Function Catalog API

Authority:

- `openhcs.processing.backends.lib_registry.registry_service.RegistryService`
- existing LLM function documentation builder logic
- `openhcs.processing.func_registry`

DTOs in `openhcs/agent/dto/functions.py`:

```python
@dataclass(frozen=True, slots=True)
class FunctionCatalogEntry:
    function_id: str
    name: str
    module: str
    library: str
    import_path: str
    signature: str
    summary: str | None
    backend_tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FunctionParameterSpec:
    name: str
    annotation: str | None
    default_repr: str | None
    required: bool
    description: str | None = None


@dataclass(frozen=True, slots=True)
class FunctionDetail:
    entry: FunctionCatalogEntry
    parameters: tuple[FunctionParameterSpec, ...]
    doc: str | None
```

Service methods:

```python
FunctionCatalogService.search(query=None, library=None, limit=50) -> FunctionCatalogPage
FunctionCatalogService.get(function_id: str) -> FunctionDetail
FunctionCatalogService.resolve(function_id: str) -> Callable
```

MCP resources:

```text
openhcs://functions/catalog
openhcs://functions/{function_id}
```

MCP tools:

```text
openhcs_search_functions
openhcs_describe_function
```

## Config API

Authorities:

- `openhcs.core.config.GlobalPipelineConfig`
- generated `PipelineConfig`
- ObjectState lazy dataclass machinery:
  - `LazyDataclassFactory`
  - `get_base_type_for_lazy`
  - `resolve_field_inheritance`
  - `DataclassFieldAccess`
  - `ObjectState` flat parameter extraction where appropriate
- `openhcs.serialization.pycodify_formatters.LazyDataclassFormatter`

DTOs in `openhcs/agent/dto/config.py`:

```python
@dataclass(frozen=True, slots=True)
class ConfigRef:
    config_id: str
    config_type: str
    uri: str


@dataclass(frozen=True, slots=True)
class ConfigFieldSchema:
    path: str
    type_repr: str
    default_repr: str | None
    required: bool
    description: str | None
    enum_values: tuple[str, ...] = ()
    ui_hidden: bool = False
    lazy: bool = False


@dataclass(frozen=True, slots=True)
class ConfigSchema:
    schema_version: str
    config_type: str
    fields: tuple[ConfigFieldSchema, ...]


@dataclass(frozen=True, slots=True)
class ConfigPatch:
    config_type: str
    values: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ConfigValidationResult:
    valid: bool
    errors: tuple[AgentError, ...] = ()
    warnings: tuple[AgentWarning, ...] = ()
    config_ref: ConfigRef | None = None
```

Service methods:

```python
ConfigService.describe_schema(config_type: str) -> ConfigSchema
ConfigService.create(config_type: str, patch: ConfigPatch | None = None) -> ConfigRef
ConfigService.validate_patch(config_type: str, patch: ConfigPatch) -> ConfigValidationResult
ConfigService.resolve(global_config: ConfigRef, pipeline_config: ConfigRef | None, field_path: str) -> object
ConfigService.render_source(config_ref: ConfigRef, clean: bool = True) -> AgentResourceRef
ConfigService.diff(left: ConfigRef, right: ConfigRef) -> ConfigDiff
```

MCP resources:

```text
openhcs://schemas/config/global
openhcs://schemas/config/pipeline
openhcs://configs/{config_id}
openhcs://configs/{config_id}/source
```

MCP tools:

```text
openhcs_describe_config_schema
openhcs_create_config
openhcs_validate_config_patch
openhcs_resolve_config_value
openhcs_render_config_source
openhcs_diff_configs
```

Implementation note:

- Do not manually mirror every config field.
- Reflect schema from dataclasses/ObjectState.
- Preserve lazy semantics: unset lazy fields should stay unset, not be
  materialized as concrete defaults.

## Pipeline Authoring API

Authorities:

- `openhcs.core.steps.function_step.FunctionStep`
- `openhcs.core.steps.abstract.AbstractStep`
- `FunctionStepTransportAuthority`
- function catalog service
- pycodify formatters

DTOs in `openhcs/agent/dto/pipeline.py`:

```python
@dataclass(frozen=True, slots=True)
class FunctionSpecRef:
    function_id: str
    kwargs: Mapping[str, object] = field(default_factory=dict)
    runtime_options: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FunctionStepSpec:
    step_id: str
    name: str
    function: FunctionSpecRef | tuple[FunctionSpecRef, ...]
    description: str | None = None
    enabled: bool = True
    debug_pause: bool = False
    dtype_config: ConfigPatch | None = None
    processing_config: ConfigPatch | None = None
    source_bindings: Mapping[str, object] | None = None
    step_well_filter_config: ConfigPatch | None = None
    step_materialization_config: ConfigPatch | None = None
    napari_streaming_config: ConfigPatch | None = None
    fiji_streaming_config: ConfigPatch | None = None


@dataclass(frozen=True, slots=True)
class PipelineSpec:
    pipeline_id: str
    steps: tuple[FunctionStepSpec, ...]
    global_config: ConfigRef | None = None
    pipeline_config: ConfigRef | None = None
```

Service methods:

```python
PipelineAuthoringService.create_pipeline(steps=()) -> PipelineRef
PipelineAuthoringService.add_step(pipeline_ref, step_spec, index=None) -> PipelineRef
PipelineAuthoringService.update_step(pipeline_ref, step_id, patch) -> PipelineRef
PipelineAuthoringService.remove_step(pipeline_ref, step_id) -> PipelineRef
PipelineAuthoringService.reorder_steps(pipeline_ref, ordered_step_ids) -> PipelineRef
PipelineAuthoringService.validate(pipeline_ref) -> PipelineValidationResult
PipelineAuthoringService.to_function_steps(pipeline_ref) -> list[FunctionStep]
PipelineAuthoringService.from_function_steps(steps) -> PipelineRef
```

MCP resources:

```text
openhcs://schemas/pipeline/function-step
openhcs://pipelines/{pipeline_id}
openhcs://pipelines/{pipeline_id}/summary
openhcs://pipelines/{pipeline_id}/source
```

MCP tools:

```text
openhcs_create_pipeline
openhcs_add_function_step
openhcs_update_function_step
openhcs_remove_function_step
openhcs_reorder_pipeline_steps
openhcs_validate_pipeline
openhcs_render_pipeline_source
openhcs_save_pipeline_source
```

Important behavior:

- Function references are by `function_id`, not arbitrary Python import text.
- The service resolves `function_id` through `FunctionCatalogService`.
- Before compile/execute/source render, call
  `FunctionStepTransportAuthority.normalize_pipeline(...)`.
- `openhcs_render_pipeline_source` uses pycodify and returns an artifact/ref.
- `openhcs_save_pipeline_source` writes only under an allowed output root.
- Loading arbitrary user Python source into live objects is not v1. If added,
  it must be an explicit dangerous tool with path policy and clear side effects.

## Orchestrator API

Authority:

- `PipelineOrchestrator`
- `InputWorkspacePreparationRequest`
- `prepare_cellprofiler_input_workspace(...)`
- `ZMQExecutionClient`
- `OpenHCSExecutionSubmission`
- `ExecutionRuntimeProjection`

Expose orchestrator lifecycle through handles, not object methods.

DTOs in `openhcs/agent/dto/orchestrator.py`:

```python
@dataclass(frozen=True, slots=True)
class OrchestratorSessionRef:
    orchestrator_id: str
    plate_path: str
    execution_plate_path: str | None
    selected_pipeline_path: str | None
    state: str
    uri: str


@dataclass(frozen=True, slots=True)
class PlateProbeSummary:
    orchestrator_id: str | None
    microscope_type: str | None
    axes: tuple[str, ...]
    channels: tuple[str, ...]
    sites: tuple[str, ...]
    z_indexes: tuple[str, ...]
    timepoints: tuple[str, ...]
    warnings: tuple[AgentWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class CompileSubmission:
    execution_id: str
    status: str
    compile_artifact_id: str | None = None
```

Service methods:

```python
OrchestratorService.create_session(plate_path, selected_pipeline_path=None, global_config=None, pipeline_config=None) -> OrchestratorSessionRef
OrchestratorService.initialize(orchestrator_id) -> OrchestratorSessionRef
OrchestratorService.probe(orchestrator_id) -> PlateProbeSummary
OrchestratorService.compile(orchestrator_id, pipeline_ref, axis_filter=(), retain_artifact=True) -> CompileSubmission
OrchestratorService.execute(orchestrator_id, pipeline_ref, compile_artifact_id=None) -> ExecutionSubmission
OrchestratorService.status(execution_id=None) -> ExecutionStatusSummary
OrchestratorService.progress(execution_id=None) -> ExecutionProgressSummary
OrchestratorService.dispose(orchestrator_id) -> AgentOperationReceipt
```

MCP resources:

```text
openhcs://orchestrators/{orchestrator_id}
openhcs://orchestrators/{orchestrator_id}/probe
openhcs://runtime/executions/{execution_id}/status
openhcs://runtime/executions/{execution_id}/progress
```

MCP tools:

```text
openhcs_create_orchestrator_session
openhcs_initialize_orchestrator
openhcs_probe_orchestrator
openhcs_compile_pipeline
openhcs_submit_execution
openhcs_get_execution_status
openhcs_get_execution_progress
openhcs_dispose_orchestrator
```

Implementation notes:

- Sessions are local server handles with TTL and explicit disposal.
- Session handles are not durable public IDs.
- Direct `PipelineOrchestrator` creation is allowed only inside
  `OrchestratorService`.
- ZMQ compile/execute should be preferred where possible because it already
  matches the app runtime path.
- Direct local compile is acceptable for `probe` and tests if it uses the same
  orchestrator methods and config context setup.

## Dataset, Benchmark, Result APIs

Keep the APIs from the main MCP plan, but wire them into the common capability,
path, job, and resource systems.

Services:

```python
DatasetService.list_catalog(...)
DatasetService.plan_acquisition(...)
DatasetService.acquire(...)
BenchmarkService.build_manifest(...)
BenchmarkService.plan_run(...)
BenchmarkService.run(...)
ResultService.summarize_benchmark_results(...)
```

Benchmark execution must be a job. Do not block a tool call for a large suite.

## Agent Store

MCP needs a process-local store for handles:

```python
class AgentObjectStore:
    configs: dict[str, object]
    pipelines: dict[str, PipelineSpec]
    orchestrators: dict[str, PipelineOrchestrator]
    manifests: dict[str, Path]
    results: dict[str, Path]
```

Requirements:

- deterministic IDs from content hashes where possible;
- opaque random IDs for live orchestrator sessions;
- TTL cleanup for live sessions;
- no persisted secrets;
- no raw model-provided Python source stored as executable state.

## Path Policy

Create `openhcs/agent/path_policy.py`.

API:

```python
AgentPathPolicy.from_environment(...)
AgentPathPolicy.assert_readable(path) -> Path
AgentPathPolicy.assert_writable(path) -> Path
AgentPathPolicy.describe() -> PathPolicySummary
```

Rules:

- resolve before checking;
- reject symlink escapes;
- default read roots: repo root, `/tmp`, configured cache/output roots;
- default write roots: `/tmp`, configured cache/output roots;
- no destructive delete tool in v1.

## MCP Registration

Use the official Python MCP SDK in `openhcs/mcp/server.py`.

Startup:

```python
def build_server(config: OpenHCSMCPServerConfig) -> FastMCP:
    context = OpenHCSMCPContext.from_config(config)
    register_resources(mcp, context)
    register_tools(mcp, context)
    register_prompts(mcp, context)
    return mcp
```

The MCP adapter functions should be thin:

```python
@mcp.tool()
def openhcs_describe_config_schema(config_type: str) -> ConfigSchema:
    return ctx.config_service.describe_schema(config_type)
```

If a tool contains workflow logic beyond argument validation and service
delegation, move that logic into `openhcs.agent`.

## Implementation Phases

### Phase 0: Scaffold And Boundaries

Files:

- `openhcs/agent/contracts.py`
- `openhcs/agent/capabilities.py`
- `openhcs/agent/path_policy.py`
- `openhcs/agent/dto/common.py`
- `openhcs/mcp/server.py`

Tests:

- capability registry loads;
- path policy allows repo/tmp and rejects outside roots;
- MCP server imports without PyQt, Napari, Fiji, or CellProfiler extras.

### Phase 1: Function Catalog And LLM Context Extraction

Move reusable prompt/catalog code out of PyQt:

- create `FunctionCatalogService`;
- create `LLMContextService`;
- make PyQt `LLMPipelineService` consume the shared builder.

Tests:

- function search returns real registry entries;
- internal params are filtered;
- prompt context includes `FunctionStep` guidance and registered function docs;
- PyQt LLM tests still pass with updated imports.

### Phase 2: Config Service

Implement schema/patch/render for:

- `GlobalPipelineConfig`;
- `PipelineConfig`;
- lazy config dataclasses used by `FunctionStep`.

Tests:

- schema includes nested/lazy fields;
- unset lazy fields stay unset;
- explicit patch values survive source render/reload where existing pycodify
  supports it;
- invalid enum/path/type patches return structured errors.

### Phase 3: Pipeline Authoring Service

Implement structured `PipelineSpec` and conversion to/from `FunctionStep`.

Tests:

- create pipeline with one function step;
- function ID resolves through catalog;
- kwargs validate against function signature;
- pipeline validates through current compiler where possible;
- rendered source compiles;
- `FunctionStepTransportAuthority` is invoked before render/compile.

### Phase 4: MCP Read-Only Adapter

Expose resources/tools for:

- capabilities;
- function catalog/detail;
- config schemas;
- pipeline source rendering;
- authoring context;
- benchmark result summary.

Tests:

- every registered tool has `AgentCapabilitySpec`;
- resources have no side effects;
- schemas do not expose raw internal class names as callable APIs;
- server can run in stdio mode.

### Phase 5: Orchestrator Sessions And Compile

Implement:

- create/init/probe/dispose session;
- compile via ZMQ if available;
- direct compile only as a controlled service path for tests or no-ZMQ mode.

Tests:

- session lifecycle with temporary fixture plate;
- path-denied before orchestrator construction;
- compile failure maps to `AgentError`;
- status/progress projections are stable.

### Phase 6: Controlled Execution And Benchmarks

Add:

- execution submit/status/progress;
- benchmark manifest/run jobs;
- dataset acquisition with explicit network permission.

Tests:

- mocked jobs;
- small local fixture benchmark;
- no network by default;
- output roots enforced.

## Initial Tool Set

First shippable read-only/authoring tools:

```text
openhcs_health_check
openhcs_list_capabilities
openhcs_search_functions
openhcs_describe_function
openhcs_get_authoring_context
openhcs_describe_config_schema
openhcs_create_config
openhcs_validate_config_patch
openhcs_create_pipeline
openhcs_add_function_step
openhcs_validate_pipeline
openhcs_render_pipeline_source
openhcs_summarize_benchmark_results
```

Second wave:

```text
openhcs_create_orchestrator_session
openhcs_initialize_orchestrator
openhcs_probe_orchestrator
openhcs_compile_pipeline
openhcs_get_execution_status
openhcs_get_execution_progress
openhcs_dispose_orchestrator
```

Third wave:

```text
openhcs_plan_dataset_acquisition
openhcs_acquire_dataset
openhcs_build_benchmark_manifest
openhcs_plan_benchmark_run
openhcs_run_benchmark
openhcs_get_job_status
openhcs_cancel_job
```

## Advisor Rules To Add Later

After the first implementation lands, add advisor checks:

- MCP tools must have `AgentCapabilitySpec`.
- Mutating MCP tools must declare side effects.
- Network-capable MCP tools must declare `requires_network`.
- MCP DTOs must not expose `PipelineOrchestrator`, `FileManager`,
  `ProcessingContext`, or raw `StorageRegistry` fields.
- MCP tool functions should delegate to `openhcs.agent.services`.
- Pipeline render/compile paths must pass through
  `FunctionStepTransportAuthority`.

## Answer To The API Exposure Question

Yes: the MCP feature should expose the `PipelineOrchestrator`,
`GlobalPipelineConfig`, `PipelineConfig`, and `FunctionStep` APIs in the sense
that agents can inspect, construct, validate, render, compile, and execute those
things.

No: it should not expose live Python objects, arbitrary method calls, or raw
Python execution as the public contract.

The implementation should expose them through stable DTOs, handles, schemas,
and controlled service methods backed by the existing OpenHCS authorities.
