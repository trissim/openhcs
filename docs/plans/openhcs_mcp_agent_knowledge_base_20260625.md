# OpenHCS MCP Agent Knowledge Base

Status: source-backed draft integrated with the MCP/agent knowledge surface.
Created: 2026-06-25.
Scope: orient an agent to OpenHCS before it works on the MCP server, UI bridge,
viewer validation, or agent-facing API.

This is a knowledge base, not a task handoff. Runtime status and live process
state belong in `docs/plans/mcp_ui_current_handoff.md`.

## How To Use This Document

Start here when you need to understand what the MCP server should teach an
agent about OpenHCS:

1. Read the project overview and architecture themes.
2. Use the example corpus map before assuming a domain workflow needs to be
   invented from scratch.
3. Use the authority map to find the owner of a behavior.
4. Use the MCP surface map to understand the current tool/resource boundary.
5. Use the operating rules before adding or changing MCP tools.

Do not treat MCP as the domain authority. MCP is the transport adapter over the
headless `openhcs.agent` API.

## MCP Knowledge Surface

This document is now part of an allowlisted, read-only documentation surface:

- resource: `openhcs://knowledge`
- tool: `openhcs_list_knowledge_documents`
- tool: `openhcs_get_knowledge_document`
- tool: `openhcs_search_knowledge`

The implementation lives in `openhcs.agent.services.knowledge_base_service` and
returns DTOs from `openhcs.agent.dto.knowledge`. The MCP server only delegates
to that service. It does not accept arbitrary documentation paths from clients.

The stale-source watchlist includes both the KB service/DTO code and existing
allowlisted documentation files, so `openhcs_health_check` can report when a
running MCP server is serving outdated docs.

Fresh-checkout user access is available through `openhcs.mcp.dev_client`:

```bash
python -m openhcs.mcp.dev_client knowledge
python -m openhcs.mcp.dev_client knowledge-document openhcs_agent_mcp_overview --section-id mcp-knowledge-surface
python -m openhcs.mcp.dev_client knowledge-search viewer
```

## Source Evidence

Closed PR descriptions were used to reconstruct the project-level architecture
arc:

| PR | Theme | URL |
| --- | --- | --- |
| #4 | Registry, metadata, lazy config, materialization, compiler/orchestrator split | https://github.com/OpenHCSDev/openhcs/pull/4 |
| #9 | PyQt parameter form system, service extraction, fail-loud UI architecture | https://github.com/OpenHCSDev/openhcs/pull/9 |
| #12 | `AllComponents` vs `VariableComponents`, dynamic component metaprogramming | https://github.com/OpenHCSDev/openhcs/pull/12 |
| #14 | Napari streaming and dual-axis configuration resolution | https://github.com/OpenHCSDev/openhcs/pull/14 |
| #20 | Generic `openhcs.config_framework` with contextvars and MRO resolution | https://github.com/OpenHCSDev/openhcs/pull/20 |
| #23 | OMERO, virtual backends, ZMQ execution framework | https://github.com/OpenHCSDev/openhcs/pull/23 |
| #30 | Virtual workspace backend and zarr input conversion | https://github.com/OpenHCSDev/openhcs/pull/30 |
| #44 | UI anti-duck-typing refactor, ABC contracts, service layer | https://github.com/OpenHCSDev/openhcs/pull/44 |
| #45 | Lazy auto-discovery registry framework via `AutoRegisterMeta` | https://github.com/OpenHCSDev/openhcs/pull/45 |
| #51 | GUI performance, cross-window sync, live context, runtime cleanup | https://github.com/OpenHCSDev/openhcs/pull/51 |
| #58 | ObjectState extraction, DAG time-travel, snapshot/branch model | https://github.com/OpenHCSDev/openhcs/pull/58 |
| #69 | Writer-based materialization with type-safe options | https://github.com/OpenHCSDev/openhcs/pull/69 |

Local source/docs used:

- `openhcs/mcp/server.py`
- `openhcs/mcp/context.py`
- `openhcs/agent/capabilities.py`
- `openhcs/agent/dto/*.py`
- `openhcs/agent/services/*.py`
- `docs/source/development/mcp_development.rst`
- `docs/source/guides/example_corpus_map.rst`
- `docs/source/guides/complete_examples.rst`
- `docs/source/user_guide/production_examples.rst`
- `docs/plans/openhcs_mcp_server_plan_20260616.md`
- `docs/plans/openhcs_mcp_api_exposition_investigation_20260616.md`
- `docs/plans/openhcs_mcp_ui_capability_registry_20260617.md`
- `docs/plans/openhcs_mcp_ui_code_document_bridge_20260617.md`
- NominalRefactorAdvisor style references in
  `/home/ts/code/projects/nominal-refactor-advisor/README.md`,
  `nominal_refactor_advisor/models.py`,
  `nominal_refactor_advisor/semantic_inspection.py`, and
  `nominal_refactor_advisor/patterns.py`.

Current documents in the allowlisted KB should be treated as either current
guidance or historical evidence:

- Current guidance: this overview, `docs/source/development/mcp_development.rst`,
  `docs/source/development/respecting_codebase_architecture.rst`,
  `docs/source/guides/example_corpus_map.rst`,
  `docs/source/guides/complete_examples.rst`, and
  `docs/source/user_guide/production_examples.rst`.
- Historical planning evidence:
  `docs/plans/openhcs_mcp_server_plan_20260616.md`,
  `docs/plans/openhcs_mcp_api_exposition_investigation_20260616.md`,
  `docs/plans/openhcs_mcp_implementation_blueprint_20260616.md`,
  `docs/plans/openhcs_mcp_ui_capability_registry_20260617.md`, and
  `docs/plans/openhcs_mcp_ui_code_document_bridge_20260617.md`.

Historical planning docs are searchable because they preserve design rationale,
but current behavior should be verified against the capability registry and
agent services before implementation.

## Project Overview

OpenHCS is a high-content screening pipeline system. Its current architecture
has evolved around these load-bearing ideas:

- Pipeline behavior is declared as typed, serializable configuration and
  `FunctionStep`-style pipeline declarations.
- Configuration resolution is generic and MRO/context-driven, not ad hoc
  widget state.
- Function discovery, storage backends, microscope handlers, and runtime
  servers use registries or metaclass-backed closed families where possible.
- Storage may be physical, virtual, zarr-backed, OMERO-backed, or memory-backed;
  FileManager/backend abstractions own the I/O boundary.
- Runtime execution can happen through a ZMQ execution server with progress and
  status projection.
- Viewer state is an inspectable runtime surface, not just a screenshot.
- PyQt UI state is owned by ObjectState, windows, and bridge providers; MCP
  projects those surfaces through DTOs and transport.

The recurring architecture theme is not "wrap everything." It is:

```text
semantic authority -> typed service/DTO projection -> adapter transport
```

For MCP work, that becomes:

```text
OpenHCS internals -> openhcs.agent services -> openhcs.agent DTOs -> openhcs.mcp
```

## Technical Operator Map

The knowledge base should let an agent become an OpenHCS technical operator for
a domain expert, not just a chatbot that summarizes docs. Before designing,
debugging, or running a microscopy workflow, an agent should understand these
current architecture contracts:

### Example Corpus First

OpenHCS has practical starting material. The KB should surface the in-tree
CellProfiler examples under `benchmark/cellprofiler_pipelines`, the thirty
scoped native CellProfiler references under
`benchmark/native_refs/official30_scoped_rows`, native benchmark pipelines under
`benchmark/pipelines`, preset pipelines under
`openhcs/processing/presets/pipelines`, and the complete/production example
docs. Use these as semantic anchors for domain tasks such as nuclei
segmentation, illumination correction, colocalization, translocation, quality
control, neurite outgrowth, worm analysis, and wound healing.

The corpus does not replace live plate-layout validation. Example matching tells
an agent what workflow to try; a plate-inspection tool should still report the
actual wells, sites, channels, timepoints, files, microscope handler confidence,
and warnings for a user-supplied folder.

### Pipeline Input Routing

`FunctionStep` input routing is declarative. Normal steps consume the previous
step output. Steps that need original input data, such as position generation
or quality-control comparisons, use `InputSource.PIPELINE_START`. Do not infer
chain-breaking from decorators or ad hoc function names; routing belongs to the
step declaration and compiler/path-planning contract.

### Component And Axis Model

OpenHCS separates the full internal component space from user-selectable step
axes. `AllComponents` names the complete internal dimensions. `VariableComponents`
names the user-facing dimensions a step can vary over, such as site, channel,
Z index, timepoint, and well. `GroupBy` aligns with the variable-component
surface for dictionary function routing, and `MULTIPROCESSING_AXIS` owns the
axis used for execution partitioning. Agents should not hardcode WELL or
parallel component tuples when compiler/orchestrator APIs expose the axis.

### Configuration Resolution Current Model

`GlobalPipelineConfig` carries concrete application or orchestrator defaults.
`PipelineConfig` and step-level lazy configs can use `None` to mean "inherit".
Resolution is dual-axis:

- context axis: step -> pipeline/orchestrator -> global -> static defaults;
- type axis: Python MRO across registered config families.

The current config framework uses `contextvars` and MRO resolution. It is not a
stack-introspection or widget-state system. Direct `object.__getattribute__`
reads the stored field; resolved `getattr`/ObjectState access is what applies
inheritance. Compiler paths should use the saved, resolved values that will
actually execute.

### ObjectState Semantics

ObjectState is the UI/model authority for scoped configuration state. It stores
flat dotted-path parameters, saved and live resolved values, dirty state, scope
identity, snapshots, branches, and DAG time travel. UI bridge tools expose DTO
surfaces, revision tokens, snapshots, branches, and code documents. They should
not expose raw `ObjectState`, raw widgets, or direct in-process mutation hooks.

### Registry And Custom Functions

Function discovery, storage backends, microscope handlers, and related plugin
families are registry-backed. `AutoRegisterMeta` and `__registry_key__` style
contracts are semantic declarations, not optional convenience. Custom
functions become OpenHCS functions by using memory decorators such as `@numpy`,
`@cupy`, or `@torch`; validation, persistence, registry injection, and GUI
availability are part of the custom-function management system.

### Special Outputs And Materialization Writers

Analysis functions can declare special outputs and materialization behavior.
Modern materialization is writer-based and type-driven: `MaterializationSpec`
uses typed output options for formats such as CSV, JSON, ROI ZIP, TIFF, and
text. Agents should prefer typed presets/options over legacy string handlers,
and should remember that path planning and special-I/O dependency checks happen
at compile time.

### Runtime And Execution

OpenHCS compiles before it executes. The compiler initializes step plans,
declares zarr stores, plans materialization, validates memory contracts, and
assigns GPU resources before workers run. Runtime execution may be local or
through ZMQ execution services. For MCP, compile/run jobs and status polling
belong to `ExecutionSessionService` and runtime-server tools, not to raw
orchestrator mutation.

### Viewer Inspection

Napari and Fiji are runtime inspection surfaces. They run out of process,
receive streamed images/payloads over ZMQ-backed viewer protocols, and should be
inspected through semantic state/payload tools before screenshots. Napari layer
identity and Fiji hyperstack behavior depend on component/axis semantics,
including source axes for repeated runs. Agents should use
`openhcs_get_viewer_window_state`, `openhcs_get_viewer_window_payloads`, and
`openhcs_validate_viewer_window_state` before treating a screenshot as evidence.

## Architecture Themes To Preserve

### Fail-Loud Contracts

OpenHCS deliberately removes defensive `hasattr`/`getattr` probes when a
contract guarantees a field or method. MCP should preserve this. If an
agent-facing DTO says a field exists, produce it or return a structured
`AgentError`; do not silently omit it through fallback chains.

### Nominal Boundaries

Closed semantic families should be named. Prefer enums, dataclasses, ABCs, and
`AutoRegisterMeta` families over repeated strings and branch ladders. Examples
already in the current MCP/agent slice:

- `CapabilityKind`, `AgentCapabilitySpec`, `AgentCapabilityRegistry`
- `UiSelectedPlateWorkflowKind`
- `UiCodeDocumentId`, `UiStateSurfaceId`, `UiWidgetId`
- viewer result factory registration in `openhcs.agent.dto.viewer`
- dev-client command specs using `AutoRegisterMeta`

### Stable DTO Reports

The advisor codebase projects results through frozen records with stable fields,
source/evidence identifiers, and explicit serialization. OpenHCS agent DTOs
should follow that shape:

- schema version on public result payloads;
- bounded result pages/summaries;
- typed request/result records;
- explicit errors and warnings;
- source-backed architecture/symbol summaries instead of raw object dumps.

### Authority Over Convenience

Do not create local MCP patches that preserve parallel semantic paths. If a
tool needs a new concept, first decide which OpenHCS authority owns it:

- capability catalog -> `openhcs.agent.capabilities`
- config schema/source rendering -> `ConfigService`
- function discovery -> `FunctionCatalogService`
- pipeline drafts -> `PipelineAuthoringService`
- execution sessions/jobs -> `ExecutionSessionService`
- runtime server status -> `RuntimeServerService`
- UI bridge state/actions/code/snapshots -> `UiBridgeService`
- viewer window state/payloads/validation -> `ViewerWindowService`
- documentation knowledge base -> `KnowledgeBaseService`
- architecture/context help -> `ArchitectureProjectionService` and
  `AgentAuthoringContextService`

## MCP Public Boundary

Current adapter owner:

- `openhcs/mcp/server.py`

Current context owner:

- `openhcs/mcp/context.py`
- `OpenHCSAgentContext`

`OpenHCSAgentContext` composes:

- `AgentPathPolicy`
- `FunctionCatalogService`
- `ConfigService`
- `ArchitectureProjectionService`
- `KnowledgeBaseService`
- `PipelineAuthoringService`
- `AgentAuthoringContextService`
- `ExecutionSessionService`
- `RuntimeServerService`
- `UiBridgeService`
- `ViewerWindowService`

MCP registers tool functions, resolves sparse command arguments, applies
bounded timeout policies, and converts service DTOs with `to_jsonable`. It
should not own compile semantics, UI mutation semantics, viewer layer semantics,
or function-registry semantics.

## Current MCP Resources

Declared in `openhcs/mcp/server.py`:

- `openhcs://capabilities`
  - projects `get_capability_registry()`
  - canonical list of resources/tools, side effects, requirements, and contract
    names.
- `openhcs://architecture/topics`
  - projects `ArchitectureProjectionService.list_topics()`.
- `openhcs://knowledge`
  - projects `KnowledgeBaseService.list_documents()`.
  - lists source-backed, allowlisted documentation available to agents.

If adding resources, first add or extend a service/DTO authority under
`openhcs.agent`. Resources should be bounded and JSON-friendly.

## Agent Capability Registry Snapshot

Generated from `openhcs.agent.capabilities.get_capability_registry()` on
2026-06-25.

Total surface:

- 61 capabilities.
- 3 resources.
- 58 tools.

Capabilities by service:

| Service | Count | Role |
| --- | ---: | --- |
| `architecture_projection` | 4 | source-backed architecture topics and internal-symbol projection |
| `capability_registry` | 3 | health and canonical capability discovery |
| `config` | 4 | config schema, validation, draft refs, source rendering |
| `execution_session` | 7 | orchestrator sessions, compile/run jobs, artifact plans |
| `function_catalog` | 2 | registry-backed function search and detail |
| `knowledge_base` | 4 | source-backed documentation catalog, document reads, and search |
| `llm_context` | 1 | bounded authoring context |
| `pipeline_authoring` | 4 | in-memory pipeline draft construction and rendering |
| `runtime_server` | 3 | ZMQ execution-server scan/status |
| `ui_bridge` | 24 | running UI state, actions, windows, code documents, snapshots |
| `viewer_window` | 5 | running viewer state, payloads, screenshots, validation |

Side-effect-bearing capabilities:

| Capability | Side effects | Runtime requirement |
| --- | --- | --- |
| `openhcs_create_config` | creates in-memory config ref | none |
| `openhcs_create_pipeline` | creates in-memory pipeline ref | none |
| `openhcs_add_function_step` | mutates in-memory pipeline ref | none |
| `openhcs_create_orchestrator_session` | creates in-memory execution session | none |
| `openhcs_create_orchestrator_session_from_pipeline_source` | creates in-memory execution session | none |
| `openhcs_submit_compile` | submits ZMQ compile job | execution server connection |
| `openhcs_submit_pipeline_execution` | submits ZMQ execution job | execution server connection |
| `openhcs_viewer_snapshot_window` | writes agent output file | running OpenHCS viewer server |
| `openhcs_ui_invoke_action` | may mutate running UI state; may start UI workflow | running OpenHCS UI bridge |
| `openhcs_ui_selected_plate_workflow` | may mutate running UI state; may start UI workflow | running OpenHCS UI bridge |
| `openhcs_ui_focus_window` | changes running UI focus | running OpenHCS UI bridge |
| `openhcs_ui_navigate_window` | changes running UI focus; may open UI window | running OpenHCS UI bridge |
| `openhcs_ui_close_window` | closes running UI window | running OpenHCS UI bridge |
| `openhcs_ui_snapshot_window` | writes agent output file | running OpenHCS UI bridge |
| `openhcs_ui_apply_code_document` | mutates running UI state | running OpenHCS UI bridge |
| `openhcs_ui_restore_snapshot` | mutates running UI state; time-travels UI state | running OpenHCS UI bridge |
| `openhcs_ui_time_travel_head` | mutates running UI state; time-travels UI state | running OpenHCS UI bridge |
| `openhcs_ui_switch_branch` | mutates running UI state; time-travels UI state | running OpenHCS UI bridge |

Risk rule: if a tool appears in the side-effect table, an agent should first
read the relevant state/capability surface and preserve the returned receipt,
job id, revision token, operation id, or snapshot id.

## Current MCP Tool Families

### Health And Discovery

- `openhcs_health_check`
- `openhcs_list_capabilities`
- `openhcs_get_authoring_context`
- `openhcs_list_architecture_topics`
- `openhcs_explain_architecture`
- `openhcs_describe_internal_symbol`

Use these to orient agents before mutating anything. `openhcs_health_check`
stays callable when source freshness checks would block normal tools.

### Knowledge Base

- `openhcs_list_knowledge_documents`
- `openhcs_get_knowledge_document`
- `openhcs_search_knowledge`

Authority: `KnowledgeBaseService`.

Purpose: expose an allowlisted documentation catalog, bounded document/section
reads, and source-backed search through stable agent DTOs. Clients pass
document ids and section ids, not filesystem paths.

### Function Catalog

- `openhcs_search_functions`
- `openhcs_describe_function`

Authority: `FunctionCatalogService`.

Purpose: discover registry-backed processing functions by name/library/docs and
inspect parameters/signatures without dumping the full registry into context.

### Config Drafts

- `openhcs_describe_config_schema`
- `openhcs_create_config`
- `openhcs_validate_config_patch`
- `openhcs_render_config_source`

Authority: `ConfigService`.

Purpose: let agents build `GlobalPipelineConfig` or `PipelineConfig` patches
through reflected schemas and render reviewable Python source.

### Pipeline Drafts

- `openhcs_create_pipeline`
- `openhcs_add_function_step`
- `openhcs_validate_pipeline`
- `openhcs_render_pipeline_source`

Authority: `PipelineAuthoringService`.

Purpose: create in-memory pycodified pipeline drafts from registry-backed
function references. This is the safe path for agent-authored pipelines.

### Orchestrator Sessions And Execution

- `openhcs_create_orchestrator_session`
- `openhcs_create_orchestrator_session_from_pipeline_source`
- `openhcs_get_orchestrator_session`
- `openhcs_inspect_pipeline_source_artifact_plan`
- `openhcs_submit_compile`
- `openhcs_submit_pipeline_execution`
- `openhcs_get_execution_status`

Authority: `ExecutionSessionService`.

Purpose: bind plate path, pipeline/config refs or source, execution connection,
and runtime job refs. Compile/run are represented through session/job handles,
not raw `PipelineOrchestrator` objects.

### Runtime Server Status

- `openhcs_scan_runtime_servers`
- `openhcs_get_runtime_server_info`
- `openhcs_get_runtime_server_execution_status`

Authority: `RuntimeServerService`.

Purpose: inspect ZMQ execution servers and execution status without scraping UI
rows.

### Viewer Windows

- `openhcs_viewer_snapshot_window`
- `openhcs_get_viewer_window_state`
- `openhcs_get_viewer_window_payloads`
- `openhcs_probe_viewer_window`
- `openhcs_validate_viewer_window_state`

Authority: `ViewerWindowService`.

Purpose: inspect running viewer windows through layer, component, axis, image,
and shape payload records. Screenshots are useful but not sufficient; validation
should inspect axes/payloads/slices.

### Running UI Bridge

- `openhcs_ui_list_bridges`
- `openhcs_ui_bridge_status`
- `openhcs_ui_list_code_documents`
- `openhcs_ui_list_state_surfaces`
- `openhcs_ui_list_actions`
- `openhcs_ui_invoke_action`
- `openhcs_ui_selected_plate_workflow`
- `openhcs_ui_list_windows`
- `openhcs_ui_focus_window`
- `openhcs_ui_navigate_window`
- `openhcs_ui_close_window`
- `openhcs_ui_snapshot_window`
- `openhcs_ui_get_widget_tree`
- `openhcs_ui_list_object_state_scopes`
- `openhcs_ui_get_state_surface`
- `openhcs_ui_get_code_document`
- `openhcs_ui_validate_code_document`
- `openhcs_ui_apply_code_document`
- `openhcs_ui_list_snapshots`
- `openhcs_ui_restore_snapshot`
- `openhcs_ui_time_travel_head`
- `openhcs_ui_list_branches`
- `openhcs_ui_switch_branch`
- `openhcs_ui_get_operation_status`

Authority: `UiBridgeService`, with the PyQt side providing state/action/window
surfaces. Do not expose raw `QWidget`, raw `ObjectState`, or raw
`WindowManager` objects.

## UI Bridge Mental Model

The UI bridge is a cross-process control surface for a running PyQt UI. The
bridge descriptor under `/run/user/1000/openhcs/ui-bridge` identifies the
running UI bridge instance and its auth/transport details.

Important distinctions:

- UI bridge: the running PyQt-side control server.
- Fresh-process MCP dev client: launches a new current-source stdio MCP server
  for a single command.
- Codex embedded MCP connection: a Codex-owned stdio child that can become
  stale after code edits.

For live status and stale-process debugging, consult
`docs/plans/mcp_ui_current_handoff.md`.

## Code Documents And ObjectState

The first important UI-owned code document is:

- `plate_manager.orchestrator_config`

Agent-safe edit flow:

1. list/read code document;
2. validate source;
3. apply source through the UI bridge;
4. receive revision/snapshot/receipt/operation metadata;
5. poll relevant state surfaces or operation status;
6. use ObjectState snapshots/branches for undo or restore.

Do not add a second mutation path for PlateManager. The bridge must call the
existing PlateManager code workflow and ObjectState snapshot authority.

## Viewer Inspection Rules

Viewer bugs are often semantic-layer bugs, not screenshot bugs. Prefer:

1. `openhcs_probe_viewer_window`
2. `openhcs_get_viewer_window_state`
3. `openhcs_get_viewer_window_payloads`
4. `openhcs_validate_viewer_window_state`
5. screenshot only as final human evidence

Preserve the viewer architecture conclusions from prior work:

- `component_axis_semantics` owns viewer layout, role policy, and value-domain
  ownership.
- `AllComponents` is the durable source for component ordering/defaults.
- `source_mode=STACK` changes topology; it is not a title-only fix.

## Agent Workflow Examples

Use these as first moves for common MCP sessions. They name the public tools,
not private implementation shortcuts.

### Learn The Current Surface

1. Call `openhcs_health_check`.
2. If `server_source_changed_since_import` is true, restart the MCP process
   before trusting non-health tools.
3. Call `openhcs_list_capabilities`.
4. Call `openhcs_list_knowledge_documents`.
5. Use `openhcs_search_knowledge` for domain terms before opening broad docs.
6. Use `openhcs_get_knowledge_document` with `section_id` when the catalog
   already points at the relevant section.

### Discover Functions And Author A Pipeline

1. Call `openhcs_get_authoring_context`.
2. Search functions with `openhcs_search_functions`.
3. Inspect selected functions with `openhcs_describe_function`.
4. Create a draft with `openhcs_create_pipeline`.
5. Add steps with `openhcs_add_function_step`.
6. Validate with `openhcs_validate_pipeline`.
7. Render reviewable source with `openhcs_render_pipeline_source`.
8. For source-mode workflows, inspect artifact behavior with
   `openhcs_inspect_pipeline_source_artifact_plan` before submitting execution.

### Read And Apply UI Code Documents

1. Check `openhcs_ui_bridge_status`.
2. List documents with `openhcs_ui_list_code_documents`.
3. Read a document with `openhcs_ui_get_code_document`.
4. Validate proposed edits with `openhcs_ui_validate_code_document`.
5. Apply with `openhcs_ui_apply_code_document` using the observed revision
   token and a request token.
6. Preserve the returned snapshot/undo targets.
7. Poll `openhcs_ui_get_state_surface` or `openhcs_ui_get_operation_status`
   until the affected UI state reaches the expected status.

### Inspect Viewer State

1. Probe liveness with `openhcs_probe_viewer_window`.
2. Read semantic state with `openhcs_get_viewer_window_state`.
3. Pull bounded payload records with `openhcs_get_viewer_window_payloads`.
4. Use `openhcs_validate_viewer_window_state` for expected layer counts, axis
   labels, and nonzero payload requirements.
5. Capture a screenshot only after semantic checks point to a visual issue or a
   human-readable artifact is needed.

### Query Runtime Execution

1. Find a server with `openhcs_scan_runtime_servers`.
2. Inspect one server with `openhcs_get_runtime_server_info`.
3. Create or retrieve an orchestrator session.
4. Submit compile or execution through `openhcs_submit_compile` or
   `openhcs_submit_pipeline_execution`.
5. Poll job status with `openhcs_get_execution_status`.
6. Use `openhcs_get_runtime_server_execution_status` for server-level status
   when the execution id is known or discoverable.

## Development Workflow

Follow `docs/source/development/mcp_development.rst`.

Recommended local loop:

```bash
. .venv/bin/activate
python -m openhcs.mcp.dev_client health
python -m openhcs.mcp.dev_client tools
python -m openhcs.mcp.dev_client ui-status --timeout-ms 1000
```

When editing `openhcs/mcp` or `openhcs/agent`, use a fresh-process dev client
for validation. Do not trust a Codex embedded MCP child to hot-reload changed
tool functions.

## Advisor-Style Documentation Rules For MCP KB Work

Borrow these habits from NominalRefactorAdvisor:

- use stable record names and explicit fields;
- list the authority for each behavior;
- include evidence paths or PR links;
- distinguish candidate observations from confirmed facts;
- define the first move for a reader;
- avoid generic prose when a concrete symbol map exists.

Good MCP documentation shape:

```text
Problem -> authority -> public DTO/tool shape -> side effects -> validation
```

Avoid:

- unbounded lists of raw internals;
- "this probably calls X" without source evidence;
- transport-specific wrappers that just rename a service method;
- compatibility shims unless the caller must support legacy data.

## Maintenance Rules

- Treat this file as current guidance only when its capability snapshot matches
  `openhcs.agent.capabilities.get_capability_registry()`.
- Keep the Sphinx-facing page at
  `docs/source/development/mcp_knowledge_base.rst` aligned with the dev-client
  commands here.
- If changing production MCP/agent files, finish with focused tests and an
  advisor run on production files touched.
