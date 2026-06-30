# MCP Tool Surface Grouping Plan

Date: 2026-06-29

Status: Implemented. Remaining work is optional workflow-guide expansion and
copy refinement.

## Implementation Progress

Implemented 2026-06-29:

- Added a nominal `AgentCapabilityExposition` object as the single grouping
  authority for each capability family.
- Added nominal exposition enums for workflow group, workflow stage, target
  context, visibility, and role.
- Added capability-family inheritance such as `PlatePathCapability`,
  `UiSelectedPlateCapability`, `UiWindowCapability`,
  `UiWidgetFallbackCapability`, `UiCodeDocumentCapability`,
  `UiObjectStateCapability`, `HeadlessExecutionCapability`, and
  `ViewerWindowCliConnectionCapability`.
- Updated leaf capability classes to inherit the appropriate family instead of
  participating in a separate grouping table.
- Added generated `AgentCapabilityRegistry.groups`; groups are derived from
  registered capability specs and their inherited `AgentCapabilityExposition`.
- Updated the dev-client `tools` command to render grouped output by default and
  preserve `--flat` for machine-oriented inspection.
- Replaced hardcoded prompt function inventories with registry-backed
  projections through `FunctionCatalogService`; PyQt and MCP authoring context
  now get function signatures from the same function-catalog authority.
- Moved shared LLM prompt resources into `openhcs.agent.services` so PyQt prompt
  generation reuses the agent function-catalog/context infrastructure instead
  of owning a parallel function-info surface.
- Added tests that verify all tools declare exposition, similar tool names are
  disambiguated by target context/role, registry groups are generated from
  declarations, and grouped/flat dev-client rendering both work.

Canonical rule:

- Do not maintain a production dict, list, or switch mapping tool names to
  groups.
- A capability is grouped by its nominal declaration class and inherited
  `AgentCapabilityExposition`.
- JSON fields and rendered headings are projections from those declarations.
- Do not maintain handpicked processing-function lists in prompt builders.
  Function examples may be illustrative, but available function names,
  signatures, imports, and summaries must come from the function catalog.
- Do not maintain handpicked "important enum" lists in prompt builders. Enum
  usage should be visible through registry signatures, config schemas, and
  authoring-context rules that point back to those authorities.

## Goal

Organize the current flat MCP tool surface into agent-facing groups so similar
tool names are understandable, true redundancies are visible, and intentional
mode variants are not mistaken for duplication.

This document is not a new semantic authority. The source of truth must remain
the `AgentCapabilityDeclaration` registry built by `AutoRegisterMeta` in
`openhcs/agent/capabilities.py`. Any durable grouping should be declared on
capability classes and projected into docs, CLI help, and MCP resources from the
registry.

## Inventory Snapshot

Current registry snapshot from `get_capability_registry()`:

- 79 total capabilities.
- 76 tools.
- 3 resources:
  - `openhcs://capabilities`
  - `openhcs://knowledge`
  - `openhcs://architecture/topics`

The current declarations already expose useful metadata:

- `service`
- `cli_command`
- `cli_aliases`
- `mutating`
- `side_effects`
- `runtime_requirements`
- `data_exposure`
- `input_contract`
- `output_contract`

Implemented for first-use ergonomics:

- workflow group
- workflow stage
- primary/fallback role
- target context, such as path plate, UI-selected plate, UI window, viewer
  window, runtime server, or submitted job
- beginner/expert visibility

Still optional:

- recommended predecessor and successor tools, if encoded as nominal workflow
  declarations rather than a hand-maintained sequence table

## Current Service Buckets

These are the current registry `service` values. They are useful implementation
clusters, but they are not enough as the only agent-facing organization.

| Service | Tools | Notes |
| --- | ---: | --- |
| `architecture_projection` | 3 | Coherent. |
| `capability_registry` | 2 | Coherent. |
| `config` | 4 | Coherent but should be shown next to pipeline authoring. |
| `execution_session` | 7 | Coherent, but status/session/artifact-plan tools need workflow ordering. |
| `function_catalog` | 3 | Coherent. |
| `knowledge_base` | 3 | Coherent. |
| `llm_context` | 1 | Should become a front door for workflow guides. |
| `pipeline_authoring` | 4 | Coherent but should be shown next to config tools. |
| `plate_inspection` | 3 | Mixed with selected-plate UI variant. Needs target-context metadata. |
| `plate_streaming` | 1 | Should group with path-based plate inventory. |
| `runtime_server` | 3 | Coherent. |
| `selected_plate` | 4 | Mixed with path-based `inspect_plate_path`. Needs target-context metadata. |
| `synthetic_plate_generation` | 1 | Belongs in plate setup/data preparation. |
| `ui_bridge` | 28 | Too broad. Needs subgroups. |
| `viewer_window` | 9 | Coherent, but similar verbs overlap with UI-window names. |

Current weak spots:

- `ui_bridge` is the largest bucket and hides several distinct domains:
  bridge discovery, UI windows, semantic actions, selected-plate workflow,
  ObjectState, code documents, snapshots, and widget fallback controls.
- `plate_inspection` and `selected_plate` mix target contexts. A first-use
  agent cannot reliably infer whether a tool acts on an explicit path or the
  currently selected UI plate from service alone.
- Several `get_*_status` and `*_snapshot_window` names are not redundant, but
  the target object is not prominent enough in a flat alphabetical list.

## Proposed Agent-Facing Groups

### 1. Discovery And Health

Purpose: establish what server is running, what it exposes, and whether the
agent is looking at current-source tooling.

Tools:

- `openhcs_health_check`
- `openhcs_list_capabilities`

Resources:

- `openhcs://capabilities`

Redundancy judgment:

- Not redundant. `health_check` is diagnostics and freshness. `list_capabilities`
  is the exposed API catalog.

Exposition action:

- Keep both.
- Make this the first group shown in any compact capability index.
- Surface restart/fresh-process guidance through health output when the server
  is stale.

### 2. Knowledge, Architecture, And Authoring Guidance

Purpose: give a contextless agent enough OpenHCS model knowledge before it starts
editing or running workflows.

Tools:

- `openhcs_list_knowledge_documents`
- `openhcs_get_knowledge_document`
- `openhcs_search_knowledge`
- `openhcs_list_architecture_topics`
- `openhcs_explain_architecture`
- `openhcs_describe_internal_symbol`
- `openhcs_get_authoring_context`

Resources:

- `openhcs://knowledge`
- `openhcs://architecture/topics`

Redundancy judgment:

- Not redundant. The knowledge tools answer document lookup/search. The
  architecture tools answer source-backed design questions. `get_authoring_context`
  is a bounded prompt/context entry point.

Exposition action:

- Add workflow-guide kinds to `openhcs_get_authoring_context`, or add a typed
  workflow-guide capability projected from the same declaration registry.
- Recommended kinds:
  - `first_use`
  - `folder_onboarding`
  - `domain_expert_assisted_setup`
  - `ui_visible_workflow`
  - `headless_execution`
  - `viewer_review`
  - `objectstate_editing`
  - `cellprofiler_translation`

### 3. Function Catalog And Custom Functions

Purpose: find registered processing functions, inspect call signatures, and
register new functions when needed.

Tools:

- `openhcs_search_functions`
- `openhcs_describe_function`
- `openhcs_register_custom_function`

Redundancy judgment:

- Not redundant. Search returns candidates. Describe returns a specific function
  contract. Register mutates the custom-function registry.

Exposition action:

- Mark `search_functions` then `describe_function` as the normal sequence before
  draft pipeline edits.
- Keep `register_custom_function` advanced and mutating.

### 4. Config And Pipeline Drafting

Purpose: build or validate OpenHCS configs and pipeline drafts without touching a
running UI.

Tools:

- `openhcs_describe_config_schema`
- `openhcs_create_config`
- `openhcs_validate_config_patch`
- `openhcs_render_config_source`
- `openhcs_create_pipeline`
- `openhcs_add_function_step`
- `openhcs_validate_pipeline`
- `openhcs_render_pipeline_source`
- `openhcs_inspect_pipeline_source_artifact_plan`

Redundancy judgment:

- The config and pipeline tools are parallel patterns, not redundant:
  config tools own typed config patches; pipeline tools own in-memory
  `FunctionStep` drafts.
- `inspect_pipeline_source_artifact_plan` overlaps with compile setup but is
  read-only and should remain a planning/validation tool.

Exposition action:

- Present as one authoring group with two lanes:
  - config lane: describe, validate, create, render
  - pipeline lane: create, add step, validate, render, artifact-plan
- Keep `inspect_pipeline_source_artifact_plan` before session creation in
  recommended workflows.

### 5. Plate Data: Explicit Path Mode

Purpose: inspect, query, sample, generate, or stream plate data by explicit local
path, without depending on the PyQt UI selection.

Tools:

- `openhcs_generate_synthetic_plate`
- `openhcs_inspect_plate_path`
- `openhcs_query_plate_files`
- `openhcs_sample_plate_image`
- `openhcs_stream_plate_files_to_viewer`

Redundancy judgment:

- Not redundant with selected-plate tools. These operate from explicit
  `plate_path` input and are the right tools when the user says "I have this
  folder with images".

Exposition action:

- Mark these with target context `plate_path`.
- Make this the default non-UI data onboarding lane.
- Move `inspect_plate_path` out of the `selected_plate` service bucket or add
  enough target metadata that service is no longer interpreted as public
  grouping.

### 6. Plate Data: UI-Selected Mode

Purpose: inspect, query, sample, stream, initialize, compile, and run the plate
currently selected in the running PyQt UI.

Tools:

- `openhcs_ui_inspect_selected_plate_images`
- `openhcs_ui_query_selected_plate_files`
- `openhcs_ui_sample_selected_plate_image`
- `openhcs_ui_stream_selected_plate_files_to_viewer`
- `openhcs_ui_selected_plate_workflow`
- `openhcs_ui_list_state_surfaces`
- `openhcs_ui_get_state_surface`

Redundancy judgment:

- Not redundant with explicit path mode. These tools use the live UI bridge and
  preserve user-visible workflow state.

Exposition action:

- Mark these with target context `ui_selected_plate`.
- Make `openhcs_ui_selected_plate_workflow` the primary mutation path for
  `init_plate`, `compile_plate`, and `run_plate` when the user should see the UI.
- Explain that state surfaces are the poll/read path after workflow dispatch.

### 7. Headless Sessions And Jobs

Purpose: create execution sessions, submit compile/run jobs, and poll submitted
job status without using the visible UI.

Tools:

- `openhcs_create_orchestrator_session`
- `openhcs_create_orchestrator_session_from_pipeline_source`
- `openhcs_get_orchestrator_session`
- `openhcs_submit_compile`
- `openhcs_submit_pipeline_execution`
- `openhcs_get_execution_status`

Redundancy judgment:

- The two session constructors are mode variants:
  - draft-backed session
  - pycodified source-backed session
- `get_execution_status` is submitted-job status, not runtime-server status and
  not UI-operation status.

Exposition action:

- Group under target context `headless_session`.
- Rename in docs, not necessarily ABI, as:
  - "Create session from draft"
  - "Create session from source"
  - "Poll submitted job"
- In beginner guidance, prefer the UI-selected workflow when the user expects to
  watch the application.

### 8. Runtime Server Diagnostics

Purpose: discover and inspect live ZMQ runtime servers.

Tools:

- `openhcs_scan_runtime_servers`
- `openhcs_get_runtime_server_info`
- `openhcs_get_runtime_server_execution_status`

Redundancy judgment:

- Not redundant with headless job status. These tools query a runtime server
  endpoint directly.

Exposition action:

- Mark target context `runtime_server`.
- Document this as diagnostics and takeover, not the normal first-use execution
  path.

### 9. UI Bridge Discovery, Windows, And Actions

Purpose: connect to the running PyQt UI, inspect windows/actions, invoke semantic
actions, and use generic widget operations only as a fallback.

Tools:

- `openhcs_ui_list_bridges`
- `openhcs_ui_bridge_status`
- `openhcs_ui_list_windows`
- `openhcs_ui_focus_window`
- `openhcs_ui_navigate_window`
- `openhcs_ui_close_window`
- `openhcs_ui_snapshot_window`
- `openhcs_ui_list_actions`
- `openhcs_ui_invoke_action`
- `openhcs_ui_get_widget_tree`
- `openhcs_ui_invoke_widget_action`
- `openhcs_ui_get_operation_status`

Redundancy judgment:

- `ui_focus_window` and `ui_navigate_window` overlap, but are not identical:
  focus only versus focus plus reveal field/item.
- `ui_invoke_action` and `ui_invoke_widget_action` overlap intentionally:
  semantic UI actions are primary; generic widget actions are fallback for
  controls that have no semantic action yet.
- `ui_snapshot_window` is not redundant with `viewer_snapshot_window`; they
  capture different window domains.
- `ui_get_operation_status` is UI bridge operation status, not submitted
  execution-job status.

Exposition action:

- Split the current `ui_bridge` service bucket into subgroups in projected help:
  - bridge discovery
  - UI windows
  - semantic actions
  - widget fallback
  - UI operation status
- Mark widget tree/action tools as fallback/advanced.

### 10. UI Code Documents, ObjectState, Snapshots, And Branches

Purpose: read, validate, apply, inspect, and mutate the typed state surfaces used
by the running UI.

Tools:

- `openhcs_ui_list_code_documents`
- `openhcs_ui_get_code_document`
- `openhcs_ui_validate_code_document`
- `openhcs_ui_apply_code_document`
- `openhcs_ui_list_object_state_scopes`
- `openhcs_ui_get_object_state_fields`
- `openhcs_ui_describe_object_state_field`
- `openhcs_ui_mutate_object_state_field`
- `openhcs_ui_list_snapshots`
- `openhcs_ui_restore_snapshot`
- `openhcs_ui_time_travel_head`
- `openhcs_ui_list_branches`
- `openhcs_ui_switch_branch`

Redundancy judgment:

- Code documents and ObjectState mutation both edit UI-backed state, but they
  operate at different abstraction levels:
  - code documents preserve pycodified workflow surfaces;
  - ObjectState field mutation edits typed field values directly.
- Snapshot and branch tools are not duplicate state mutation. They are
  time-travel controls.

Exposition action:

- Make code documents the primary editable surface for pipeline/config state.
- Keep direct ObjectState mutation advanced and field-scoped.
- Mark snapshot/branch tools as expert and mutating when applicable.

### 11. Viewer And Napari Review

Purpose: open, validate, navigate, sample, and inspect viewer payloads after
streaming files or workflow outputs.

Tools:

- `openhcs_probe_viewer_window`
- `openhcs_viewer_snapshot_window`
- `openhcs_get_viewer_window_state`
- `openhcs_get_viewer_window_payloads`
- `openhcs_sample_viewer_window_image`
- `openhcs_summarize_viewer_window_rois`
- `openhcs_navigate_viewer_window`
- `openhcs_isolate_viewer_window_layers`
- `openhcs_validate_viewer_window_state`

Redundancy judgment:

- `probe`, `get_state`, and `validate_state` are different readiness depths:
  reachability, inspection, and assertion.
- `get_payloads`, `sample_image`, and `summarize_rois` are different data
  projections from viewer layers.
- `navigate` and `isolate` both mutate viewer presentation; `isolate` is a
  higher-level convenience command over layer visibility plus optional
  navigation.

Exposition action:

- Mark target context `viewer_window`.
- Show the normal sequence:
  stream files -> probe viewer -> validate state -> inspect payloads/ROIs ->
  navigate or isolate.

## Similar-Name Clusters

### `list_*`

Tools:

- `openhcs_list_capabilities`
- `openhcs_list_knowledge_documents`
- `openhcs_list_architecture_topics`
- `openhcs_ui_list_bridges`
- `openhcs_ui_list_code_documents`
- `openhcs_ui_list_state_surfaces`
- `openhcs_ui_list_actions`
- `openhcs_ui_list_windows`
- `openhcs_ui_list_object_state_scopes`
- `openhcs_ui_list_snapshots`
- `openhcs_ui_list_branches`

Judgment:

- Not redundant. These list different registries or live UI inventories.

Plan:

- Expose target context in capability metadata so "list" is not the only visible
  differentiator.

### `get_*`

Tools:

- `openhcs_get_authoring_context`
- `openhcs_get_knowledge_document`
- `openhcs_get_orchestrator_session`
- `openhcs_get_execution_status`
- `openhcs_get_runtime_server_info`
- `openhcs_get_runtime_server_execution_status`
- `openhcs_get_viewer_window_state`
- `openhcs_get_viewer_window_payloads`
- `openhcs_ui_get_state_surface`
- `openhcs_ui_get_widget_tree`
- `openhcs_ui_get_object_state_fields`
- `openhcs_ui_get_code_document`
- `openhcs_ui_get_operation_status`

Judgment:

- Mostly not redundant. The confusing part is the three status/info domains:
  submitted job, runtime server, and UI operation.

Plan:

- Add target context and status domain metadata.
- In docs/rendered help, label as:
  - submitted job status
  - runtime server status
  - UI bridge operation status

### `create_*`

Tools:

- `openhcs_create_config`
- `openhcs_create_pipeline`
- `openhcs_create_orchestrator_session`
- `openhcs_create_orchestrator_session_from_pipeline_source`

Judgment:

- Not redundant. These create different artifacts: config draft, pipeline draft,
  draft-backed session, and source-backed session.

Plan:

- Group config and pipeline creation in authoring.
- Group orchestrator session creation in headless execution.

### `inspect/query/sample/stream` Plate Tools

Tools:

- `openhcs_inspect_plate_path`
- `openhcs_query_plate_files`
- `openhcs_sample_plate_image`
- `openhcs_stream_plate_files_to_viewer`
- `openhcs_ui_inspect_selected_plate_images`
- `openhcs_ui_query_selected_plate_files`
- `openhcs_ui_sample_selected_plate_image`
- `openhcs_ui_stream_selected_plate_files_to_viewer`

Judgment:

- Intentional mode variants. The split is explicit path versus UI-selected
  plate.

Plan:

- Keep both modes.
- Project them as one "plate data" concept with two sources:
  - explicit `plate_path`
  - current UI selection

### `validate_*`

Tools:

- `openhcs_validate_config_patch`
- `openhcs_validate_pipeline`
- `openhcs_validate_viewer_window_state`
- `openhcs_ui_validate_code_document`

Judgment:

- Not redundant. These validate different contracts.

Plan:

- Group by object being validated, not by verb.

### `snapshot/window` Tools

Tools:

- `openhcs_ui_snapshot_window`
- `openhcs_viewer_snapshot_window`

Judgment:

- Not redundant. UI windows and viewer windows are separate control endpoints.

Plan:

- Keep both.
- Make target context explicit in grouped help.

### `navigate/isolate/focus/close`

Tools:

- `openhcs_ui_focus_window`
- `openhcs_ui_navigate_window`
- `openhcs_ui_close_window`
- `openhcs_navigate_viewer_window`
- `openhcs_isolate_viewer_window_layers`

Judgment:

- Not redundant. These mutate different presentation targets.

Plan:

- UI window controls belong to UI bridge.
- Viewer layer controls belong to viewer review.
- `isolate_viewer_window_layers` is a convenience command, not a duplicate of
  `navigate_viewer_window`.

## Actual Redundancy Candidates

These are not necessarily deletions. They are places where the public projection
should make primary/fallback or mode-variant status explicit.

| Cluster | Tools | Keep? | Reason |
| --- | --- | --- | --- |
| UI semantic action vs widget fallback | `openhcs_ui_invoke_action`, `openhcs_ui_invoke_widget_action` | Keep both | Semantic actions are stable primary controls. Widget actions are fallback for generic Qt controls. |
| UI focus vs navigate | `openhcs_ui_focus_window`, `openhcs_ui_navigate_window` | Keep both | Focus is simple window activation. Navigate can create/focus and reveal a field/item. |
| Viewer navigate vs isolate | `openhcs_navigate_viewer_window`, `openhcs_isolate_viewer_window_layers` | Keep both | Isolate is a higher-level layer visibility operation. |
| Draft session vs source session | `openhcs_create_orchestrator_session`, `openhcs_create_orchestrator_session_from_pipeline_source` | Keep both | Different authoring modes feed the same execution system. |
| Plate path tools vs selected-plate tools | path-based and `openhcs_ui_*selected_plate*` tools | Keep both | Same operations, different authority for the target plate. |
| Status tools | `openhcs_get_execution_status`, `openhcs_get_runtime_server_execution_status`, `openhcs_ui_get_operation_status` | Keep all | Different status domains. Needs clearer grouping. |
| Window snapshots | `openhcs_ui_snapshot_window`, `openhcs_viewer_snapshot_window` | Keep both | Different window servers. |

No current cluster looks like an obvious safe deletion based only on tool name.
Most confusion is exposition and target-context metadata, not duplicate behavior.

## Recommended Registry Additions

Add declaration-owned metadata to capability classes. The names below are
suggested fields, not required ABI names:

```python
workflow_group: ClassVar[AgentWorkflowGroup]
workflow_stage: ClassVar[AgentWorkflowStage]
target_context: ClassVar[AgentCapabilityTargetContext]
visibility: ClassVar[AgentCapabilityVisibility]
role: ClassVar[AgentCapabilityRole]
precedes: ClassVar[tuple[type[AgentCapabilityDeclaration], ...]]
follows: ClassVar[tuple[type[AgentCapabilityDeclaration], ...]]
```

These should be nominal enums or nominal declaration classes, not strings.
Projected strings can be emitted only at the final JSON/MCP boundary.

Suggested values:

- workflow groups:
  - discovery
  - knowledge
  - function_authoring
  - pipeline_authoring
  - plate_data
  - ui_selected_plate
  - headless_execution
  - runtime_diagnostics
  - ui_control
  - ui_state_editing
  - viewer_review
- target contexts:
  - server
  - knowledge_base
  - architecture_model
  - function_registry
  - config_draft
  - pipeline_draft
  - plate_path
  - ui_selected_plate
  - headless_session
  - submitted_job
  - runtime_server
  - ui_bridge
  - ui_window
  - ui_object_state
  - ui_code_document
  - viewer_window
- roles:
  - primary
  - mode_variant
  - fallback
  - diagnostic
  - expert

Implementation constraint:

- Do not add a separate hand-maintained grouping table.
- Use inheritance/mixins for repeated metadata, for example:
  - `PlatePathCapability`
  - `UiSelectedPlateCapability`
  - `ViewerWindowCapability`
  - `UiObjectStateCapability`
  - `HeadlessExecutionCapability`
- The grouped index should be generated from
  `AgentCapabilityDeclaration.__registry__`.

## Proposed First-Use Workflows

### User Has A Folder Of Images

1. `openhcs_health_check`
2. `openhcs_get_authoring_context(kind="folder_onboarding")`
3. `openhcs_inspect_plate_path`
4. `openhcs_query_plate_files`
5. `openhcs_sample_plate_image`
6. `openhcs_search_functions`
7. `openhcs_describe_function`
8. `openhcs_create_pipeline`
9. `openhcs_add_function_step`
10. `openhcs_validate_pipeline`
11. `openhcs_inspect_pipeline_source_artifact_plan`
12. `openhcs_create_orchestrator_session`
13. `openhcs_submit_compile`
14. `openhcs_submit_pipeline_execution`
15. `openhcs_get_execution_status`
16. `openhcs_stream_plate_files_to_viewer`
17. `openhcs_validate_viewer_window_state`

### User Wants To See The UI Work

1. `openhcs_health_check`
2. `openhcs_ui_bridge_status`
3. `openhcs_ui_list_windows`
4. `openhcs_ui_list_code_documents`
5. `openhcs_ui_get_code_document`
6. `openhcs_ui_validate_code_document`
7. `openhcs_ui_apply_code_document`
8. `openhcs_ui_selected_plate_workflow`
9. `openhcs_ui_get_operation_status`
10. `openhcs_ui_get_state_surface`
11. `openhcs_ui_query_selected_plate_files`
12. `openhcs_ui_stream_selected_plate_files_to_viewer`
13. `openhcs_validate_viewer_window_state`

### Agent Needs To Review Viewer Results

1. `openhcs_probe_viewer_window`
2. `openhcs_get_viewer_window_state`
3. `openhcs_validate_viewer_window_state`
4. `openhcs_get_viewer_window_payloads`
5. `openhcs_sample_viewer_window_image`
6. `openhcs_summarize_viewer_window_rois`
7. `openhcs_navigate_viewer_window`
8. `openhcs_isolate_viewer_window_layers`
9. `openhcs_viewer_snapshot_window`

### Agent Needs Runtime Diagnostics

1. `openhcs_scan_runtime_servers`
2. `openhcs_get_runtime_server_info`
3. `openhcs_get_runtime_server_execution_status`
4. `openhcs_get_orchestrator_session`
5. `openhcs_get_execution_status`

## Documentation And UI Projection Plan

1. Add nominal grouping metadata to capability declarations.
2. Generate a grouped capability resource from the registry:
   - compact beginner view
   - full expert view
   - workflow guide view
3. Teach `openhcs_list_capabilities` to optionally return grouped projection.
4. Teach the dev client `tools` command to render grouped output by default.
5. Keep a flat output mode for machine diffing and regression tests.
6. Add tests that every tool declares:
   - workflow group
   - target context
   - role
   - mutating and side effects when applicable
7. Add tests that no capability group is maintained outside declaration-owned
   metadata.

## Decision Rules

- Same verb does not imply redundancy.
- A tool is redundant only if it has the same target context, same authority,
  same side effects, and same output contract as another tool.
- UI-selected plate tools and path-based plate tools are mode variants.
- UI-window and viewer-window tools are separate domains.
- Submitted-job status, runtime-server status, and UI-operation status are
  separate domains.
- Generic widget tools are fallback controls; semantic UI actions remain primary.
- The registry declaration is the only durable place to encode grouping.
