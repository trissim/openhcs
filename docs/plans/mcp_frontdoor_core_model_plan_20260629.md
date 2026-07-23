# MCP Front-Door Core Model Plan

Date: 2026-06-29

Status: Implemented first pass. Remaining work is optional copy refinement and
an optional architecture-topic symbol index.

## Goal

Make MCP explain what OpenHCS is at the core before it exposes the full tool
surface.

The first response a contextless agent gets should not only say "here are 76
tools". It should teach the operational model:

OpenHCS is a compiler/runtime for high-content microscopy workflows. It turns
microscope folders and metadata into a typed virtual workspace, executes
declared `FunctionStep` pipelines over semantic axes, resolves lazy config and
artifact contracts at compile time, runs through a headless runtime or UI-owned
ObjectState workflow, and makes outputs reviewable through inventory and viewer
payload tools.

The conceptual explanation should be source-backed by the knowledge base. MCP
authoring contexts should project that knowledge, not become a second prose
authority.

## Existing Sources Found

### Current And Useful

- `docs/plans/openhcs_mcp_agent_knowledge_base_20260625.md`
  - Best current high-level agent/operator map.
  - Explains the architecture theme: semantic authority -> typed service/DTO ->
    adapter transport.
  - Good for MCP and developer boundaries, but too MCP-internal as the first
    explanation of OpenHCS itself.

- `docs/source/guide_for_biologists/domain_expert_onboarding.rst`
  - Good domain-entry page for "I have a folder of microscopy images".
  - Already explains fit, first questions, first workflow, CellProfiler mental
    model, and example corpus.
  - Should remain the domain-facing workflow authority.

- `docs/source/guides/example_corpus_map.rst`
  - Good authority for "do not invent from scratch".
  - Already maps CellProfiler examples, official30, native OpenHCS examples,
    presets, production examples, and live data inspection.

- `docs/source/architecture/pipeline_compilation_system.rst`
  - Good compiler model authority.
  - Explains declaration, compilation, execution, compiler phases, step plans,
    materialization, memory contracts, and GPU assignment.

- `docs/source/development/runtime_system_assembly_rules.rst`
  - Good agent/developer guardrail document.
  - Explains authority order, FunctionStep runtime shape, assembly flow, source
    binding, artifacts, runtime policies, and MCP usage rules.

- `docs/source/architecture/configuration_framework.rst`
  - Existing authority for `GlobalPipelineConfig`, `PipelineConfig`, lazy
    dataclasses, and inheritance semantics.

- `docs/source/architecture/special_io_system.rst`
  - Existing authority for artifact inputs/outputs, special IO, and
    materialization.

- `docs/source/guides/viewer_management.rst`,
  `docs/source/architecture/napari_streaming_system.rst`, and
  `docs/source/architecture/fiji_streaming_system.rst`
  - Existing viewer/review authorities.

### Useful But Needs Refresh Before Promotion

- `docs/source/concepts/data_dimensions.rst`
  - Useful axis mental model.
  - Needs current wording around `variable_components` and `group_by`.
  - Examples still show direct `FunctionStep(..., variable_components=...)`
    style instead of current `processing_config=LazyProcessingConfig(...)`.

- `docs/source/concepts/pipelines_and_steps.rst`
  - Useful pipeline/step model.
  - Needs current API examples and clearer compiler/runtime separation.
  - Some examples still present older direct `variable_components` usage.

- `docs/source/concepts/building_intuition.rst`
  - Good mental-model intent.
  - Needs current API examples and less metaphor-heavy prose for agent use.

- `docs/source/architecture/code_ui_interconversion.rst`
  - Correct topic, stale framing.
  - It still describes the system primarily as TUI/external-editor
    interconversion.
  - Needs update to current PyQt/ObjectState/code-document model:
    UI state is ObjectState-scoped; code documents are typed pycodified
    projections with revision tokens; `get-code-document`,
    `validate-code-document`, and `apply-code-document` are the MCP/UI bridge
    route; snapshots/branches/time-travel preserve provenance.

- `docs/source/architecture/microscope_handler_integration.rst`
  - Useful microscope abstraction source.
  - Should be checked against current source-binding/virtual-workspace behavior
    before making it a front-door authority.

## Proposed Knowledge-Base Source Shape

### Add Or Promote A Core Model Document

Create a concise source-backed knowledge document, preferably:

- `docs/source/concepts/core_model.rst`
- manifest id: `openhcs_core_model`

This document should be the front-door conceptual authority. It should be short
enough to include at the top of `openhcs_get_authoring_context(kind="first_use")`
without truncating the operational guidance.

Required sections:

1. **What OpenHCS Is**
   - A compiler/runtime for high-content microscopy workflows.
   - Not just a GUI, not just a function library, not just a viewer.

2. **Data And Source Model**
   - Plates/folders/OMERO sources are interpreted by microscope handlers,
     metadata handlers, source bindings, and virtual workspaces.
   - Agents inspect real inventory first; they do not guess filenames.

3. **Axis And Component Model**
   - Wells, sites, channels, Z, timepoint, and component identity.
   - `variable_components` define what is stacked into callable inputs.
   - `group_by` routes/fans out dictionary function patterns.

4. **Pipeline And Function Model**
   - Pipelines are ordered `FunctionStep` declarations.
   - Functions come from the registry or custom-function manager.
   - Function signatures/contracts are source of truth for agent-authored kwargs.

5. **Configuration And ObjectState Model**
   - `GlobalPipelineConfig`, `PipelineConfig`, step lazy configs, inheritance,
     resolved values, saved values.
   - ObjectState owns UI/edit/provenance state, snapshots, branches, and dirty
     markers.

6. **Compiler And Artifact Model**
   - Compile resolves source bindings, lazy config, axis/group semantics,
     artifact contracts, special IO, materialization, memory contracts, and
     resource plans before runtime execution.

7. **Runtime And UI Ownership Model**
   - Headless sessions execute without updating PlateManager.
   - UI-owned runs preserve selected plate rows, ObjectState snapshots,
     visible status, and output auto-add behavior.

8. **UI <-> Code Biconversion Model**
   - Code documents are bidirectional typed projections of UI state.
   - Agents use read/validate/apply with revision tokens, not raw widget state.
   - Reviewable Python is an interchange/provenance surface; ObjectState remains
     the UI state authority.

9. **Review Model**
   - Outputs are verified through plate inventory, materialized artifacts,
     viewer layers, payload records, ROI summaries, and sampled pixels.

10. **Agent Operating Rule**
    - Inspect inventory, query registries, compile/inspect plans, execute
      bounded tests, then validate outputs.

### Keep Existing Documents As Deep Authorities

The new core document should not duplicate every detail. It should link to
existing KB documents:

- `openhcs_domain_expert_onboarding`
- `openhcs_example_corpus_map`
- `openhcs_data_dimensions`
- `openhcs_pipelines_and_steps`
- `openhcs_function_patterns`
- `openhcs_configuration_framework`
- `openhcs_pipeline_compilation_system`
- `openhcs_special_io_system`
- `openhcs_code_ui_interconversion`
- `openhcs_viewer_management`
- `openhcs_runtime_system_assembly_rules`

## MCP Projection Plan

### 1. Knowledge Base Manifest

Add `openhcs_core_model` to
`docs/source/development/mcp_knowledge_base_manifest.json` with tags such as:

- `core model`
- `compiler runtime`
- `virtual workspace`
- `FunctionStep`
- `ObjectState`
- `code documents`
- `UI code interconversion`
- `viewer review`

This keeps the document discoverable through `openhcs_list_knowledge_documents`
and `openhcs_search_knowledge`.

### 2. First-Use Authoring Context

Change `openhcs_get_authoring_context(kind="first_use")` so the first section is
the core model, before capability groups.

Target ordering:

1. `OPENHCS CORE MODEL`
2. `FIRST-USE OPERATIONAL ROUTES`
3. `FOLDER ONBOARDING WORKFLOW`
4. `UI-VISIBLE WORKFLOW`
5. `VIEWER REVIEW WORKFLOW`
6. `CAPABILITY GROUPS`

The first section should be sourced from the KB document or generated by a
single declaration that names the KB document. Do not create a second long
static copy of the core model in `llm_context_service.py`.

### 3. Folder-Onboarding Authoring Context

Prepend a smaller source/data model excerpt to
`kind="folder_onboarding"`:

- source/metadata/virtual workspace;
- axis/component semantics;
- examples-first rule;
- compile before run.

Then keep the current concrete workflow steps.

### 4. UI-Visible Authoring Context

Update `kind="ui_visible_workflow"` and `kind="objectstate_editing"` to explain
UI/code biconversion upfront:

- ObjectState is the UI state authority.
- Code documents are typed projections, not freeform files.
- Revision tokens protect edits.
- Snapshots/branches/time-travel explain provenance and rollback.
- `ui_get_state_surface` is how agents observe workflow results.

### 5. Architecture Topic Option

Optionally add `openhcs_explain_architecture(topic_id="core_model")` as a
source-backed symbol map that references the same model:

- `FunctionStep`
- `GlobalPipelineConfig`
- `PipelineConfig`
- microscope/source binding declarations
- artifact/materialization declarations
- ObjectState/code document DTOs
- runtime submission DTOs
- viewer payload DTOs

This should be a symbol index, not the main prose authority.

## Documentation Refresh Plan

Implemented:

1. Added `docs/source/concepts/core_model.rst` and exposed its concise
   `core-summary` section through `first_use`.
2. Added `openhcs_core_model` to the KB manifest.
3. Updated `first_use` authoring context to start with the KB-backed core
   model, then operational routes, then workflows, then capability groups.
4. Updated `folder_onboarding`, `ui_visible_workflow`, and ObjectState/code
   roundtrip context text to frontload source, axis, ObjectState, and
   UI/code ownership.
5. Replaced stale `docs/source/architecture/code_ui_interconversion.rst` with
   the current PyQt/ObjectState/code-document model.
6. Replaced stale front-door versions of `data_dimensions.rst` and
   `pipelines_and_steps.rst` with current processing-config/compiler/runtime
   wording.
7. Added tests for KB discovery/search, stale-source watchlist coverage, and
   first-use core-model ordering.
8. Promoted first-class CellProfiler compatibility into the front-door model:
   `first_use` now explains that CellProfiler `.cppipe` modules, Images,
   Objects, Measurements, SaveImages, and exports compile into OpenHCS source
   bindings, `FunctionStep` declarations, artifact contracts, runtime values,
   materialization, and measurements.
9. Added the artifact sidecar/source-universe model to the same front door:
   source bindings resolve against compile-planned source universes, runtime
   adapters consume `SourceUniverseRequest`/strategy results, and sidecar
   artifacts are typed `ArtifactSpec`/`ArtifactSidecarRole` contracts rather
   than ad hoc companion files.
10. Raised the default authoring-context bound to 16k and made the dev-client
    authoring-context command reuse the DTO default, so `first_use` renders the
    full front-door model by default while `--max-chars` remains the explicit
    escape hatch.
11. Made the first-use prerequisite explicit: agents that do not already know
    OpenHCS should call/read `openhcs_get_authoring_context(kind="first_use")`
    before choosing tools, including from the domain-expert setup context and
    the MCP tool description itself.
12. Sharpened the UI/code biconversion model: code mode is a live reflected
    ObjectState/code-document projection over UI-owned objects, with
    validation, revision tokens, snapshots, and real-time running-UI updates;
    it is not an export/re-import script workflow.

Remaining optional work:

1. Further edit `docs/source/concepts/building_intuition.rst` before promoting
   it into any front-door context.
2. Audit `docs/source/architecture/microscope_handler_integration.rst` against
   current source-binding/virtual-workspace behavior before using it as a
   front-door authority.
3. Optionally add `openhcs_explain_architecture(topic_id="core_model")` as a
   source-symbol index over the same model.

Original planned steps:

1. Add `docs/source/concepts/core_model.rst`.
2. Update `docs/source/architecture/code_ui_interconversion.rst` from TUI-first
   language to current PyQt/ObjectState/code-document biconversion.
3. Update `docs/source/concepts/data_dimensions.rst` examples to current
   `LazyProcessingConfig`/`processing_config` usage and correct
   `variable_components` semantics.
4. Update `docs/source/concepts/pipelines_and_steps.rst` examples and compiler
   language.
5. Add `openhcs_core_model` to the KB manifest.
6. Update authoring-context section ordering and add a KB-backed core-model
   section.
7. Add tests:
   - KB catalog contains `openhcs_core_model`.
   - `first_use` context starts with core model, not capability groups.
   - `first_use` includes UI/code biconversion terms.
   - `folder_onboarding` still includes concrete workflow tools.
   - `code_ui_interconversion.rst` no longer presents TUI as the only/current
     authority.
8. Smoke test:
   - `python -m openhcs.mcp.dev_client knowledge-search "what is openhcs core model"`
   - `python -m openhcs.mcp.dev_client knowledge-document openhcs_core_model --max-chars 4000`
   - `python -m openhcs.mcp.dev_client authoring-context --kind first_use --max-chars 3000`
   - `python -m openhcs.mcp.dev_client authoring-context --kind ui_visible_workflow --max-chars 2500`

## Acceptance Criteria

A contextless agent that calls only `openhcs_get_authoring_context(kind="first_use")`
should be able to answer:

- What is OpenHCS?
- What is the data/source model?
- What is a `FunctionStep`?
- What do `variable_components` and `group_by` mean?
- Why does compile happen before run?
- What is ObjectState?
- How does UI <-> code biconversion work?
- When should it use UI-owned versus headless execution?
- How should outputs be validated?

The answer should be operationally useful without reading the whole tool list.
The tool list should become reference material after the core model and golden
paths are clear.

## Risks

- Stale docs may be accidentally promoted as current if refreshed piecemeal.
  Mitigation: update stale docs before adding them to the front-door flow.
- Duplicated conceptual prose can drift between docs and authoring contexts.
  Mitigation: project from the KB or keep the authoring-context section as a
  short pointer plus excerpt, not a second independent essay.
- A large first-use context can crowd out the golden path.
  Mitigation: keep the core model concise and move detail into KB follow-up
  documents.
