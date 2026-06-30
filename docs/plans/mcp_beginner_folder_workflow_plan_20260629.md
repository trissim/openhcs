# MCP Beginner Folder Workflow Plan

Date: 2026-06-29

## Problem

The critical first-use workflow is: "I have this folder with microscopy images;
help me set it up and validate it." Agents need a reliable path that works for a
domain expert who does not know OpenHCS internals.

This should be a composed workflow over existing services. It should not become
an MCP wizard that bypasses plate inspection, config authoring, function search,
compile inspection, execution, or viewer validation.

## Existing Authorities

- `PlateInspectionService`
- `PipelineAuthoringService`
- function search and function detail services
- source-binding config/schema/view model
- knowledge docs
- execution session service
- artifact-plan inspection
- UI selected-plate workflow services
- viewer validation/sample/ROI services

## Target Shape

The first-use path should be explicit in docs and authoring context:

```text
inspect folder
    -> infer microscope/source model
    -> inspect/query/sample images
    -> search functions/examples
    -> draft pipeline/config
    -> compile artifact plan
    -> run bounded workflow
    -> inspect outputs and viewer payloads
```

MCP can expose a recipe projection or authoring-context section, but each step
must name the existing capability/service that performs the work.

## Nominal Iteration Authority

If implementation needs workflow steps with tool names, iterate
`AgentCapabilityDeclaration` registered capability classes and their
`cli_command`, workflow group, workflow stage, and target context declarations.
Do not keep an MCP-local tool list for the beginner workflow.

If implementation needs plate/onboarding state, iterate records returned by
`PlateInspectionService`, `PipelineAuthoringService`, function catalog services,
execution services, and viewer services. Do not create a wizard-owned state
model.

If implementation needs docs/recipe sections, iterate knowledge-base document
specs from `KnowledgeBaseService`. Do not embed a second recipe corpus in MCP.

## Implementation Dry Run

See `docs/plans/mcp_agent_experience_implementation_dry_run_20260629.md`.
The dry run confirmed `agent_capability_declarations()` exposes the capability
set and `KnowledgeBaseService` exposes the example/official30 docs needed for a
source-backed beginner route. The implementation must still dry-run an actual
synthetic plate workflow before claiming this user experience is complete.

## Implementation Steps

1. Update first-use authoring context.
   - Put the folder workflow before the flat tool list.
   - Include exact capability sequence.
2. Add a source-backed recipe doc.
   - Keep conceptual text in `docs/source`.
   - Reference `openhcs_core_model`, example corpus, source bindings, config
     schema, and artifact-plan inspection.
3. Add optional recipe projection service only if docs are insufficient.
   - If added, derive steps from capability declarations, not a hardcoded tool
     list.
4. Add dev-client quick path.
   - Render a concise sequence with commands.
   - Command names must come from capability declarations.
5. Test with deterministic service-level flow.
   - Generate synthetic plate.
   - Inspect.
   - Sample.
   - Draft a tiny pipeline.
   - Compile artifact plan.

## Mirror Traps To Avoid

- Do not create a wizard that writes pipeline/config side channels.
- Do not hardcode a separate list of tool names if capability declarations can
  provide them.
- Do not infer microscope/source identity from filenames in MCP.
- Do not skip compile inspection before execution.
- Do not treat viewer screenshots as the only validation evidence.

## Semantic Mirroring Audit

Audit questions:

- Is each workflow step backed by an existing service or capability declaration?
- Are command names derived from `AgentCapabilityDeclaration.cli_command` where
  possible?
- Does folder interpretation route through `PlateInspectionService` and source
  projection, not MCP guesses?
- Does validation include compile inspection and structured output checks before
  visual review?

Hard failures:

- A workflow helper writes pipeline/config state outside
  `PipelineAuthoringService`, config services, or UI ObjectState/code-document
  APIs.
- A static MCP recipe owns a list of tool names that can drift from capability
  declarations.
- A first-use path infers microscope type or source bindings from filename
  strings in MCP.
- The recipe says "run" without an artifact-plan validation step.

AST/rg audit:

```bash
rg -n "AgentCapabilityDeclaration|cli_command|PlateInspectionService|PipelineAuthoringService|artifact-plan|inspect_pipeline_source_artifact_plan" openhcs/agent openhcs/mcp docs/source
rg -n "workflow.*= \\{|tools.*= \\[|openhcs_.*openhcs_" openhcs/agent openhcs/mcp
rg -n "parse_filename|infer.*microscope|guess.*source" openhcs/mcp openhcs/agent
```

Allowed static recipe text must name capability declarations or source-backed
docs as the authority.

## Verification

Search gates:

```bash
rg -n "folder workflow|first use|domain expert|inspect folder" docs/source openhcs/agent/services/llm_context_service.py
rg -n "openhcs_inspect_plate_path|openhcs_search_functions|openhcs_inspect_pipeline_source_artifact_plan" openhcs/agent/services/llm_context_service.py docs/source
```

Focused tests:

```bash
XDG_CACHE_HOME=/tmp/openhcs-test-cache .venv/bin/python -m pytest \
  tests/unit/agent/test_agent_services.py \
  tests/unit/agent/test_knowledge_base_service.py \
  tests/unit/agent/test_capabilities.py
```

Manual MCP dry run:

```bash
.venv/bin/python -m openhcs.mcp.dev_client authoring-context --kind first_use
.venv/bin/python -m openhcs.mcp.dev_client generate-synthetic-plate /tmp/openhcs_mcp_beginner_plate
.venv/bin/python -m openhcs.mcp.dev_client inspect-plate /tmp/openhcs_mcp_beginner_plate
```
