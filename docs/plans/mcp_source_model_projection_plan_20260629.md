# MCP Source Model Projection Plan

Date: 2026-06-29

## Problem

Agents need to understand the source model before they can safely help a
biologist with a folder of images. Today the MCP surface can expose plate
inspection and artifact-plan output, but it does not give a single typed view of
how source bindings, source inventories, source contexts, and virtual workspace
projection relate.

The fix must not make MCP a second source-binding implementation. MCP must not
parse filenames, read `openhcs_metadata.json` directly, or restate binding
semantics in DTO glue.

## Existing Authorities

- `openhcs/core/source_bindings.py`
  - `SourceBindingsConfig`
  - `StepSourceBindingsConfig`
  - `CompiledSourceBindingPlan`
  - `CompiledSourceUniversePlan`
  - `SourceBindingRuntimeContext`
- `openhcs/core/source_bindings_view.py`
  - `SourceBindingsViewModel`
  - `SourceInventoryProvider`
  - concrete inventory providers registered with `AutoRegisterMeta`
- `openhcs/core/source_binding_context.py`
  - `SourceBindingContext`
- `openhcs/core/source_workspace_projection.py`
  - `VirtualWorkspaceSourceProjection`
  - `VirtualWorkspaceSourceProjectionAuthority`
- `openhcs/core/orchestrator/orchestrator.py`
  - `source_workspace_projection()`
  - `source_workspace_files()`

## Target Shape

Add an agent-level source-model projection service that composes existing
authorities:

```text
SourceBindingContext
    -> SourceInventoryProvider.inventory()
    -> SourceBindingsViewModel.from_schema_and_bindings()
    -> optional VirtualWorkspaceSourceProjection summary
    -> bounded MCP DTO
```

The service should be UI-neutral and transport-neutral. It should be usable by
MCP, docs/tests, and eventually UI diagnostics without adding another binding
engine.

Suggested DTO sections:

- `schema`: source schema summary from the source-binding context.
- `inventory`: bounded source inventory from the selected inventory provider.
- `bindings`: source-binding view model rows.
- `virtual_workspace`: bounded projection summary if a projection exists.
- `provenance`: which authority produced each section.

## Nominal Iteration Authority

If implementation needs to enumerate source inventory behavior, iterate
`SourceInventoryProvider` registered subclasses through its `AutoRegisterMeta`
registry. Do not maintain a list of inventory provider names in MCP.

If implementation needs to enumerate source-binding table rows, iterate the
typed rows already returned by `SourceBindingsViewModel`. Do not rebuild rows
from source-binding field names.

If implementation needs virtual source records, iterate
`VirtualWorkspaceSourceProjection.pipeline_start_files()` and resolve each record
through `source_path_for()` and `source_metadata_for()`. Do not iterate raw
metadata JSON keys.

## Implementation Dry Run

See `docs/plans/mcp_agent_experience_implementation_dry_run_20260629.md`.
The dry run confirmed five registered `SourceInventoryProvider` implementations,
the typed `SourceBindingsViewModel.from_schema_and_bindings` constructor,
`SourceBindingContext` dataclass fields, and the required
`VirtualWorkspaceSourceProjection` methods.

## Implementation Steps

1. Add source-model DTOs under `openhcs/agent/dto`.
   - Keep DTOs descriptive and bounded.
   - Include `authority` fields such as `SourceBindingsViewModel` or
     `VirtualWorkspaceSourceProjection`.
   - Do not add behavior or parsing to DTOs.
2. Add `SourceModelProjectionService` or an equivalent method on an existing
   source/plate service.
   - Accept typed context inputs, not raw config dictionaries.
   - Ask `SourceBindingContext.inventory(bindings)` for inventory.
   - Ask `SourceBindingsViewModel.from_schema_and_bindings(...)` for binding
     presentation.
   - Ask `VirtualWorkspaceSourceProjection` for virtual-to-source records.
3. Expose the projection through MCP only after the agent service exists.
   - The tool wrapper should call one service method and serialize the DTO.
   - The wrapper must not inspect source-binding config fields itself.
4. Update first-use authoring context to point agents to this projection when a
   folder has ambiguous source identity.

## Mirror Traps To Avoid

- Do not parse microscope filenames in MCP.
- Do not inspect `openhcs_metadata.json` in MCP.
- Do not copy `SourceBindingOrigin`, selector, filter, or metadata-rule
  semantics into agent DTO code.
- Do not create MCP-only source inventory providers.
- Do not use string field names to decide whether a binding is pipeline-level or
  step-level.

## Semantic Mirroring Audit

Audit questions:

- Does every binding row come from `SourceBindingsViewModel`, not from MCP
  walking `SourceBindingsConfig` fields?
- Does every source inventory row come from `SourceInventoryProvider.inventory`
  through `SourceBindingContext`, not from MCP filesystem scans?
- Does every virtual-to-physical mapping come from
  `VirtualWorkspaceSourceProjection`, not from reading metadata JSON directly?
- Are source-binding origin, selector, filter, and metadata-rule names only
  final projected values from core objects?

Hard failures:

- Any MCP wrapper imports `OpenHCSMetadataHandler.METADATA_FILENAME` or opens
  `openhcs_metadata.json`.
- Any MCP or agent service calls a microscope filename parser directly for this
  projection.
- Any new dictionary maps source-binding field names, origins, selectors,
  filters, or metadata rules to behavior.
- Any implementation chooses pipeline vs step scope by checking a string field
  name instead of the typed config/context.

AST/rg audit:

```bash
rg -n "openhcs_metadata.json|METADATA_FILENAME|parse_filename|source_metadata_for" openhcs/mcp openhcs/agent
rg -n "SourceBindingOrigin|SourceSelector|SourceFilterClause|MetadataExtractionRule" openhcs/mcp openhcs/agent
rg -n "\\bsource_.*= \\{|\\bbindings.*= \\{|\\.get\\(\"origin\"|\\.get\\(\"bindings\"" openhcs/mcp openhcs/agent
```

Allowed matches are imports of DTO/service types, calls into
`SourceBindingsViewModel`, `SourceInventoryProvider`, `SourceBindingContext`, or
`VirtualWorkspaceSourceProjection`, and display-only rendering of DTO values.

## Verification

Search gates:

```bash
rg -n "openhcs_metadata.json|METADATA_FILENAME|parse_filename|source_metadata_for" openhcs/mcp openhcs/agent
rg -n "SourceBindingOrigin|SourceSelector|SourceFilterClause|MetadataExtractionRule" openhcs/mcp openhcs/agent
```

Expected result after implementation:

- MCP wrappers have no matches except imports of DTO/service types.
- Agent services only call the core authority methods; they do not reimplement
  matching, parsing, or metadata extraction.

Focused tests:

```bash
XDG_CACHE_HOME=/tmp/openhcs-test-cache .venv/bin/python -m pytest \
  tests/unit/agent/test_agent_services.py \
  tests/unit/agent/test_mcp_server.py \
  tests/unit/agent/test_knowledge_base_service.py
```
