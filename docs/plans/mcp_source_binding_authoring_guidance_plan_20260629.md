# MCP Source-Binding Authoring Guidance Plan

Date: 2026-06-29

## Problem

Agents need enough source-binding guidance to author a pipeline config for a
real folder of images. The guidance must be practical, but it must not become a
parallel source-binding schema. Field names, defaults, lazy inheritance, enabled
behavior, filters, selectors, metadata rules, and match plans already have typed
owners.

## Existing Authorities

- `SourceBindingsConfig`
- `StepSourceBindingsConfig`
- generated lazy config classes from the config framework
- `ConfigService.describe_schema`
- `dataclass_parameter_descriptions`
- `SourceBindingsViewModel`
- source-backed knowledge docs and manifest entries

## Target Shape

Make authoring guidance source-backed:

```text
dataclass docs and generated config schema
    -> knowledge docs with conceptual examples
    -> source-binding view model for table-shaped previews
    -> MCP authoring context/renderers
```

The MCP should tell agents:

- pipeline-level `SourceBindingsConfig` describes how arbitrary sources become a
  workspace when the pipeline config selects the source-bindings microscope;
- step-level `StepSourceBindingsConfig` inherits the payload fields and carries
  step enablement;
- fields are edited through config/object-state machinery, not custom MCP tables;
- source-binding tables shown in the UI are views over dataclasses, not the data
  authority.

## Nominal Iteration Authority

If implementation needs source-binding fields, iterate dataclass fields on
`SourceBindingsConfig` and `StepSourceBindingsConfig` through config schema
extraction. Do not maintain a source-binding field list in authoring context,
MCP server code, or renderers.

If implementation needs config types, use the existing config declaration path
and generated lazy config classes. Do not introduce a separate MCP config-kind
registry for source bindings.

If implementation needs table-shaped presentation, iterate
`SourceBindingsViewModel` rows. Do not derive table rows from field names or UI
labels.

## Implementation Dry Run

See `docs/plans/mcp_agent_experience_implementation_dry_run_20260629.md`.
The dry run confirmed the source-binding view model and context authorities used
by this plan. Implementation still needs to verify config-schema extraction on
`SourceBindingsConfig` and `StepSourceBindingsConfig` in the specific patch that
edits authoring docs or schema output.

## Implementation Steps

1. Refresh source-binding config docstrings.
   - Each field should explain intent, inheritance behavior, and common values.
   - Do this on the dataclasses, not in MCP.
2. Ensure config schema projection exposes those descriptions.
   - `openhcs_describe_config_schema` should show useful descriptions for both
     source-binding classes.
3. Add or update source-backed docs.
   - Keep examples in `docs/source`, referenced by the knowledge-base manifest.
   - Link to core model, source model, and folder onboarding docs.
4. Add examples that are rendered from real config/code paths where feasible.
   - Prefer pycodified `PipelineConfig(...)` examples.
   - Avoid copied JSON schemas.
5. Update authoring-context text to point to:
   - config schema;
   - source-model projection;
   - source-binding docs;
   - compile artifact-plan validation.

## Mirror Traps To Avoid

- Do not create a second source-binding schema in MCP docs.
- Do not hand-code field lists in `llm_context_service.py`.
- Do not describe lazy inheritance in a way that diverges from ObjectState.
- Do not special-case step vs pipeline fields by string name.
- Do not add MCP-specific source-binding examples that cannot be compiled.

## Semantic Mirroring Audit

Audit questions:

- Do field descriptions come from dataclass docstrings/signature docs through
  config schema extraction?
- Do examples instantiate real `SourceBindingsConfig`,
  `StepSourceBindingsConfig`, or `PipelineConfig` objects?
- Does the guidance tell agents to query config schema and source-model
  projection instead of copying a schema from docs?
- Does lazy inheritance wording match ObjectState/config behavior, not a
  hand-authored approximation?

Hard failures:

- A static MCP schema lists source-binding fields and allowed values.
- A docs-only example cannot be imported, rendered, or compiled with current
  config classes.
- A service or renderer manually lists source-binding table columns as semantic
  data instead of using the source-binding view model.

AST/rg audit:

```bash
rg -n "source_bindings.*schema|bindings.*schema|field.*payload|allowed.*Source" openhcs/mcp openhcs/agent docs/source
rg -n "SourceBindingsConfig|StepSourceBindingsConfig|dataclass_parameter_descriptions|ConfigFieldSchema" openhcs/agent docs/source
rg -n "\\[\".*bindings|\\(\"bindings\"|\\.get\\(\"bindings\"" openhcs/mcp openhcs/agent
```

Allowed matches are config-schema extraction, source-backed docs, and examples
using real config classes.

## Verification

Search gates:

```bash
rg -n "bindings.*dict|source_bindings.*schema|Step Source Bindings|Pipeline Source Bindings" openhcs/mcp openhcs/agent
rg -n "SourceBindingsConfig|StepSourceBindingsConfig" docs/source docs/plans openhcs/agent/services/config_service.py
```

Expected result:

- Guidance references dataclass/config-schema authorities.
- No MCP file owns a parallel source-binding schema.

Focused tests:

```bash
XDG_CACHE_HOME=/tmp/openhcs-test-cache .venv/bin/python -m pytest \
  tests/unit/agent/test_agent_services.py \
  tests/unit/agent/test_knowledge_base_service.py \
  tests/unit/pyqt_gui/test_source_bindings_editor.py
```

Fresh MCP checks:

```bash
.venv/bin/python -m openhcs.mcp.dev_client config-schema SourceBindingsConfig
.venv/bin/python -m openhcs.mcp.dev_client config-schema StepSourceBindingsConfig
.venv/bin/python -m openhcs.mcp.dev_client knowledge-search "source bindings arbitrary folder"
```
