# MCP Artifact Source Coverage Plan

Date: 2026-06-29

## Problem

`openhcs_inspect_pipeline_source_artifact_plan` is the right validation surface,
but source coverage is currently too narrow for first-use autonomy. A contextless
agent can see an artifact plan and still be unsure whether an empty
`source_workspace.file_count` means:

- no physical input images exist;
- the virtual workspace was not initialized;
- pipeline-start files are empty but physical inventory exists;
- a step-level source universe will select files later;
- source bindings require runtime selector resolution.

The fix must expand the existing artifact-plan projection. It must not add raw
file counters in MCP to paper over ambiguity.

## Existing Authorities

- `openhcs/core/compiled_step_plan.py`
  - `CompiledStepPlan`
  - `source_binding_plan`
  - `source_universe_plan`
  - `source_load_plan`
  - artifact input/output plans
- `openhcs/agent/services/execution_session_service.py`
  - `artifact_plan_inspection_from_compilation`
  - `_bounded_step_summaries`
  - `_source_workspace_summary`
- `openhcs/core/source_binding_selection.py`
  - `SourceUniverseRequest`
  - `SourceUniverseStrategy`
  - step-input and pipeline-start universe strategies
- `openhcs/core/source_workspace_projection.py`
  - `VirtualWorkspaceSourceProjection`
- `openhcs/core/source_load_plan.py`
  - `SourceLoadPlan`

## Target Shape

Extend `ArtifactPlanInspection` with source-coverage sections derived from the
compiled context:

```text
CompiledExecutionBundle
    -> ProcessingContext.step_plans
    -> CompiledStepPlan.source_binding_plan
    -> CompiledStepPlan.source_universe_plan
    -> CompiledStepPlan.source_load_plan
    -> VirtualWorkspaceSourceProjection
    -> SourceUniverseRequest/Strategy summaries where runtime state exists
```

Recommended DTO additions:

- `source_workspace`: keep the existing virtual workspace summary.
- `physical_inventory`: optional bounded summary from the same plate inspection
  authority used elsewhere, if available through the request path.
- `source_universes`: per-step summary of required source-universe behavior.
- `source_bindings`: per-step summary of compiled bindings and whether selector
  resolution is required.
- `source_load`: per-step source-load plan summary.
- `compiled_context`: optional generic bounded compiled-context tree for
  discovery and debugging.

The generic compiled-context tree should be type-driven:

- dataclasses: traverse `dataclasses.fields`;
- enums/paths/callables: reuse `openhcs.agent.serialization.to_jsonable`;
- mappings/sequences: bounded previews;
- runtime-only objects: emit type/import path, not internals;
- domain-specific enrichments: registered by type, not by field-name strings.

## Nominal Iteration Authority

If implementation needs to enumerate compiled steps, iterate
`CompiledExecutionBundle.runtime_contexts` and each
`ProcessingContext.step_plans.values()`. Do not keep a separate step index or
step-field list in MCP.

If implementation needs to enumerate source plans, iterate each
`CompiledStepPlan` instance and read its typed fields:
`source_binding_plan`, `source_universe_plan`, and `source_load_plan`.
The 2026-06-29 dry run showed `SourceLoadPlan` currently exposes only
`zarr_config`; do not promise richer source-load facts until the compiler adds
them to that plan.

If implementation needs to enumerate source-universe strategy behavior, iterate
the nominal `SourceUniverseRequest`/`SourceUniverseStrategy` registration and
matching machinery. Do not mirror strategy names or physical/virtual/pipeline
start categories in MCP.

If implementation needs a generic compiled-context tree, iterate dataclass
fields with `dataclasses.fields()` and registered projection policies by target
type. Do not use a string allowlist of important `CompiledStepPlan` fields.

## Implementation Dry Run

See `docs/plans/mcp_agent_experience_implementation_dry_run_20260629.md`.
The dry run confirmed the compiled source/artifact fields on `CompiledStepPlan`,
the three current `CompiledSourceUniversePlan` flags, the two
`SourceUniverseRequest` types, and six concrete `SourceUniverseStrategy` types.
It also confirmed `SourceLoadPlan` is currently only `zarr_config`.

## Implementation Steps

1. Add bounded source-coverage DTOs to `openhcs/agent/dto/execution.py`.
   - Use names that describe projection, not new semantics.
   - Keep counts and examples bounded.
2. Extend `artifact_plan_inspection_from_compilation`.
   - Read `CompiledStepPlan` fields directly.
   - Reuse `_source_workspace_summary` for virtual workspace projection.
   - Add source-universe and source-load summaries from compiled plans.
3. Add a generic compiled-context projection helper.
   - Make it a generic serialization/projection utility, not a source-binding
     specific function.
   - Allow depth/item limits.
   - Include declared/runtime type names and provenance.
4. Add optional request controls.
   - `include_compiled_context`
   - `max_depth`
   - `max_items`
   - optional `field_path_contains`
5. Update dev-client artifact-plan renderer.
   - Render source coverage before artifact output paths.
   - Explain empty counts by showing which authority was empty.

## Mirror Traps To Avoid

- Do not count raw files in MCP independently of plate/source authorities.
- Do not re-run source-universe selection in the renderer.
- Do not add string lists of `CompiledStepPlan` fields to keep.
- Do not hardcode special cases for source-binding field names.
- Do not expose mutable `ProcessingContext` directly.

## Semantic Mirroring Audit

Audit questions:

- Does source coverage come from `CompiledStepPlan.source_binding_plan`,
  `source_universe_plan`, `source_load_plan`, `VirtualWorkspaceSourceProjection`,
  or plate inventory services?
- Does a generic compiled-context projection traverse dataclasses/types instead
  of maintaining a field allowlist?
- Are source-universe facts projected from `CompiledSourceUniversePlan` and
  `SourceUniverseRequest`/`SourceUniverseStrategy`, not recomputed in MCP?
- Can an empty count be traced to the authority that was empty?

Hard failures:

- MCP or renderer code uses `os.walk`, `Path.rglob`, or raw globbing to explain
  source coverage.
- A new list of `CompiledStepPlan` field strings is the discovery mechanism.
- A renderer calls source-universe strategy code to recompute selection.
- A DTO field says a universe is physical/virtual/pipeline-start without a
  corresponding compiled plan or projection authority.

AST/rg audit:

```bash
rg -n "os\\.walk|rglob\\(|glob\\(" openhcs/mcp openhcs/agent/services/execution_session_service.py
rg -n "source_binding_plan|source_universe_plan|source_load_plan|VirtualWorkspaceSourceProjection" openhcs/agent/dto openhcs/agent/services/execution_session_service.py
rg -n "CompiledStepPlan.*\\[|\\[.*step_.*plan|field.*allow|keep_fields|include_fields" openhcs/agent openhcs/mcp
```

Allowed field filtering must be generic projection control, not a list that
declares semantic importance.

## Verification

Search gates:

```bash
rg -n "source_workspace.file_count|raw file|os.walk|rglob\\(" openhcs/mcp openhcs/agent/services/execution_session_service.py
rg -n "source_binding_plan|source_universe_plan|source_load_plan" openhcs/agent/dto openhcs/agent/services/execution_session_service.py
```

Expected result:

- New source summaries come from compiled-plan fields or existing inventory
  services.
- No MCP wrapper performs filesystem scans to infer source coverage.

Focused tests:

```bash
XDG_CACHE_HOME=/tmp/openhcs-test-cache .venv/bin/python -m pytest \
  tests/unit/agent/test_agent_services.py \
  tests/unit/agent/test_mcp_server.py \
  tests/unit/test_path_planner_materialization.py
```

Live check:

```bash
XDG_CACHE_HOME=/tmp/openhcs-test-cache \
  .venv/bin/python -m openhcs.mcp.dev_client artifact-plan \
  --plate-path /path/to/test/plate \
  --pipeline-source /path/to/pipeline.py
```

The output should distinguish physical inventory, virtual workspace files,
pipeline-start files, and per-step source requirements.
