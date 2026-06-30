# MCP Runtime Sidecar Projection Plan

Date: 2026-06-29

## Problem

The MCP function and artifact-plan surfaces should explain runtime sidecars and
artifact sidecar roles. Agents need to know when an artifact output carries a
declared sidecar role. They also need adjacent materialization/provenance facts,
such as source-identity filename behavior, without confusing those facts with
sidecar roles.

The MCP must not infer this from artifact names, file extensions, or filename
conventions.

## Existing Authorities

- `openhcs/core/artifacts.py`
  - `ArtifactSidecarRole`
  - `ArtifactSpec.sidecar_role`
  - `ArtifactInputPlan`
  - `ArtifactOutputPlan`
  - `ArtifactSpec.materialization_uses_source_identity_filename()`
- function artifact contracts and module artifact contracts
- `FunctionRuntimeContractSummary`
- `FunctionArtifactSpec`
- compiled artifact input/output plans
- materialization preview in `planned_materialization_preview`

## Target Shape

Sidecar information should flow through the existing contract path:

```text
function/module declaration
    -> ArtifactSpec.sidecar_role
    -> CallableContract/module artifact contract
    -> FunctionRuntimeContractSummary
    -> CompiledStepPlan.artifact_inputs/artifact_outputs
    -> ArtifactPlanSummary
```

Required projection behavior:

- `describe-function` shows sidecar roles on artifact inputs and outputs.
- `artifact-plan` shows sidecar roles after compilation.
- materialization preview continues to show source-identity filename behavior.
- if compiled plans have lost sidecar metadata, preserve it at the artifact-plan
  construction point rather than reconstructing it later.

## Nominal Iteration Authority

If implementation needs to enumerate artifact specs for a function, iterate the
artifact spec collections already exposed by `CallableContract` and module
artifact contracts. Do not keep a sidecar role table in MCP.

If implementation needs compiled artifact sidecars, iterate
`CompiledStepPlan.artifact_inputs.values()` and
`CompiledStepPlan.artifact_outputs.values()` and project the fields carried by
`ArtifactInputPlan` and `ArtifactOutputPlan`.

If implementation needs all legal sidecar roles for display or validation, use
the `ArtifactSidecarRole` enum. Do not duplicate it as a DTO or renderer enum.

The 2026-06-29 dry run showed `ArtifactSidecarRole` currently has one value,
`crop_mask`. Measurement tables, object labels, and source identity must not be
presented as sidecar roles unless core `ArtifactSidecarRole` grows those
members.

## Implementation Dry Run

See `docs/plans/mcp_agent_experience_implementation_dry_run_20260629.md`.
The dry run confirmed `ArtifactSpec`, `ArtifactInputPlan`, and
`ArtifactOutputPlan` all carry `sidecar_role`, so MCP can project sidecars from
compiled plans without reconstructing them from names.

## Implementation Steps

1. Audit whether `ArtifactInputPlan` and `ArtifactOutputPlan` retain
   `sidecar_role`.
   - If they do, project it in `ArtifactInputPlanSummary` and
     `ArtifactPlanSummary`.
   - If they do not, extend plan construction to carry
     `ArtifactSpec.sidecar_role`.
2. Extend execution DTOs.
   - Add `sidecar_role: str | None` to artifact storage summaries if missing.
   - Keep string values as final DTO ABI values only.
3. Update `artifact_plan_inspection_from_compilation`.
   - Read sidecar role from the compiled artifact plan.
   - Do not inspect artifact names.
4. Update renderers.
   - Show sidecar role beside artifact kind and path.
   - Show source-identity filename note from existing materialization preview.
5. Add tests covering:
   - function catalog sidecar roles;
   - compiled artifact-plan sidecar roles;
   - no fallback name-based sidecar inference.

## Mirror Traps To Avoid

- Do not create an MCP map from artifact names to sidecar roles.
- Do not infer sidecars from substrings like `mask`, `labels`, or `metadata`.
- Do not add CellProfiler-module specific sidecar rules in MCP.
- Do not duplicate `ArtifactSidecarRole` as another enum.

## Semantic Mirroring Audit

Audit questions:

- Does every displayed sidecar role originate from `ArtifactSpec.sidecar_role`,
  `ArtifactInputPlan`, or `ArtifactOutputPlan`?
- Does function detail use `FunctionRuntimeContractSummary` and
  `FunctionArtifactSpec`, not a renderer-side classifier?
- Does artifact-plan inspection carry sidecar information from compiled plans
  rather than reconstructing it from artifact keys?
- Are sidecar role strings final DTO values from `ArtifactSidecarRole.value`?

Hard failures:

- Any MCP/agent map classifies artifact names into sidecar roles.
- Any substring check such as `mask`, `label`, `measurement`, or `metadata`
  decides sidecar behavior.
- Any new enum duplicates `ArtifactSidecarRole`.
- Any CellProfiler module-specific sidecar rule appears outside module/function
  declaration or artifact contract code.

AST/rg audit:

```bash
rg -n "sidecar.*= \\{|ArtifactSidecarRole\\(|class .*Sidecar|mask.*sidecar|label.*sidecar|metadata.*sidecar" openhcs/mcp openhcs/agent
rg -n "sidecar_role" openhcs/core/artifacts.py openhcs/agent/dto openhcs/agent/services
rg -n "if .*name.*(mask|label|metadata|measurement)|if .*key.*(mask|label|metadata|measurement)" openhcs/agent openhcs/mcp
```

Allowed matches should trace back to artifact spec/plan fields or display those
fields unchanged.

## Verification

Search gates:

```bash
rg -n "sidecar.*dict|mask.*sidecar|labels.*sidecar|metadata.*sidecar" openhcs/mcp openhcs/agent
rg -n "sidecar_role" openhcs/core/artifacts.py openhcs/agent/dto openhcs/agent/services
```

Expected result:

- Sidecar values are projected from artifact specs/plans.
- No MCP code infers sidecar behavior from names.

Focused tests:

```bash
XDG_CACHE_HOME=/tmp/openhcs-test-cache .venv/bin/python -m pytest \
  tests/unit/agent/test_agent_services.py \
  tests/unit/agent/test_mcp_server.py \
  tests/unit/test_path_planner_materialization.py
```
