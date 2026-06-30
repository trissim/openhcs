# MCP VFS And Storage Model Plan

Date: 2026-06-29

## Problem

Agents need to understand why OpenHCS uses a virtual filesystem, storage
backends, source workspaces, materialized outputs, and viewer streams. The MCP
front door should explain this model well enough for a domain-expert workflow,
but it must not create MCP-specific storage rules that drift from path planning
and runtime materialization.

## Existing Authorities

- `PathPlanningConfig`
- `VFSConfig`
- `PlateFileInventory`
- `PlateInspectionService`
- `CompiledStepPlan`
- `RuntimeArtifactMaterializationPlan`
- `planned_materialization_preview`
- storage, special IO, pipeline compilation, and viewer docs

## Target Shape

The MCP should project storage facts from existing systems:

```text
config schema and docs
    -> path planning and VFS config explanation
    -> plate inventory
    -> compiled materialization plans
    -> artifact-plan/dev-client renderers
```

Agents should be able to answer:

- where the source files are;
- whether files are physical, virtual workspace, or materialized outputs;
- which backend is used for read and write;
- whether analysis outputs are persistent;
- whether runtime metadata can refine materialized paths;
- what viewer streaming validates.

## Nominal Iteration Authority

If implementation needs config fields, iterate `PathPlanningConfig` and
`VFSConfig` through config schema extraction. Do not list storage defaults or
backend options in MCP.

If implementation needs physical or result files, iterate records returned by
`PlateFileInventory` through `PlateInspectionService` or related plate services.
Do not walk the filesystem in MCP wrappers.

If implementation needs materialized output paths, iterate compiled artifact
plans and `planned_materialization_preview()` results. Do not infer
materialization from output path strings.

## Implementation Dry Run

See `docs/plans/mcp_agent_experience_implementation_dry_run_20260629.md`.
The dry run confirmed `PathPlanningConfig`, `VFSConfig`, `PlateFileInventory`,
and `planned_materialization_preview` import and expose the expected fields or
methods.

## Implementation Steps

1. Refresh docs and first-use context.
   - Link VFS/storage explanation to config schema and compiler artifact plan.
   - Do not duplicate detailed defaults in static text.
2. Enrich artifact-plan renderer.
   - Use existing `ArtifactMaterializationPlanSummary`.
   - Show read/write backends, persistent backend, and analysis output directory.
3. Confirm `PlateFileInventory` remains the file listing authority.
   - MCP should call plate inventory services for file queries.
   - It should not scan paths itself.
4. Add source model cross-link.
   - Virtual workspace explanation should point to source-model projection.
5. Add tests for renderer/service output.
   - Verify config schema contains VFS/path planning docs.
   - Verify artifact-plan summaries expose materialization facts.

## Mirror Traps To Avoid

- Do not restate VFS backend defaults in MCP code.
- Do not scan plate folders in MCP wrappers.
- Do not infer persistent output behavior from path strings.
- Do not treat viewer streaming as the artifact authority.
- Do not add a storage glossary separate from source-backed docs.

## Semantic Mirroring Audit

Audit questions:

- Are storage defaults and meanings projected from `PathPlanningConfig`,
  `VFSConfig`, config schema, or source-backed docs?
- Are file records produced by `PlateFileInventory`/`PlateInspectionService`?
- Are output paths and persistence facts produced by compiled plans and
  `planned_materialization_preview`?
- Are viewer tools presented as validation/review, not storage authority?

Hard failures:

- MCP code declares backend defaults or path-planning rules.
- MCP wrappers scan folders directly to explain storage.
- A renderer infers persistence or materialization from path substrings.
- A docs section contains storage rules without linking to config/schema or
  compiled-plan authority.

AST/rg audit:

```bash
rg -n "PathPlanningConfig|VFSConfig|planned_materialization_preview|PlateFileInventory" openhcs/agent openhcs/mcp docs/source
rg -n "os\\.walk|rglob\\(|glob\\(" openhcs/mcp openhcs/agent/services
rg -n "backend.*default|persistent.*default|if .*path.*analysis|if .*path.*output" openhcs/mcp openhcs/agent
```

Allowed backend/default text must come from config schema or docs that cite the
config authority.

## Verification

Search gates:

```bash
rg -n "PathPlanningConfig|VFSConfig|planned_materialization_preview|PlateFileInventory" openhcs/agent openhcs/mcp docs/source
rg -n "os.walk|rglob\\(|glob\\(" openhcs/mcp openhcs/agent/services
```

Expected result:

- File queries route through inventory services.
- Storage explanation comes from config docs and compiled materialization plans.

Focused tests:

```bash
XDG_CACHE_HOME=/tmp/openhcs-test-cache .venv/bin/python -m pytest \
  tests/unit/agent/test_agent_services.py \
  tests/unit/agent/test_knowledge_base_service.py \
  tests/unit/test_path_planner_materialization.py
```
