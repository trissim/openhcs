# MCP Agent Experience Plan Set

Date: 2026-06-29

## Purpose

This plan set covers the next MCP ergonomics batch for agents that are helping
domain experts use OpenHCS. The common rule is that MCP must project existing
OpenHCS authorities. It must not parse filenames, duplicate source-binding
semantics, maintain module lists, or build MCP-local registries of behavior that
already belongs to core declarations.

Current target architecture:

```text
core declarations, compiler plans, ObjectState, runtime services
    -> openhcs.agent public operation declarations, services, and DTO projections
    -> MCP transport wrappers
    -> dev-client renderers
```

The current MCP implementation has already moved a long way toward this shape:
capabilities are declaration-backed, MCP bindings are mostly generated, the dev
client derives commands from capability declarations, and output renderers bind
to DTO types. The remaining correction is to stop treating MCP capabilities as
the final generic center. Capabilities should become projections of the public
operation model in `openhcs.agent`.

This plan set must not re-add existing functionality under new names. For every
item below, implementation starts by inventorying the existing authority and
then filling only the missing projection or exposition gap. Existing capability
declarations, UI bridge contracts, ObjectState code-document routes,
source-binding view models, compiler artifact-plan inspection, knowledge
documents, examples, dev-client generators, and renderers must be reused rather
than duplicated.

## Plan Files

0. `docs/plans/mcp_public_python_api_projection_plan_20260629.md`
   - Reinvestigate the plan set under the public-Python-API-first lens and
     define how the current generic MCP/capability/dev-client infrastructure
     should collapse behind public operation declarations.
1. `docs/plans/mcp_source_model_projection_plan_20260629.md`
   - Expose source model and source bindings through typed projections over
     `SourceBindingsViewModel`, `SourceInventoryProvider`,
     `SourceBindingContext`, and `VirtualWorkspaceSourceProjection`.
2. `docs/plans/mcp_artifact_source_coverage_plan_20260629.md`
   - Expand artifact-plan inspection so source coverage is visible through
     compiled plans and source-universe authorities.
3. `docs/plans/mcp_source_binding_authoring_guidance_plan_20260629.md`
   - Make source-binding authoring guidance source-backed through config schema,
     dataclass docs, source-binding view models, and knowledge documents.
4. `docs/plans/mcp_runtime_sidecar_projection_plan_20260629.md`
   - Surface runtime sidecar roles from artifact contracts and compiled artifact
     plans.
5. `docs/plans/mcp_vfs_storage_model_plan_20260629.md`
   - Explain the virtual filesystem and storage model through existing configs,
     inventories, materialization previews, and docs.
6. `docs/plans/mcp_function_contract_axis_semantics_plan_20260629.md`
   - Enrich function detail output from callable contracts, processing
     declarations, registered signatures, and CellProfiler declarations.
7. `docs/plans/mcp_beginner_folder_workflow_plan_20260629.md`
   - Make the "I have a folder of images" workflow explicit without bypassing
     `PlateInspectionService`, `PipelineAuthoringService`, function search, or
     execution services.
8. `docs/plans/mcp_cellprofiler_example_search_plan_20260629.md`
   - Improve searchable access to existing CellProfiler and native OpenHCS
     examples without adding a static MCP list.
9. `docs/plans/mcp_dev_client_ergonomics_plan_20260629.md`
   - Fix dev-client affordances by deriving command behavior from capability and
     command declarations, not by changing server APIs.
10. `docs/plans/mcp_agent_experience_implementation_dry_run_20260629.md`
    - Import and introspection dry run of the nominal authorities named by this
      plan set, including concrete implementation-time gaps found before coding.

## Cross-Cutting Rules

- Do not add MCP-local semantic dictionaries.
- Do not add filename parsing or `openhcs_metadata.json` reading to MCP.
- Do not add hardcoded function, module, artifact, or config inventories.
- Do not make dev-client UX friction a server API problem.
- Use class/type authority, dataclass fields, AutoRegisterMeta registries,
  compiled plans, and existing service APIs as the source of truth.
- String names are acceptable only as final ABI values emitted by typed
  declarations.

## Semantic Mirroring Audit Protocol

Every child plan must be auditable with the same four-question review:

1. What existing class, dataclass field, enum, registry, compiled plan, or
   service method owns the semantic fact?
2. Does the proposed MCP/agent code query that owner directly, or does it encode
   a parallel list, dictionary, string test, filename parser, or fallback?
3. If the owner changes, will the MCP output update automatically through type
   lookup, dataclass traversal, registry iteration, or compiled-plan projection?
4. Can a reviewer prove the answer with `rg` or AST inspection without knowing
   private implementation history?

Audit failure conditions:

- A new MCP or agent dictionary maps names to behavior that is already declared
  by a core, UI, CellProfiler, config, or capability class.
- A renderer decides domain behavior instead of formatting a DTO field.
- A DTO stores a semantic category that cannot be traced to an authority field or
  declaration.
- A helper has to be manually edited whenever a new function, module, artifact,
  source-binding type, config field, or capability is added.
- A test asserts copied behavior instead of proving projection from the owner.

Minimum evidence for each implementation:

```bash
rg -n "getattr\\(|hasattr\\(|\\.get\\(\"|\\bin \\{\" openhcs/mcp openhcs/agent
rg -n "SourceBindingOrigin|ArtifactSidecarRole|CallableContract|CompiledStepPlan|AgentCapabilityDeclaration|public_operation" openhcs/mcp openhcs/agent
```

The first command is a triage list, not an automatic failure. Each match must be
classified as transport/display plumbing, generic serialization, or semantic
mirroring. The second command must show imports/calls to authorities, not copied
surrogate declarations.

Additional gate for the new API-first lens:

```bash
rg -n "GeneratedMcp.*ToolBinding|Mcp.*ToolBindingABC|CapabilityCliConnectionProfile" openhcs/mcp openhcs/agent
```

Matches are allowed only while they are current migration scaffolding or final
transport projection mechanics. They must not own operation semantics that could
belong to a public operation declaration, request DTO, service, or renderer.

## Implementation Dry-Run Rule

Before implementing any child plan, run the dry-run import/introspection checks
in `docs/plans/mcp_agent_experience_implementation_dry_run_20260629.md` or an
updated equivalent. If the named authority does not import, does not expose the
expected registry, or only exposes narrower state than the plan assumes, patch
the plan first. Do not fill the gap with an MCP-side list.

## Completion Gates

The plan set is ready for implementation when each child plan:

- names its existing authority;
- names the projection point in `openhcs.agent` or `openhcs.mcp`;
- states what must not be mirrored;
- includes deterministic implementation steps;
- includes removal/search gates for old mirrors;
- includes focused tests or live MCP checks.
