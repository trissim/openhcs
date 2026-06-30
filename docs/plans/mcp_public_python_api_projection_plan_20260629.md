# MCP Public Python API Projection Plan

Date: 2026-06-29

## Purpose

Reinvestigate the MCP ergonomics plans under the stricter lens that the best
long-term agent surface is a public Python API first, with MCP, dev-client, and
knowledge prompts projected from that API.

This is not a request to make a second generic registry. The current MCP already
tried to be generic, and much of that work is valuable. The problem is that the
generic layer is still shaped around MCP capabilities and transport binding
families. The next refactor should promote the operation model into
`openhcs.agent` as the stable Python API boundary, then make MCP a mechanical
transport projection.

## Current Evidence

The current `openhcs.agent` package already declares its intent:

```text
Headless agent-facing API for OpenHCS.
This package owns the stable projection used by MCP, future CLIs, and automated
review agents. It intentionally avoids PyQt imports.
```

The current implementation has real generic infrastructure:

- `AgentCapabilityDeclaration(ABC, metaclass=AutoRegisterMeta)` is the current
  declaration authority for 79 agent-facing capabilities.
- `AgentCapabilitySpec` is a DTO projection, not the registration authority.
- `McpDevCommandSpec` and generated command profiles project CLI commands from
  capability declarations.
- `McpDevOutputRenderer` is keyed by output DTO type, not raw MCP tool names.
- `UiBridgeOperationContractABC` is an AutoRegisterMeta-backed operation
  contract registry for 26 UI bridge operations.
- `OpenHCSAgentContext` derives source-watch owner types from dataclass field
  annotations.

Implementation dry run from this checkout:

```text
AgentCapabilityDeclaration registry: 79 declarations
UI bridge operation contracts: 26 operations
MCP explicit binding leaves: 7
MCP generated binding families: 9
Capability invocation counts:
  config_patch_request: 2
  connection: 2
  connection_request: 23
  connection_scalar: 1
  dataclass_request: 12
  from_fields_request: 13
  no_invocation: 7
  scalar: 3
  viewer_request: 8
```

AST evidence for the current MCP transport exceptions:

```text
HealthCheckMcpToolBinding
UiListCodeDocumentsMcpToolBinding
UiListStateSurfacesMcpToolBinding
UiListActionsMcpToolBinding
UiListWindowsMcpToolBinding
UiGetWidgetTreeMcpToolBinding
ViewerProbeMcpToolBinding
```

Import-time caveat:

- broad registry discovery can still warn about a partial
  `openhcs.agent.dto` circular import before loading cached capability
  declarations. A public API operation surface should not depend on cached
  registry luck.

Reproduce the registry counts:

```bash
. .venv/bin/activate
python - <<'PY'
from collections import Counter
from openhcs.agent.capabilities import agent_capability_declarations
from openhcs.agent.capabilities import (
    AgentConnectionRequestServiceInvocation,
    AgentConnectionScalarServiceInvocation,
    AgentConnectionServiceInvocation,
    AgentConfigPatchServiceInvocation,
    AgentDataclassRequestServiceInvocation,
    AgentFromFieldsServiceInvocation,
    AgentScalarServiceInvocation,
    AgentViewerWindowRequestServiceInvocation,
)
from openhcs.agent.services.ui_bridge_service import UiBridgeOperationContractABC
from openhcs.mcp.server import (
    McpNoArgumentToolBindingABC,
    McpUiConnectionToolBindingABC,
    McpUiRequestToolBindingABC,
    McpScalarInputToolBindingABC,
    McpUiScalarInputToolBindingABC,
    McpConfigPatchToolBindingABC,
    McpFromFieldsToolBindingABC,
    McpDataclassRequestToolBindingABC,
    McpViewerRequestToolBindingABC,
)

invocation_attrs = (
    "no_argument_invocation",
    "connection_invocation",
    "connection_request_invocation",
    "connection_scalar_invocation",
    "scalar_invocation",
    "request_invocation",
)
invocation_types = {
    AgentConnectionRequestServiceInvocation: "connection_request",
    AgentConnectionScalarServiceInvocation: "connection_scalar",
    AgentConnectionServiceInvocation: "connection",
    AgentConfigPatchServiceInvocation: "config_patch_request",
    AgentDataclassRequestServiceInvocation: "dataclass_request",
    AgentFromFieldsServiceInvocation: "from_fields_request",
    AgentScalarServiceInvocation: "scalar",
    AgentViewerWindowRequestServiceInvocation: "viewer_request",
}
counts = Counter()
for declaration in agent_capability_declarations():
    matched = False
    for attr in invocation_attrs:
        value = next(
            (
                base.__dict__[attr]
                for base in declaration.__mro__
                if attr in base.__dict__
            ),
            None,
        )
        if value is None:
            continue
        matched = True
        for invocation_type, label in invocation_types.items():
            if isinstance(value, invocation_type):
                counts[label] += 1
    if not matched:
        counts["no_invocation"] += 1

print("capability declarations", len(agent_capability_declarations()))
print("invocation counts", dict(sorted(counts.items())))
print("ui bridge operations", len(UiBridgeOperationContractABC.__registry__))
for binding in (
    McpNoArgumentToolBindingABC,
    McpUiConnectionToolBindingABC,
    McpUiRequestToolBindingABC,
    McpScalarInputToolBindingABC,
    McpUiScalarInputToolBindingABC,
    McpConfigPatchToolBindingABC,
    McpFromFieldsToolBindingABC,
    McpDataclassRequestToolBindingABC,
    McpViewerRequestToolBindingABC,
):
    print(binding.__name__, len(binding.__registry__))
PY
```

Reproduce the explicit MCP binding-leaf inventory:

```bash
. .venv/bin/activate
python - <<'PY'
import ast
from pathlib import Path

tree = ast.parse(Path("openhcs/mcp/server.py").read_text())
binding_bases = {
    "McpNoArgumentToolBindingABC",
    "McpUiConnectionToolBindingABC",
    "McpUiRequestToolBindingABC",
    "McpScalarInputToolBindingABC",
    "McpUiScalarInputToolBindingABC",
    "McpConfigPatchToolBindingABC",
    "McpFromFieldsToolBindingABC",
    "McpDataclassRequestToolBindingABC",
    "McpViewerRequestToolBindingABC",
}
for cls in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
    bases = {ast.unparse(base).split("[", 1)[0] for base in cls.bases}
    if bases & binding_bases:
        print(cls.name, sorted(bases & binding_bases))
PY
```

## What The Current MCP Did Right

Keep these patterns:

- class declarations, not string tables, own exposed operation identity;
- `AgentCapabilitySpec` is a projection;
- DTO input/output classes own schema shape;
- service classes own business behavior;
- UI bridge operation contracts are headless and PyQt-free;
- dev-client commands and renderers are UX projections;
- output renderers bind to output DTO types;
- source freshness and context ownership derive from typed owners where
  possible.

The current generic MCP work should be treated as a successful intermediate
step, not thrown away.

## Non-Goal: Do Not Re-Add Existing Features

This plan is a refactor and projection plan, not a feature reimplementation
plan. Before adding any new operation, DTO, knowledge section, source model
view, compiled-plan view, UI route, dev-client command, or renderer, the
implementation must first prove whether the capability already exists.

Existing load-bearing features must be reused in place:

- capability declarations and exposition metadata;
- MCP-generated tool registration;
- dev-client command and renderer generation;
- `UiBridgeOperationContractABC` and UI bridge DTOs;
- ObjectState code-document read/validate/apply operations;
- selected-plate workflow operations;
- source-binding view models and source-inventory providers;
- compiler artifact-plan inspection;
- knowledge-base manifest documents, official30 examples, and native examples;
- function catalog signatures and callable/processing contracts.

Allowed work:

- rename or promote the existing owner so it becomes the public operation
  authority;
- add a small projection DTO when an existing authority has data but no bounded
  agent-facing view;
- add a test that proves an MCP/dev-client/knowledge surface is generated from
  the existing owner;
- move behavior out of MCP leaves into the existing DTO, service, operation, or
  renderer owner.

Disallowed work:

- a second list of operations, examples, CellProfiler modules, source-binding
  features, UI operations, or compiled-plan fields;
- a new MCP tool that duplicates an existing capability under a nicer name;
- a knowledge document that manually inventories operations instead of querying
  operation declarations;
- a source/artifact projection that parses or reconstructs facts already owned
  by the compiler, source-binding layer, or inventory layer;
- a UI route that bypasses ObjectState/code-document/bridge contracts when those
  contracts already cover the action.

Implementation rule:

```text
Inventory existing authority -> identify projection gap -> patch owner or
projection -> prove no duplicate authority was added.
```

## Remaining Architectural Problem

The current generic center is still capability/MCP-shaped:

- `AgentCapabilityDeclaration` owns tool/resource/prompt identity, CLI metadata,
  invocation slots, output contracts, and exposition metadata in one class.
- `openhcs.mcp.server` still has nine generated binding families selected by
  MCP transport shape.
- MCP explicit leaves still own response compaction or request-shape exceptions.
- `CapabilityCliConnectionProfile` is a CLI/MCP routing concept sitting on the
  same declaration that should be a public operation contract.
- There is no single public Python call surface that says: here is the stable
  OpenHCS operation, here is its typed request, here is its typed response, here
  is how to execute it from an `OpenHCSAgentContext`.

The result is generic, but from the transport inward. The target is generic from
the OpenHCS operation outward.

## Target Shape

Use `openhcs.agent` as the stable public operation boundary.

The public operation declaration should own:

- stable operation name;
- operation kind, such as tool, resource, or prompt projection;
- request DTO type, scalar contract, or no-input marker;
- response DTO type;
- public Python invocation object;
- mutation, side effects, data exposure, security, and runtime requirements;
- exposition metadata for first-use grouping and workflow ordering;
- optional transport projection hints when the transport cannot derive them from
  the request/invocation type.

MCP should project this declaration mechanically:

```text
OpenHCS semantic owners
    -> openhcs.agent public operation declarations and DTOs
    -> MCP transport signatures
    -> dev-client command/render UX
```

Do not expose arbitrary importable functions dynamically. Only curated public
operation declarations are API.

## Refactor Strategy

### 1. Promote Capability Declarations In Place

Do not add a parallel operation registry.

Refactor the current capability declaration system so the declaration is the
public operation authority. If a rename is worth doing, use a mechanical AST
rename from `AgentCapabilityDeclaration` to an operation-root name only after
the behavior is already correct. Until then, the current class can remain the
registry owner, but the code and docs should treat it as the public operation
declaration, and `AgentCapabilitySpec` should remain only a capability-list DTO
projection.

Required invariant:

```text
len(public_operation_declarations()) == len(agent_capability_declarations())
```

unless a declaration explicitly opts out of MCP exposure.

### 2. Collapse Invocation Slots Behind One Invocation Object

Current declarations have multiple mutually exclusive slots:

```text
no_argument_invocation
connection_invocation
connection_request_invocation
connection_scalar_invocation
scalar_invocation
request_invocation
```

These are already nominal, but the owner still has to know the whole family.
Move toward one declared `operation_invocation` value whose concrete type owns
the execution shape. Existing invocation classes can become that family instead
of being listed as separate slots.

MCP and dev-client then ask the invocation object for typed ABI parts:

- whether it needs a UI bridge connection;
- whether it needs viewer/runtime connection fields;
- which DTO factory or dataclass fields form the public request;
- which timeout profile applies;
- how to execute against `OpenHCSAgentContext`.

This should replace repeated generated-family selection in `server.py` with a
registry of transport projectors keyed by invocation type, not by capability
name.

### 3. Make MCP Binding A Transport Projector Registry

Replace the nine generated binding family loops in `openhcs.mcp.server` with:

1. iterate public operation declarations;
2. resolve one `McpOperationProjector` by operation invocation type and kind;
3. build the FastMCP function signature from the DTO/invocation authority;
4. call the operation invocation mechanically;
5. serialize the typed response.

The projector registry must be type keyed and AutoRegisterMeta-backed, for
example:

```text
AgentRequestServiceInvocation -> request DTO tool projector
AgentConnectionRequestServiceInvocation -> UI connection request projector
AgentViewerWindowRequestServiceInvocation -> viewer request projector
AgentScalarServiceInvocation -> scalar tool projector
```

No projector should know specific operation names. If it has to branch on a
specific name, the missing owner is the operation declaration, DTO, or service.

### 4. Move The Seven MCP Exceptions To Their Owners

Current explicit MCP binding leaves should be eliminated or justified by moving
their non-transport behavior to the public operation layer:

- `HealthCheckMcpToolBinding`
  - Health and stale-source behavior belongs to a health/status operation
    service or operation invocation, not an MCP leaf.
- `UiListCodeDocumentsMcpToolBinding`,
  `UiListStateSurfacesMcpToolBinding`, `UiListActionsMcpToolBinding`,
  `UiListWindowsMcpToolBinding`
  - Catalog identity flattening is output projection policy. Put it on the DTO
    projection or renderer, not MCP execution.
- `UiGetWidgetTreeMcpToolBinding`
  - `compact_actions` is a response projection option. Either it becomes a
    field on `UiWidgetTreeRequest` / a typed projection request, or it stays as
    dev-client rendering policy. It should not require a bespoke MCP execution
    class.
- `ViewerProbeMcpToolBinding`
  - Probe should be a normal viewer request DTO/operation whose connection-only
    request shape is declared by the viewer operation, not a binding exception.

Gate:

```bash
. .venv/bin/activate
python - <<'PY'
import ast
from pathlib import Path

tree = ast.parse(Path("openhcs/mcp/server.py").read_text())
binding_bases = {
    "McpNoArgumentToolBindingABC",
    "McpUiConnectionToolBindingABC",
    "McpUiRequestToolBindingABC",
    "McpScalarInputToolBindingABC",
    "McpUiScalarInputToolBindingABC",
    "McpConfigPatchToolBindingABC",
    "McpFromFieldsToolBindingABC",
    "McpDataclassRequestToolBindingABC",
    "McpViewerRequestToolBindingABC",
}
leaves = []
for cls in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
    bases = {ast.unparse(base).split("[", 1)[0] for base in cls.bases}
    if bases & binding_bases:
        leaves.append(cls.name)
print(leaves)
PY
```

Expected after the refactor: no capability-specific MCP binding leaves, or only
transport-policy leaves with a documented nominal owner.

### 5. Dev Client Projects Public Operations

The dev client should stop thinking in MCP tool terms except at the final
transport call.

Keep:

- `McpDevCommandSpec` as CLI UX declaration;
- output DTO renderer registry;
- command profiles for direct, UI bridge, viewer, and runtime routes.

Change:

- generated command profiles should select from public operation invocation
  type or transport profile, not from an MCP/capability-specific enum;
- CLI argument factories should be declared by request DTOs or invocation
  projectors;
- command help should use operation exposition metadata;
- multi-call workflow commands should name operation declarations, not MCP tool
  strings.

### 6. Knowledge And Authoring Context Project Operations

First-use guidance should be generated from the same public operation catalog
plus source-backed knowledge documents.

The knowledge layer can still write prose, but it must not keep a separate list
of available operations, CellProfiler examples, source-binding features, or UI
routes. It should query:

- public operation declarations for capability/tool availability;
- knowledge manifest specs for docs and example corpora;
- function catalog for registered callable signatures;
- compiled plans for source/artifact/runtime facts.

### 7. Compiled Context Discovery Belongs In Public Operations

If agents need compiled-context visibility, add public operations that expose
bounded compiled plan DTOs. Do not let MCP inspect compiler internals directly.

Nominal authorities to project:

- `CompiledStepPlan`
- `FunctionStepExecutionPlan`
- `CompiledSourceUniversePlan`
- `SourceLoadPlan`
- `ArtifactInputPlan`
- `ArtifactOutputPlan`
- runtime sidecar roles and materialization preview records

The operation should execute compiler/inspection services and return bounded
DTOs. MCP should only serialize the DTO.

## Plan Set Reinterpretation

The existing MCP plan set remains useful, but its authority changes:

- `mcp_frontdoor_core_model_plan_20260629.md`
  - supplies the conceptual knowledge content.
- `mcp_tool_surface_grouping_plan_20260629.md`
  - grouping metadata should live on public operation declarations.
- `mcp_capability_tool_binding_refactor_20260629.md`
  - becomes the migration history from manual capability tables to declaration
    authority.
- `mcp_ui_bridge_authority_refactor_20260629.md`
  - UI bridge operation contracts become one family of public operation
    invocations.
- `mcp_dev_client_schema_derivation_refactor_20260629.md`
  - dev-client projection remains valid, but its source should become public
    operation declarations instead of MCP capabilities.
- Source model, artifact, VFS, sidecar, function-contract, example-search, and
  beginner workflow plans define operation families and acceptance criteria, not
  MCP-local features.

## Implementation Order

1. Add a public operation registry accessor in `openhcs.agent`.
   - It may initially return `AgentCapabilityDeclaration.__registry__.values()`.
   - It must be documented as the public operation authority.
   - It must not create a second registry.

2. Add an operation catalog DTO separate from capability-list compatibility.
   - Include request/response contract names, operation kind, side effects,
     target context, workflow group/stage, and invocation shape.
   - Generate the existing capability registry from it.

3. Refactor `server.py` projector selection.
   - Introduce type-keyed MCP operation projectors.
   - Move generated family logic into projector classes.
   - Keep old family names only until the new projector tests pass, then delete
     them in one batch.

4. Move the seven explicit MCP leaves to their real owners.
   - Health to health service/operation.
   - Catalog flattening to DTO projection/render policy.
   - Widget compaction to request/projection policy.
   - Viewer probe to a viewer request DTO.

5. Refactor dev-client profiles to use public operation declarations.
   - Preserve command UX.
   - Remove capability/MCP-specific routing from command generation.

6. Update knowledge and first-use context to describe public operations.
   - Keep all conceptual prose source-backed.
   - Link operation groups to knowledge docs by operation exposition metadata.

7. Run import-order and stale-cache tests.
   - Public operation discovery must succeed in a fresh interpreter without
     relying on registry cache fallback.

## Semantic Mirroring Audit Gates

Run these before implementation and after each batch:

```bash
rg -n "AgentCapabilityDeclaration|agent_capability_declarations|CapabilityCliConnectionProfile" openhcs/mcp openhcs/agent
rg -n "GeneratedMcp.*ToolBinding|Mcp.*ToolBindingABC" openhcs/mcp/server.py
rg -n "if .*capability|if .*declaration|isinstance\\(.*Agent.*Invocation" openhcs/mcp openhcs/agent
rg -n "openhcs_[a-z0-9_]+" openhcs/mcp openhcs/agent/services openhcs/agent/dto
```

Expected disposition:

- final ABI strings only on public operation declarations;
- transport projectors keyed by invocation type, not operation name;
- dev-client commands reference operation declarations or output DTOs;
- no MCP-local operation behavior tables;
- no knowledge-layer hand lists of operation names;
- no import warning during public operation discovery.

## Tests

Focused tests after each step:

```bash
XDG_CACHE_HOME=/tmp/openhcs-test-cache XDG_DATA_HOME=/tmp/openhcs-test-data \
  .venv/bin/python -m pytest \
  tests/unit/agent/test_capabilities.py \
  tests/unit/agent/test_mcp_server.py \
  tests/unit/agent/test_agent_services.py \
  tests/unit/agent/test_ui_bridge_service.py
```

Add tests for the new public operation boundary:

- public operation declarations import cleanly in a fresh interpreter;
- capability registry is projected from public operations;
- every MCP tool/resource is generated from exactly one public operation;
- no capability-specific MCP binding leaf exists without a documented
  operation-owner reason;
- dev-client command generation uses public operation declarations;
- first-use authoring context references operation groups generated from the
  public operation catalog.
