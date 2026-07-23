# MCP Dev Client Schema Derivation Refactor

Date: 2026-06-29

## Problem

`openhcs/mcp/dev_client.py` is a useful fresh-process test harness, but it has
grown into a second schema and tool registry:

- `McpDevCommandName` enumerates every command.
- many `SingleToolCommandSpec` leaves own raw `tool_name = "openhcs_*"`
  strings;
- command specs manually repeat argument defaults and choices already present in
  DTO request classes or backend declarations;
- renderer classes bind to raw tool names and parse response dicts directly;
- UI bridge and viewer command families repeat tool names again near the bottom
  of the file.

The dev client should be ergonomic, but it must not become the authority for MCP
tool identity, request schema, enum choices, defaults, or response shape.

## Existing Authorities

- Capability declarations should own tool identity and input/output contracts
  after `mcp_capability_tool_binding_refactor_20260629.md`.
- DTO request dataclasses own request fields and default values.
- Backend declarations own enum choices and bounds:
  - `AllComponents` for component axes;
  - `PlateFileKind` and query profiles for file kinds;
  - synthetic generation profile for synthetic defaults/bounds;
  - UI bridge operation contracts for UI bridge request DTOs.
- DTO response dataclasses own response field names.
- Renderers are CLI presentation policy; they may format typed payloads, but
  they should not define payload semantics.

## Target Shape

Keep `McpDevCommandSpec` as an AutoRegisterMeta CLI command family, but make
commands reference capability declarations.

For single-tool commands:

```python
class ListKnowledgeDocumentsCapability(AgentCapabilityDeclaration):
    name = "openhcs_list_knowledge_documents"
    cli_command = "knowledge"

class KnowledgeCommandSpec(SingleToolCommandSpec):
    capability = agent_capabilities.list_knowledge_documents
```

`SingleToolCommandSpec` derives:

- CLI command and aliases from `capability.cli_command` /
  `capability.cli_aliases`;
- MCP tool name from `capability.name`;
- default timeout from capability metadata if present;
- request DTO from `capability.input_contract`;
- output DTO from `capability.output_contract`.

Argument parsing can remain hand-tuned where UX needs aliases or compact flags,
but field choices/defaults must be read from DTO/backend declarations.

Response rendering should bind to output DTO type or capability declaration:

```python
class KnowledgeCatalogRenderer(McpResponseRenderer):
    output_contract = KnowledgeBaseCatalog
```

Avoid renderer lookup by raw tool-name strings.

## Deterministic Steps

1. Add capability references to command specs.
   - Add `capability: ClassVar[type[AgentCapabilityDeclaration]]` to
     `SingleToolCommandSpec`.
   - Delete `tool_name` from all single-tool command leaves.
   - Derive calls from `self.capability.name`.

2. Replace raw tool-name renderer registry.
   - Change renderer registry keys from `tool_name` to output DTO type.
   - Update `CallCommandSpec` to find capability by name through the capability
     registry, then renderer by the capability's output DTO type.

3. Derive choices/defaults.
   - For DTO-backed commands, add a small argparse projection helper that reads
     dataclass defaults and enum annotations.
   - Where UX uses a different flag name, bind the flag to a request DTO field
     explicitly in the command spec.
   - Do not duplicate backend enums as literal `choices=(...)` if the enum can
     be imported.

4. Split large renderers only after schema authority is fixed.
   - The file is large, but do not split it before removing mirrored schema
     authority. Splitting first would hide the mirror across more modules.
   - Once capability/DTO-driven, renderers can move to
     `openhcs/mcp/dev_client_renderers.py` without changing semantics.

5. Add consistency tests.
   - Every `SingleToolCommandSpec` has a capability.
   - No command spec leaf has a `tool_name` class attribute.
   - Every renderer points to an output DTO or capability declaration.
   - Every command-generated request can be hydrated into the declared request
     DTO before sending through MCP.

## AST Removal Gates

```bash
rg -n "tool_name: ClassVar|tool_name = \"openhcs_|for_tool_name|__registry_key__ = \"tool_name\"" openhcs/mcp/dev_client.py
rg -n "choices=\\(|default=.*SyntheticPlateGenerationDefaults|PlateFileQueryKind|UiSelectedPlateWorkflowKind" openhcs/mcp/dev_client.py
```

Expected result:

- no command leaf owns a raw MCP tool name;
- no renderer registry is keyed by raw MCP tool name;
- enum choices and defaults are imported from the semantic owner or request DTO.

## Tests

```bash
XDG_CACHE_HOME=/tmp/openhcs-test-cache .venv/bin/python -m pytest \
  tests/unit/agent/test_mcp_server.py -k dev_client \
  tests/unit/agent/test_capabilities.py
```

If no dev-client-specific unit test exists after the refactor, add one before
landing the production changes. The dev client is now large enough that manual
smoke checks are not adequate.

## Implementation Progress

2026-06-29 partial implementation:

- `SingleToolCommandSpec` and `SingleUiBridgeToolCommandSpec` now declare an
  `AgentCapabilitySpec` and derive the MCP tool name from
  `capability.name`.
- `McpDevOutputRenderer` is registered by output DTO type instead of by a raw
  `tool_name` registry key.
- Command and renderer leaves no longer declare `tool_name = "openhcs_*"`.

Verified gate:

```bash
rg -n "tool_name: ClassVar|tool_name = \"openhcs_|for_tool_name|__registry_key__ = \"tool_name\"" openhcs/mcp/dev_client.py
```

The command currently returns no matches.

Verified tests:

```bash
XDG_CACHE_HOME=/tmp/openhcs-test-cache XDG_DATA_HOME=/tmp/openhcs-test-data \
  .venv/bin/python -m pytest \
  tests/unit/agent/test_mcp_server.py \
  -k "dev_client and not launches_fresh_current_source_server"
```

This selected 168 tests and all passed. The remaining fresh-process stdio
smoke still times out during MCP `initialize`, even with a 30 second timeout.

Still open:

- Command argument defaults are mostly derived from DTO/backend declarations,
  but this should stay under review as new command flags are added.
- Fresh-process stdio initialization still needs separate transport/startup
  investigation.

2026-06-29 schema authority follow-up:

- Dev-client command and renderer leaves now reference generated
  `agent_capabilities.<name>` objects instead of raw MCP tool-name strings.
- Multi-call command specs and renderer payload lookups use
  `agent_capabilities.<name>.name` when they need the final ABI string.
- `SelectedAllSelectionMode` owns `selected` / `all` parser choices.
- `WidgetTreeOutputFormat` owns the widget-tree `json` / `outline` CLI
  presentation choices.
- The only remaining `"openhcs_*"` literal in `openhcs/mcp/dev_client.py` is
  `openhcs_format`, a synthetic generation request field, not a tool name.

Verified gates:

```bash
rg -n "get_agent_capability\\(\"openhcs_|tool_name: ClassVar|tool_name = \"openhcs_|for_tool_name|__registry_key__ = \"tool_name\"|choices=\\(" \
  openhcs/mcp/dev_client.py
# no matches

rg -n "\"openhcs_[a-z0-9_]+\"" openhcs/mcp/dev_client.py
# one match: "openhcs_format" request field
```

Verified tests:

```bash
XDG_CACHE_HOME=/tmp/openhcs-test-cache .venv/bin/python -m pytest \
  tests/unit/agent/test_mcp_server.py \
  -k "dev_client and not launches_fresh_current_source_server"
# 168 passed, 66 deselected
```

2026-06-29 command-axis cleanup:

- Removed the central `McpDevCommandName` enum. Command names are now final CLI
  ABI strings owned by each `McpDevCommandSpec` declaration and keyed directly
  by the `McpDevCommandSpec` AutoRegisterMeta registry.
- `McpDevCommandSpec.for_name()`, parser registration, and help fallback now use
  that declaration-owned string key mechanically.
- `UiStateSurfacePayloadRenderer.for_payload()` now resolves the payload's
  `surface_id` through the `UiBridgeIdentityDeclaration` registry, then looks up
  the renderer by identity declaration class. It no longer loops over renderers
  and compares identity `.value` strings.

Verified gates:

```bash
.venv/bin/python - <<'PY'
import ast
from pathlib import Path

tree = ast.parse(Path("openhcs/mcp/dev_client.py").read_text())
print(any(
    isinstance(node, ast.ClassDef) and node.name == "McpDevCommandName"
    for node in ast.walk(tree)
))
print(sum(
    1
    for node in ast.walk(tree)
    if isinstance(node, ast.Attribute)
    and node.attr == "value"
    and isinstance(node.value, ast.Attribute)
    and node.value.attr == "command"
))
PY
# False
# 0
```

Verified tests:

```bash
.venv/bin/python -m pytest \
  tests/unit/agent/test_mcp_server.py \
  -k "dev_client and not launches_fresh_current_source_server"
# 168 passed, 69 deselected

.venv/bin/python -m pytest \
  tests/unit/agent/test_mcp_server.py \
  tests/unit/agent/test_capabilities.py \
  tests/unit/agent/test_ui_bridge_service.py \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py -q
# 308 passed
```

2026-06-29 direct DTO renderer keys:

- Renderer declarations now assign `output_contract = agent_dto.<ResultDTO>`.
- No renderer declaration binds through `agent_capabilities.<tool>.output_contract`.
- `CapabilityBackedCommandSpec.render_response()` still uses the selected
  capability's output contract to find the renderer, but the renderer registry
  itself is DTO-owned.

Verified gates:

```bash
rg -n "output_contract = agent_capabilities" openhcs/mcp/dev_client.py
# no matches
```

Verified tests:

```bash
.venv/bin/python -m pytest \
  tests/unit/agent/test_mcp_server.py \
  -k "dev_client and not launches_fresh_current_source_server"
# 168 passed, 69 deselected
```

2026-06-29 generated simple command specs:

- Added `GeneratedSingleToolCommandSpec` for capabilities that declare
  `cli_command` but do not have an explicit `CapabilityBackedCommandSpec`.
- Removed the explicit health command leaf. `health` is now projected directly
  from `HealthCheckCapability.cli_command` and the generic single-tool command
  mechanics.
- `McpDevCommandSpec.all_specs()` and `for_name()` merge explicit
  AutoRegisterMeta command classes with generated capability-backed specs.
- Generated specs now cover `AgentScalarInputContract` capabilities. This
  removed explicit `ArchitectureTopicCommandSpec` and
  `InternalSymbolCommandSpec`; their positional argument name/default comes from
  the scalar input contract.
- `CapabilityBackedCommandSpec.for_capability_name()` falls back to generated
  command specs, so `call openhcs_explain_architecture` still uses the compact
  output renderer.

Verified gates:

```bash
.venv/bin/python - <<'PY'
from openhcs.mcp import dev_client

print(any(
    cls.__name__ == "GeneratedSingleToolCommandSpec"
    for cls in dev_client.McpDevCommandSpec.__registry__.values()
))
print([
    (type(spec).__name__, spec.command, spec.capability.name)
    for spec in dev_client.McpDevCommandSpec.all_specs()
    if type(spec).__name__ == "GeneratedSingleToolCommandSpec"
])
PY
# False
# [('GeneratedSingleToolCommandSpec', 'health', 'openhcs_health_check')]
```

After scalar input generation the generated set is:

```text
health -> openhcs_health_check
architecture-topic -> openhcs_explain_architecture
internal-symbol -> openhcs_describe_internal_symbol
```

Verified tests:

```bash
.venv/bin/python -m pytest \
  tests/unit/agent/test_mcp_server.py \
  -k "dev_client and not launches_fresh_current_source_server"
# 168 passed, 69 deselected
```

2026-06-29 capability-owned CLI command metadata:

- Added `cli_command` and `cli_aliases` to `AgentCapabilityDeclaration` and
  `AgentCapabilitySpec`.
- Moved the 51 capability-backed dev-client command names and 6 alias tuples
  onto the matching capability declarations.
- `McpDevCommandSpec` now has an AutoRegisterMeta key extractor that registers
  explicit custom commands from their local `command` and capability-backed
  commands from `capability.cli_command`.
- Capability-backed command leaves no longer declare local `command` or
  `aliases`; they keep only parser/projection behavior that is genuinely
  dev-client UX.
- `health` now uses the same `SingleToolCommandSpec` path as other single MCP
  tool commands, with `cli_command = "health"` declared on
  `HealthCheckCapability`.

Verified gates:

```bash
.venv/bin/python - <<'PY'
import ast
from pathlib import Path

tree = ast.parse(Path("openhcs/mcp/dev_client.py").read_text())
command_assignments = []
alias_assignments = []
for node in tree.body:
    if not (isinstance(node, ast.ClassDef) and node.name.endswith("CommandSpec")):
        continue
    has_capability = any(
        isinstance(stmt, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "capability" for target in stmt.targets)
        for stmt in node.body
    )
    if not has_capability:
        continue
    for stmt in node.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == "command":
                    command_assignments.append(node.name)
                if isinstance(target, ast.Name) and target.id == "aliases":
                    alias_assignments.append(node.name)
print(command_assignments)
print(alias_assignments)
PY
# []
# []
```

Verified tests:

```bash
.venv/bin/python -m pytest \
  tests/unit/agent/test_mcp_server.py \
  -k "dev_client and not launches_fresh_current_source_server"
# 168 passed, 69 deselected

.venv/bin/python -m pytest \
  tests/unit/agent/test_mcp_server.py \
  tests/unit/agent/test_capabilities.py \
  tests/unit/agent/test_ui_bridge_service.py \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py -q
# 308 passed
```

2026-06-29 generated UI-bridge single-tool commands:

- Added a typed `CapabilityCliConnectionProfile` to capability declarations and
  specs. This keeps CLI connection mechanics declaration-owned instead of
  deriving them from command names, service strings, or runtime-requirement
  labels.
- Added `UiBridgeCliConnectionCapability` for capability declarations whose
  generated CLI command must accept UI bridge connection options.
- The UI bridge connection profile is now inherited by all capability
  declarations whose explicit dev-client command currently calls
  `add_ui_connection_options`, including selected-plate UI commands. This makes
  the connection requirement queryable from the capability registry even while
  those commands still own custom parser/projection UX.
- Added `GeneratedUiBridgeToolCommandSpec` so no-input UI bridge commands can
  be projected from capability declarations while preserving the `connection`
  payload shape.
- `GeneratedMcpDevCommandProfile` now uses AutoRegisterMeta dispatch keyed by
  `CapabilityCliConnectionProfile`, so generated command selection is
  declaration/profile driven instead of an `if` branch over profile values.
- Removed no-behavior command leaves for `ui-status`, `windows`,
  `code-documents`, and `state-surfaces`.
- Added output DTO-owned compact renderer keys for viewer navigation, viewer
  layer isolation, runtime server scan, runtime server info, and runtime
  execution status. The dev-client renderer registry now uses a neutral
  `renderer_key` extractor so these renderers resolve from DTO-declared compact
  keys instead of direct output-contract adapter leaves.
- Extended `CapabilityCliConnectionProfile` to cover viewer-window and
  runtime-server CLI connection mechanics. Viewer-window and runtime-server
  capability declarations now inherit those profiles instead of leaving the
  connection requirement encoded only in dev-client parser classes.
- Added generated command profile leaves for viewer-window and runtime-server
  profiles so the generated command profile registry is exhaustive over the
  declared connection profile enum.
- Moved MCP tool argument projection for `SyntheticPlateGenerationRequest`,
  `PlatePathInspectionRequest`, `PlateFileQueryRequest`, and
  `PlateImageSampleRequest` onto the DTO classes via `as_tool_arguments()`.
  The dev-client commands now call the request DTO projection instead of
  copying normalized fields, enum values, and nested bounds back into JSON.

Verified gates:

```bash
.venv/bin/python - <<'PY'
from openhcs.mcp import dev_client

print([
    (type(spec).__name__, spec.command, spec.capability.name)
    for spec in dev_client.McpDevCommandSpec.all_specs()
    if type(spec).__name__ == "GeneratedUiBridgeToolCommandSpec"
])
PY
# ui-status, code-documents, state-surfaces, and windows are generated
```

2026-06-29 plate request DTO projection completion:

- Added MCP argument projection to `PlateFileStreamRequest`,
  `SelectedPlateImageInspectionRequest`, `SelectedPlateFileQueryRequest`,
  `SelectedPlateImageSampleRequest`, and `SelectedPlateFileStreamRequest`.
- Removed the dev-client-local `PlateFileStreamToolArguments`,
  `DirectPlateFileStreamToolArguments`, and
  `SelectedPlateFileStreamToolArguments` records. Direct and selected stream
  commands now hydrate the declared request DTOs and call `as_tool_arguments()`.
- Added `add_request_field_option()` usage for selected-plate inspect/query/
  sample commands and shared stream options so parser defaults and primitive
  types come from request DTO `from_fields` signatures.
- Kept the explicit-path stream CLI policy local to the command layer:
  omitted `--kind` resolves to `all` when explicit file paths are supplied,
  otherwise it resolves to the request DTO's declared `kind` default.
- Inverse sample flags (`--no-array-values` / `--include-array-values`) now
  derive their default from the DTO `include_array_values` field instead of a
  literal boolean.

Verified gates:

```bash
source .venv/bin/activate && python -m py_compile \
  openhcs/mcp/dev_client.py openhcs/agent/dto/plate.py

source .venv/bin/activate && python - <<'PY'
import ast
from pathlib import Path

tree = ast.parse(Path("openhcs/mcp/dev_client.py").read_text())
classes = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
targets = {
    "GenerateSyntheticPlateCommandSpec",
    "InspectPlateCommandSpec",
    "QueryPlateFilesCommandSpec",
    "SamplePlateImageCommandSpec",
    "SelectedPlateImagesCommandSpec",
    "SelectedPlateFilesCommandSpec",
    "SelectedPlateSampleCommandSpec",
    "StreamPlateFilesCommandSpec",
    "SelectedPlateStreamCommandSpec",
}
manual_literals = []
for class_name in sorted(targets):
    for child in ast.walk(classes[class_name]):
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "add_argument"
        ):
            for keyword in child.keywords:
                if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                    manual_literals.append((class_name, child.lineno, keyword.value.value))
print("target_literal_add_argument_defaults", manual_literals)
PY
# target_literal_add_argument_defaults []
```

Verified tests:

```bash
XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest \
  tests/unit/agent/test_plate_inspection_service.py::test_plate_request_dtos_own_mcp_tool_argument_projection -q
# 1 passed

XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest tests/unit/agent/test_mcp_server.py \
  -k "plate_files_command_projects_tool_arguments or generate_synthetic_plate_command_projects_tool_arguments or inspect_plate or query_plate_files or sample_plate_image_command_projects_tool_arguments or selected_plate_files_command_projects_tool_arguments or selected_plate_images_command_projects_tool_arguments or selected_plate_sample_command_projects_tool_arguments or selected_plate_stream_command_projects_tool_arguments" -q
# 19 passed, 219 deselected

XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest \
  tests/unit/agent/test_mcp_server.py \
  tests/unit/agent/test_capabilities.py \
  tests/unit/agent/test_ui_bridge_service.py \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py \
  tests/unit/agent/test_plate_inspection_service.py -q
# 326 passed
```

2026-06-29 generated request CLI and knowledge command cleanup:

- Added the shared `AgentCliArgumentSpec` / `AgentCliRequest` request interface
  in `openhcs.agent.dto.common`. Generated CLI commands can now ask the request
  DTO for its parser factory, argument shape, and final MCP tool arguments.
- `GeneratedSingleToolCommandSpec` and `GeneratedRuntimeServerToolCommandSpec`
  now share the same `AgentCliRequest` parser/projection path. Runtime-server
  commands only override the runtime connection field splice; they no longer
  own separate argument-spec projection helpers.
- `KnowledgeBaseDocumentRequest` owns the `document_id#section_id` CLI split,
  `--section-id`, `--max-chars`, and final `as_tool_arguments()` payload.
- `KnowledgeBaseSearchRequest` owns positional/repeated query joining,
  `--query`, `--limit`, and the missing-query validation.
- Removed explicit `KnowledgeDocumentCommandSpec` and
  `KnowledgeSearchCommandSpec`. Both commands are now generated from capability
  declarations and request DTO contracts.
- Removed dev-client-local knowledge default constants; tests now assert the
  default through `KnowledgeBaseDocumentRequest`.
- Added `docs/source/development/ast_refactoring_workflow.rst` operational
  discipline notes for active shared refactors: AST inventory/gates drive the
  batch, patches remain the reviewed source edit.

Verified gates:

```bash
source .venv/bin/activate && python - <<'PY'
import ast
from pathlib import Path

client = ast.parse(Path("openhcs/mcp/dev_client.py").read_text())
knowledge_command_classes = [
    node.name
    for node in client.body
    if isinstance(node, ast.ClassDef)
    and node.name in {"KnowledgeDocumentCommandSpec", "KnowledgeSearchCommandSpec"}
]
local_knowledge_defaults = []
for node in client.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.startswith("DEFAULT_KNOWLEDGE_"):
                local_knowledge_defaults.append((target.id, node.lineno))

knowledge = ast.parse(Path("openhcs/agent/dto/knowledge.py").read_text())
request_bases = {}
for node in knowledge.body:
    if isinstance(node, ast.ClassDef) and node.name in {
        "KnowledgeBaseDocumentRequest",
        "KnowledgeBaseSearchRequest",
    }:
        request_bases[node.name] = [ast.unparse(base) for base in node.bases]

print("knowledge_command_classes", knowledge_command_classes)
print("local_knowledge_defaults", local_knowledge_defaults)
print("knowledge_request_bases", request_bases)
PY
# knowledge_command_classes []
# local_knowledge_defaults []
# knowledge_request_bases {'KnowledgeBaseDocumentRequest': ['AgentCliRequest'], 'KnowledgeBaseSearchRequest': ['AgentCliRequest']}

source .venv/bin/activate && python - <<'PY'
from openhcs.mcp import dev_client
for command in (
    "knowledge-document",
    "knowledge-search",
    "runtime-scan",
    "runtime-info",
    "runtime-status",
):
    spec = dev_client.McpDevCommandSpec.for_name(command)
    print(command, type(spec).__name__, spec.capability.name)
PY
# knowledge-document GeneratedSingleToolCommandSpec openhcs_get_knowledge_document
# knowledge-search GeneratedSingleToolCommandSpec openhcs_search_knowledge
# runtime-scan GeneratedRuntimeServerToolCommandSpec openhcs_scan_runtime_servers
# runtime-info GeneratedRuntimeServerToolCommandSpec openhcs_get_runtime_server_info
# runtime-status GeneratedRuntimeServerToolCommandSpec openhcs_get_runtime_server_execution_status
```

Verified tests:

```bash
source .venv/bin/activate && python -m py_compile \
  openhcs/agent/dto/common.py \
  openhcs/agent/dto/knowledge.py \
  openhcs/agent/dto/execution.py \
  openhcs/mcp/dev_client.py \
  tests/unit/agent/test_mcp_server.py

XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest tests/unit/agent/test_mcp_server.py \
  -k "knowledge or runtime_scan or runtime_info or runtime_status" -q
# 13 passed, 225 deselected

XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest tests/unit/agent/test_knowledge_base_service.py -q
# 16 passed

XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest \
  tests/unit/agent/test_mcp_server.py \
  tests/unit/agent/test_capabilities.py \
  tests/unit/agent/test_ui_bridge_service.py \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py \
  tests/unit/agent/test_plate_inspection_service.py \
  tests/unit/agent/test_knowledge_base_service.py -q
# 342 passed
```

2026-06-29 viewer snapshot/state/payload DTO authority follow-up:

- Added DTO-owned `from_fields()` and `as_tool_arguments()` projection for
  `ViewerWindowSnapshotRequest`, `ViewerWindowStateRequest`, and
  `ViewerWindowPayloadRequest`.
- Moved `ViewerSnapshotMcpToolBinding`, `ViewerStateMcpToolBinding`, and
  `ViewerPayloadsMcpToolBinding` to `McpViewerRequestToolBindingABC`
  DTO-derived request signatures and request hydration.
- Removed the dev-client `ViewerPayloadArguments` mirror; `viewer-payloads`,
  `viewer-state`, and `snapshot-viewer` now hydrate the request DTO and emit
  MCP tool arguments through the DTO projection.
- `ViewerWindowControlRequest` now exposes its factory-injected field names from
  the dataclass hierarchy, so the server helper does not need to know which
  viewer fields are transport/timeout plumbing.
- `viewer-state` keeps two intentional DTO constructors: compact MCP tool
  defaults on `from_fields()` and richer human CLI defaults on
  `from_cli_fields()`. Both delegate field validation to
  `ViewerStateControlOptions`.

Verified gates:

```bash
source .venv/bin/activate && python -m py_compile \
  openhcs/agent/dto/viewer.py openhcs/mcp/server.py openhcs/mcp/dev_client.py

python - <<'PY'
from pathlib import Path
import ast

files = ["openhcs/mcp/server.py", "openhcs/mcp/dev_client.py"]
for file in files:
    tree = ast.parse(Path(file).read_text())
    print(f"## {file}")
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if "Viewer" not in node.name or not (
            node.name.endswith("McpToolBinding") or node.name.endswith("CommandSpec")
        ):
            continue
        smells = []
        for child in node.body:
            if not isinstance(child, ast.FunctionDef):
                continue
            for sub in ast.walk(child):
                if isinstance(sub, ast.Call):
                    fn = sub.func
                    name = (
                        fn.id
                        if isinstance(fn, ast.Name)
                        else fn.attr
                        if isinstance(fn, ast.Attribute)
                        else None
                    )
                    if name in {"Parameter", "option_parameters", "option_arguments"}:
                        smells.append((child.name, name, sub.lineno))
                if isinstance(sub, ast.Dict):
                    smells.append((child.name, "dict", sub.lineno))
        if smells:
            print(node.name, smells)
PY
# no viewer binding smells printed

python - <<'PY'
from pathlib import Path
import ast

server = ast.parse(Path("openhcs/mcp/server.py").read_text())
handwritten = [
    (node.name, node.lineno)
    for node in server.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    and node.name.startswith("openhcs_")
]
print("handwritten_openhcs_funcs", handwritten)
PY
# handwritten_openhcs_funcs []

git diff --check
```

Verified tests:

```bash
XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest tests/unit/agent/test_mcp_server.py \
  -k "viewer_payloads or get_viewer_window_payloads or viewer_state or get_viewer_window_state or snapshot_viewer or viewer_snapshot or viewer_tool_schema or validate_viewer or navigate_viewer or isolate_viewer or sample_viewer_image or viewer_rois" -q
# 44 passed, 194 deselected

XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest \
  tests/unit/agent/test_mcp_server.py \
  tests/unit/agent/test_capabilities.py \
  tests/unit/agent/test_ui_bridge_service.py \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py \
  tests/unit/agent/test_plate_inspection_service.py -q
# 326 passed
```

2026-06-29 compact renderer key removal:

- Removed `AgentCompactRendererKey` and `AgentCompactRenderableOutput`; compact
  presentation no longer requires DTO-side renderer-key declarations.
- Rebound the five affected dev-client renderers directly to their output DTO
  types:
  `ViewerWindowNavigationResult`, `ViewerWindowLayerIsolationResult`,
  `RuntimeServerScanResult`, `RuntimeServerInfo`, and
  `RuntimeExecutionStatus`.
- Simplified `McpDevOutputRenderer` lookup to use only the output contract
  registry. The output DTO class is now the renderer authority for those
  result shapes; there is no parallel compact-key namespace.

Verified gates:

```bash
source .venv/bin/activate && python -m py_compile \
  openhcs/agent/dto/common.py openhcs/agent/dto/execution.py \
  openhcs/agent/dto/viewer.py openhcs/mcp/dev_client.py

rg -n "AgentCompactRenderableOutput|AgentCompactRendererKey|compact_renderer_key" \
  openhcs tests
# no matches

source .venv/bin/activate && python - <<'PY'
from openhcs.mcp.dev_client import McpDevOutputRenderer
from openhcs.agent import dto as agent_dto

for contract in (
    agent_dto.ViewerWindowNavigationResult,
    agent_dto.ViewerWindowLayerIsolationResult,
    agent_dto.RuntimeServerScanResult,
    agent_dto.RuntimeServerInfo,
    agent_dto.RuntimeExecutionStatus,
):
    renderer = McpDevOutputRenderer.for_output_contract(contract)
    print(contract.__name__, renderer.__name__ if renderer else None)
PY
# ViewerWindowNavigationResult ViewerWindowNavigationResultRenderer
# ViewerWindowLayerIsolationResult ViewerWindowLayerIsolationResultRenderer
# RuntimeServerScanResult RuntimeServerScanRenderer
# RuntimeServerInfo RuntimeServerInfoRenderer
# RuntimeExecutionStatus RuntimeExecutionStatusRenderer
```

Verified tests:

```bash
XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest tests/unit/agent/test_mcp_server.py \
  -k "navigate_viewer or isolate_viewer or runtime_server or runtime_execution or scan_runtime or get_runtime or viewer_navigation" -q
# 7 passed, 231 deselected
```

2026-06-29 generated viewer probe command follow-up:

- Removed the explicit `ProbeViewerCommandSpec` class. Its behavior was exactly
  the generated viewer-window profile: viewer port/connection options plus
  JSON rendering.
- `ProbeViewerWindowCapability` now provides the CLI command/profile authority,
  and `GeneratedViewerWindowToolCommandSpec` supplies the command mechanics.

Verified gates:

```bash
source .venv/bin/activate && python -m py_compile openhcs/mcp/dev_client.py

source .venv/bin/activate && python - <<'PY'
from openhcs.mcp import dev_client

spec = dev_client.McpDevCommandSpec.for_name("probe-viewer")
print(type(spec).__name__, spec.capability.name)
parser = dev_client._build_parser()
args = parser.parse_args(("probe-viewer", "5555", "--timeout-ms", "1000"))
call = dev_client._calls_from_args(args)[0]
print(call.name, call.arguments)
PY
# GeneratedViewerWindowToolCommandSpec openhcs_probe_viewer_window
# openhcs_probe_viewer_window {'port': 5555, 'host': 'localhost', 'timeout_ms': 1000}

python - <<'PY'
from pathlib import Path
import ast

tree = ast.parse(Path("openhcs/mcp/dev_client.py").read_text())
print([
    node.name
    for node in ast.walk(tree)
    if isinstance(node, ast.ClassDef) and node.name == "ProbeViewerCommandSpec"
])
PY
# []
```

Verified tests:

```bash
XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest tests/unit/agent/test_mcp_server.py \
  -k "probe_viewer" -q
# 2 passed, 236 deselected
```

2026-06-29 runtime command DTO projection follow-up:

- Added `RuntimeServerToolRequest` as the nominal DTO contract for generated
  runtime-server dev-client commands, plus `RuntimeServerConnectionToolRequest`
  for runtime tools that use connection fields.
- `RuntimeServerInfoRequest` and `RuntimeServerExecutionStatusRequest` now
  inherit that runtime-connection request contract and own
  `as_tool_arguments()` projection.
- `RuntimeServerScanRequest` now owns its MCP tool argument projection, CLI
  port text parsing, and generated CLI argument declarations for positional
  `ports` plus repeated `--ports`.
- Removed explicit `RuntimeScanCommandSpec`, `RuntimeInfoCommandSpec`, and
  `RuntimeStatusCommandSpec`; `runtime-scan`, `runtime-info`, and
  `runtime-status` now use
  `GeneratedRuntimeServerToolCommandSpec` from their capability declarations.
- `GeneratedRuntimeServerToolCommandSpec` reflects CLI fields from the request
  DTO's nominal `mcp_dev_cli_factory()` and
  `mcp_dev_cli_argument_specs()`, so `RuntimeServerScanRequest` owns scan port
  CLI shape and `RuntimeServerExecutionStatusRequest.from_cli_fields()` owns
  the `execution_id` option and CLI timeout default.

Verified gates:

```bash
source .venv/bin/activate && python -m py_compile \
  openhcs/agent/dto/execution.py openhcs/mcp/dev_client.py

source .venv/bin/activate && python - <<'PY'
from openhcs.mcp import dev_client

parser = dev_client._build_parser()
for argv in (
    ("runtime-info", "7777"),
    ("runtime-status", "7777", "--execution-id", "run-1"),
    ("runtime-scan", "4444", "--ports", "5555,5565", "--ports", "7777"),
):
    args = parser.parse_args(argv)
    spec = dev_client.McpDevCommandSpec.for_name(args.command)
    call = dev_client._calls_from_args(args)[0]
    print(args.command, type(spec).__name__, call.name, call.arguments)
PY
# runtime-info GeneratedRuntimeServerToolCommandSpec openhcs_get_runtime_server_info {'host': 'localhost', 'port': 7777, 'transport_mode': None, 'persistent': True, 'timeout_ms': 500}
# runtime-status GeneratedRuntimeServerToolCommandSpec openhcs_get_runtime_server_execution_status {'host': 'localhost', 'port': 7777, 'transport_mode': None, 'persistent': True, 'timeout_ms': 500, 'execution_id': 'run-1'}
# runtime-scan GeneratedRuntimeServerToolCommandSpec openhcs_scan_runtime_servers {'ports': [4444, 5555, 5565, 7777], 'host': 'localhost', 'transport_mode': None, 'timeout_ms': 200}

python - <<'PY'
from pathlib import Path
import ast

tree = ast.parse(Path("openhcs/mcp/dev_client.py").read_text())
print([
    node.name
    for node in ast.walk(tree)
    if isinstance(node, ast.ClassDef)
    and node.name in {
        "RuntimeScanCommandSpec",
        "RuntimeInfoCommandSpec",
        "RuntimeStatusCommandSpec",
    }
])
node = next(
    node for node in ast.walk(tree)
    if isinstance(node, ast.ClassDef)
    and node.name == "GeneratedRuntimeServerToolCommandSpec"
)
print([
    call.lineno
    for call in ast.walk(node)
    if isinstance(call, ast.Call)
    and isinstance(call.func, ast.Attribute)
    and call.func.attr == "from_payload"
    and call.args
    and isinstance(call.args[0], ast.Dict)
])
PY
# []
# []
```

Verified tests:

```bash
XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest tests/unit/agent/test_mcp_server.py \
  -k "runtime_scan or runtime_info or runtime_status or runtime_server or runtime_execution" -q
# 6 passed, 232 deselected
```

2026-06-29 viewer navigation/isolation DTO projection follow-up:

- Added `from_fields()` and `as_tool_arguments()` to
  `ViewerWindowNavigationRequest` and `ViewerWindowLayerIsolationRequest`.
- The navigation DTO now owns the public MCP default policy for `visible=True`
  and `selected=True`, while still preserving explicit `None` in
  `as_tool_arguments()` for "no visible/selection change".
- Updated `ViewerNavigationMcpToolBinding` and
  `ViewerLayerIsolationMcpToolBinding` to derive FastMCP signatures and request
  construction from those DTOs.
- Updated `navigate-viewer` and `isolate-viewer` dev-client commands to hydrate
  the DTOs and call `as_tool_arguments()`. The command classes retain only CLI
  conveniences: positional route-key aliases and semantic `--axis-index`
  parsing.

Verified gates:

```bash
source .venv/bin/activate && python -m py_compile \
  openhcs/agent/dto/viewer.py openhcs/mcp/server.py openhcs/mcp/dev_client.py

source .venv/bin/activate && python - <<'PY'
import ast
from pathlib import Path

for file, class_names in {
    "openhcs/mcp/server.py": (
        "ViewerNavigationMcpToolBinding",
        "ViewerLayerIsolationMcpToolBinding",
    ),
    "openhcs/mcp/dev_client.py": (
        "NavigateViewerCommandSpec",
        "IsolateViewerCommandSpec",
    ),
}.items():
    tree = ast.parse(Path(file).read_text())
    print(file)
    for class_name in class_names:
        cls = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        parameter_calls = []
        literal_defaults = []
        dict_literals = []
        for child in ast.walk(cls):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name) and child.func.id == "Parameter":
                    parameter_calls.append(child.lineno)
                if (
                    isinstance(child.func, ast.Attribute)
                    and child.func.attr == "add_argument"
                ):
                    for keyword in child.keywords:
                        if (
                            keyword.arg == "default"
                            and isinstance(keyword.value, ast.Constant)
                        ):
                            literal_defaults.append((child.lineno, keyword.value.value))
            if isinstance(child, ast.Dict):
                dict_literals.append(child.lineno)
        print(
            class_name,
            "Parameter_calls",
            parameter_calls,
            "literal_defaults",
            literal_defaults,
            "dict_literals",
            dict_literals,
        )
PY
# ViewerNavigationMcpToolBinding Parameter_calls [] literal_defaults [] dict_literals []
# ViewerLayerIsolationMcpToolBinding Parameter_calls [] literal_defaults [] dict_literals []
# NavigateViewerCommandSpec Parameter_calls [] literal_defaults [] dict_literals []
# IsolateViewerCommandSpec Parameter_calls [] literal_defaults [] dict_literals []
```

Verified tests:

```bash
XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest tests/unit/agent/test_mcp_server.py \
  -k "navigate_viewer or isolate_viewer or viewer_navigation" -q
# 7 passed, 231 deselected

XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest \
  tests/unit/agent/test_mcp_server.py \
  tests/unit/agent/test_capabilities.py \
  tests/unit/agent/test_ui_bridge_service.py \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py \
  tests/unit/agent/test_plate_inspection_service.py -q
# 326 passed
```

2026-06-29 viewer sample/ROI server binding follow-up:

- Added a DTO-derived request option projection helper to
  `McpViewerRequestToolBindingABC`. It derives public request parameters from
  `request_type.from_fields()` and excludes inherited viewer-control fields by
  reflecting `ViewerWindowControlRequest`, not by a parallel string table.
- Updated `ViewerImageSampleMcpToolBinding` and
  `ViewerRoiSummaryMcpToolBinding` to derive their FastMCP signatures and
  request construction from `ViewerWindowImageSampleRequest.from_fields()` and
  `ViewerWindowRoiSummaryRequest.from_fields()`.
- Moved list/tuple axis-index edge normalization onto
  `ViewerWindowControlRequest`, so the server binding no longer has a local
  `axis_indices` conversion branch.
- Extended the same pattern to viewer validation: `ViewerWindowValidationRequest`
  now owns field construction/projection for route-key, validation policy, and
  include-state controls, and `ViewerValidationMcpToolBinding` derives its MCP
  signature and request construction from that DTO.
- Updated the `validate-viewer` dev-client command to hydrate
  `ViewerWindowValidationRequest` and call `as_tool_arguments()`. The command
  keeps only CLI-specific label parsing and the `--require-components`
  convenience expansion.

Verified gates:

```bash
source .venv/bin/activate && python -m py_compile \
  openhcs/mcp/server.py openhcs/agent/dto/viewer.py openhcs/mcp/dev_client.py

source .venv/bin/activate && python - <<'PY'
import ast
from pathlib import Path

tree = ast.parse(Path("openhcs/mcp/server.py").read_text())
for class_name in (
    "ViewerImageSampleMcpToolBinding",
    "ViewerRoiSummaryMcpToolBinding",
    "ViewerValidationMcpToolBinding",
):
    cls = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    manual_params = []
    constructors = []
    for child in ast.walk(cls):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name) and child.func.id == "Parameter":
                manual_params.append(child.lineno)
            if isinstance(child.func, ast.Name) and child.func.id.startswith("ViewerWindow"):
                constructors.append((child.func.id, child.lineno))
    print(class_name, "Parameter_calls", manual_params, "ViewerWindow_constructors", constructors)
PY
# ViewerImageSampleMcpToolBinding Parameter_calls [] ViewerWindow_constructors []
# ViewerRoiSummaryMcpToolBinding Parameter_calls [] ViewerWindow_constructors []
# ViewerValidationMcpToolBinding Parameter_calls [] ViewerWindow_constructors []
```

Verified tests:

```bash
XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest tests/unit/agent/test_mcp_server.py \
  -k "viewer_tool_schema or sample_viewer_image or viewer_rois or mcp_viewer_rois" -q
# 18 passed, 220 deselected

XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest tests/unit/agent/test_mcp_server.py \
  -k "viewer_tool_schema or validate_viewer or viewer_validation or validate_viewer_window_state" -q
# 7 passed, 231 deselected

source .venv/bin/activate && python - <<'PY'
import ast
from pathlib import Path

tree = ast.parse(Path("openhcs/mcp/dev_client.py").read_text())
for class_name in (
    "ViewerRoisCommandSpec",
    "SampleViewerImageCommandSpec",
    "ValidateViewerCommandSpec",
):
    cls = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    literal_defaults = []
    dict_literals = []
    for child in ast.walk(cls):
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "add_argument"
        ):
            for keyword in child.keywords:
                if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                    literal_defaults.append((child.lineno, keyword.value.value))
        if isinstance(child, ast.Dict):
            dict_literals.append(child.lineno)
    print(class_name, "literal_defaults", literal_defaults, "dict_literals", dict_literals)
PY
# ViewerRoisCommandSpec literal_defaults [] dict_literals []
# SampleViewerImageCommandSpec literal_defaults [] dict_literals []
# ValidateViewerCommandSpec literal_defaults [] dict_literals []

XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest \
  tests/unit/agent/test_mcp_server.py \
  tests/unit/agent/test_capabilities.py \
  tests/unit/agent/test_ui_bridge_service.py \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py \
  tests/unit/agent/test_plate_inspection_service.py -q
# 326 passed
```

2026-06-29 viewer sample/ROI DTO projection follow-up:

- Added `from_fields()` and `as_tool_arguments()` to
  `ViewerWindowImageSampleRequest` and `ViewerWindowRoiSummaryRequest`.
- Updated `sample-viewer-image` and `viewer-rois` dev-client commands to derive
  parser defaults from those DTO `from_fields` signatures and project tool
  arguments through the DTOs.
- The remaining viewer command-specific parser logic is CLI UX only: positional
  route-key aliases, semantic axis-index parsing, and renderer options.

Verified gates:

```bash
source .venv/bin/activate && python -m py_compile \
  openhcs/agent/dto/viewer.py openhcs/mcp/dev_client.py

source .venv/bin/activate && python - <<'PY'
import ast
from pathlib import Path

tree = ast.parse(Path("openhcs/mcp/dev_client.py").read_text())
classes = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
for class_name in ("ViewerRoisCommandSpec", "SampleViewerImageCommandSpec"):
    literals = []
    for child in ast.walk(classes[class_name]):
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "add_argument"
        ):
            for keyword in child.keywords:
                if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                    literals.append((child.lineno, keyword.value.value))
    print(class_name, literals)
PY
# ViewerRoisCommandSpec []
# SampleViewerImageCommandSpec []
```

Verified tests:

```bash
XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest tests/unit/agent/test_mcp_server.py \
  -k "viewer_rois_command_projects_tool_arguments or viewer_rois_command_projects_semantic_axis_indices or viewer_rois_command_accepts_route_key_option or viewer_rois_command_allows_route_discovery or sample_viewer_image_command_projects_tool_arguments or sample_viewer_image_command_allows_omitted_route_key or sample_viewer_image_command_projects_semantic_axis_indices or sample_viewer_image_command_can_include_array_values" -q
# 8 passed, 230 deselected

XDG_CACHE_HOME=/tmp/openhcs-test-cache pytest \
  tests/unit/agent/test_mcp_server.py \
  tests/unit/agent/test_capabilities.py \
  tests/unit/agent/test_ui_bridge_service.py \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py \
  tests/unit/agent/test_plate_inspection_service.py -q
# 326 passed
```

2026-06-29 module split checkpoint:

- `openhcs/mcp/dev_client.py` is now the CLI composition entrypoint, parser
  builder, and `main()` wrapper. It no longer contains renderer bodies,
  command-framework base classes, or domain command declarations.
- Shared transport, typed response records, workflow polling, and parser helper
  primitives moved to `openhcs/mcp/dev_client_core.py`.
- `McpDevCommandSpec`, generated command profiles, and command-registry
  mechanics moved to `openhcs/mcp/dev_client_commanding.py`.
- Domain command declarations moved to:
  - `openhcs/mcp/dev_client_commands/knowledge_pipeline.py`
  - `openhcs/mcp/dev_client_commands/plate.py`
  - `openhcs/mcp/dev_client_commands/ui.py`
  - `openhcs/mcp/dev_client_commands/viewer.py`
- Shared render options and renderer registration moved to
  `openhcs/mcp/dev_client_rendering.py`.
- Domain renderers moved to:
  - `openhcs/mcp/dev_client_renderers/knowledge.py`
  - `openhcs/mcp/dev_client_renderers/pipeline.py`
  - `openhcs/mcp/dev_client_renderers/plate.py`
  - `openhcs/mcp/dev_client_renderers/ui_bridge.py`
  - `openhcs/mcp/dev_client_renderers/object_state.py`
  - `openhcs/mcp/dev_client_renderers/viewer.py`

Current size gate:

```bash
wc -l openhcs/mcp/dev_client.py \
  openhcs/mcp/dev_client_core.py \
  openhcs/mcp/dev_client_commanding.py \
  openhcs/mcp/dev_client_commands/*.py \
  openhcs/mcp/dev_client_rendering.py \
  openhcs/mcp/dev_client_renderers/*.py
```

`openhcs/mcp/dev_client.py` is 190 lines after import pruning. No remaining
module is an 8k-line catch-all; the largest current modules are domain-local
(`dev_client_core.py`, plate/UI/viewer renderer modules, and UI/plate command
modules).

Verified tests after the split:

```bash
. .venv/bin/activate && pytest tests/unit/agent/test_mcp_server.py \
  -k "dev_client and not launches_fresh_current_source_server" -q
# 169 passed, 69 deselected
```

Additional gates after import pruning:

```bash
.venv/bin/python -m ruff check openhcs/mcp/dev_client.py \
  openhcs/mcp/dev_client_core.py openhcs/mcp/dev_client_commanding.py \
  openhcs/mcp/dev_client_commands openhcs/mcp/dev_client_rendering.py \
  openhcs/mcp/dev_client_renderers tests/unit/agent/test_mcp_server.py \
  --select F401,F821,F811
# All checks passed

. .venv/bin/activate && pytest tests/unit/agent/test_mcp_server.py -q
# 238 passed
```

Test patch-target note:

- Tests that patch command-framework transport now patch
  `openhcs.mcp.dev_client_commanding.call_fresh_mcp_server`, because the command
  framework owns the call site.
- Tests that patch stdio/session/tool-call internals now patch
  `openhcs.mcp.dev_client_core`, because fresh-process transport is owned by the
  core helper module.
- `openhcs.mcp.dev_client` still composes and re-exports the core records used
  by existing dev-client tests, but production command execution is not routed
  through the old monolith.

2026-06-29 fresh-process MCP smoke update:

- The earlier fresh-process stdio initialize timeout is no longer reproduced in
  the current checkout.
- `openhcs.mcp.dev_client health` starts a fresh `python -m openhcs.mcp`
  server, initializes, calls `openhcs_health_check`, and exits cleanly.
- `openhcs.mcp.dev_client tools` lists 76 generated tools after resource/tool
  separation.
- `openhcs.mcp.dev_client knowledge` lists 38 source-backed knowledge
  documents and renders the first 20 compactly.
- `openhcs.mcp.dev_client call openhcs_list_capabilities --json` returns the
  capability registry with resource declarations and generated tool
  declarations.

Verified commands:

```bash
XDG_CACHE_HOME=/tmp/openhcs-test-cache \
XDG_DATA_HOME=/tmp/openhcs-test-data \
  .venv/bin/python -m openhcs.mcp.dev_client health
# status ok, restart_required=false

XDG_CACHE_HOME=/tmp/openhcs-test-cache \
XDG_DATA_HOME=/tmp/openhcs-test-data \
  .venv/bin/python -m openhcs.mcp.dev_client tools
# Tools: matched=76 total=76 shown=76

XDG_CACHE_HOME=/tmp/openhcs-test-cache \
XDG_DATA_HOME=/tmp/openhcs-test-data \
  .venv/bin/python -m openhcs.mcp.dev_client knowledge
# Knowledge documents: matched=38 shown=20
```
