# MCP Semantic Authority Audit

Date: 2026-06-29

## Purpose

This audit covers the current MCP, agent, and UI bridge layer after the large
agent-facing surface expansion. It is a plan-only artifact. Another agent can
keep using the current MCP implementation while this document set defines the
next refactor batch.

The desired end state is:

```text
backend / UI semantic declarations
    -> openhcs.agent typed services and DTO projections
    -> openhcs.mcp transport wrappers
    -> optional dev-client UX
```

The MCP layer must not become a second authority for OpenHCS semantics. String
names are acceptable only as final ABI field values emitted by a typed
declaration. They are not acceptable as decision tables, hand-maintained
registries, or copied semantic rules.

## Current-State Scan

The audit used AST over:

- `openhcs/mcp`
- `openhcs/agent`
- `openhcs/pyqt_gui/services`

The scan reported:

```text
AUDITED_FILES 73
RECORDS 1675
KIND class 545
KIND uppercase_constant 512
KIND call 472
KIND decorated_func 88
KIND enum 39
KIND dict_literal 14
KIND sequence_constant 5
```

Largest mirror clusters:

```text
351 openhcs/mcp/dev_client.py
212 openhcs/mcp/server.py
142 openhcs/agent/dto/ui_bridge.py
127 openhcs/agent/services/viewer_window_service.py
95  openhcs/agent/dto/plate.py
72  openhcs/pyqt_gui/services/ui_bridge_windows.py
50  openhcs/agent/capabilities.py
41  openhcs/agent/services/plate_inspection_service.py
33  openhcs/agent/services/ui_bridge_service.py
30  openhcs/pyqt_gui/services/ui_bridge_server.py
```

The AST pass is reproducible with this shape:

```bash
.venv/bin/python - <<'PY'
import ast, pathlib, collections
roots = [pathlib.Path("openhcs/mcp"), pathlib.Path("openhcs/agent"), pathlib.Path("openhcs/pyqt_gui/services")]
files = [p for root in roots for p in root.rglob("*.py") if p.is_file()]
records = []
for p in sorted(files):
    tree = ast.parse(p.read_text(), filename=str(p))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bases = ",".join(ast.unparse(b) for b in node.bases)
            if "Enum" in bases or any(token in node.name for token in ("Policy", "Spec", "Kind", "Name", "Field", "Action", "Registry", "Projection", "Payload", "Request", "Response", "Default", "Manifest")):
                records.append(("class", str(p), node.lineno, node.name, bases))
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if names and (any(n.isupper() for n in names) or isinstance(node.value, ast.Dict)):
                records.append(("assignment", str(p), node.lineno, ",".join(names), type(node.value).__name__))
        elif isinstance(node, ast.Call):
            fn = ast.unparse(node.func)
            if fn in {"getattr", "hasattr", "isinstance"} or fn.endswith(".register") or "openhcs_tool" in fn:
                records.append(("call", str(p), getattr(node, "lineno", 0), fn))
print("AUDITED_FILES", len(files))
print("RECORDS", len(records))
for kind, count in collections.Counter(r[0] for r in records).most_common():
    print(kind, count)
for path, count in collections.Counter(r[1] for r in records).most_common(25):
    print(count, path)
PY
```

## Coverage and Disposition

| Area | Audited files | Disposition |
| --- | --- | --- |
| Capability and MCP registration | `openhcs/agent/capabilities.py`, `openhcs/mcp/server.py` | Needs typed capability/tool binding refactor. |
| Fresh-process CLI | `openhcs/mcp/dev_client.py` | Needs capability/request/response schema derivation. |
| UI bridge DTO/client/server | `openhcs/agent/dto/ui_bridge.py`, `openhcs/agent/services/ui_bridge_service.py`, `openhcs/agent/services/ui_bridge_transport.py`, `openhcs/pyqt_gui/services/ui_bridge_server.py` | Needs shared typed operation contracts. |
| UI bridge providers/actions/windows | `ui_agent_bridge.py`, `ui_bridge_plate_manager.py`, `ui_bridge_pipeline_editor.py`, `ui_bridge_windows.py`, `ui_window_ids.py` | Needs provider/action/window authority cleanup. `PipelineEditorAction` is the positive pattern. |
| Plate/synthetic/streaming services | `agent/dto/plate.py`, `plate_inspection_service.py`, `plate_streaming_service.py`, `synthetic_plate_service.py` | Needs derivation from component, inventory, microscope, and generator profile authorities. |
| Viewer/runtime projections | `viewer_window_service.py`, `runtime_server_service.py` | Needs runtime protocol and server-role authority cleanup. |
| Agent service projections | `config_service.py`, `function_catalog_service.py`, `mcp/context.py`, selected helpers in `mcp/server.py` | Needs smaller service-projection cleanup. |
| Knowledge base | `knowledge_base_service.py`, `agent/dto/knowledge.py` | Mostly acceptable: manifest fields are local knowledge-base format authority. Search stop words should stay display/search policy only. |
| Path policy and stdio | `path_policy.py`, `agent/services/stdio.py`, `mcp/control_timeout.py` | Operational/security policy, not backend semantic mirroring in this audit. Keep separate from domain declarations. |
| Serialization | `agent/serialization.py` | Generic DTO serialization. Watch for future per-domain coercion, but no current domain authority move is required. |

## Refactor Plan Set

Execute these plans in order:

0. `docs/plans/mcp_public_python_api_projection_plan_20260629.md`
   - Reframe the already-generic MCP/capability/dev-client infrastructure as a
     projection of public operation declarations in `openhcs.agent`, not as the
     final authority itself.
1. `docs/plans/mcp_capability_tool_binding_refactor_20260629.md`
   - Collapse capability contracts, MCP tool registration, and dev-client command
     tool names onto typed capability declarations. This is now migration
     history and scaffold for the public-operation refactor, not the final
     endpoint.
2. `docs/plans/mcp_ui_bridge_authority_refactor_20260629.md`
   - Move UI bridge operation/action/window/code-document semantics to their
     existing UI declarations and typed operation contracts.
3. `docs/plans/mcp_plate_viewer_projection_refactor_20260629.md`
   - Remove plate, synthetic, viewer, and runtime-server semantic mirrors from
     agent DTO/services by deriving them from backend owners.
4. `docs/plans/mcp_dev_client_schema_derivation_refactor_20260629.md`
   - Keep the dev client as UX, but derive tool identities, argument schemas,
     and response renderers from capability/request/response declarations.
5. `docs/plans/mcp_agent_service_projection_refactor_20260629.md`
   - Remove smaller agent-service mirrors: config-kind lookup, function
     parameter supplier labels, custom-function IDs, MCP context source
     freshness, and enum coercion helpers.

## Cross-Cutting Rules

- Do not add MCP-local shims or fallback maps.
- Do not add a new semantic hub beside existing backend/UI declarations.
- Do not use `getattr` or `hasattr` as a semantic decision mechanism.
- Do not use name-token heuristics to determine mutation or side effects.
- Do not duplicate DTO class names as strings in a separate enum.
- Do not let `openhcs.mcp.server` inspect backend object shapes beyond simple
  transport coercion.
- Do not let `openhcs.mcp.dev_client` become the authority for tool names,
  defaults, enum choices, or response payload shape.

## Completion Gates

The refactor is not complete until these checks are true:

```bash
rg -n "class AgentContractName|MutatingCapabilityNamePolicy|MUTATING_CAPABILITY_NAME_POLICY" openhcs/agent/capabilities.py
rg -n "tool_name = \"openhcs_|UiBridgeOperationRequestPayload|PlateInspectionComponentKind|SyntheticPlateGenerationDefaults" openhcs/agent openhcs/mcp openhcs/pyqt_gui/services
rg -n "class ViewerControlField|class ViewerLayerField|class ViewerPayloadField|class ViewerDescriptorField|class ViewerPayloadSummaryField" openhcs/agent/services/viewer_window_service.py
rg -n "PLATE_MANAGER_ACTION_SIDE_EFFECTS|PLATE_MANAGER_ACTION_OPERATIONS|MANAGED_WINDOW_ACTION_SPECS|UiCodeDocumentSourcePolicy.expected_assignments" openhcs/pyqt_gui/services
rg -n "class AgentConfigKind|OPENHCS_AGENT_CONTEXT_SOURCE_TYPES|agent_supplier|f\"openhcs:\\{registered_function.__name__\\}\"" openhcs/agent openhcs/mcp
```

Expected result after implementation: no matches for the removed mirrors, except
for final ABI values declared on the real owner. Protocol-owned viewer field
enums may be imported and used by `ViewerWindowService`; the removed mirror is
the local class declaration.

Focused tests after implementation:

```bash
XDG_CACHE_HOME=/tmp/openhcs-test-cache .venv/bin/python -m pytest \
  tests/unit/agent/test_capabilities.py \
  tests/unit/agent/test_mcp_server.py \
  tests/unit/agent/test_agent_services.py \
  tests/unit/agent/test_plate_inspection_service.py \
  tests/unit/agent/test_plate_streaming_service.py \
  tests/unit/agent/test_ui_bridge_service.py \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py \
  tests/unit/pyqt_gui/test_config_window_state_resolution.py \
  tests/unit/pyqt_gui/test_object_state_callable_preview.py
```

## Final Verification

2026-06-29 verification pass:

```bash
rg -n "class AgentContractName|MutatingCapabilityNamePolicy|MUTATING_CAPABILITY_NAME_POLICY" \
  openhcs/agent/capabilities.py
# no matches

rg -n "tool_name = \"openhcs_|UiBridgeOperationRequestPayload|PlateInspectionComponentKind|SyntheticPlateGenerationDefaults" \
  openhcs/agent openhcs/mcp openhcs/pyqt_gui/services
# no matches

rg -n "class ViewerControlField|class ViewerLayerField|class ViewerPayloadField|class ViewerDescriptorField|class ViewerPayloadSummaryField" \
  openhcs/agent/services/viewer_window_service.py
# no matches

rg -n "PLATE_MANAGER_ACTION_SIDE_EFFECTS|PLATE_MANAGER_ACTION_OPERATIONS|MANAGED_WINDOW_ACTION_SPECS|UiCodeDocumentSourcePolicy.expected_assignments" \
  openhcs/pyqt_gui/services
# no matches

rg -n "class AgentConfigKind|OPENHCS_AGENT_CONTEXT_SOURCE_TYPES|agent_supplier|f\"openhcs:\\{registered_function.__name__\\}\"" \
  openhcs/agent openhcs/mcp
# no matches

rg -n "get_agent_capability\\(\"openhcs_|tool_name: ClassVar|tool_name = \"openhcs_|for_tool_name|__registry_key__ = \"tool_name\"|choices=\\(" \
  openhcs/mcp/dev_client.py
# no matches
```

## Public API Lens Update

2026-06-29 reinvestigation:

- `AgentCapabilityDeclaration` is currently the operation-like authority for
  the exposed agent/MCP surface, with 79 declarations.
- `UiBridgeOperationContractABC` is already a headless operation contract
  registry for 26 UI bridge operations.
- `openhcs.mcp.server` is mostly generic, but still owns nine generated binding
  families and seven explicit binding leaves.
- The next cleanup should not add another MCP registry. It should promote the
  operation model into `openhcs.agent` and make MCP/dev-client/knowledge
  projections consume that public operation catalog.

Relevant plan:

- `docs/plans/mcp_public_python_api_projection_plan_20260629.md`

Focused tests:

```bash
XDG_CACHE_HOME=/tmp/openhcs-test-cache XDG_DATA_HOME=/tmp/openhcs-test-data \
  .venv/bin/python -m pytest \
  tests/unit/agent/test_capabilities.py \
  tests/unit/agent/test_mcp_server.py \
  tests/unit/agent/test_agent_services.py \
  tests/unit/agent/test_plate_inspection_service.py \
  tests/unit/agent/test_plate_streaming_service.py \
  tests/unit/agent/test_agent_serialization.py
# 327 passed

QT_QPA_PLATFORM=offscreen \
XDG_CACHE_HOME=/tmp/openhcs-test-cache \
XDG_DATA_HOME=/tmp/openhcs-test-data \
  .venv/bin/python -m pytest \
  tests/unit/agent/test_ui_bridge_service.py \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py \
  tests/unit/pyqt_gui/test_config_window_state_resolution.py \
  tests/unit/pyqt_gui/test_object_state_callable_preview.py \
  -k "not descriptor_resolution_uses_process_advertised_descriptor"
# 74 passed, 1 deselected

cd external/zmqruntime && ../../.venv/bin/python -m pytest tests/test_messages.py
# 4 passed
```

Unfiltered UI bridge service note:

- `test_descriptor_resolution_uses_process_advertised_descriptor` fails in this
  live environment because a real `/run/user/1000/openhcs/ui-bridge/...json`
  descriptor is discoverable and takes precedence over the fake proc descriptor
  built by the test.

2026-06-29 commit-readiness audit update:

- A changed-file AST scan found no `getattr` / `hasattr` use, no DTO identity
  imports outside the DTO re-export module, and no PlateManager action imports
  from the widget module.
- User-facing `plate_manager.state` and `plate_manager.orchestrator_config`
  strings in selected-plate guidance, dev-client defaults, LLM context, and UI
  code-document hints now derive from UI identity declarations.
- `SyntheticPlateGenerationIssueCode` now follows the same `str, Enum` pattern
  as `PlateInspectionIssueCode`.
- `ExecutionSessionService` now uses
  `OpenHCSMetadataHandler.METADATA_FILENAME` for the OpenHCS metadata filename
  fallback instead of duplicating the filename.
- Official30 knowledge-document rendering is content-shape driven by the JSON
  recipe manifest rather than gated by a hardcoded document id constant.
- MCP resources are generated from `CapabilityKind.RESOURCE` declarations and
  no longer appear as tools.
- `OpenHCSUiWindowId.plate_manager` and `.pipeline_editor` now derive from the
  same UI widget identity declarations used by the agent bridge.

Current changed-file gates:

```bash
.venv/bin/python - <<'PY'
from __future__ import annotations
import ast
import subprocess
from pathlib import Path

prod = [
    Path(p)
    for p in (
        subprocess.check_output(["git", "diff", "--name-only"], text=True).splitlines()
        + subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], text=True).splitlines()
    )
    if p.endswith(".py") and p.startswith("openhcs/")
]
findings = []
for path in sorted(prod):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"getattr", "hasattr"}:
            findings.append((path, node.lineno, node.func.id))
print("dynamic_attr_findings", findings)
print("changed_production_python_files", len(prod))
PY
# dynamic_attr_findings []
# changed_production_python_files 71
```

Current verification:

```bash
.venv/bin/python - <<'PY'
from __future__ import annotations
import py_compile
import subprocess
from pathlib import Path
files = [
    Path(p)
    for p in (
        subprocess.check_output(["git", "diff", "--name-only"], text=True).splitlines()
        + subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], text=True).splitlines()
    )
    if p.endswith(".py") and p.startswith(("openhcs/", "tests/"))
]
for path in files:
    py_compile.compile(str(path), doraise=True)
print(f"compiled {len(files)} changed python files")
PY
# compiled 82 changed python files

.venv/bin/python -m ruff check <changed openhcs/tests python files> --select F401,F821,F811
# All checks passed

QT_QPA_PLATFORM=offscreen \
XDG_CACHE_HOME=/tmp/openhcs-test-cache \
XDG_DATA_HOME=/tmp/openhcs-test-data \
  .venv/bin/python -m pytest \
  tests/unit/agent/test_capabilities.py \
  tests/unit/agent/test_mcp_server.py \
  tests/unit/agent/test_knowledge_base_service.py -q
# 260 passed

QT_QPA_PLATFORM=offscreen \
XDG_CACHE_HOME=/tmp/openhcs-test-cache \
XDG_DATA_HOME=/tmp/openhcs-test-data \
  .venv/bin/python -m pytest \
  tests/unit/agent/test_ui_bridge_service.py \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py \
  tests/unit/pyqt_gui/test_object_state_callable_preview.py \
  tests/unit/pyqt_gui/test_plate_manager_widget.py -q
# 82 passed

cd external/zmqruntime && ../../.venv/bin/python -m pytest tests/test_messages.py -q
# 4 passed
```
