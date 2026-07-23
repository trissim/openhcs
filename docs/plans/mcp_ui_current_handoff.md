# OpenHCS MCP/UI Handoff

Last updated: 2026-06-25 21:02 America/Toronto

This file is meant to let a fresh Codex thread continue without carrying the huge
transcript. It intentionally includes architectural reasoning and anti-patterns,
not just mechanical status.

## How To Resume

Use a new Codex CLI thread in `/home/ts/code/projects/openhcs-benchmark-platform`
and start with:

```text
Continue OpenHCS MCP/UI robustness work from docs/plans/mcp_ui_current_handoff.md.
Source .venv before Python commands. Use fresh-process MCP dev_client, not the
embedded Codex MCP transport, while editing MCP code.
```

Always source the repo venv:

```bash
. .venv/bin/activate
```

Do not assume this file proves completion. Re-check the current worktree and
runtime state first.

## Active Objective

Make OpenHCS UI and MCP robust enough that an agent can operate the UI with a
seamless user-facing workflow:

- inspect UI state and windows
- edit/apply code documents with ObjectState snapshot/revision/undo contracts
- initialize, compile, and eventually run selected plates through MCP
- inspect screenshots/widget trees/viewer payloads instead of guessing
- use subagents for orthogonal work when useful
- use advisor as a guardrail, but do not disappear into advisor-only refactoring
- avoid introducing new debt or compatibility shims

The user especially cares that agents do not write "local patches" that preserve
parallel semantic paths. The work should move toward centralized, nominal,
load-bearing abstractions.

## Current Runtime State

Verified immediately before writing this file:

- Repo: `/home/ts/code/projects/openhcs-benchmark-platform`
- Branch: `benchmark-platform`
- HEAD: `a129f479`
- UI process: PID `2615980`, command `python -m openhcs.pyqt_gui.__main__`
- UI bridge descriptor:
  `/run/user/1000/openhcs/ui-bridge/ui_bridge_ui-29ee7f52-e754-4028-bc3d-b8accb3bcedd.json`
- UI bridge instance id: `ui-29ee7f52-e754-4028-bc3d-b8accb3bcedd`
- UI bridge connection: host `127.0.0.1`, port `7888`, transport `ipc`
- Execution server: PID `2361533`, port `7777`, IPC transport
- Napari viewer server: PID `2372206`, port `5555`, IPC transport
- There is also a standalone `.venv/bin/python -m openhcs.mcp` process
  (`2665208` at handoff time). Treat it as possibly stale; verify before using
  or kill it if it is not intentionally needed.

Fresh-process MCP state poll showed one selected plate:

```json
{
  "name": "BeginnerSegmentation / segmentation_final",
  "initialized": true,
  "compiled": true,
  "orchestrator_state": "compiled",
  "status_prefix": "✅ Compiled 100.0%"
}
```

Selected scope:

```text
/tmp/openhcs_benchmark_dataset_cache_last8/CellProfiler_tutorials/data/BeginnerSegmentation#openhcs-cppipe=segmentation_final.cppipe
```

The UI is visible on workspace 5. A previous screenshot was successfully saved
to:

```text
/tmp/openhcs_mcp_screenshots/20260626T002925313154Z_plate_manager_Plate_Manager.png
```

## Current Worktree State

There are broad uncommitted changes. Do not revert unrelated files. Work with
the current state.

Production areas with changes include:

- `openhcs/mcp/dev_client.py`
- `openhcs/mcp/server.py`
- `openhcs/agent/dto/ui_bridge.py`
- `openhcs/agent/services/ui_bridge_service.py`
- `openhcs/pyqt_gui/services/ui_agent_bridge.py`
- `openhcs/pyqt_gui/services/ui_bridge_plate_manager.py`
- `openhcs/pyqt_gui/services/ui_bridge_object_state.py`
- `openhcs/pyqt_gui/services/ui_bridge_windows.py`
- `openhcs/pyqt_gui/widgets/plate_manager.py`
- `openhcs/pyqt_gui/widgets/pipeline_editor.py`
- many CellProfiler runtime/backend files
- external submodules `external/ObjectState` and `external/pyqt-reactive`

New untracked production files include:

- `openhcs/core/function_contract_metadata.py`
- `openhcs/core/special_output_declarations.py`
- `openhcs/interop/cellprofiler/runtime/bound_parameters.py`
- `openhcs/mcp/control_timeout.py`
- `openhcs/pyqt_gui/services/ui_window_ids.py`

The diff is large: about 47 files, roughly 4048 insertions and 1383 deletions
at handoff time.

## Verified MCP/UI Work Before Latest Partial Edit

These were verified before the latest in-progress `dev_client.py` polling edit:

- Fresh-process MCP dev client works; do not use the stale Codex embedded MCP
  transport while actively editing the MCP server.
- UI bridge status works and advertises:
  - UI code documents
  - UI state surfaces
  - UI actions
  - UI windows/navigation/snapshots
  - widget tree projection
  - ObjectState scopes/snapshots/branches
  - selected plate workflows
  - operation status
- `plate_manager.orchestrator_config` code document can be read with zero plates.
- Code-document apply can add the BeginnerSegmentation final `.cppipe` plate and
  returns explicit revision/snapshot/receipt/operation-id data.
- Selected workflow `init_plate` succeeded by MCP.
- Selected workflow `compile_plate` succeeded by MCP.
- `openhcs_ui_snapshot_window` captured the `plate_manager` window.
- Empty selection compile rejects with `plate_selection_required`, includes a
  selection revision token, and includes `plate_manager.state` as polling surface.
- Confirmation-required selected workflow rejection preserves:
  - selected target scope ids
  - selection revision token
  - `plate_manager.state` polling surface

Focused tests run before the latest partial polling edit:

```bash
. .venv/bin/activate
PYTEST_ADDOPTS='--no-cov' python -m pytest -q \
  tests/unit/agent/test_mcp_server.py \
  tests/unit/agent/test_ui_bridge_service.py \
  tests/unit/pyqt_gui/test_ui_agent_bridge.py \
  tests/unit/pyqt_gui/test_plate_manager_widget.py::TestPlateManagerWidget::test_compile_validator_requires_initialized_orchestrator
```

Result before latest partial edit: `60 passed`.

## Debugging MCP Connection Failures

Added by takeover thread at 2026-06-25 20:55 America/Toronto, then corrected
after stale Codex-owned MCP children were found and killed.

Do not treat all "MCP is broken" symptoms as the same failure. There are three
separate layers:

1. **OpenHCS UI bridge**: the running PyQt UI control server advertised by the
   descriptor under `/run/user/1000/openhcs/ui-bridge`.
2. **Fresh-process MCP dev client**: a new current-source stdio MCP subprocess
   launched by `python -m openhcs.mcp.dev_client ...`.
3. **Codex embedded MCP connection**: a Codex-managed stdio child process
   (`python -m openhcs.mcp`) owned by one Codex CLI instance.

The Codex embedded MCP process is not a shared daemon. Each Codex CLI owns its
own stdio child. If that child imports old code or its transport closes, Codex
may keep a dead/stale connection until the client restarts. This can happen even
when the UI bridge and fresh-process MCP are healthy.

### First Diagnostic Commands

Run these from `/home/ts/code/projects/openhcs-benchmark-platform`:

```bash
. .venv/bin/activate

ps -eo pid,ppid,stat,etime,cmd \
  | rg 'python -m openhcs.pyqt_gui|python -m openhcs.mcp|zmq_execution_server|napari_viewer' \
  | rg -v 'rg '

find /run/user/1000/openhcs/ui-bridge -maxdepth 1 -type f -name '*.json' \
  -print -exec sed -n '1,160p' {} \;

python -m openhcs.mcp.dev_client health
python -m openhcs.mcp.dev_client ui-status --timeout-ms 1000
```

Expected healthy results:

- `python -m openhcs.mcp.dev_client health` returns `status: "ok"` and
  `server_source_changed_since_import: false`.
- `python -m openhcs.mcp.dev_client ui-status --timeout-ms 1000` returns
  `reachable: true`, `descriptor_status: "ok"`, and the current bridge id.
- The UI process is still `python -m openhcs.pyqt_gui.__main__`.
- The descriptor PID exists and matches the running UI process.

If these fresh-process commands pass but Codex MCP tools fail, the bug is the
Codex embedded MCP connection, not OpenHCS runtime. Restart that Codex CLI or
kill only its stale MCP child process.

### Cleaning Stale Codex-Owned MCP Children

If `ps` shows one or more long-lived `.venv/bin/python -m openhcs.mcp` children
under Codex, inspect ownership:

```bash
pstree -asp <MCP_PID>
readlink /proc/<MCP_PID>/cwd
tr '\0' '\n' < /proc/<MCP_PID>/cmdline
```

If the process is just a stale Codex-owned stdio MCP child, it is safe to kill
that MCP child only:

```bash
kill <MCP_PID>
```

Do **not** kill:

- the PyQt UI process
- the ZMQ execution server
- the Napari viewer server

After killing stale MCP children, immediately re-run:

```bash
. .venv/bin/activate
python -m openhcs.mcp.dev_client health
python -m openhcs.mcp.dev_client ui-status --timeout-ms 1000
```

At 2026-06-25 20:53 America/Toronto, this cleanup was performed. Fresh-process
MCP then returned healthy:

```json
{
  "status": "ok",
  "server_source_changed_since_import": false
}
```

and UI status returned:

```json
{
  "reachable": true,
  "bridge": "ui-29ee7f52-e754-4028-bc3d-b8accb3bcedd",
  "descriptor_status": "ok"
}
```

Later check at 2026-06-25 21:00 America/Toronto from the restricted managed
Codex tool sandbox did **not** reproduce the healthy result:

```bash
. .venv/bin/activate
XDG_CACHE_HOME=/tmp/openhcs-test-cache \
  timeout 15 python -m openhcs.mcp.dev_client health
```

returned:

```json
{
  "errors": [
    {
      "code": "mcp_transport_failed",
      "phase": "initialize",
      "exception_type": "ExceptionGroup",
      "causes": [{"exception_type": "TimeoutError", "message": ""}]
    }
  ],
  "results": []
}
```

The same tool sandbox could read the UI bridge descriptor file, but `ps` inside
that sandbox did not show descriptor PID `2615980` or the expected host
OpenHCS/UI processes. The visible process table only showed the sandbox wrapper
and the current command, which means process-state diagnostics from this managed
tool sandbox are not equivalent to host-shell diagnostics.

After switching this thread to full filesystem/process permissions, the same
diagnostics did reproduce the healthy result at 2026-06-25 21:02
America/Toronto:

```bash
. .venv/bin/activate
python -m openhcs.mcp.dev_client health
python -m openhcs.mcp.dev_client ui-status --timeout-ms 1000
```

Fresh `health` returned:

```json
{
  "status": "ok",
  "server_source_changed_since_import": false,
  "stale_source_paths": []
}
```

Fresh `ui-status` returned:

```json
{
  "reachable": true,
  "bridge_instance_id": "ui-29ee7f52-e754-4028-bc3d-b8accb3bcedd",
  "descriptor_status": "ok"
}
```

The same full-permission process listing showed:

```text
2361533 ... python -m openhcs.runtime.zmq_execution_server_launcher --port 7777 ...
2372206 ... run_napari_viewer_process ... 5555 ...
2615980 ... python -m openhcs.pyqt_gui.__main__
```

Conclusion: the 21:00 initialize timeout was caused by the restricted managed
tool sandbox/process namespace, not by OpenHCS MCP or the UI bridge. For
diagnosing live MCP/UI runtime, use a context with host process visibility and
normal subprocess stdio behavior.

### Interpreting Timeout Failures

Do not keep increasing MCP initialize timeouts. If `ClientSession.initialize()`
times out at 5 seconds and also at 60 seconds, collect evidence for which layer
is stuck:

- Does `python -m openhcs.mcp.dev_client health` fail? Then fresh stdio MCP is
  failing before UI access. Check import errors, cache permissions, and stdio
  subprocess behavior.
- Does `health` pass but `ui-status` fail? Then MCP starts, but UI bridge
  descriptor/connection is stale or the UI bridge is down.
- Do both fresh-process commands pass, but Codex MCP tools fail? Then the Codex
  embedded MCP connection is stale. Restart that Codex client or kill its MCP
  child.

One prior takeover attempt saw napari theme-cache writes fail under
`/home/ts/.cache` in a read-only sandbox. If tests fail before MCP assertions
with a cache write error, rerun with:

```bash
XDG_CACHE_HOME=/tmp/openhcs-test-cache
```

Do not confuse that environment failure with an OpenHCS MCP protocol failure.

## In-Progress Edit: dev_client Workflow Polling

`openhcs/mcp/dev_client.py` currently has an in-progress implementation of:

```bash
python -m openhcs.mcp.dev_client selected-workflow compile_plate --poll-state ...
```

Relevant symbols currently present:

- `WorkflowStatePollPolicy`
- `call_selected_workflow_with_state_poll`
- `workflow_poll_summary_result`
- `--poll-state`
- `--poll-selection-mode`
- `--poll-interval-seconds`
- `--poll-timeout-seconds`

Syntactic check passed after this partial edit:

```bash
. .venv/bin/activate
python -m compileall -q openhcs/mcp/dev_client.py
```

But tests have not been updated or rerun after this latest polling edit. Treat
the polling feature as unfinished until the next thread adds focused tests and
live-runs it.

Suggested next steps for this slice:

1. Add tests in `tests/unit/agent/test_mcp_server.py` for parser/projection:
   - `--poll-state`
   - poll interval/timeout args
   - selected workflow call plus follow-up `openhcs_ui_get_state_surface`
   - summary result marks timeout as `mcp_error=true`
2. Add small unit tests for:
   - `WorkflowStatePollPolicy` terminal state criteria for init/compile/run
   - `workflow_poll_has_reached_terminal_state`
3. Run focused test file:

   ```bash
   . .venv/bin/activate
   PYTEST_ADDOPTS='--no-cov' python -m pytest -q tests/unit/agent/test_mcp_server.py
   ```

4. Run live dev-client smoke against the current UI:

   ```bash
   . .venv/bin/activate
   python -m openhcs.mcp.dev_client selected-workflow compile_plate \
     --poll-state \
     --descriptor-file-path /run/user/1000/openhcs/ui-bridge/ui_bridge_ui-29ee7f52-e754-4028-bc3d-b8accb3bcedd.json \
     --timeout-ms 1000
   ```

5. Run advisor only on production files touched, not tests:

   ```bash
   . .venv/bin/activate
   PYTHONPATH=/home/ts/code/projects/nominal-refactor-advisor \
     python -m nominal_refactor_advisor --no-auto-context-root openhcs/mcp/dev_client.py
   ```

Do not add server-side blocking workflow semantics just to avoid polling in the
client. The MCP tool should remain a dispatch/status primitive; the dev client
can compose dispatch plus polling for development ergonomics.

## Important Architecture Reasoning

### Public Boundary

The MCP public split must stay:

```text
OpenHCS internals -> openhcs.agent services -> openhcs.agent DTOs -> MCP adapter
```

`openhcs.mcp` is the transport adapter, not the domain authority. If a concept
needs to be part of the agent-facing API, define it in `openhcs.agent` DTOs or
services first, then expose it over MCP.

Agent-facing payloads should be bounded and versioned with
`schema_version="openhcs.agent.v1"`. Do not expose raw `PipelineOrchestrator`,
`FileManager`, arbitrary PyQt objects, giant raw artifacts, or unbounded host
paths as first-class API. Prefer opaque IDs, bounded summaries, and explicit
resources/tools.

### UI Bridge

`UiAgentBridgeService` has too much authority currently. Advisor flags it as a
larger authority-boundary refactor target. Do not get stuck in that now unless a
user-facing bug requires it.

Recent fixed behavior:

- `selected_plate_workflow()` now resolves implicit current selection into
  explicit target scope ids and revision token before invoking the generic action
  path.
- Rejected action results now preserve target scope ids and selection revision
  token when available.
- `PlateManagerActionProvider` projects disabled action reasons from the central
  `PlateOperationValidator`; it should not invent parallel workflow precondition
  logic.
- `CompilePlateOperationValidator` requires an orchestrator whose state has
  completed initialization. A created-but-not-initialized plate must reject
  compile with `orchestrator_not_initialized` and recovery `init_plate`.

Design principle: agents plan from `list_actions` and state surfaces before
invoking. Disabled/error metadata must be as informative as invocation errors.

### ObjectState / Code Documents

The code-document bridge is meant to support UI<->code round-tripping with
snapshots:

- read code document
- validate source
- apply source
- get `receipt`, `operation_id`, `current_revision_token`,
  `current_snapshot`, `pre_apply_snapshot`, `post_apply_snapshot`,
  `undo_snapshot`

The snapshot/revision/undo contract is important. Do not weaken it or hide it in
transport-specific wrappers.

The PlateManager code document must be able to send code with no plates. Empty
plate state is a valid state, not an error.

### Viewer / Napari Streaming

Important prior conclusions:

- `component_axis_semantics` is the shared carrier for viewer layout, role
  policy, and value-domain ownership.
- `AllComponents` is the durable source for component ordering/defaults.
- Do not hardcode component names like `"channel"` or `"site"` in generic code.
  If a module needs component-specific behavior, it should declare it nominally.
- `source_mode=STACK` changes topology; it is not just a nicer title format.
- `source_<step>` naming came from the layer identity contract, not just a UI
  string bug.
- Validate viewer bugs by inspecting actual layer axes/payloads/slices, not just
  end-state screenshots.

### CellProfiler Runtime

The user wants long-term centralized fixes, not instance patches:

- `.cppipe` is only an input format. It should parse into regular pycodified
  pipeline/function-step declarations. Do not make `.cppipe` special at runtime.
- Runtime logic should be computed at compile time from `FunctionStep`
  declarations plus pipeline/global config.
- Module-specific settings belong in the module declaration/metadata. Avoid
  scattering module semantics across adapter, generator, and runtime execution.
- Do not duplicate parameter-name facts in multiple physical locations. Derive
  from existing declarations.
- Hidden runtime artifact binding in `FunctionStep` is suspect if users are not
  meant to edit it. Prefer deriving hidden/runtime-only details in the compiler
  from declarations.

### Advisor Usage

Use the advisor on production files you touch, but do not run it on tests.

Advisor should be treated as:

- a semantic landscape and gate
- a way to identify authority-boundary refactors
- not a license to nibble one local finding at a time

Latest advisor run before this handoff, on production files touched before the
polling edit, was still active:

- 10 semantic candidates
- 4 SSOT-critical signals
- main target: `openhcs/pyqt_gui/services/ui_agent_bridge.py:UiAgentBridgeService`
- next larger refactor: collapse action invocation/mutation/result construction
  authority inside or below `UiAgentBridgeService`

Do not enter that refactor loop unless the current user-facing task is blocked
by it.

## Anti-Slop Guardrails

The user repeatedly corrected these patterns. Preserve these rules:

- No broad `hasattr` / `getattr` defensive probing when contracts can be nominal.
- No `Any` or `object` as a way to avoid typing semantics.
- No compatibility shims unless explicitly required; migrate properly.
- No duplicated hardcoded defaults in multiple files.
- No repeated literal component axes in generic code.
- No private dangling predicate/helper piles where a nominal class/hook owns the
  behavior.
- No local fallback chains like `or ""` everywhere; fail loud with contracts.
- No per-module special cases in generic runtime paths unless the module owns a
  nominal hook/declaration.
- If the same edit is needed in many files, suspect a missing inheritance,
  registry, or shared authority.
- Prefer `AutoRegisterMeta` and registry/MRO ordering where the codebase already
  uses those patterns. Do not invent priority tables when MRO/nominal ordering is
  the semantic contract.
- Reduce OpenHCS UI surface area where possible by pushing generic widget/window
  composition into `pyqt-reactive`.
- Avoid new wrappers that only rename or pass through behavior. Refactors should
  collapse parallel paths and reduce indirection.

## MCP/Runtime Testing Loop

Use fresh-process dev client:

```bash
. .venv/bin/activate
python -m openhcs.mcp.dev_client health
python -m openhcs.mcp.dev_client ui-status --timeout-ms 1000
python -m openhcs.mcp.dev_client windows --timeout-ms 1000
python -m openhcs.mcp.dev_client state-surface --selection-mode all --timeout-ms 1000
```

When targeting the currently open UI explicitly:

```bash
DESCRIPTOR=/run/user/1000/openhcs/ui-bridge/ui_bridge_ui-29ee7f52-e754-4028-bc3d-b8accb3bcedd.json

python -m openhcs.mcp.dev_client state-surface \
  --descriptor-file-path "$DESCRIPTOR" \
  --selection-mode all \
  --timeout-ms 1000
```

Do not use the Codex embedded MCP connection as the development oracle while
actively editing MCP. It does not reliably reconnect to fresh source and can
produce stale `Transport closed` behavior. Fresh-process `dev_client` is the
correct development loop.

## Subagent State

At handoff time, no subagent from this thread is active.

Previous subagent `Lovelace` completed and was closed. It fixed the dev-client
`state-surface` argument mismatch:

- command now uses `--revision-token`
- MCP arg is `revision_token`
- not `base_revision_token`

The attempted MCP knowledgebase/onboarding subagent was interrupted before it
started. The user indicated they may open another Codex CLI to work on that in
parallel. Treat MCP knowledgebase/onboarding as potentially owned by another
agent; coordinate before touching the same files.

## Useful Live Commands

Check UI bridge descriptor:

```bash
find /run/user/1000/openhcs/ui-bridge -maxdepth 1 -type f -name '*.json' \
  -print -exec sed -n '1,120p' {} \;
```

Check relevant processes:

```bash
ps -eo pid,ppid,stat,etime,cmd \
  | rg 'python -m openhcs.pyqt_gui|python -m openhcs.mcp|zmq_execution_server|napari_viewer' \
  | rg -v 'rg '
```

Capture PlateManager screenshot:

```bash
. .venv/bin/activate
python -m openhcs.mcp.dev_client call openhcs_ui_snapshot_window --arguments '{
  "window_id": "plate_manager",
  "capture_scope": "window",
  "output_dir_path": "/tmp/openhcs_mcp_screenshots",
  "connection": {
    "descriptor_file_path": "/run/user/1000/openhcs/ui-bridge/ui_bridge_ui-29ee7f52-e754-4028-bc3d-b8accb3bcedd.json",
    "timeout_ms": 1000
  }
}'
```

Read selected orchestrator code:

```bash
. .venv/bin/activate
python -m openhcs.mcp.dev_client call openhcs_ui_get_code_document --arguments '{
  "document_id": "plate_manager.orchestrator_config",
  "selection_mode": "selected",
  "connection": {
    "descriptor_file_path": "/run/user/1000/openhcs/ui-bridge/ui_bridge_ui-29ee7f52-e754-4028-bc3d-b8accb3bcedd.json",
    "timeout_ms": 1000
  }
}'
```

## Next Best Work

Short-term, user-facing:

1. Finish and test `selected-workflow --poll-state` in `openhcs/mcp/dev_client.py`.
2. Re-run focused MCP/UI tests.
3. Run advisor only on `openhcs/mcp/dev_client.py` if that is the only
   production file touched.
4. Live-test `--poll-state` against the current UI.

Parallel/orthogonal:

1. MCP knowledgebase/onboarding surface:
   - inspect `openhcs_list_capabilities`
   - inspect `openhcs_get_authoring_context`
   - inspect `openhcs_explain_architecture`
   - improve capability/architecture summaries so agents immediately understand
     OpenHCS concepts and controls
2. Viewer validation:
   - use viewer payload APIs to inspect layers/axes/images/shapes
   - validate Napari streaming across CellProfiler benchmark pipelines
   - avoid just eyeballing final screenshots
3. Larger architecture debt:
   - `UiAgentBridgeService` action/mutation authority boundary
   - CellProfiler runtime/module execution declaration ownership
   - viewer component axis semantics consistency

Do not mark the active goal complete until the UI/MCP experience is proved
end-to-end for code editing, init/compile/run, state polling, screenshots/widget
navigation, and viewer inspection, with no new semantic debt introduced.
