# Global UI and ZMQ Config Tabs Plan

**Date:** 2026-07-19
**Status:** implementation in progress
**Scope:** Global Config window tabs, one UI settings authority, one audited ZMQ
settings authority, ObjectState/config integration, and live MCP UI acceptance

## Goals

1. Make Global Config an object-driven two-tab editor: the existing
   `GlobalPipelineConfig` is tab one and one nominal `UIConfig` object is tab two.
2. Consolidate ZMQ settings onto one nominal authority at the scope proven by
   lifecycle and inheritance evidence, then delete superseded parallel stores.
3. Preserve live ObjectState behavior, global inheritance, source generation,
   serialization, and the canonical UI-to-ZMQ execution path.
4. Prove the result in the actual running PyQt UI through the MCP dev client.
5. Prove a named-source-binding code-mode save and same-path revert rebuild one
   canonical plate/microscope/component state and propagate it in-process to Image
   Browser, Metadata Browser, function-list choices, and group-by/component
   selectors through the existing plate/config state event.

## Non-Goals

- No changes to code-mode path factoring, relative path compilation, Artifact-tab
  semantics, CZI/source-binding behavior, CellProfiler lowering, or benchmark
  kernels.
- No copied field maps, UI-owned mirrors, compatibility settings stores,
  heuristic environment fallbacks, or manual synchronization.
- No additional public config surface beyond the single necessary `UIConfig` and
  the single necessary ZMQ config authority.

## Active Coordination

- Adjacent UI owner: worker `019f7961-e6f2-7111-8355-a06d075bbedf`, tracked in
  `docs/plans/code_mode_paths_and_artifact_tab_plan_20260719.md`.
- Bio-Formats/CZI owner: worker `019f799a-c5c0-7400-b5e9-1e7f92c70e60`, tracked
  in `docs/plans/czi_source_bindings_and_zarr_audit_20260719.md`.
- Shared ledger: `.agents/global-ui-zmq-config-tabs.md`.

## Authority Inventory

- `GlobalPipelineConfig` (`openhcs/core/config.py:216`) is the true owner for
  pipeline-global execution semantics. `@auto_create_decorator` and the
  `@global_pipeline_config` injections create `PipelineConfig` plus lazy nested
  config inheritance. `ObjectState` scope `""` and the saved/live global config
  context carry this root through the application (`openhcs/pyqt_gui/app.py:117`,
  `external/ObjectState/src/objectstate/global_config.py:19`).
- `PyQtGUIConfig` (`openhcs/pyqt_gui/config.py:447`) is already the true UI
  settings object graph. It owns performance, progress, bridge, window, style,
  logging, debug, update, and plugin settings. Adding a second `UIConfig` beside
  it would be a nominal mirror; the migration renames this owner to `UIConfig`
  and deletes the old name rather than aliasing it.
- `PyQtGuiRuntimeContext` (`openhcs/pyqt_gui/config.py:488`) is a startup/runtime
  composition value, not a settings owner. It will continue to hold the exact
  `UIConfig` and `GlobalPipelineConfig` objects.
- `ConfigWindow` (`openhcs/pyqt_gui/windows/config_window.py:149`) is currently a
  single-object `ObjectState` editor. `ActionTabbedWindowBody` and `ActionTabSpec`
  in pyqt-reactive are the existing tab/layout authority used by
  `DualEditorWindow`; this window will reuse them. A config tab will receive the
  authoritative config object and derive its dataclass type, ObjectState,
  ParameterFormManager, tree, and code document from that object.
- `OPENHCS_ZMQ_CONFIG` (`openhcs/runtime/zmq_config.py:13`) is currently only an
  instance of zmqruntime's transport-topology dataclass. Its defaults are
  duplicated in `openhcs/constants/constants.py:339`, UI client services
  (`zmq_client_service.py:25`, `plate_manager.py:793`), client/server constructors,
  the standalone launcher, server-browser scans, and compiled-artifact TTL.
- zmqruntime `ZMQConfig` (`external/zmqruntime/src/zmqruntime/config.py:15`) is the
  generic transport-topology owner. The OpenHCS runtime config will extend this
  owner with OpenHCS endpoint/control/lifecycle defaults; it will not introduce a
  parallel topology object.
- `ExecutionConnectionSpec` (`openhcs/agent/dto/execution.py:34`) is a per-request
  endpoint identity/override. It is not configuration authority and remains
  separate: explicit remote host/port requests must not mutate process defaults.
- `StreamingDefaults`/`StreamingConfig` (`openhcs/core/config.py:733`) own
  per-pipeline viewer transport, persistence, host, and viewer ports. Those are
  actual pipeline outputs and remain registered/inheritable. They are not moved
  into execution ZMQ settings.
- `AgentUiBridgeConfig` (`openhcs/pyqt_gui/config.py:351`) owns one distinct UI
  bridge endpoint plus bridge security/limits. It uses shared ZMQ topology but is
  not the execution server endpoint and is not folded into execution settings.
- The AST inventory command walked `openhcs/runtime`, `openhcs/agent`, and
  `openhcs/pyqt_gui`, collecting defaulted parameters/assignments whose declared
  names contain port, timeout, retry, persistent, transport, host, IPC, ACK,
  poll, TTL, or linger. It found no execution retry policy to migrate; none will
  be invented. It identified the execution defaults listed above plus separate
  viewer/UI-bridge protocol timeouts that remain with their owning protocols.

## Audited Ownership Decision

### UI settings

`UIConfig` is application-global process state. It does not participate in
pipeline compilation or lazy `PipelineConfig` inheritance. The existing
`PyQtGUIConfig` object graph is renamed in place and registered as its own
ObjectState for editing; it is not decorated with `@global_pipeline_config` and
does not widen the pipeline public API.

### ZMQ settings

Execution ZMQ settings are process/application transport settings, not pipeline
semantics. A pipeline cannot select the transport needed to deliver its own
serialized `GlobalPipelineConfig`/`PipelineConfig`; registering ZMQ under
`GlobalPipelineConfig` would create a circular and misleading ownership model.

One `OpenHCSZMQConfig` dataclass will extend zmqruntime's existing `ZMQConfig`
topology owner with OpenHCS defaults for client/server endpoints, default
transport, persistence, control/connect/scan timing, server polling, scan width,
and compiled-artifact lifetime. The exact object is nested in `UIConfig` for the
PyQt process and explicitly accepted by execution clients, servers, launch, and
UI workflow services. The module-level `OPENHCS_ZMQ_CONFIG` remains only the
default instance of this class for headless callers; it is not a second schema.

Explicit `ExecutionConnectionSpec` values continue to override endpoint fields
for one request. Viewer streaming and UI bridge settings remain with their
existing nominal owners because they identify different processes/protocols.

### Config window

The Global Config window edits two authoritative objects in one window. Each tab
is built from the object itself, not a copied field map. The existing generic
ParameterFormManager/ObjectState/code-document paths are reused per object. Save,
reset, cancel, dirty state, and code mode operate on both exact ObjectStates.

## Exact API and File Migration

- Rename `PyQtGUIConfig` to `UIConfig` in `openhcs/pyqt_gui/config.py` and all
  typed consumers; do not leave an alias.
- Add `UIConfig.zmq: OpenHCSZMQConfig`; keep `PyQtGuiRuntimeContext` as the exact
  composition of `UIConfig` and `GlobalPipelineConfig`, with methods to replace
  either object immutably.
- Extend `openhcs/runtime/zmq_config.py` with `OpenHCSZMQConfig` and make
  `OPENHCS_ZMQ_CONFIG` its default instance. Remove execution ZMQ constants from
  `openhcs/constants/constants.py` and migrate consumers to the config object.
- Make `ZMQExecutionClient` and `ZMQExecutionServer` accept the exact config
  object. Resolve omitted endpoint/timing values from it. Pass the full object
  across spawned-server process launch without enumerating fields.
- Migrate `ZMQClientService`, PlateManager construction, server manager/scanning,
  reactor log scanning, kill service, image browser, streaming-port enumeration,
  TUI defaults, agent runtime gateways, and bridge transport topology to the
  config object or the exact UI-provided instance.
- Refactor the Global Config creation path in
  `openhcs/pyqt_gui/services/window_handlers.py` to provide both the current
  `GlobalPipelineConfig` and current `UIConfig`, and persist/publish each owner.
- Refactor `ConfigWindow` around object-driven tab editors using
  `ActionTabbedWindowBody`; PipelineConfig windows remain one-tab object editors,
  while the global window receives the two objects.
- Register one `UIConfig` ObjectState during application startup at its explicit
  non-pipeline scope and update `PyQtGuiRuntimeContext` on save. Preserve the
  existing `GlobalPipelineConfig` root scope and inheritance unchanged.
- Add generic typed config-cache entry points if persistence requires them;
  remove the PyQt hardcoded alias/parallel cache wrapper rather than adding a
  second UI-only persistence implementation.

## Executable Plan

- [x] Complete AST/call-site inventory and classify every candidate owner.
- [x] Record the ownership decision and exact migration/deletion map.
- [x] Implement object-driven Global Config tabs.
- [x] Implement and integrate `UIConfig` through the established ObjectState
  configuration mechanisms.
- [x] Implement and migrate the ZMQ config at the audited scope.
- [x] Delete all superseded settings/config paths.
- [x] Remove the incomplete main-window `global_config` mirror migration and
  route PyQt services through `PyQtGuiRuntimeContext.pipeline_runtime`.
- [x] Add focused architecture, ObjectState, serialization, and PyQt coverage.
- [x] Delete the rejected `ConfigWindowStateResolver` dispatch layer. Make
  `ConfigWindowTabSpec` carry only the exact caller-owned registered `ObjectState`
  and optional save callback, deriving object/type/label/scope from that authority.
- [ ] Migrate GlobalPipelineConfig, UIConfig, and delegated PipelineConfig callers
  and focused tests; keep missing/wrong-state checks at those owning boundaries
  and rerun the focused ObjectState/PyQt/static gate.
- [ ] Verify scoped tests and static architecture constraints.
- [ ] Verify the live running UI through MCP and canonical ZMQ execution.
- [ ] Through public pipeline-config/code mode, save a named-source-binding edit,
  reinitialize the selected plate, and verify immediate microscope metadata and
  visible component-domain changes; revert through the same save path and verify
  exact state return.
- [ ] For both transitions, verify Image Browser, Metadata Browser, function-list
  choices, and group-by/component selectors consume the identical rebuilt state
  through the canonical plate/config event, with no copied cache or per-widget
  refresh logic.
- [ ] After the source-mapping/UI critical path, run official30 F5: cold-launch
  the execution server and prove the first compile submission waits on
  zmqruntime's existing lifecycle ready signal, without retries, sleeps, or a
  second readiness flag.

## Progress

- **2026-07-19 07:13 EDT - config-window authority correction:** Parent review
  classified `ConfigWindowStateResolver` as forbidden caller-side dispatch. The
  existing `ObjectState`, `ParameterFormManager`, `ConfigHierarchyTreeHelper`,
  and `ActionTabbedWindowBody` authorities already provide state, form, hierarchy,
  and tab composition. The next production batch deletes the resolver, removes
  parallel tab `config`/`scope_id` storage, passes exact registered states from
  Global/UI and plate owners, and migrates focused tests before any live retry.
  Source acceptance remains one canonical rebuilt plate metadata event across
  Image Browser, Metadata Browser, exact function-list parameter choices, and
  group-by/component selectors; no fuzzy discovery or widget-local refresh is in
  scope.
- **2026-07-19 07:13 EDT - state-only config tabs implemented:** Deleted the
  resolver and concrete config imports/branches from the generic editor.
  `ConfigWindowTabSpec` now stores only the exact `ObjectState` and callback;
  object/type/label/scope and delegated restore behavior are derived from that
  state. Global/UI setup states and validated plate orchestrator states are passed
  directly, and PlateManager no longer converts a missing scope to global scope.
  Focused tests now assert owner-boundary missing/wrong-state failures and exact
  state identity. Next action is focused verification.
- **2026-07-19 07:11 EDT - expanded source-state propagation gate:** Coordinated
  the pre-ZMQ `SourcePixelRef` initialization blocker directly into Kepler's CZI/
  source-binding plan. This worker owns the real MCP code-mode save -> reinitialize
  -> visible state proof and same-path revert. Acceptance now requires Image
  Browser, Metadata Browser, function-list choices, and group-by/component
  selectors to update in-process from the same rebuilt microscope/component
  metadata via the existing plate/config state event. No copied UI cache or
  widget-specific refresh is permitted. The old leased GUI imported pre-change
  source and will be replaced before retry.
- **2026-07-19 07:12 EDT - clean current-source process:** Terminated only the
  leased stale GUI PID `1873825`; no execution/viewer child existed. Started the
  current tree as PID `1919463` on bridge `7891`, descriptor
  `ui_bridge_ui-3189dd3c-0d30-4723-8122-8b2b188ff7b0.json`, with log
  `/tmp/openhcs_global_ui_zmq_live_retry.log`. This clean process imports Kepler's
  current physical-store-first microscope routing and is the acceptance target.
- **2026-07-19 07:13 EDT - source handoff and F5 ordering:** Kepler records the
  exact built-in plate with 216 structured `SourcePixelRef` payloads plus focused
  handler/VFS and direct plane-load evidence, so the fresh process can retry Init.
  The cold ZMQ official30 F5 proof remains after source-binding/UI propagation and
  must use only zmqruntime's authoritative lifecycle ready signal.
- **2026-07-19 07:14 EDT - source blocker cleared live:** Reapplied the exact
  eight-step built-in source through the writable orchestrator code document;
  validation returned `valid` in 2.59s and async apply completed `applied`. The
  real MCP `selected-workflow init_plate` then passed in 3.85s with
  `initialized=true` and `orchestrator_state=ready`. No UI fallback or metadata
  mirror was needed. Next is baseline capture and the reversible named-binding
  save/reinitialize propagation proof.

- **2026-07-19 EDT - assignment and boundary setup:** Read current agent
  guidance, enumerated active peer ownership, inspected the existing code-mode
  and CZI plans, located the primary config/ZMQ/window files, and created the
  required shared ledger. No production or test edits have been made.
- **2026-07-19 EDT - authority audit complete:** Used `rg` plus an AST walk over
  runtime/agent/PyQt declarations and defaults. Resolved `PyQtGUIConfig` as the
  existing UI owner to rename, zmqruntime `ZMQConfig` as transport topology to
  extend, `ExecutionConnectionSpec` as a per-request override, and
  `StreamingConfig` as the distinct pipeline-level viewer owner. Decided that
  execution ZMQ settings are process-global and nested in `UIConfig`, not
  registered into `GlobalPipelineConfig`/`PipelineConfig`. Next action is the
  production migration.
- **2026-07-19 06:04 EDT - owner and tab migration gate:** Renamed the existing
  GUI root in place to `UIConfig`, nested exact shortcut/ZMQ config objects,
  removed the parallel shortcut singleton and unused shortcut service, added the
  immutable `OpenHCSZMQConfig` extension of zmqruntime topology, and converted
  `ConfigWindow` to object-derived tab specs/sessions using
  `ActionTabbedWindowBody`. An import cleanup left two stale `Optional[int]`
  annotations; both were migrated to `int | None` with no alias/import ceremony.
  Direct import/default construction passed and focused PyQt collection found 20
  tests. A first collection attempt briefly observed the adjacent CZI worker's
  in-flight adapter rename, then passed after that worker completed the matching
  edit. Next action: migrate all production constructors and ZMQ consumers.
- **2026-07-19 06:15 EDT - nominal scope and first transport gate:** Removed the
  free-standing `UI_CONFIG_SCOPE_ID`. `UIConfig.object_state_scope_id()` now
  derives its ObjectState scope from the owning module-qualified nominal type;
  application registration and window lookup call that owner directly. Extended
  `OpenHCSZMQConfig` with the previously copied connection timing and migrated
  the canonical execution client/server, spawned launcher, plate workflow,
  server browser/kill/progress clients, reactor scan provider, and UI bridge to
  receive the exact resolved config. The spawned server receives one pycodified
  config object instead of enumerated fields. Focused tests passed `21 passed in
  11.50s`; direct pycodify round-trip equality passed.
- **2026-07-19 06:18 EDT - structured remaining-consumer inventory:** Re-read all
  active peer ownership and parent-note sections before continuing. Ran the
  updated NominalRefactorAdvisor `e8a3c50` over the ZMQ/UI owner files with
  `--context-root openhcs --json --json-payload loop`; it emitted no actionable
  owner findings, so no advisor-generated rewrite was applied. A stdlib-AST walk
  then covered every `.py` under `openhcs/runtime`, `openhcs/pyqt_gui`,
  `openhcs/agent`, and `openhcs/textual_tui`, inventorying nominal ZMQ constants
  and calls to the five client/server/service constructors. It identified 20
  files: actual PyQt execution consumers now carry `config`; remaining default
  constant uses are headless/viewer/agent construction boundaries, plus the
  image-browser and TUI copied defaults still to migrate. The mechanical batch
  shape is constructor config injection and deletion of copied endpoint/control
  arguments, with protocol-specific viewer and UI-bridge settings retained on
  their nominal owners.
- **2026-07-19 06:29 EDT - resumed diff/advisor/live-startup audit:** Re-read root
  guidance and active peer boundaries, then audited the current owner diff and
  running processes. The only live NominalRefactorAdvisor workers are orphaned
  forkserver processes for the CZI peer's source-projection scan; they have no
  UI/ZMQ result stream, so they were left untouched. The recorded `e8a3c50`
  UI/ZMQ scan is complete and remains empty. Focused config/ZMQ tests pass
  `23 passed in 9.22s`, but a real offscreen app construction fails before the
  main window opens because `PyQtServiceAdapter.get_global_config()` still reads
  the removed `main_window.global_config` parallel store. The live inventory also
  found `ImageBrowserWidget` omitting the required ZMQ config, execution-port
  constants still mixed into generic streaming discovery, the TUI hardcoding
  `7777`, and agent execution/runtime gateways constructing clients without one
  injected `OpenHCSZMQConfig`. This supersedes the narrower 20-file inventory;
  next is one coherent object-propagation and deletion batch.
- **2026-07-19 06:35 EDT - consumer/deletion and startup gate:** Routed the PyQt
  service adapter through the runtime context and deleted all remaining
  main-window global-config mirror access. Passed the exact ZMQ config through
  reactor scanning, embedded/floating server managers, image browsers,
  plate/live-result viewers, TUI execution, agent client factories, and runtime
  server discovery. Generic streaming discovery now owns only registered viewer
  ports; callers prepend the execution port from the config object. Deleted the
  old constants and the import-time `POLYSTORE_ZMQ_*` environment mirror.
  Spawned execution servers now receive one `replace(...)`-resolved pycodified
  object and launcher flags are optional explicit overrides. `py_compile` passed
  across the batch. The exact offscreen startup probe that previously failed now
  creates `OpenHCSMainWindow` successfully and reports `[7777, 5555, 5556]` from
  the resolved config. Next action is test migration and expanded owner gates.
- **2026-07-19 06:39 EDT - owner regression gate:** Migrated tests away from the
  deleted timeout/topology constants and added coverage for exact nested
  ObjectState reconstruction, UIConfig/OpenHCSZMQConfig pycodify equality, saving
  both object-driven tabs, runtime-context-only propagation, exact config
  injection into both agent gateways, and dynamic default port/timeout use.
  Static AST/deletion tests reject old config names, environment mirrors,
  main-window stores, and any production ZMQ/UI browser constructor without a
  config object. The new batch passes `25 passed in 12.73s`; the migrated focused
  UI/ZMQ/streaming suite passes `41 passed in 6.83s`. Next action is broad
  agent/PyQt/static verification.
- **2026-07-19 06:40 EDT - broad test classification:** The full agent service
  file passes `70 passed in 4.61s`. The broader PyQt/serialization batch produced
  `94 passed, 20 failed`; all failures are stale constructor fixtures, with three
  execution-control tests missing `config=` and the shared plate-manager harness
  missing `gui_config=`. No runtime assertion failed. These fixtures will receive
  the exact default object explicitly; production constructors remain strict and
  no compatibility fallback will be introduced.
- **2026-07-19 06:43 EDT - broad regression rerun:** Migrated only the stale test
  harnesses to pass the exact default config object. The same broad
  PyQt/ObjectState/serialization/ZMQ command now passes `114 passed in 13.65s`;
  the external `zmqruntime` suite passes `33 passed in 1.29s`. Re-read the shared
  Parent Notes and peer boundaries. Remaining work is the focused external
  ObjectState gate, scoped static/diff review, and live MCP UI acceptance.
- **2026-07-19 06:44 EDT - ObjectState gate:** The focused external ObjectState
  config, field-access, restore, instance-update, and subfield-semantics suite
  passes `23 passed in 1.95s`. Next is the MCP/UI bridge regression file and live
  GUI exercise.
- **2026-07-19 06:45 EDT - UI bridge gate:** The complete PyQt UI-agent bridge
  regression file passes `70 passed in 7.95s`. Re-read Parent Notes and the
  remaining executable checklist. Next is scoped lint/compile/static/diff
  verification, then the live MCP client.
- **2026-07-19 06:49 EDT - static/diff gate:** All migrated production files
  compile; the owner modules and focused tests pass Ruff. Deleted-authority
  searches, copied-field-map inspection, and scoped `git diff --check` are clean.
  A broad touched-file Ruff run still reports existing lint debt in shared large
  modules; changed-line review isolated and removed the two imports made stale by
  this migration. Next is live MCP UI acceptance.
- **2026-07-19 06:51 EDT - expanded live acceptance:** Before closure, the user
  expanded the gate to the existing built-in test plate through the real UI ->
  ZMQ compile/execute path. Acceptance now requires completed analysis,
  inspectable measurement snapshots, compiled/runtime artifact provenance,
  Napari ROI layers and coherent visibility/state, plus exact MCP commands,
  timings, screenshots, logs, and owner-classified failures. No new fixture will
  be created and unrelated runtime failures will not be patched in this slice.
- **2026-07-19 06:55 EDT - process coordination:** Read
  `.agents/builtin-testplate-mcp-e2e.md` and recorded a disjoint lease: this
  worker owns live PyQt PID `1873825`, bridge port `7891`, and all ZMQ/Napari
  children from the current run; the new worker owns only
  `scripts/mcp_thesis_demo_live.py` and its focused tests. The existing
  ImageXpress acceptance plate is loaded and selected but not initialized. No
  rehearsal-script file will be edited.
- **2026-07-19 06:58 EDT - remaining fallback audit:** Parent review identified
  `LiveMeasurementsWindow.__init__` still defaulting to `OPENHCS_ZMQ_CONFIG`.
  Constructor inventory found one production caller, which already passes
  `self.gui_config.zmq`; no headless owner exists for this dialog. Removed the
  fallback/import and added a signature plus production-call AST gate so direct
  tests must inject explicitly. The focused authority/plate-manager batch passes
  `26 passed in 15.69s`; scoped Ruff and `py_compile` pass.
- **2026-07-19 06:59 EDT - built-in live preflight:** Re-read both coordination
  ledgers and confirmed the leased PyQt PID `1873825` and bridge `7891` are live,
  with no execution-server or Napari child. `ui-smoke` passed in 2.44s. MCP state
  identifies the selected, uninitialized existing ImageXpress `zstack_plate` and
  the exact output root; the pipeline state returned all eight expected steps in
  2.25s, ending in the public `count_cells_single_channel` Cell Counting step.
  Snapshot `20260719T105848841312Z_pipeline_editor_Pipeline_Editor.png` (SHA-256
  `04028f3a0808e5ba5b3fa99b343f85f2ab4958f45e897b2e70d4670efe753e72`)
  visibly agrees and shows the correct pre-init action state. Next is the real UI
  selected-workflow init, compile, and execute path; no fixture or direct runtime
  bypass will be used.
- **2026-07-19 07:00 EDT - built-in init owner blocker:** The exact real UI command
  `.venv/bin/python -m openhcs.mcp.dev_client selected-workflow init_plate
  --poll-state --poll-selection-mode selected --poll-interval-seconds 0.5
  --poll-timeout-seconds 180 --timeout-seconds 210 --descriptor-file-path
  /run/user/1000/openhcs/ui-bridge/ui_bridge_ui-1fc460b3-f0f3-4537-b0ea-e506efd665df.json
  --timeout-ms 2000 --json` was accepted and reached `init_failed` in 3.24s.
  `OpenHCSMicroscopeHandler` reaches `VirtualWorkspaceBackend._load_mapping()`,
  where `SourcePixelRef.from_workspace_mapping()` rejects the built-in fixture's
  legacy string-valued `workspace_mapping` with `TypeError: SourcePixelRef
  workspace mapping must be structured.` This occurs before any ZMQ server,
  compile, analysis, artifact runtime update, measurement snapshot, or Napari
  process. It is assigned to the active CZI/source-binding/virtual-workspace owner;
  this slice did not alter the fixture or runtime. Snapshot
  `20260719T110013514184Z_plate_manager_Plate_Manager.png` (SHA-256
  `42113f368a3d894defb61aed564eabc2b3788e003fc67f34fa259f9d265ed0a8`)
  visibly records `Init Failed` and zero initialized plates. The downstream live
  gate remains blocked pending that owner boundary.
- **2026-07-19 07:04 EDT - guided UI explanation under failure:** Continued only
  through non-runtime public MCP surfaces. Selected `functionstep_7` from the
  pipeline-editor widget tree, invoked `edit_step`, and opened the visible Cell
  Counting editor. The clean callable code document returned the exact watershed
  config (area 40..200, preprocessing false, segmentation mask true) in 2.49s;
  ObjectState returned 76 fields in 4.73s and resolved inherited Napari streaming
  to enabled, persistent IPC port 5555. The Step Settings screenshot (SHA-256
  `3470af4f7e01338aaf8e00156ff0e580012597cb981c928fb94a5a7d0d3d54e3`)
  agrees with the selected public step. MCP switched to the Artifacts tab and its
  screenshot (SHA-256
  `2f5abf4cf9f97b13e5eed1de145b5a97bf2ed029f34d90f8c681f0625ae63103`)
  correctly renders `No compiled artifact plan.` Compile/Run remain disabled with
  an explicit Init-first explanation. The discoverable docs/function/config/UI
  responses are coherent and avoid stale success claims; compiled/runtime artifact
  provenance, measurements, and Napari layer acceptance remain blocked by init.

## Changed Files

- `.agents/global-ui-zmq-config-tabs.md`
- `docs/plans/global_ui_zmq_config_tabs_plan_20260719.md`
- `external/zmqruntime/src/zmqruntime/config.py`
- `openhcs/runtime/zmq_config.py`
- `openhcs/pyqt_gui/config.py`
- `openhcs/pyqt_gui/main.py`
- `openhcs/pyqt_gui/launch.py`
- `openhcs/pyqt_gui/testing/event_recorder.py`
- `openhcs/pyqt_gui/services/shortcut_manager.py` (deleted)
- `openhcs/pyqt_gui/services/reactor_providers.py`
- `openhcs/pyqt_gui/services/ui_bridge_server.py`
- `openhcs/pyqt_gui/services/plate_manager_batch_workflow.py`
- `openhcs/pyqt_gui/windows/config_edit_session.py`
- `openhcs/pyqt_gui/windows/config_window.py`
- `openhcs/pyqt_gui/widgets/plate_manager.py`
- `openhcs/pyqt_gui/widgets/shared/zmq_server_manager.py`
- `openhcs/pyqt_gui/widgets/shared/server_browser/server_kill_service.py`
- `openhcs/pyqt_gui/widgets/shared/services/zmq_client_service.py`
- `openhcs/pyqt_gui/widgets/shared/services/execution_control_service.py`
- `openhcs/pyqt_gui/widgets/shared/services/batch_workflow_components.py`
- `openhcs/runtime/zmq_execution_client.py`
- `openhcs/runtime/zmq_execution_server.py`
- `openhcs/runtime/zmq_execution_server_launcher.py`
- focused PyQt config-window/ZMQ tests
- typed UI config symbol migrations in production/tests
- agent DTO/service timeout and exact-config migrations
- image-browser, managed-window, plate/live-viewer, reactor, TUI, and streaming
  config propagation
- focused owner/static authority tests and exact-object fixture migrations

## Verification

- UI/ZMQ/config-window `py_compile`: passed.
- `config.py` AST parse and no-`Optional` assertion: passed.
- Direct `UIConfig` import/default construction: passed.
- Focused PyQt collection: 20 tests collected.
- Focused config-window/ZMQ/UI-bridge tests: `21 passed in 11.50s`.
- `OpenHCSZMQConfig` pycodify/exec round trip: passed exact equality.
- NominalRefactorAdvisor owner scan: completed with no emitted actionable finding.
- AST remaining-consumer inventory: 20 files across four production roots.
- Broad PyQt/ObjectState/serialization/ZMQ regression batch: `114 passed in
  13.65s`.
- External `zmqruntime` suite: `33 passed in 1.29s`.
- Focused external ObjectState suite: `23 passed in 1.95s`.
- PyQt UI-agent bridge suite: `70 passed in 7.95s`.
- Owner/focused scoped Ruff: passed; migrated production `py_compile`: passed.
- Deleted-authority/copy-map searches and scoped `git diff --check`: passed.
- Strict `LiveMeasurementsWindow` injection gate: `26 passed in 15.69s`;
  scoped Ruff and `py_compile` passed.
- Live MCP built-in-plate preflight: health 2.44s, selected state 2.82s,
  eight-step pipeline state 2.25s, and visually inspected snapshot 2.56s.
- Built-in selected-workflow init: accepted then `init_failed` in 3.24s; exact GUI
  traceback and visually inspected failure snapshot captured; no ZMQ/Napari child.
- Guided Cell Counting MCP tour: exact scope selection, editor open, clean public
  config source, inherited viewer config, action preconditions, and two visually
  inspected tab snapshots agree with the failed/precompile lifecycle state.
