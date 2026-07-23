# Code-Mode Paths, Relative Compilation Paths, and Artifact Tab Plan

**Date:** 2026-07-19
**Status:** owned implementation complete; final live runtime-enrichment retry blocked by an active peer's transient UI startup failure
**Scope:** source generation, public path declarations and compilation, compiled-artifact inspection, and Artifact-tab static/runtime dataflow

## Goals

1. Generated code factors semantically identified absolute paths into stable `Path` bindings. Repeated plate paths and their longest coherent common base appear once, so moving an orchestrator document between machines requires changing one root binding rather than every plate entry. The same mechanism covers declared path arguments inside pipeline steps and `Path` fields in serialized configuration.
2. Public pipeline APIs accept declared relative paths. Compilation resolves each relative path against the exact root of the plate being compiled, through the existing compilation scope and VFS authority. Authored `FunctionStep`, configuration, `ObjectState`, and generated source remain relative.
3. The Artifact tab renders the exact artifact plans and edges in the compiled execution bundle. Existing runtime-value observations and debug snapshots enrich those rows with latest values; the UI does not reconstruct artifact semantics.

## Non-Goals

- Do not reinterpret arbitrary strings as paths, even when a field or variable name ends in `path`, `file`, `folder`, or `directory`.
- Do not make artifact declarations a substitute for compiled plans in the Artifact tab.
- Do not change source-binding URI semantics, output-artifact planning, materialization-result placement, or backend addressing.
- Do not mutate authored pipeline/config state when compilation resolves paths.
- Do not introduce an artifact registry, a path-field registry, a UI-owned artifact DTO, or a compatibility fallback.

## Architectural Constraints

- Existing nominal owners remain authoritative. Consumers query `CallableContract`, `CompilationPlateScope`, `FileManagerLike`, `PathPlannerPathAuthority`, `ArtifactPlan`/`CompiledStepPlan`, `RuntimeValueStore`, and their existing registries/MRO strategies.
- No mirrored semantic dicts, sets, lists, field-name tables, artifact-name tables, priority tables, or fallback chains.
- No `getattr`/`setattr` fallback, class-name matching, UI-label parsing, path-string heuristics, regex source rewriting, or backend-specific path branches.
- No `Path.cwd()`, `os.getcwd()`, process working directory, environment-variable expansion, ambient current plate, or other process state may affect resolution or factoring.
- Resolution must pass an explicit plate scope and VFS/file-manager instance. Source formatting must pass an explicit factoring plan.
- New AST validation recognizes syntax shapes and previously bound names; it must not duplicate the payload enum or declared callable/config path fields.

## Audit Progress

- [x] Read repository `AGENTS.md`, including nominal-owner, registry/MRO, deletion-gate, and shared-worktree requirements.
- [x] Read active `.agents` coordination documents to avoid ownership conflicts and identified the in-flight artifact-contract collapse; this plan does not restore the removed `ModuleArtifactContract` lattice.
- [x] Searched `__registry__`, `AutoRegisterMeta`, `RegistryConfig`, `RegistryFamily`, `MostDerivedContextStrategyMixin`, `NominalTypeKeyedStrategyMixin`, and `EnumKeyedStrategyMixin` before assigning new ownership.
- [x] Traced code-mode generation through `FunctionStepTransportAuthority`, pycodify formatters, `OrchestratorCodeSource`, plate-manager `ObjectState`, `PlateManagerCodeNamespace`, `OpenHCSCodegenProvider`, and the agent bridge AST validator.
- [x] Inspected the vendored pycodify two-pass import/render model and confirmed that it formats `Path` values but has no document-level binding or common-base mechanism.
- [x] Used Python AST inspection to inventory public callable parameters annotated as `Path`/`Optional[Path]`; classified each against its owning callable instead of using parameter names.
- [x] Traced compilation from plate workspace preparation through `CompilationPlateScope`, config inheritance, `FileManagerLike.resolve_address`, source-binding workspaces, `PathPlannerPathAuthority`, function compilation, and debug-bundle output.
- [x] Resolved the output-path boundary: artifact/materialization output locations remain owned by `PathPlannerPathAuthority`; relative public input/output parameters and the two compiler config locations use the compilation plate scope.
- [x] Traced static artifact data from `ArtifactSpec`/`ArtifactPlan` through `CompiledFunctionInvocation`, `CompiledStepPlan`, `CompiledExecutionBundle`, the ZMQ compile artifact record, compile workflows, and the current declaration-only Artifact preview.
- [x] Traced dynamic artifact data through `RuntimeValueStore`, observation cursors, `RuntimeArtifactAddress`, `DebugArtifactRef`, debug snapshot events, progress notifications, and dual-editor refresh wiring.
- [x] Implemented the audited migrations and ran every acceptance command; the
  unrelated shared-tree documentation findings are recorded below.

No production code or tests were edited during this audit.

## Implementation Checklist

- [x] Re-read `AGENTS.md`, this plan, the live worktree, relevant memory guidance,
  and every active peer's status/ownership before implementation.
- [x] Reconcile the plan with the completed artifact-contract collapse and active
  parity/object-label work; preserve their nominal owners and avoid their files.
- [x] Implement and verify document-level source path factoring and code-mode AST
  validation.
- [x] Implement and verify declaration-owned relative-path resolution in the
  compiler.
- [x] Implement and verify compiled artifact inspection transport and typed plate
  compile state.
- [x] Replace the declaration-only Artifact tab with compiled plans and wire debug
  and normal runtime enrichment.
- [x] Run focused source/compiler/artifact/UI tests, architecture/deletion gates,
  documentation validation, and diff checks.
- [x] Record final changed files, exact command results, deviations, blockers, and
  remaining work.
- [x] Prove compiled inspection and runtime observations traverse the canonical
  UI client -> ZMQ server -> compiler/orchestrator -> spawned worker transport.
- [x] Exercise code editing, relative compilation, compiled Artifact inspection,
  invalidation, and recompile refresh against the live PyQt UI through the
  fresh-process MCP dev client and canonical execution server.
- [ ] Complete the final live runtime-enrichment retry after the active Global
  Config worker restores UI startup; transport-level spawned-worker runtime
  enrichment is already covered by the canonical ZMQ integration test.

## Active Coordination

- `.agents/next-batch-parity-and-performance.md` remains active and owns canonical
  parity integration. This implementation does not edit `pipeline_import.py`,
  CellProfiler module artifact declarations, parity kernels, benchmark timing, or
  cached references.
- `.agents/unify-object-label-measurement-runtime.md` remains in progress and owns
  generic artifact payload normalization plus its analysis producers. This work
  reads compiled plans but does not change `ArtifactType`, `ArtifactOutputPlan`
  normalization, runtime output matching, or those producers.
- `.agents/artifact-contract-collapse.md` has completed its production migration
  and is running final parity acceptance. Its current authority is adopted: exact
  `CallableContract` metadata and compiled invocation edges/output plans are the
  only artifact declarations/plans. No removed module-contract wrapper is restored.
- `.agents/global-ui-zmq-config-tabs.md` is active under worker
  `019f79bf-030b-7a90-a5f7-5c83837ec5c1` and owns `UIConfig`, Global Config tabs,
  config-window/global-settings files, and ZMQ configuration. This plan has not
  edited those files and treats its current startup failure as an external blocker.
- No subagent/worker-spawn capability is available in this session. The parent
  therefore owns each batch serially and uses this document as the shared ledger;
  final acceptance remains serialized with concurrent parity runs.

## Implementation Progress

- **2026-07-19 04:28 EDT - live-tree re-audit:** Re-read `AGENTS.md`, this plan,
  current `git status`, memory guidance, and all peer status/ownership headers.
  Read the complete active integration and object-label plans plus artifact-collapse
  parent notes. Confirmed a heavily shared dirty tree and adopted a no-revert,
  owner-preserving boundary. The source-rendering, explicit compilation-path,
  compiled-inspection transport, and GUI presentation files remain implementable
  without editing concurrent `pipeline_import.py`/module-artifact work. Next action:
  AST/API inventory of the first source-rendering batch, then production edits and
  focused tests.
- **2026-07-19 04:31 EDT - concurrent parity coordination:** User confirmed an
  active external lane in `pipeline_import.py`, module artifact declarations, and
  export observation closure. PercentPositive currently lacks an exact
  ExportToSpreadsheet dependency edge from its externally observed file bundle to
  selected measurement inputs. This implementation will neither diagnose nor
  revert that class. Focused gates exclude it; broader failures with that signature
  are recorded here and reported as overlapping parity work.
- **2026-07-19 04:38 EDT - source batch gate 1:** Added pycodify context
  extensions, nominal path declarations, `CallableContract` path discovery,
  document-level factoring, owner-based plate-scope rendering, declarative AST
  validation, and shared source-renderer delegation. Removed the obsolete
  singular `pipeline_config` codegen fallback from the external provider protocol.
  `py_compile` passed for the changed source batch. Direct generated-source probe
  produced one `/media/alice/T7/screen` root and executable substitutions.
  OpenHCS focused tests passed `11 passed, 62 deselected`. Vendored pycodify passed
  `7 passed`. The combined vendored/repository pytest command cannot collect both
  top-level `tests` packages in one process, so acceptance uses separate commands.
  Plate-manager code-mode tests initially reported `7 passed, 1 failed`; the one
  failure referenced the concurrently deleted `CallableContract.module_artifact_contract`
  API in a stale adjacent assertion. The test was migrated to its actual purpose,
  exact preservation of the public callable identity; no removed contract was
  restored. Next action: rerun the complete source batch, then begin compiler path
  resolution.
- **2026-07-19 04:47 EDT - source batch complete/compiler path gate 1:** The
  complete source command
  `PYTHONPATH=external/pycodify/src:external/pyqt-reactive/src:external/PolyStore/src:. QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/unit/test_source_path_factoring.py tests/unit/test_pycodify_formatters.py tests/unit/test_function_step_transport.py tests/unit/pyqt_gui/test_plate_manager_widget.py tests/unit/pyqt_gui/test_ui_agent_bridge.py -q`
  first reached `123 passed, 1 failed`: the migrated transport assertion lacked
  its `CallableContract` import. That local test defect was corrected; the rerun
  is the final source gate in progress. The structured callable inventory command
  used `ast.walk` over `openhcs/processing/**/*.py` to enumerate function
  parameters whose names ended in `_path`, `_directory`, or `_folder`; it found
  the three multi-template callables, RunImageJMacro, two self-supervised model
  inputs, N2V2 model input/output, two results-directory callables, and one
  non-pipeline consolidation helper. Only the callable owners were annotated;
  the helper stayed ordinary. Added `PlatePathDeclaration` leaves, explicit VFS
  resolution through `CompilationPlateScope`, immutable dataclass-copy traversal,
  callable default resolution, and PathPlanner's compiled-absolute assertion.
- **2026-07-19 05:02 EDT - slots dataclass regression:** Concurrent integration
  exposed `TypeError: vars() argument must have __dict__ attribute` while
  `PipelineCompiler.compile_pipelines` recursively resolved a slots dataclass in
  `resolve_declared_dataclass_paths`. Replaced `vars()` traversal with
  `DataclassFieldAccess.raw_value` over `dataclasses.fields`, preserving the
  nominal dataclass field authority without `getattr` or copied field names.
  Converted the path-resolution test configs to `slots=True`. The exact reported
  regression plus the compilation-path suite passed `8 passed, 1 warning in
  8.62s` with
  `PYTHONPATH=external/pycodify/src:external/pyqt-reactive/src:external/PolyStore/src:. .venv/bin/python -m pytest tests/unit/test_cellprofiler_generated_pipeline_execution.py::test_compiler_derives_runtime_executor_after_generic_transport tests/unit/test_compilation_paths.py -q`.
- **2026-07-19 05:03 EDT - artifact batch audit/migration:** Verified the
  compiled-inspection test migration survived the interrupted edit and the old
  declaration-preview production/test files are deleted. Corrected its runtime
  fixture to use the nominal `RuntimeExecutionAxisScope.from_raw` constructor,
  migrated the stale dual-editor declaration-preview test to compiled-plan
  invalidation, and moved debug-only store construction into
  `DebugControlMessageSupportMixin` so the generic registered strategy root does
  not own debug semantics. Next action: run focused artifact/control/UI tests and
  correct integration failures at their owning APIs.
- **2026-07-19 05:04 EDT - compiler path batch complete:** The broad compiler
  and planner command covering the exact reported slots regression,
  `test_compilation_paths`, compilation sessions, function patterns/contracts,
  axis/materialization planning, ZMQ compilation, MFD presets, and conditional
  CellProfiler images passed `204 passed, 1 warning in 13.92s`. The warning is
  the existing skimage rank-filter bin-count warning. The existing batch workflow
  suite also passed `24 passed in 6.66s`. The latter lacked assertions for typed
  inspection retrieval/publication and generic runtime artifact delivery, so
  those regressions are being added before closing the artifact checklist.
- **2026-07-19 05:05 EDT - compiled-state lifecycle correction:** Re-auditing
  invalidation showed that authored pipeline changes deleted `plate_compiled_data`
  without notifying open Artifact tabs and that a failed recompile could leave an
  older state visible. Consolidated map mutation and signal publication in the
  existing `emit_compiled_state` manager boundary: compile submission and authored
  invalidation publish `None`, successful inspection retrieval publishes a typed
  `PlateCompiledState`. The dual editor accepts only that nominal state or the
  explicit clear event. Added focused retrieval, publication, clear, and generic
  runtime progress listener regressions. Next action: run this focused workflow
  batch, then complete server/client and UI integration gates.
- **2026-07-19 05:10 EDT - artifact integration/static gates:** Added the two
  integration tests named by this plan. Generated public source now executes and
  compiles one relative callable path under two explicit plate roots from two
  unrelated cwd values. The retained compiled bundle flows through registered
  control dispatch, compile workflow state, static Artifact rows, and exact
  runtime enrichment. Added AST gates for reflection, ambient cwd, concrete
  backend imports, deleted preview/router files, and duplicate control routers.
  The focused path/artifact command passed `20 passed in 2.66s`; the two
  integration tests passed `2 passed in 2.64s`.
- **2026-07-19 05:12 EDT - final source and artifact suites:** Vendored pycodify
  passed `9 passed, 1 coverage warning in 0.31s`. The full source/code-mode suite
  initially exposed one stale `PlateManagerCodeWorkflowHarness` lacking the new
  typed publication boundary; the harness was migrated without a fallback and
  the rerun passed `128 passed in 9.48s`. The combined artifact/control/runtime/
  UI/integration suite passed `96 passed in 4.15s`. The repository CellProfiler
  static deletion gate passed `26 passed in 45.47s`; no PercentPositive/export
  parity failure appeared in these gates.
- **2026-07-19 05:15 EDT - static acceptance:** Focused Ruff passed for every new
  owner and its tests. `py_compile` passed for all changed source/compiler/control/
  GUI modules. Scoped root and vendored-pycodify `git diff --check` commands
  passed. Repository-wide `scripts/validate_docs.py` ran and reported 11 findings,
  all in `docs/source/architecture/abstraction_lattices.rst`: stale references to
  concurrently deleted CellProfiler/converter/runtime files and three existing
  non-parsing examples. No plan-owned document was reported, so those files were
  not edited.
- **2026-07-19 05:18 EDT - final API reconciliation:** Renamed the inspection
  constructor to `CompiledArtifactInspection.from_execution_bundle` and added
  `CompileWorkflowService.inspect_compile_artifact`; no aliases were retained.
  Moved absolute-scope validation to `CompilationPlateScope.__post_init__` so an
  invalid root fails at construction. The exact user-reported slots regression
  plus the expanded path suite passed `9 passed, 1 existing skimage warning in
  6.48s`; the final API-focused artifact rerun passed `35 passed in 5.26s`.
- **2026-07-19 05:19 EDT - ledger close:** Ran the literal whole-tree
  `git diff --check`; it reported only trailing whitespace in concurrently
  modified generated CellProfiler reference CSVs under `benchmark/native_refs`.
  Re-ran structural searches and confirmed no unchecked implementation item,
  legacy preview/router reference, or old inspection constructor remains. Added
  the exact changed-file inventory, verification results, deviations, external
  hygiene findings, and remaining-work statement below. Implementation is
  complete; no parity-owned file was diagnosed, reverted, or modified.
- **2026-07-19 05:20 EDT - boundary verification reopened:** User required an
  explicit transport proof beyond the existing direct-router and fake-client
  integration tests. Reopened implementation to verify the canonical UI -> ZMQ
  execution client -> server-held compiler artifact and spawned-worker -> progress
  transport paths. Next action: reuse the existing ZMQ server/process harness and
  add transport-level assertions; no CellProfiler importer/parity file is in scope.
- **2026-07-19 05:26 EDT - transport boundary proof:** Added live TCP ZMQ tests
  that retrieve the server-held compiled projection through
  `CompileWorkflowService`/`ZMQExecutionClient`, and that create a real
  `RuntimeValueStore` observation in a `spawn` child, pass it through
  `ZMQExecutionServer._forward_worker_progress` with worker-claim validation,
  publish it over the server PUB socket, parse it in `ProgressWorkflowService`,
  and enrich the exact Artifact row. Added AST gates proving the GUI does not
  import compiler/orchestrator owners or construct contracts, plans, inspections,
  or runtime addresses; the server control strategy and worker executor remain
  the only projection producers. Command:
  `PYTHONPATH=external/pycodify/src:external/pyqt-reactive/src:external/PolyStore/src:. QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/integration/test_compiled_artifact_inspection.py tests/unit/test_code_mode_path_artifact_architecture.py -q`.
  Result: `9 passed in 14.45s`. Exact transport tests:
  `test_artifact_ui_inspection_crosses_live_zmq_from_server_compiled_record` and
  `test_spawned_worker_runtime_observation_crosses_server_zmq_to_artifact_ui`.
- **2026-07-19 05:27 EDT - live acceptance opened:** User added a live UI gate
  covering path-factor editing, plate-relative compilation, server-held Artifact
  plans, runtime enrichment, invalidation, and recompile refresh through the MCP
  dev/UI bridge. Process and descriptor inspection found no running OpenHCS UI or
  bridge. Next action: launch the actual PyQt application with its built-in bridge
  and canonical ZMQ server integration, then drive only advertised MCP/UI actions
  and record visible results, timing, and logs.
- **2026-07-19 05:36 EDT - live bridge tab-selection gap:** The real dual editor
  visibly uses `ActionTabbedWindowBody`/`QTabBar`, while the existing bridge
  declares `WidgetActionKind.TAB_SELECTOR` only for `QTabWidget` and its mutation
  request has no target index. Consequently the MCP client can inspect the hidden
  Artifact table but cannot select that visible tab. This is a canonical bridge
  deficiency, not an artifact-semantic gap. Extending the existing nominal Qt
  projector/action kind with indexed `QTabBar` selection is now an acceptance
  prerequisite; focused projection, invocation, DTO, and dev-client tests will be
  added before restarting the same PyQt/ZMQ integration. No alternate UI or
  runtime route will be introduced.
- **2026-07-19 05:40 EDT - active peer boundary:** Worker
  `019f79bf-030b-7a90-a5f7-5c83837ec5c1` owns the Global Config dual-tab,
  `UIConfig`, and ZMQ-config slice. This plan does not touch config-window,
  global-settings, `UIConfig`, or ZMQ-config files. Its bridge prerequisite is
  confined to the generic widget projector/invoker and preserves that peer's edits.
- **2026-07-19 05:44 EDT - live runtime fixture blocker:** The canonical MCP
  `run_plate` action compiled and submitted execution through ZMQ and the spawned
  worker, but the selected multi-template fixture terminated with
  `RuntimeSliceProjectionDeclarationError: Runtime-slice projection has no nominal
  strategy for TemplateMatchResult`. UI polling ended `exec_failed` after 6.02 s.
  This is an unrelated runtime-type declaration gap and is not being patched by
  this plan. Runtime acceptance will switch to an existing artifact-producing
  callable whose declared value type already has a nominal runtime projection;
  the failed action/log evidence remains part of the ledger.
- **2026-07-19 05:47 EDT - live path/static/invalidation acceptance:** Drove the
  actual offscreen PyQt application through fresh
  `.venv/bin/python -m openhcs.mcp.dev_client` processes connected to its built-in
  TCP bridge at `127.0.0.1:7888`. `code-document` visibly returned one generated
  `path_root`, `path_1`, and `path_2`; changing only `path_root` from
  `/tmp/openhcs_path_artifact_acceptance` to the copied `_moved` root produced a
  one-line `diff -u`, and `apply-code-document` rebuilt both plate rows. The source
  retained `Path('templates/template.tif')`. `selected-workflow init_plate` then
  `compile_plate` compiled plate A through the normal UI ZMQ client; UI log
  `/tmp/openhcs_code_mode_artifact_ui_20260719_restart.log` records submission at
  `05:41:52.805` and success at `05:41:53.419` (0.614 s). The server cwd had no
  `templates/template.tif`; compilation succeeded against the plate-local file.
  `invoke-widget-action <step-window> 1.0.0.0 --action-kind tab_selector
  --target-index 2` selected the visible `Artifacts` tab. `widget-tree` showed the
  compiled row `A01 | multi_template_crop_reference_channel | output | special |
  match_results | yes | channel: default |
  /tmp/.../plate_a_openhcs/results/A01_match_results_step0.pkl`. Snapshot
  `/tmp/openhcs-mcp-window-snapshots/20260719T094326770158Z_tmp_openhcs_path_artifact_acceptance_moved_plate_a_functionstep_0_Edit_Step_1._Plate-relative_template_match.png`
  has SHA-256 `826f4fd4696df2ec6ba1dcc7d12edf1baab01284e7192cbc574a27c6e27f8bc9`.
  Applying a `skeletonize_and_save` pipeline while that window remained open made
  its table empty immediately; recompiling repopulated exact
  `skeleton_measurements` and `skeleton_rois` rows. This verifies authored-change
  invalidation and server-held recompile refresh without local UI reconstruction.
- **2026-07-19 05:48 EDT - live boundary evidence:** The live UI log contains
  `CompileWorkflowService: Submit compile` and `ZMQExecutionClient: Serialize
  task`; the per-port server log contains `PipelineCompiler`, artifact storage,
  and the compile artifact ID. The UI process did not run compiler logs or create
  plans. This agrees with the AST transport gates: GUI modules import neither
  compiler nor orchestrator owners and construct no `ArtifactPlan`,
  `CallableContract`, `CompiledArtifactInspection`, or runtime address. The static
  Artifact view therefore came from UI -> ZMQ client -> server-held compiled
  bundle -> typed inspection, not local compilation or declaration reconstruction.
- **2026-07-19 05:50-06:02 EDT - runtime acceptance attempts:** Registered a
  disposable acceptance callable through the public MCP custom-function manager,
  reapplied it through `window_code_document:pipeline_editor`, and compiled it in
  the same UI/server path. The corrected-module compile submitted at `05:58:50.108`
  and the UI reported success at `05:58:56.043`; server execution ID
  `3c4f62f6-d711-4f00-bbbf-e8b4180ededd` completed compilation in 4.9 s. The
  visible static row was `A01 | live_artifact_passthrough | output | image |
  runtime_preview | yes | channel: default | ...`; snapshot
  `20260719T094640431756Z_...Executable_artifact_observation.png` has SHA-256
  `a15037fb6c190d700d581aa8d8de3e966e8821bcc2a203f83548b9556231889a`.
  `selected-workflow run_plate` crossed compile-before-run and spawned-worker
  transport but failed after materializing the artifact because an image artifact
  requires a declared source-context relation. The exact server error was
  `Image output ... runtime_preview consumes a runtime stack without a declared
  source-context relation`; no core special case was added.
- **2026-07-19 06:04 EDT - concurrent startup blocker:** Replaced only the
  disposable fixture with the repository-supported `SpecialArtifactType` ABI
  (`return image, {"acceptance": "observed"}`), persisted it through
  `register-custom-function`, stopped the owned UI and orphaned port-7777 server,
  and restarted through the normal PyQt integration. Startup now fails before the
  bridge exists at `ServiceAdapter.get_global_config()` with
  `AttributeError: 'OpenHCSMainWindow' object has no attribute 'global_config'`.
  `.agents/global-ui-zmq-config-tabs.md` records that its worker changed
  `UIConfig`/`main.py` at `06:04`; those files are outside this plan and were not
  diagnosed, edited, or reverted. Log:
  `/tmp/openhcs_code_mode_artifact_ui_20260719_final4.log`. The final runtime
  enrichment retry is blocked only by that active shared-tree startup regression;
  canonical spawned-worker/ZMQ enrichment remains proven by the passing transport
  integration test named above.

## Current Authority Inventory

| Concern | Existing nominal owner | Current behavior/gap |
|---|---|---|
| Pipeline source | `FunctionStepTransportAuthority` in `openhcs/core/function_step_transport.py` | Emits a complete pipeline assignment with pycodify; no document-level path factoring. |
| Plate-manager source | `OrchestratorCodeSource` and `PlateManagerCodeDocumentContext` in `openhcs/pyqt_gui/widgets/plate_manager.py` | Correctly reads live per-plate/global `ObjectState`; emits four independent assignments containing repeated absolute paths. |
| Code-mode payload | `PlateManagerCodeNamespaceField`, `PlateManagerCodeNamespace`, and `PlateManagerOrchestratorCodePayload` in `openhcs/pyqt_gui/widgets/shared/services/plate_manager_workflows.py` | Own the four payload values. The AST validator currently rejects helper path bindings and `/` expressions. |
| Generic source rendering | `FormatContext`, `Assignment`, `CodeBlock`, and `generate_python_source` in `external/pycodify/src/pycodify` plus OpenHCS formatters in `openhcs/serialization/pycodify_formatters.py` | Two-pass imports are suitable for explicit document context; `PathFormatter` only emits `Path('...')`. |
| Canonical source expression | `PythonSourceLiteral` in `openhcs/core/python_source_literal.py` | Existing object for an owned Python source expression; currently not used to coordinate document-level path references. |
| Callable signature | `CallableContract` in `openhcs/core/callable_contract.py` | Owns public signature/kwargs validation; it does not retain `Annotated` path metadata or resolve declared path arguments. |
| Plate compilation root | `CompilationPlateScope` in `openhcs/core/pipeline/compilation_session.py` | Carries the exact plate path but has no public path-resolution operation. |
| Backend addressing | `FileManagerLike.resolve_address` in `openhcs/core/vfs_protocol.py` and the PolyStore file manager | Already distinguishes absolute and relative addresses when passed an explicit `base_path`. |
| Source bindings | `SourceBindingWorkspace`, `ImagePlaneSource.resolved`, `ImportedMetadataTable.resolved`, and `resolve_source_file` | Already resolve through an explicit source root and VFS; they remain separate from callable/config path declarations. |
| Planned output paths | `PathPlannerPathAuthority` in `openhcs/core/pipeline/path_planner.py` | Owns artifact output roots, result paths, and materialization paths. These semantics must not move into the new plate-relative resolver. |
| Static artifact truth | `ArtifactSpec`, `ArtifactPlan`, `ArtifactInputPlan`, `ArtifactOutputPlan`, `InvocationArtifactInputEdgePlan`, `CompiledFunctionInvocation`, and `CompiledStepPlan` | Exact compiled plans exist in each execution context but are not transported to the GUI. |
| Compiled aggregate | `CompiledExecutionBundle` in `openhcs/core/compiled_execution.py` | Contains runtime contexts and compiled step plans. The server stores it in `ZMQCompileArtifactRecord`; the GUI retains only a definition-pipeline marker. |
| Runtime truth | `RuntimeValueStore`, `StoredRuntimeValue`, and `RuntimeArtifactAddress` in `openhcs/core/runtime_stores.py` | Own latest runtime values and already support observation deltas. Only measurement-specific progress and debug artifact refs reach current GUI views. |
| Debug artifact projection | `DebugArtifactRef`/`DebugArtifactIdentity` and debug snapshots | Already project exact plans plus runtime type/shape and provide typed identity matching. |
| Artifact UI | `ArtifactContractPreviewWidget` in `openhcs/pyqt_gui/widgets/artifact_contract_preview.py` | Iterates authored function syntax and declarations, not compiled inputs, outputs, plans, paths, or runtime values. |

## Implementation File Map

| File | Concrete change |
|---|---|
| `external/pycodify/src/pycodify/core.py` and `external/pycodify/src/pycodify/__init__.py` | Preserve the existing API while adding immutable render-context extensions and context-preserving two-pass rendering. |
| `external/pyqt-reactive/src/pyqt_reactive/protocols/codegen_provider.py`, `external/pyqt-reactive/src/pyqt_reactive/core/code_generator.py`, and `external/pyqt-reactive/src/pyqt_reactive/widgets/editors/simple_code_editor.py` | Delete the obsolete singular `pipeline_config` compatibility argument and require the canonical per-plate payload. |
| `openhcs/serialization/source_path_factoring.py` (new) | Own `SourcePathOccurrenceCollector`, `SourcePathFactoringPlan`, and `OpenHCSPythonSourceDocument`. |
| `openhcs/serialization/pycodify_formatters.py` | Collect actual `Path` values, apply factored expressions, and ask `CallableContract` for declared path arguments. |
| `openhcs/core/python_source_literal.py` | Retain the canonical source-expression type; no second expression wrapper is introduced. |
| `openhcs/core/function_step_transport.py` | Render pipeline source through `OpenHCSPythonSourceDocument`. |
| `openhcs/pyqt_gui/widgets/plate_manager.py` | Make `OrchestratorCodeSource` the sole complete orchestrator renderer and publish compiled/runtime artifact signals. |
| `openhcs/pyqt_gui/services/plate_scope_identity.py` | Add owner method `code_value()` for normal versus opaque pipeline scopes. |
| `openhcs/pyqt_gui/widgets/shared/services/plate_manager_workflows.py` | Normalize `Path`/`str` payload keys once and enforce exact plate key-set identity. |
| `openhcs/pyqt_gui/services/ui_agent_bridge.py` | Validate declarative path bindings structurally. |
| `openhcs/pyqt_gui/services/reactor_providers.py` | Delegate orchestrator code generation and delete the duplicate assignment construction. |
| `openhcs/agent/services/source_rendering_service.py` | Route `PythonSourceAssignment` through the shared OpenHCS source document. |
| `openhcs/pyqt_gui/services/pycodified_window_code_document.py` | Route `PycodifiedObjectDocumentSpec` through the same source document. |
| `openhcs/core/vfs_protocol.py` | Own the `PlatePathDeclaration` nominal root, four behavior-owning declaration leaves, and four public `Annotated` aliases. |
| `openhcs/core/callable_contract.py` | Preserve `Annotated` metadata and own declared callable path discovery/resolution. |
| `openhcs/core/pipeline/compilation_session.py` | Extend `CompilationPlateScope`; add `CompilationPathResolver`; carry it in the compilation session/request. |
| `openhcs/core/function_patterns.py` | Thread the resolver into `compile_function_pattern`, `CompileFunctionGroupAuthority`, and `_compile_invocation`. |
| `openhcs/core/pipeline/compiler.py` | Resolve inherited config copies and callable kwargs before producing compiled plans; remove debug-bundle cwd dependence. |
| `openhcs/core/pipeline/path_planner.py` | Consume an already absolute `global_output_folder`; retain all output/result planning. |
| `openhcs/core/config.py` | Add declaration metadata to `global_output_folder` and `compiled_execution_bundle_path`; leave output-root-owned and host paths unmarked. |
| `openhcs/processing/backends/analysis/multi_template_matching.py` | Annotate all three `template_path` parameters. |
| `openhcs/processing/backends/cellprofiler/imagej_macro.py` | Annotate `macro_path`. |
| `openhcs/processing/backends/assemblers/self_supervised_stitcher.py` | Annotate both optional model input paths. |
| `openhcs/processing/backends/enhance/n2v2_processor_torch.py` | Annotate the optional model input and model output paths. |
| `openhcs/processing/backends/analysis/consolidate_special_outputs.py` and `openhcs/processing/backends/analysis/consolidate_analysis_results.py` | Annotate the two pipeline-callable results-directory inputs; keep non-pipeline helpers ordinary. |
| `openhcs/core/artifact_inspection.py` (new) | Own the pickle-safe compiled-plan inspection projection and typed control payloads. |
| `openhcs/runtime/zmq_control.py` (new) | Generalize the existing registered control strategy/router and add compile inspection. |
| `openhcs/runtime/zmq_debug_control.py` (delete) | Remove after debug leaves and support mixin move to the generic router module. |
| `openhcs/runtime/zmq_execution_server.py`, `openhcs/runtime/zmq_execution_client.py`, and `openhcs/runtime/zmq_compilation.py` | Route generic control messages, expose inspection retrieval, and read the existing compile artifact record. |
| `openhcs/pyqt_gui/widgets/shared/services/compile_workflow_service.py` | Expose typed asynchronous inspection retrieval. |
| `openhcs/pyqt_gui/widgets/shared/services/compile_batch_workflow_service.py` | Publish `PlateCompiledState` after batch wait and inspection retrieval. |
| `openhcs/pyqt_gui/widgets/artifact_plan_view.py` (new) | Own the typed static/runtime artifact presentation model and widget. |
| `openhcs/pyqt_gui/widgets/artifact_contract_preview.py` (delete) | Remove the authored-declaration preview. |
| `openhcs/pyqt_gui/windows/dual_editor_tab_builder.py`, `openhcs/pyqt_gui/windows/dual_editor_window.py`, and `openhcs/pyqt_gui/windows/dual_editor_session.py` | Bind the Artifact tab to canonical plate scope, step index, compile state, and runtime events; remove syntax-preview refresh. |
| `openhcs/core/progress/runtime_artifacts.py` (new) | Own the generic runtime-artifact progress context payload. |
| `openhcs/core/orchestrator/worker_execution.py` | Project one `RuntimeValueStore` observation delta into generic artifact and existing measurement contexts. |
| `openhcs/pyqt_gui/widgets/shared/services/runtime_artifact_progress_service.py` (new) | Parse and notify typed runtime-artifact progress. |
| `openhcs/pyqt_gui/widgets/shared/services/batch_workflow_components.py` and `progress_workflow_service.py` | Construct, inject, parse, and forward the new notification service. |
| `openhcs/pyqt_gui/services/plate_manager_batch_workflow.py` | Expose runtime-artifact listeners through the existing facade. |
| `openhcs/core/runtime_stores.py` | Own exact runtime-address-to-plan matching. |

## 1. Document-Level Path Factoring

### Public Rendering API

Extend pycodify's existing render context rather than post-processing generated text:

```python
# external/pycodify/src/pycodify/core.py
@dataclass(frozen=True)
class FormatContext:
    indent: int = 0
    clean_mode: bool = False
    name_mappings: Mapping[tuple[str, str], str] = field(default_factory=dict)
    extensions: Mapping[type[object], object] = field(default_factory=dict)

def generate_python_source(
    obj: object,
    header: str = "",
    clean_mode: bool = False,
    *,
    context: FormatContext | None = None,
) -> str: ...
```

`generate_python_source` must preserve its existing positional `header` and `clean_mode` API. When `context` is supplied it owns clean mode and extensions; otherwise the function constructs the current default context. Import collection uses `dataclasses.replace(context, name_mappings=resolved_names)` for the final pass, preserving extensions. Existing callers therefore retain current behavior. Export the extended API from `external/pycodify/src/pycodify/__init__.py`.

Add the OpenHCS document plan in `openhcs/serialization/source_path_factoring.py`:

```python
@dataclass(frozen=True)
class SourcePathOccurrence:
    value: Path

@dataclass(frozen=True)
class SourcePathFactoringPlan:
    bindings: tuple[Assignment, ...]

    @classmethod
    def from_occurrences(
        cls, occurrences: Iterable[SourcePathOccurrence]
    ) -> "SourcePathFactoringPlan": ...

    def expression_for(self, value: Path) -> PythonSourceLiteral | None: ...

@dataclass(frozen=True)
class OpenHCSPythonSourceDocument:
    body: CodeBlock
    path_occurrences: tuple[SourcePathOccurrence, ...] = ()
    header: str = ""
    clean_mode: bool = False

    def render(self) -> str: ...
```

Occurrence order is the stable pycodify traversal order. The plan's private path-to-expression map is an immutable rendering result and does not become a runtime path registry.

Add an OpenHCS `Path` formatter in `openhcs/serialization/pycodify_formatters.py`. During the discovery pass it appends each actual `Path` to a `SourcePathOccurrenceCollector` extension. During final rendering it checks the `SourcePathFactoringPlan` extension; a matched value emits the plan-owned `PythonSourceLiteral`, and an unmatched `Path` retains normal pycodify formatting. The formatter never scans source text.

### Structural Path Collection

`OpenHCSPythonSourceDocument.render` first calls `to_source` with an occurrence collector and discards that provisional fragment. It builds the factoring plan from the collected order, prepends the plan's bindings, and calls `generate_python_source` with that exact plan. Each current source owner contributes only paths it can identify structurally:

- Extend `PlateScopeIdentity` with `code_value() -> Path | str`. A normal filesystem plate returns `plate_root`; a CellProfiler-pipeline scope returns its opaque `scope_id` unchanged. `OrchestratorCodeSource` rekeys the three plate payload structures through this owner method before formatting, so normal plate entries become actual `Path` values without parsing synthetic IDs.
- `OrchestratorCodeSource` contributes every normal filesystem plate path/map key, every dataclass field whose runtime value is a `Path`, and every callable argument declared with the public path metadata defined in Section 2.
- `FunctionPatternTupleFormatter` asks `CallableContract.declared_path_parameters`; it does not contain a parameter-name list.
- `LazyDataclassFormatter` traverses actual dataclass fields and contributes values whose runtime type is `Path`. It does not reinterpret strings from serialized configuration.
- `FunctionStepTransportAuthority`, `PythonSourceAssignment`, and `PycodifiedObjectDocumentSpec` construct `OpenHCSPythonSourceDocument` and delegate rendering to it.
- `OpenHCSCodegenProvider.generate_complete_orchestrator_code` delegates to `OrchestratorCodeSource`. Delete its local four-assignment renderer.

The plate-manager collector receives the four authoritative payload values from `PlateManagerCodeDocumentContext`; it does not repeat the `PlateManagerCodeNamespaceField` names. `PlateManagerCodeNamespace` accepts `str | Path` plate entries and map keys, normalizes each once through `PlateScopeIdentity` to canonical `scope_id: str`, and validates exact key-set equality across `plate_paths`, `pipeline_data`, and `per_plate_configs`. `PlateManagerOrchestratorCodePayload` remains the canonical string-keyed application payload.

### Deterministic Factoring Algorithm

1. Consider only absolute `Path` occurrences discovered structurally. Relative paths stay inline and unchanged.
2. Lexically normalize separators and `.` segments using `PurePath` operations. Do not call `resolve`, inspect the filesystem, expand users/environment variables, or consult cwd.
3. Partition paths by anchor and first component after the anchor. This prevents a meaningless `/` binding and prevents unrelated volumes/trees from sharing a root.
4. For every partition containing at least two distinct values, choose the deepest common ancestor by component equality. Emit a root only when that ancestor is stricter than the anchor and is an ancestor of at least one value.
5. Emit common-root bindings in first-occurrence traversal order as `path_root`, `path_root_2`, and so on. Any exact absolute value occurring at least twice receives `path_1`, `path_2`, and so on, built from its root when one exists. A partition containing only one repeated value gets only `path_1 = Path(<absolute value>)`, not a redundant root plus full-path pair.
6. Emit descendants as `/` composition of the selected binding and literal path components. Never construct path expressions with string concatenation.
7. Render bindings before the four payload assignments. The same factoring plan is used in both pycodify passes, so imports and names cannot diverge.

This algorithm has no plate-manager special case: plate paths merely create frequent semantically typed occurrences and therefore naturally produce a useful common root.

### Generated Source Example

Before:

```python
plate_paths = [
    "/media/alice/T7/screen/plate_A",
    "/media/alice/T7/screen/plate_B",
]
pipeline_data = {
    "/media/alice/T7/screen/plate_A": [
        FunctionStep(func=(multi_template_crop, {
            "template_path": Path("/media/alice/T7/screen/templates/fiducial.png")
        }))
    ],
    "/media/alice/T7/screen/plate_B": [],
}
per_plate_configs = {
    "/media/alice/T7/screen/plate_A": PipelineConfig(...),
    "/media/alice/T7/screen/plate_B": PipelineConfig(...),
}
global_config = GlobalPipelineConfig(...)
```

After:

```python
from pathlib import Path

path_root = Path("/media/alice/T7/screen")
path_1 = path_root / "plate_A"
path_2 = path_root / "plate_B"

plate_paths = [path_1, path_2]
pipeline_data = {
    path_1: [
        FunctionStep(func=(multi_template_crop, {
            "template_path": path_root / "templates" / "fiducial.png"
        }))
    ],
    path_2: [],
}
per_plate_configs = {
    path_1: PipelineConfig(...),
    path_2: PipelineConfig(...),
}
global_config = GlobalPipelineConfig(...)
```

Changing `path_root` is sufficient to relocate every plate and the shared template on another PC. A path on another anchor/tree remains its own literal or deterministic second root.

### Code-Mode Validation

Extend `UiCodeDocumentSourcePolicy` in `openhcs/pyqt_gui/services/ui_agent_bridge.py` with a `DeclarativePathBindingValidator`:

- Permit exactly `from pathlib import Path`; do not broadly permit standard-library imports.
- Keep the existing four `PlateManagerCodeNamespaceField` payload assignments unchanged.
- Permit a non-payload assignment only when its expression is `Path(<absolute literal>)` or an `ast.BinOp` using only `/`, a previously validated path binding, and literal relative components.
- Validate bindings topologically. Reject forward references, rebinding, duplicate names, attributes, method calls, comprehensions, f-strings, addition, arbitrary constructors, and all ambient lookups.
- Permit validated path names/expressions wherever the existing payload grammar permits the equivalent literal `str`/`Path` value.
- Execute only after the complete AST has passed policy and then normalize the four payload values through `PlateManagerCodeNamespace`.

Helper names are syntax-local and are never added to `PlateManagerCodeNamespaceField`, a mirrored allowlist, or semantic state.

## 2. Relative Public Paths and Compilation Resolution

### Declaration Owner

Add declaration metadata and public aliases to `openhcs/core/vfs_protocol.py`, beside the VFS address contract:

```python
class PlatePathDeclaration(ABC):
    @abstractmethod
    def validate_target(
        self,
        target: Path,
        *,
        filemanager: FileManagerLike,
        backend: str,
    ) -> None: ...

@dataclass(frozen=True)
class PlateInputFileDeclaration(PlatePathDeclaration): ...

@dataclass(frozen=True)
class PlateInputDirectoryDeclaration(PlatePathDeclaration): ...

@dataclass(frozen=True)
class PlateOutputFileDeclaration(PlatePathDeclaration): ...

@dataclass(frozen=True)
class PlateOutputDirectoryDeclaration(PlatePathDeclaration): ...

PlateInputFile = Annotated[Path, PlateInputFileDeclaration()]
PlateInputDirectory = Annotated[Path, PlateInputDirectoryDeclaration()]
PlateOutputFile = Annotated[Path, PlateOutputFileDeclaration()]
PlateOutputDirectory = Annotated[Path, PlateOutputDirectoryDeclaration()]
```

This is a true-owner nominal family, not a registry: metadata carries the exact declaration instance, and `CompilationPathResolver` invokes its polymorphic `validate_target` hook. Input leaves validate file/directory existence through `FileManagerLike`; output leaves deliberately validate nothing and create nothing because the writer owns creation. There is no enum branch or caller-side dispatch. These declarations are the single source of truth shared by callable validation, source collection, compilation resolution, and compile-time checks.

Migrate these currently audited public parameters:

| Owner | Parameter(s) | Declaration |
|---|---|---|
| Three multi-template processing functions | `template_path` | `PlateInputFile` |
| Fiji macro callable | `macro_path` | `PlateInputFile` |
| Self-supervised stitcher | `encoder_path`, `homography_net_path` | optional `PlateInputFile` |
| Noise2Void/N2V2 callable | `model_path` | optional `PlateInputFile` |
| Noise2Void/N2V2 callable | `save_model_path` | optional `PlateOutputFile` |
| Two consolidate-results callables | `results_directory` | `PlateInputDirectory` |
| `PathPlanningConfig` | `global_output_folder` | optional `PlateOutputDirectory` |
| `CompilationDebugConfig` | `compiled_execution_bundle_path` | optional `PlateOutputFile` |

Do not annotate these as plate-relative:

- `GlobalPipelineConfig.materialization_results_path`: it is relative to the PathPlanner-owned output plate root.
- CellProfiler spreadsheet `output_directory`: it is a logical directory inside the CellProfiler output bundle.
- save-images output fields: their owning module defines output semantics and current ignored/derived behavior.
- `FijiStreamingConfig.fiji_executable_path`: it is a host executable location, not plate input. An absolute value may still be factored during source rendering because its runtime type is `Path`.
- source-binding URIs/files: their existing source-root and VFS owner remains authoritative.

### Callable Contract API

Extend `CallableContract` in `openhcs/core/callable_contract.py`:

```python
@property
def declared_path_parameters(self) -> Mapping[str, PlatePathDeclaration]: ...

def resolve_declared_paths(
    self,
    kwargs: Mapping[str, object],
    resolver: "CompilationPathResolver",
) -> dict[str, object]: ...
```

Construction uses `typing.get_type_hints(callable, include_extras=True)` and the inspected signature. It unwraps `Annotated`, optional/union wrappers, and preserves one declaration per actual parameter. Conflicting declarations are a contract-construction error. `resolve_declared_paths` changes only declared, present, non-`None` arguments and then passes the result through existing public-kwargs validation.

No consumer stores a second list of path parameter names. `FunctionPatternTupleFormatter` queries this same property for source factoring.

### Compilation Scope and Resolver

Extend `CompilationPlateScope` in `openhcs/core/pipeline/compilation_session.py`:

```python
def resolve_address(
    self,
    value: str | Path,
    *,
    filemanager: FileManagerLike,
    backend: str,
) -> Path:
    """Resolve through FileManagerLike using this scope as explicit base_path."""
```

Add the immutable resolver in the same module:

```python
@dataclass(frozen=True)
class CompilationPathResolver:
    plate_scope: CompilationPlateScope
    filemanager: FileManagerLike
    backend: str

    def resolve(self, value: str | Path, declaration: PlatePathDeclaration) -> Path: ...
```

`CompilationPlateScope.resolve_address` delegates to `filemanager.resolve_address(value, backend=backend, base_path=self.path)`. It requires an absolute backend result. `CompilationPathResolver.resolve` calls `declaration.validate_target` on that result. `PlateInputFileDeclaration` requires an existing non-directory; `PlateInputDirectoryDeclaration` requires an existing directory. Both output declaration leaves return without creating or probing a target, leaving creation and parent policy with the owning writer.

Absolute addresses pass through the same VFS method and remain absolute. Relative addresses are based only on `CompilationPlateScope.path`.

### Compiler Integration

1. In `PipelineCompiler`, after `_capture_pipeline_config` and inherited/effective configuration are finalized, construct one `CompilationPathResolver` from the current orchestrator workspace's `CompilationPlateScope`, file manager, and backend.
2. Resolve only fields carrying `PlatePathDeclaration` into immutable per-compilation config copies. Keep `ObjectState`, authored global/per-plate config, and source documents unchanged.
3. Carry the resolved configuration and resolver in `AxisCompilationRequest`/the existing compilation session. Do not add path fields to `ArtifactDeclarationStepContext`; that context remains artifact-specific.
4. Extend `CompileFunctionGroupAuthority` and `compile_function_pattern` with an explicit `path_resolver: CompilationPathResolver | None`. `_compile_invocation` calls `CallableContract.resolve_declared_paths` before existing public-kwargs validation. Non-compiler callers pass no resolver and retain literal authored values.
5. Store the resulting absolute `Path` values in `CompiledFunctionInvocation.kwargs`. Runtime execution therefore never depends on cwd.
6. Pass the already resolved `PathPlanningConfig.global_output_folder` to `PathPlannerPathAuthority`. `_cached_output_plate_root` asserts that a configured value is absolute; it does not resolve it locally.
7. Pass the already resolved `CompilationDebugConfig.compiled_execution_bundle_path` to the debug-bundle writer. The writer no longer opens a relative process path.

`PathPlannerPathAuthority` continues to compute output plate roots, artifact locations, and `materialization_results_path`. Source-binding workspaces continue to resolve their own addresses against source roots.

### Public API Example

```python
# Public declaration
def multi_template_crop(
    image: np.ndarray,
    *,
    template_path: PlateInputFile,
    ...,
) -> np.ndarray: ...

# Portable authored/generated pipeline
FunctionStep(func=(multi_template_crop, {
    "template_path": Path("templates/fiducial.png"),
}))
```

For compilation of `/data/experiment/plate_A`, the compiled invocation contains:

```python
compiled_invocation.kwargs["template_path"] == Path(
    "/data/experiment/plate_A/templates/fiducial.png"
)
```

The authored `FunctionStep`, its `ObjectState`, and regenerated source still contain `Path("templates/fiducial.png")`.

### Compile-Time Validation

- Reject relative resolution when `CompilationPlateScope.path` or the VFS result is not absolute.
- Reject missing or wrong-kind read inputs before constructing a runnable compiled execution bundle. Include plate scope, callable/config owner, parameter/field, authored value, and resolved address in the typed compilation error.
- Reject conflicting or malformed `Annotated` declarations when constructing `CallableContract`.
- Reject a declared relative callable path when compilation is invoked without a `CompilationPathResolver`; direct non-compiling introspection/formatting remains allowed.
- Do not perform output creation while validating.
- Do not fall back from one root, backend, or config layer to another.

## 3. Compiled and Runtime Artifact UI

### Static Inspection Projection

Add `openhcs/core/artifact_inspection.py`:

```python
@dataclass(frozen=True)
class CompiledArtifactStepState:
    context_key: str
    axis_id: str
    step_index: int
    artifact_inputs: tuple[ArtifactInputPlan, ...]
    artifact_outputs: tuple[ArtifactOutputPlan, ...]
    compiled_function_pattern: CompiledFunctionPattern | None

@dataclass(frozen=True)
class CompiledArtifactInspection:
    steps: tuple[CompiledArtifactStepState, ...]

    @classmethod
    def from_execution_bundle(
        cls, bundle: CompiledExecutionBundle
    ) -> "CompiledArtifactInspection": ...

    def for_step(self, step_index: int) -> tuple[CompiledArtifactStepState, ...]: ...
```

The projection iterates `bundle.transport_contexts`, the bundle's existing pickle-safe compiled contexts, and preserves their exact `ArtifactInputPlan`, `ArtifactOutputPlan`, `CompiledFunctionPattern`, invocation edge, and invocation output-plan objects. It does not flatten them into UI strings or copy their semantic fields into a second registry. Ordering is execution-context order, then `ProcessingContext.step_plans` order.

Add typed compile-inspection request/response payloads in the same module. Refactor `DebugControlMessageRouter` from `openhcs/runtime/zmq_debug_control.py` into the generic `OpenHCSControlMessageRouter` and `OpenHCSControlMessageStrategy` in `openhcs/runtime/zmq_control.py`. Move `snapshot_store_for_request` and `artifact_filemanager` off the generic strategy and onto `DebugControlMessageSupportMixin`; existing debug leaves inherit that mixin and remain registered leaves. Add `CompiledArtifactInspectionMessageStrategy` as a direct generic-strategy leaf. Delete the old debug-only router module/import path after all callers migrate; do not add a parallel router.

The new strategy:

1. Receives a typed compile artifact ID.
2. Looks up the existing `ZMQCompileArtifactRecord`.
3. Builds `CompiledArtifactInspection` from `record.compilation.execution_bundle`.
4. Returns a typed response or a typed unknown/expired-artifact error.

Add `ZMQExecutionClient.get_compiled_artifact_inspection(compile_artifact_id)` and `CompileWorkflowService.inspect_compile_artifact(compile_artifact_id)`. In `CompileBatchWorkflowService.compile_plates`, stop publishing compiled state from `_on_wait_success`. After `BatchSubmitWaitEngine.run` returns its successful scope-ID-to-artifact-ID mapping, fetch inspections asynchronously, create `PlateCompiledState`, and only then publish the compiled UI state:

```python
@dataclass(frozen=True)
class PlateCompiledState:
    compile_artifact_id: str
    artifact_inspection: CompiledArtifactInspection
```

Replace the current untyped `plate_compiled_data: Dict[str, tuple]` marker with `dict[str, PlateCompiledState]`, retaining the existing canonical scope-ID keys used by plate-manager state projections and execution checks. The run path continues to use returned compile artifact IDs directly. Add `compiled_artifacts_changed = pyqtSignal(str, object)`, carrying the canonical scope ID and `PlateCompiledState | None`, on publication/invalidation.

### Artifact View

Replace `ArtifactContractPreviewWidget` and `artifact_contract_preview.py` with `ArtifactPlanWidget` in `openhcs/pyqt_gui/widgets/artifact_plan_view.py`. Delete declaration-preview tests and the function-syntax refresh path.

Public widget/model API:

```python
def set_compiled_step_states(
    self, states: tuple[CompiledArtifactStepState, ...]
) -> None: ...

def apply_debug_artifact_refs(
    self, refs: tuple[DebugArtifactRef, ...]
) -> None: ...

def apply_runtime_addresses(
    self, addresses: tuple[RuntimeArtifactAddress, ...]
) -> None: ...

def clear(self) -> None: ...
```

The static model iterates the plans and each compiled invocation's exact `artifact_input_edges` and `artifact_output_plans`. Rows expose typed data for context, axis, invocation key, input/output role, artifact type/name, planned location or group locations, source edge, required/materialized status, and runtime enrichment. Display labels are produced last from those typed values. Labels never drive selection or matching.

`DualEditorTabBuilder` receives the plate scope, step index, and compiled-state provider/signal. `DualEditorWindow` calls `inspection.for_step(self._step_index)` and updates the widget when its plate's compiled state changes. Editing a function invalidates or marks the compiled view stale; `DualEditorFunctionPatternController` no longer refreshes a declaration preview from `FunctionPatternSyntax`.

### Runtime Enrichment

Keep `RuntimeValueStore` as the sole runtime-value authority.

Debug route:

1. The worker/debug service continues to create `DebugArtifactRef` from exact `ArtifactPlan` plus the matching `StoredRuntimeValue`.
2. `debug_snapshot_available` reaches the plate manager and dual editor through the existing notification route.
3. The widget accepts only snapshots whose typed plate/cursor step matches its `PlateScopeIdentity` and step index.
4. Match refs with `DebugArtifactIdentity.from_artifact_plan(plan).matches(ref.identity)`, never with row labels or names alone.

Normal execution route:

1. Add `RuntimeArtifactProgressPayload` in `openhcs/core/progress/runtime_artifacts.py`, containing the ordered tuple of `RuntimeArtifactAddress` objects observed from `RuntimeValueStore` around a completed step.
2. Generalize the existing worker observation-delta compositor so one store observation produces the generic runtime-artifact payload and the existing live-measurement payload. Measurement preview remains a specialized consumer, not the artifact authority.
3. Add `RuntimeArtifactProgressNotificationService` alongside `LiveMeasurementProgressNotificationService`; expose it through `BatchWorkflowComponents` and inject it into `ProgressWorkflowService` using the same listener-service pattern. There is no progress-notification strategy registry in the current code, so this plan does not invent or claim one.
4. `ProgressWorkflowService` emits a typed runtime-artifact event; the plate manager forwards `runtime_artifacts_available(PlateScopeIdentity, int, tuple)` to open dual editors.
5. Add `RuntimeArtifactAddress.matches_plan(plan: ArtifactPlan) -> bool` in `openhcs/core/runtime_stores.py`. It compares exact `ArtifactKey` type/name/scope plus runtime location against the plan's path or resolved group path. The UI calls this owner method and does not recreate matching logic.
6. Apply a runtime value only when the current plate, step, context/axis, and exact plan identity select one row. Ambiguous or unmatched values remain visible in a typed unmatched section and are not guessed onto a plan.

The model caches only the latest presentation enrichment keyed by `DebugArtifactIdentity`/exact plan identity. It is cleared on compile invalidation and new execution. This is view state, not a mirrored artifact registry.

### End-to-End Dataflow

Static compile flow:

```text
PipelineCompiler
  -> CompiledExecutionBundle.runtime_contexts[*].step_plans
  -> ZMQCompileArtifactRecord
  -> CompiledArtifactInspectionMessageStrategy
  -> CompileWorkflowService
  -> PlateCompiledState
  -> compiled_artifacts_changed
  -> DualEditorWindow(step index)
  -> ArtifactPlanWidget(exact plans and invocation edges)
```

Normal runtime flow:

```text
RuntimeValueStore observation delta
  -> RuntimeArtifactAddress tuple
  -> RuntimeArtifactProgressPayload / ProgressEvent
  -> RuntimeArtifactProgressNotificationService
  -> ProgressWorkflowService / plate-manager signal
  -> ArtifactPlanWidget exact plan matching
```

Debug runtime flow:

```text
RuntimeValueStore + exact ArtifactPlan
  -> DebugArtifactRef / DebugSnapshot
  -> existing debug snapshot notification
  -> DualEditorWindow typed plate/step filter
  -> ArtifactPlanWidget exact DebugArtifactIdentity matching
```

## Abstractions to Extend, Collapse, and Delete

### Extend

- pycodify `FormatContext`/`generate_python_source` with explicit immutable document extensions.
- `PythonSourceLiteral` usage as the canonical factored expression output.
- `FunctionStepTransportAuthority`, OpenHCS pycodify formatters, and `OrchestratorCodeSource` with one document-level factoring plan.
- `UiCodeDocumentSourcePolicy` with structurally validated declarative path bindings.
- `CallableContract` with `Annotated` path declaration introspection/resolution.
- `CompilationPlateScope` with explicit VFS resolution and `CompilationPathResolver`.
- `ZMQExecutionClient`/compile workflow with typed compiled-artifact inspection.
- `BatchWorkflowComponents` and `ProgressWorkflowService` with generic runtime-artifact observations beside the existing measurement listener service.
- `RuntimeArtifactAddress` and `DebugArtifactIdentity` as typed plan/value match authorities.

### Collapse

- Collapse `OpenHCSCodegenProvider`'s duplicate orchestrator source construction into `OrchestratorCodeSource`.
- Collapse debug-only and compile-inspection ZMQ control dispatch into one registered `OpenHCSControlMessageRouter`.
- Collapse artifact-tab static semantics onto compiled plans; do not retain an authored-declaration parallel view under the Artifact label.
- Collapse worker runtime observation into one delta that feeds generic artifacts and specialized measurement consumers.

### Delete

- Delete the local four-assignment source renderer in `OpenHCSCodegenProvider`.
- Delete cwd-relative handling in `_cached_output_plate_root` and `_write_compilation_debug_bundle_if_configured`; both consume resolved config.
- Delete `openhcs/runtime/zmq_debug_control.py` after the generic router migration.
- Delete `ArtifactContractPreviewWidget`, `artifact_contract_preview.py`, its declaration-only model/tests, and `refresh_artifact_contract_preview` wiring.
- Delete the untyped `plate_compiled_data` definition-pipeline marker and replace it with `PlateCompiledState`.
- Preserve and extend static deletion gates so removed artifact-contract classes, preview names, and duplicate routers cannot return.

## Migration Sequence

1. **Freeze owner contracts.** Add the pycodify context extension, `PlatePathDeclaration` aliases, `CallableContract` introspection API, `CompilationPathResolver`, and compiled-artifact inspection types with focused tests. No caller migration begins until these APIs pass.
2. **Implement source factoring.** Add document collection/factoring, formatter integration, plate-manager normalization, AST policy, and renderer delegation. Remove the duplicate provider renderer in the same change set.
3. **Implement relative compilation paths.** Annotate the audited callable/config fields, resolve inherited config copies, thread the resolver through compilation, and add validation. Remove both local cwd-relative config uses in the same change set.
4. **Transport static artifact inspection.** Generalize the control router, add the compile-inspection strategy/client, publish `PlateCompiledState`, and verify local/remote compiled plans are identical.
5. **Replace the Artifact tab.** Install `ArtifactPlanWidget`, bind it to plate/step compiled state, remove declaration-preview wiring/files, and add stale/empty/error states.
6. **Add runtime enrichment.** Reuse debug refs first, then add generic runtime-artifact progress from the existing store observation and notification registries.
7. **Complete deletion/migration gates.** Run AST checks for duplicate semantics and ambient path state, remove transitional imports, update user/developer documentation, and run the complete acceptance suite.

After Step 1, Steps 2, 3, and 4 can proceed in parallel with disjoint ownership:

- Source worker: `external/pycodify`, serialization/source rendering, code-mode validator.
- Compiler worker: VFS declarations, callable contracts, compilation session/compiler, audited processing/config annotations.
- Artifact transport worker: core inspection projection, generic ZMQ control router, compile client/workflow.

Step 5 depends on Step 4. Step 6 depends on the static widget model from Step 5 but its worker/progress payload can be prepared in parallel. Changes to `CallableContract`, `FunctionPatternTupleFormatter`, plate-manager workflow state, and shared control routing are integration boundaries and must have one owner at a time.

## Test Plan

### Source Factoring Unit Tests

- Deepest common base, repeated exact paths, mixed roots/anchors, no `/` root, relative paths left inline, first-occurrence name stability, and clean-mode stability.
- pycodify import collection and final render use the same factoring plan.
- Pipeline callable path arguments and dataclass `Path` fields are collected structurally; string lookalikes are not.
- Plate-manager output has one editable root and round-trips to the same `PlateManagerOrchestratorCodePayload`.
- AST policy accepts only topologically valid `Path` bindings and `/` composition and rejects every other expression form.
- Agent and GUI code generation produce byte-identical documents from the same context.

### Relative Path Unit/Integration Tests

- `CompilationPlateScope` delegates relative and absolute addresses to a fake `FileManagerLike` with the explicit base/backend.
- Every `PlatePathDeclaration` leaf validates its owned existence/kind policy correctly; output leaves create and probe nothing.
- `CallableContract` discovers nested optional `Annotated` paths and rejects conflicting metadata.
- Function compilation changes only declared path kwargs; authored steps/config/ObjectState remain relative.
- Two plates using the same relative source compile to different absolute invocation kwargs.
- Compile the same plate after changing process cwd to two unrelated directories; compiled paths and generated bundle fingerprints are identical.
- Config inheritance resolves the effective `global_output_folder` and debug bundle path once; materialization result paths still use the output plate root.
- Existing source-binding and PathPlanner materialization suites remain unchanged and pass.

### Artifact Static/UI Tests

- `CompiledArtifactInspection.from_execution_bundle` preserves context/axis/step order and exact plan/edge values.
- Generic control dispatch returns the typed inspection for a valid artifact ID and typed failure for unknown/expired IDs.
- Compile workflow publishes `PlateCompiledState` only after compile success plus inspection retrieval, and clears it on invalidation/failure.
- Artifact widget rows come from compiled input/output plans and invocation edges, including metadata-satisfied inputs and produced-artifact reuse.
- Editing authored syntax does not silently replace the compiled view with declarations.
- Dual editors select by typed plate scope and step index and update through the compiled-state signal.
- Offscreen Qt tests cover empty, compiling, compiled, stale, error, grouped-output, and multiple-context states without overlapping text.

### Runtime Tests

- One `RuntimeValueStore` observation delta emits generic artifact addresses and preserves existing live-measurement behavior.
- Debug refs match with `DebugArtifactIdentity`; normal addresses match through `RuntimeArtifactAddress.matches_plan`.
- Plate, step, context, and axis isolation prevents cross-editor enrichment.
- Latest event replaces presentation data; compile invalidation/new execution clears it.
- Ambiguous/unmatched observations are not attached by name or UI label.

### Static Architecture Gates

Add AST-based gates that reject:

- path-parameter/field-name tables outside the owning declarations;
- artifact semantic dicts/registries in GUI modules;
- `getattr`/`setattr` fallback in the new paths;
- `Path.cwd`, `os.getcwd`, bare filesystem `resolve`, or ambient root lookup in compilation;
- imports of concrete backend modules by generic resolver/compiler/UI code;
- `isinstance`/class-name/UI-string dispatch over artifact plan subclasses;
- restoration of removed declaration-preview or artifact-contract classes;
- a second ZMQ control router.

## Acceptance Commands

Run from the repository root in the project environment:

```bash
python -m pytest external/pycodify/tests -q
python -m pytest tests/unit/test_pycodify_formatters.py tests/unit/test_function_step_transport.py tests/unit/pyqt_gui/test_plate_manager_widget.py tests/unit/pyqt_gui/test_ui_agent_bridge.py -q
python -m pytest tests/unit/test_compilation_paths.py tests/unit/test_callable_contract.py tests/unit/test_compilation_session.py tests/unit/test_path_planner_materialization.py tests/unit/test_function_patterns.py -q
QT_QPA_PLATFORM=offscreen python -m pytest tests/unit/test_artifact_plan_view.py tests/unit/pyqt_gui/test_dual_editor_window_artifact_refresh.py tests/unit/pyqt_gui/test_batch_workflow_compile_engine.py -q
python -m pytest tests/integration/test_generated_source_relative_paths.py tests/integration/test_compiled_artifact_inspection.py -q
python -m pytest tests/unit/test_cellprofiler_static_deletion_gates.py -q
python scripts/validate_docs.py
git diff --check
```

For a final GUI acceptance run, compile two plates whose generated document uses one common absolute root and relative template/model paths. Change only `path_root`, reload the document, compile on the second machine/root, and verify:

- each compiled invocation contains absolute paths under its own plate root;
- the Artifact tab immediately shows exact static plans after compilation;
- stepping or running enriches those same rows with current runtime values;
- changing cwd before either compile has no effect;
- no declaration-only artifacts or guessed UI matches appear.

## Final Changed Files

Source generation and compiler paths:

```text
external/pycodify/src/pycodify/core.py
external/pycodify/tests/test_core.py
external/pyqt-reactive/src/pyqt_reactive/protocols/codegen_provider.py
external/pyqt-reactive/src/pyqt_reactive/services/widget_tree_projection.py
external/pyqt-reactive/tests/test_widget_tree_projection.py
openhcs/agent/services/source_rendering_service.py
openhcs/agent/dto/ui_bridge.py
openhcs/core/callable_contract.py
openhcs/core/config.py
openhcs/core/function_patterns.py
openhcs/core/function_step_transport.py
openhcs/core/pipeline/compilation_session.py
openhcs/core/pipeline/compiler.py
openhcs/core/pipeline/path_planner.py
openhcs/core/vfs_protocol.py
openhcs/processing/backends/analysis/consolidate_analysis_results.py
openhcs/processing/backends/analysis/consolidate_special_outputs.py
openhcs/processing/backends/analysis/multi_template_matching.py
openhcs/processing/backends/assemblers/self_supervised_stitcher.py
openhcs/processing/backends/cellprofiler/imagej_macro.py
openhcs/processing/backends/enhance/n2v2_processor_torch.py
openhcs/pyqt_gui/services/plate_scope_identity.py
openhcs/pyqt_gui/services/pycodified_window_code_document.py
openhcs/pyqt_gui/services/reactor_providers.py
openhcs/pyqt_gui/services/ui_agent_bridge.py
openhcs/pyqt_gui/services/ui_bridge_windows.py
openhcs/serialization/pycodify_formatters.py
openhcs/serialization/source_path_factoring.py
openhcs/mcp/dev_client_commands/ui.py
```

Artifact compiler/runtime/UI dataflow:

```text
openhcs/core/artifact_inspection.py
openhcs/core/orchestrator/worker_execution.py
openhcs/core/progress/runtime_artifacts.py
openhcs/core/runtime_stores.py
openhcs/pyqt_gui/services/plate_manager_batch_workflow.py
openhcs/pyqt_gui/widgets/artifact_plan_view.py
openhcs/pyqt_gui/widgets/pipeline_editor.py
openhcs/pyqt_gui/widgets/plate_manager.py
openhcs/pyqt_gui/widgets/shared/services/batch_workflow_components.py
openhcs/pyqt_gui/widgets/shared/services/compile_batch_workflow_service.py
openhcs/pyqt_gui/widgets/shared/services/compile_workflow_service.py
openhcs/pyqt_gui/widgets/shared/services/plate_manager_workflows.py
openhcs/pyqt_gui/widgets/shared/services/progress_workflow_service.py
openhcs/pyqt_gui/widgets/shared/services/runtime_artifact_progress_service.py
openhcs/pyqt_gui/windows/dual_editor_session.py
openhcs/pyqt_gui/windows/dual_editor_tab_builder.py
openhcs/pyqt_gui/windows/dual_editor_window.py
openhcs/runtime/zmq_control.py
openhcs/runtime/zmq_execution_client.py
openhcs/runtime/zmq_execution_server.py
```

Deleted production paths:

```text
openhcs/pyqt_gui/widgets/artifact_contract_preview.py
openhcs/runtime/zmq_debug_control.py
```

Tests and coordination ledger:

```text
docs/plans/code_mode_paths_and_artifact_tab_plan_20260719.md
tests/integration/test_compiled_artifact_inspection.py
tests/integration/test_generated_source_relative_paths.py
tests/unit/agent/test_mcp_server.py
tests/unit/pyqt_gui/test_batch_workflow_compile_engine.py
tests/unit/pyqt_gui/test_dual_editor_window_artifact_refresh.py
tests/unit/pyqt_gui/test_plate_manager_widget.py
tests/unit/pyqt_gui/test_ui_agent_bridge.py
tests/unit/test_artifact_plan_view.py
tests/unit/test_code_mode_path_artifact_architecture.py
tests/unit/test_compilation_paths.py
tests/unit/test_debug_runtime.py
tests/unit/test_function_step_transport.py
tests/unit/test_path_planner_materialization.py
tests/unit/test_pycodify_formatters.py
tests/unit/test_source_path_factoring.py
```

Deleted test path:

```text
tests/unit/test_artifact_contract_preview.py
```

## Final Verification

- `PYTHONPATH=external/pycodify/src .venv/bin/python -m pytest external/pycodify/tests -q`
  passed `9 passed` with one coverage-configuration warning.
- The source/code-mode acceptance command over source factoring, pycodify
  formatters, function transport, plate manager, and UI agent bridge passed
  `128 passed in 9.48s`.
- The broad compiler/path command including the exact slots-dataclass regression,
  compilation sessions, function contracts, path planning, ZMQ compilation,
  presets, and conditional images passed `204 passed in 13.92s` with one existing
  skimage warning. The final exact regression plus expanded path suite passed
  `9 passed in 6.48s` with the same warning.
- The combined artifact/control/runtime/offscreen-Qt/integration command passed
  `96 passed in 4.15s`; the final public-API subset passed `35 passed in 5.26s`.
- The indexed `QTabBar` bridge extension passed `5 passed in 3.68s` across
  `tests/unit/agent/test_mcp_server.py` and
  `tests/unit/pyqt_gui/test_ui_agent_bridge.py`. Its vendored descriptor projection
  suite passed `6 passed in 8.61s` with `external/ObjectState/src` included in
  `PYTHONPATH`.
- `tests/unit/test_cellprofiler_static_deletion_gates.py` passed
  `26 passed in 45.47s`.
- Focused Ruff, `py_compile`, scoped root `git diff --check`, and scoped vendored
  pycodify `git diff --check` passed.
- Whole-tree `git diff --check` was also run and is nonzero only for trailing
  whitespace in concurrently modified generated CSV references under
  `benchmark/native_refs/official30_scoped_rows/`; those files were not edited.
- `scripts/validate_docs.py` was run. It is nonzero for 11 shared-tree findings in
  `docs/source/architecture/abstraction_lattices.rst` involving concurrently
  deleted files and existing invalid examples; this plan file had no finding.

## Deviations And Corrections

- The audited `SourcePathOccurrence` wrapper was collapsed during implementation.
  It had no behavior or identity beyond a `Path`; the nominal collector stores
  ordered `Path` values directly and the immutable `SourcePathFactoringPlan`
  remains the sole rendering result.
- The live registered router is named `ZMQControlMessageRouter`, matching the
  current server boundary. The old `DebugControlMessageRouter` was deleted and no
  parallel `OpenHCSControlMessageRouter` alias was added.
- `BatchSubmitWaitEngine` retains submission IDs but discards wait return values.
  Inspection retrieval therefore occurs in the owned compile-wait boundary before
  success publication, rather than in a second post-batch loop. The public
  `CompileWorkflowService.inspect_compile_artifact` API is still explicit.
- Exact normal-runtime matching remains on `RuntimeValueStore.address_matches_plan`,
  the existing runtime-value authority, rather than adding matching behavior or a
  registry to the UI.
- The cross-machine scenario was exercised in the actual offscreen PyQt UI through
  the MCP bridge in addition to deterministic two-root/two-cwd integration and
  widget tests. Static compiled rows, invalidation, and recompile refresh passed.
  Live runtime attempts reached the canonical spawned worker but fixture contracts
  failed before a success event; the final corrected fixture retry is blocked by
  the concurrent Global Config startup regression recorded above.

## Blockers And Remaining Work

No implementation work or design uncertainty remains in the owned scope. One live
acceptance action remains: rerun the corrected non-image disposable fixture after
worker `019f79bf-030b-7a90-a5f7-5c83837ec5c1` restores normal PyQt startup. The
owned transport/runtime behavior is already covered by the live TCP ZMQ and
spawned-worker integration tests. The two nonzero repository-wide hygiene commands
are external shared-tree findings listed above; resolving them belongs to the
active documentation/parity owners.

## Resolved Design Decisions

- **Root for public relative paths:** the absolute `CompilationPlateScope.path` of the execution plate workspace, not source cwd, process cwd, global output root, or source-binding root.
- **Output artifacts/results:** remain owned by `PathPlannerPathAuthority`; the generic resolver does not reinterpret them.
- **Source-binding paths:** remain owned by source-binding workspace/VFS APIs and are not folded into callable/config declarations.
- **Host executable paths:** remain host paths; source factoring may deduplicate absolute `Path` values without changing their semantics.
- **Path discovery:** runtime `Path` values plus explicit `PlatePathDeclaration`, never parameter/field names or string content.
- **Common base:** deepest lexical ancestor within anchor-and-first-component partitions containing at least two occurrences, with stable first-occurrence naming.
- **Artifact static truth:** compiled plans in `CompiledExecutionBundle`, not authored function declarations and not a GUI registry.
- **Artifact dynamic truth:** latest `RuntimeValueStore` records, transported by existing debug/progress event authorities and matched by typed identity.
- **Remote compilation:** typed inspection is retrieved from the server-held compile artifact; the GUI does not recompile or reconstruct plans locally.

## Adjacent Global Config Coordination

- **2026-07-19 EDT:** `.agents/global-ui-zmq-config-tabs.md` now owns the Global
  Config two-tab window, `UIConfig`, and audited ZMQ-config consolidation. It will
  not edit code-mode path factoring, relative path compilation, compiled Artifact
  inspection/transport, or Artifact-tab rendering. Before touching any shared
  dual-editor/config-window primitive, its owner will first record the exact file
  boundary in that ledger and preserve this plan's object/transport authorities.
