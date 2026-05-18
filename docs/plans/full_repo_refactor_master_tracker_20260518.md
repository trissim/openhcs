# Full Repo Refactor Master Tracker - 2026-05-18

## Purpose

This is the master execution tracker for the full-repo advisor campaign set. Do
not start refactoring a campaign until its row is moved from `Pending` to
`In Progress` and the first characterization/verification gate is recorded.

Raw full-repo advisor output should stay outside the repository. Commit this
tracker, the campaign plans, and concise progress notes only.

## Baseline

Advisor command:

```bash
python -m nominal_refactor_advisor openhcs
```

Last successful full-package scan:

- Date: 2026-05-18
- Findings: 1,140
- Unique finding titles: 60
- Total time: 61.032s
- Raw output: `/tmp/advisor_openhcs_repo_scan_after_speedfix.txt`

Advisor infrastructure checkpoint:

- Nominal Refactor Advisor commit: `ece7f0b Speed up repo-wide advisor scans`
- Result: full OpenHCS scan now completes within the refactor workflow budget.

OpenHCS checkpoint before these campaign plans:

- OpenHCS commit: `62c46ec4 Implement callable request binding cleanup`
- Status: request-binding/SaveImages cleanup complete and pushed.

## Status Legend

- `Pending`: planned but no implementation started.
- `In Progress`: active implementation branch/slice.
- `Blocked`: requires user decision, missing dependency, or failed gate.
- `Complete`: implemented, verified, committed, and pushed.
- `Deferred`: intentionally not being refactored now.

## Campaign Queue

| Order | Status | Campaign | Plan File | Primary Gate |
| --- | --- | --- | --- | --- |
| 0 | Complete | Callable request binding and CP threshold context cleanup | `callable_request_binding_refactor_20260518.md`, `cellprofiler_binding_context_cleanup_20260518.md` | `pytest tests/unit -q` passed; commit `62c46ec4` pushed |
| 1 | Complete | Runtime artifact query axis | `runtime_artifact_query_axis_refactor_20260518.md` | focused runtime artifact query tests + advisor on `runtime_artifact_queries.py` |
| 2 | Complete | Napari streaming handler axis | `napari_streaming_handler_axis_refactor_20260518.md` | import smoke for both Napari modules + focused advisor |
| 3 | Complete | Validation registry family | `validation_registry_family_refactor_20260518.md` | validator tests + advisor on `validation/ast_validator.py` |
| 4 | Complete | Backend parameter request records | `backend_parameter_request_records_refactor_20260518.md` | Ashlar CPU/GPU focused tests + advisor on pos-gen modules |
| 5 | Complete | Preset pipeline spec authority | `preset_pipeline_spec_authority_refactor_20260518.md` | materialization equivalence tests + advisor on presets |
| 6 | Complete | PyQt GUI decomposition | `pyqt_gui_decomposition_refactor_20260518.md` | controller tests/import smoke + advisor on `PlateViewWidget` |
| 7 | Complete | Orchestration hubs | `full_repo_orchestration_hubs_refactor_20260518.md` | orchestrator characterization tests + focused advisor |
| 8 | Pending | Full-repo triage policy and known-noise ledger | `full_repo_advisor_triage_policy_20260518.md`, `advisor_known_noise.md` | full advisor scan reviewed and ledger updated |

## Explicit Exclusions

- Deprecated Textual TUI findings are not refactor targets. They may be handled
  later by deletion/deprecation cleanup only.
- Cleanup-grade readability/blank-line findings should not interrupt campaign
  work unless they are in touched files and block review.
- CP known-noise findings in `advisor_known_noise.md` should not be silenced by
  fake inheritance or generic predicate bases.

## Per-Campaign Execution Template

Before implementation:

1. Move campaign status to `In Progress`.
2. Add a short progress note under `Execution Log`.
3. Run the campaign's first focused test/advisor baseline if not already known.

During implementation:

1. Work in small semantic slices.
2. Run focused tests after each slice.
3. Run focused advisor on touched files after each slice.
4. Do not start the next campaign until the current one is committed or marked
   `Blocked`/`Deferred`.

Before completion:

1. Run `git diff --check`.
2. Run focused campaign verification.
3. Run `.venv/bin/python -m pytest tests/unit -q`.
4. Run `python -m nominal_refactor_advisor openhcs` and record timing/counts.
5. Commit and push.
6. Move campaign status to `Complete` and record commit hash.

## Execution Log

### 2026-05-18 - Tracker Created

- Created master tracker after full-repo advisor campaign plans were drafted.
- Current new campaign docs are pending and uncommitted.
- Next action after committing this tracker set: start campaign 1,
  `Runtime artifact query axis`.

### 2026-05-18 - Runtime Artifact Query Axis Started

- Moved campaign 1 to `In Progress`.
- First slice: characterize slice/image-number axis projection wrappers, then
  introduce the typed `MeasurementTableAxisQuery` compatibility abstraction.

### 2026-05-18 - Runtime Artifact Query Axis Completed

- Added `MeasurementTableAxisQuery` as the authoritative row-axis projection
  request for slice and CellProfiler image-number filtering.
- Rewrote the four public compatibility wrappers to delegate to the query
  object, preserving existing import names.
- Added characterization tests for query projection, tuple projection, and
  wrapper delegation.
- Replaced `DataclassMeasurementColumnarRows.columns` with `AliasProperty`,
  removing a same-file descriptor-algebra finding.
- Focused verification: `37 passed`.
- Full unit verification: `1505 passed, 10 warnings`.
- Focused advisor result: property-alias finding removed; suffix-axis finding
  remains only because the public compatibility wrapper names still exist.
- Full advisor scan: 1,149 findings, 68.590s. Count includes deprecated Textual
  TUI and cleanup-grade findings tracked outside this campaign.

### 2026-05-18 - Napari Streaming Handler Axis Started

- Moved campaign 2 to `In Progress`.
- First slice: extract the duplicated `StreamingDataType` handler table into a
  shared runtime handler record while preserving module-local helper functions
  and optional Napari import behavior.

### 2026-05-18 - Napari Streaming Handler Axis Completed

- Added `openhcs/runtime/napari_streaming_handlers.py` with
  `NapariStreamingDataTypeHandler` and the canonical handler-table builder.
- Replaced duplicated `_DATA_TYPE_HANDLERS` literals in
  `napari_stream_visualizer.py` and `napari_viewer_server.py`.
- Fixed pre-existing `napari_viewer_server.py` import omissions discovered by
  mocked Napari import smoke (`register_cleanup_callback` and
  `OpenHCSTransportMode`).
- Focused verification: `tests/unit/test_napari_streaming_handlers.py` passed.
- Mocked import smoke: both Napari modules import with fake `napari` and expose
  image/points/shapes handler keys.
- Focused advisor: parallel enum-keyed table finding removed; remaining Napari
  findings are broader layer update helper ownership.
- Full unit verification: `1508 passed, 10 warnings`.
- Full advisor scan: 1,148 findings, 59.677s.

### 2026-05-18 - Validation Registry Family Started

- Moved campaign 3 to `In Progress`.
- Verified the only in-repo execution authority is `validate_file`; external
  callers consume `ValidationViolation` and `validate_file`.
- First slice: freeze violation records, add typed validation kinds, and derive
  `validate_file` execution from the registered validator family while
  preserving string compatibility aliases.

### 2026-05-18 - Validation Registry Family Completed

- Added `ValidationKind`, frozen/slotted `ValidationViolation`, and
  `ASTValidator` registry membership keyed by validation kind.
- Added `run_ast_validators(...)` as the single execution authority and rewired
  `validate_file` to use it.
- Preserved public string aliases and string-valued violation output for CLI and
  caller compatibility.
- Focused verification: `tests/unit/test_ast_validator_registry.py` passed.
- Focused advisor: no findings for `openhcs/validation/ast_validator.py`.
- Full unit verification: `1513 passed, 10 warnings`.
- Full advisor scan: 1,147 findings, 59.472s.

### 2026-05-18 - Backend Parameter Request Records Started

- Moved campaign 4 to `In Progress`.
- Focused advisor on Ashlar CPU/GPU modules reports repeated parameter bundles
  between public tile-position functions and internal aligner constructors.
- First slice: introduce a shared Ashlar alignment/request record and collapse
  CPU/GPU aligner constructors behind it while preserving public processing
  function signatures used by presets and GUI surfaces.

### 2026-05-18 - Backend Parameter Request Records Completed

- Added shared `AshlarAlignmentConfig` and `AshlarPositionRequest` records.
- Rewired CPU/GPU Ashlar aligner constructors to consume the shared alignment
  config instead of re-threading the full public parameter family.
- Kept public Ashlar processing signatures stable for presets, GUI discovery,
  and generated pipeline compatibility.
- Added focused request-projection tests for CPU and GPU paths with mocked
  aligners.
- Focused verification passed: `39 passed`.
- Focused advisor: repeated threaded parameter findings for CPU/GPU Ashlar
  request flow removed; remaining focused findings are the larger CPU/GPU
  aligner ABC extraction plus blank-line/layout cleanup.
- Full unit verification: `1518 passed, 10 warnings`.
- Full advisor scan: 1,145 findings, 60.637s.

### 2026-05-18 - Preset Pipeline Spec Authority Started

- Moved campaign 5 to `In Progress`.
- Focused advisor on the MFD preset pipeline wrappers reported duplicated
  cross-module step families and literal `FunctionStep` declarations.
- First slice: replace MFD preset files with compatibility wrappers backed by a
  typed preset-spec authority while preserving file-level `pipeline_steps`.

### 2026-05-18 - Preset Pipeline Spec Authority Completed

- Added `openhcs/processing/presets/mfd_specs.py` as the typed materialization
  authority for MFD crop/analyze and stitch presets.
- Replaced the four MFD preset pipeline files with thin compatibility wrappers
  that expose the same `pipeline_steps` variable from typed preset keys.
- Represented shared step binding metadata and stitch-family repetition as
  explicit spec/template records instead of copied `FunctionStep` literals.
- Added focused wrapper/materialization characterization tests, including the
  hyphenated DAPI/FITC/CY5 preset filename.
- Focused verification passed: `tests/unit/test_mfd_preset_specs.py` passed.
- Focused advisor: no findings for the new MFD spec authority and wrappers.
- Full unit verification: `1522 passed, 10 warnings`.
- Full advisor scan: 1,142 findings, 60.135s.

### 2026-05-18 - PyQt GUI Decomposition Started

- Moved campaign 6 to `In Progress`.
- Focused advisor on `PlateViewWidget` confirmed the primary target:
  `eventFilter` concentrated drag selection, rectangle selection, event
  routing, signal publication, and mouse-grab lifecycle.
- First slice: extract event routing and selection interaction lifecycle while
  preserving the public widget facade.

### 2026-05-18 - PyQt GUI Decomposition Completed

- Added typed event target/mode/state records for plate-view selection
  interaction.
- Extracted `PlateSelectionEventController` so `PlateViewWidget.eventFilter`
  delegates through one routing boundary instead of owning all event phases.
- Collapsed row/column selection into a shared axis-selection helper.
- Replaced string/numeric UI state dispatch with explicit enum/table-backed
  declarations for subdirectory display mode and well-button style.
- Removed the unreferenced drag-selection private method during controller
  extraction.
- Focused smoke passed under `QT_QPA_PLATFORM=offscreen`.
- Focused advisor: original `eventFilter` orchestration-hub finding removed;
  remaining focused findings are broader widget/service decomposition items.
- Full unit verification: `1522 passed, 10 warnings`.
- Full advisor scan: 1,138 findings, 61.458s.

### 2026-05-18 - Orchestration Hubs Started

- Moved campaign 7 to `In Progress`.
- Focused advisor on `PipelineOrchestrator.execute_compiled_plate` confirmed
  the method remains the largest runtime hub.
- First slice: extract request validation/defaults, visualizer bootstrap, and
  worker-lane execution identity rather than line-range helper chunks.

### 2026-05-18 - Orchestration Hubs Completed

- Added `CompiledPlateExecutionRequest` and
  `ValidatedCompiledPlateExecution` records.
- Added `CompiledPlateExecutionValidator` for initialized-state, pipeline,
  compiled-context, progress invariant, execution identity, and worker-count
  defaults.
- Added `ExecutionVisualizerBootstrap` for streaming viewer discovery, launch
  progress, readiness polling, timeout reporting, and viewer-state cleanup.
- Added `WorkerLaneExecutionIdentity`, `WorkerLaneExecutionPlan`, and
  `WorkerLaneExecutor` so inline, fork-inherited, and submitted workers share a
  nominal lane execution boundary.
- Added `PIPELINE_PROGRESS_STEP_NAME` and descriptor aliases for direct
  orchestrator state/cache projections.
- Focused verification passed: orchestrator/debug/CP compatibility tests
  `62 passed`.
- Focused advisor: worker-lane parameter-family and reused-private-helper
  findings removed; `execute_compiled_plate` remains a large hub and is tracked
  as residual staged-split debt.
- Full unit verification: `1522 passed, 10 warnings`.
- Full advisor scan: 1,133 findings, 66.622s.
