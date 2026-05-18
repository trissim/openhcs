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
| 3 | Pending | Validation registry family | `validation_registry_family_refactor_20260518.md` | validator tests + advisor on `validation/ast_validator.py` |
| 4 | Pending | Backend parameter request records | `backend_parameter_request_records_refactor_20260518.md` | Ashlar CPU/GPU focused tests + advisor on pos-gen modules |
| 5 | Pending | Preset pipeline spec authority | `preset_pipeline_spec_authority_refactor_20260518.md` | materialization equivalence tests + advisor on presets |
| 6 | Pending | PyQt GUI decomposition | `pyqt_gui_decomposition_refactor_20260518.md` | controller tests/import smoke + advisor on `PlateViewWidget` |
| 7 | Pending | Orchestration hubs | `full_repo_orchestration_hubs_refactor_20260518.md` | orchestrator characterization tests + focused advisor |
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
