# Remaining Advisor Campaign Master - 2026-05-18

## Source Evidence

Current full scan:

```bash
timeout 240 .venv/bin/python -m nominal_refactor_advisor openhcs > /tmp/advisor_openhcs_remaining_after_cp_public_20260518.txt
```

Snapshot:

- Total findings: 1,042
- Active non-TUI/test-recorder findings after policy filtering: 760
- Deprecated Textual TUI and testing-recorder findings are excluded from active
  production campaigns unless a deletion/deprecation campaign explicitly owns
  them.

## Current Top Active Finding Families

- 84 reused private helpers need nominal owners.
- 50 trivial forwarding wrappers should collapse into delegate authorities.
- 45 manual public API surfaces should derive from explicit export authorities.
- 44 enum strategy ladders need nominal strategy families.
- 41 nonsemantic blank regions and 40 overcompressed source layouts need cleanup.
- 38 repeated threaded semantic parameter families need request/context records.
- 34 anti-unified compound blocks need one derived algebra.
- 30 repeated non-orthogonal method skeletons need template-method extraction.
- 27 repeated field assignments need authoritative builders.
- 25 closed-family string dispatches and 12 inline literal dispatches need
  closed dispatch authorities.

## Campaign Files

| Order | Status | Plan | Primary Scope | Gate |
| --- | --- | --- | --- | --- |
| 1 | Priority | `remaining_cellprofiler_backend_authority_20260518.md` | CP morphology, thresholding, grid, zernike, illumination, colocalization, watershed, granularity | CP compatibility/generated-pipeline tests |
| 2 | Priority | `remaining_format_microscope_authority_20260518.md` | Plate/result readers, microscope filename parsers, BBBC/OpenHCS parser families | reader/parser unit tests + import smoke |
| 3 | Priority | `remaining_gui_runtime_authority_20260518.md` | Active PyQt services/dialogs/widgets and runtime server/viewer helpers | PyQt focused tests + runtime import smokes |
| 4 | Pending | `remaining_backend_dispatch_projection_20260518.md` | Processors, assemblers, JAX/CuPy/NumPy/Torch/pyclesperanto backend dispatch and projection families | focused processor tests/imports + advisor on changed files |
| 5 | Pending | `remaining_public_registry_export_authority_20260518.md` | `__all__`, registry surfaces, callable/contract surfaces, runtime artifact queries | import-surface tests + registry tests |
| 6 | Pending | `remaining_cleanup_noise_calibration_20260518.md` | Safe cleanup, false positives, deprecated exclusions, advisor detector follow-ups | advisor delta + targeted tests |

## Priority Override

User priority order for the next autonomous execution pass:

1. CellProfiler backend authority cleanup.
2. Microscope/format authority cleanup.
3. Active PyQt/runtime GUI authority cleanup.

Backend processor dispatch, generic public/export cleanup, and cleanup/noise
calibration should only interrupt the priority lanes when they are direct
blockers or cheap prerequisite fixes.

## Execution Rules

1. Work one coherent checkpoint at a time; commit and push after targeted
   verification.
2. Do not refactor deprecated TUI code unless the plan explicitly says delete or
   quarantine it.
3. Prefer nominal request/context records for repeated semantic parameter
   bundles; do not use generic dict bags.
4. Preserve public compatibility wrappers unless a plan adds a replacement
   public API and tests the transition.
5. When advisor suggestions would hide nominal class identity through dynamic
   class materialization, reject that route and update the advisor/noise plan
   instead.
6. For CP runtime behavior changes, rerun the CP compatibility/generated-pipeline
   test slice before committing.

## Completion Definition

- Each campaign has either implemented checkpoints with tests and advisor deltas
  or a documented reason for deferral/noise classification.
- Full active scan is materially reduced and the remaining list is organized by
  explicit campaign ownership.
- The master tracker and campaign files match the current codebase, not stale
  pre-refactor notes.
