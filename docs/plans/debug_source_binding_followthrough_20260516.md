# Debug and Source-Binding Followthrough Plan

Date: 2026-05-16

## Problem

The debug and source-binding plans have been substantially implemented, but remaining work is mostly UX hardening and end-to-end workflow coverage. This should be tracked separately from runtime parity and advisor-count cleanup.

## Current State

Implemented foundations include:

- Debug model/control records.
- ZMQ command path and paused-worker command coverage.
- Warm debug compile artifact reuse.
- Artifact hydration with fail-loud missing-output behavior.
- CP renderer families beyond a single generic renderer.
- Source-binding semantic dialogs and table-backed editors.
- Typed GUI requests for inspector export/open flows.

Remaining work should polish these flows without destabilizing runtime/planner semantics.

## Remaining Work

### Paused Worker UX Tests

- Add live GUI/ZMQ workflow tests for pause, step, continue, stop, inspect, and export.
- Verify worker lifetime across multiple commands.
- Keep these tests isolated from official30 benchmark runs.

### Artifact Replay Identity

- Strengthen replay validation beyond artifact name/kind/group identity where practical.
- Include content/settings identity when available.
- Fail loudly when identity cannot be validated and the replay would otherwise be stale.

### Inspector Details

- Add invocation kwargs and per-function snapshot metadata in inspector summaries.
- Add more module-specific thumbnails/table previews through renderer specs, not one-off conditionals.
- Surface export/open results with typed status records.

### Source-Binding Dialog Polish

- Replace remaining text-area-plus-suggestions flows with structured row editors.
- Add enum combo cells for filter subjects/operators.
- Add metadata-field picker cells.
- Add validation hints before accepting a binding.

## Boundaries

- Do not change CellProfiler runtime execution semantics in this plan.
- Do not add CellProfiler-only GUI branches where source-binding core abstractions can express the same concept.
- Do not mix GUI polish with planner/runtime parity commits.

## Verification

- Existing PyQt unit tests for debug toolbar, inspector, and source-binding widgets.
- Focused ZMQ debug workflow tests.
- Full `tests/unit` after GUI behavior changes.

## Completion Criteria

- Live debug command-loop coverage exists for the full paused-worker UX.
- Artifact replay rejects stale or missing warm outputs deterministically.
- Inspector export/open is host-visible and typed.
- Source-binding dialogs expose semantic controls instead of free-form string cells where the model is typed.

