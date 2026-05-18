# Active Non-TUI Cleanup Batch - 2026-05-18

## Full-Scan Evidence

The full scan contains many cleanup-grade findings outside deprecated Textual
TUI:

- overcompressed source layout;
- nonsemantic blank regions;
- dangling private helpers;
- unreferenced private functions;
- trivial forwarding wrappers;
- small repeated method templates.

Active non-TUI examples include:

- `openhcs/formats/experimental_analysis.py`
- `openhcs/processing/backends/analysis/self_supervised_segmentation_3d.py`
- `openhcs/processing/backends/analysis/dxf_mask_pipeline.py`
- `openhcs/processing/backends/enhance/self_supervised_3d_deconvolution.py`
- `openhcs/pyqt_gui/testing/event_recorder.py`
- `openhcs/core/runtime_artifact_queries.py`
- `openhcs/core/equivalence/*`

## Policy

This campaign is for cleanup that is local, behavior-preserving, and easy to
verify. It must not absorb architecture-grade work that deserves its own plan.

Do not include deprecated Textual TUI cleanup unless the action is deletion or a
deprecation boundary update.

## Phases

1. Batch readability/blank-region cleanup in active files already touched by
   architecture campaigns.
2. Delete unreferenced private helpers only after `rg` confirms no references
   and focused tests pass.
3. Collapse trivial wrappers only when they are not public compatibility names.
4. Collapse small repeated event/test recorder templates into typed local
   builders.
5. Keep each batch small enough to review and revert independently.

## Verification Gates

```bash
git diff --check
.venv/bin/python -m pytest tests/unit -q
timeout 180 .venv/bin/python -m nominal_refactor_advisor openhcs
```

For PyQt cleanup:

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/unit/pyqt_gui -q
```

## Completion Criteria

- Cleanup-grade findings in active non-TUI files are reduced.
- No compatibility public names are removed accidentally.
- Full tests pass after each cleanup batch.

