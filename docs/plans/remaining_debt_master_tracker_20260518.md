# Remaining Debt Master Tracker - 2026-05-18

## Purpose

This tracker supersedes ad hoc remaining-debt discussion after the first
full-repo refactor campaign set. It is derived from a full `openhcs` advisor
scan, not file-specific discovery.

## Source Scan

Command:

```bash
timeout 180 .venv/bin/python -m nominal_refactor_advisor openhcs
```

Raw output:

- `/tmp/advisor_openhcs_remaining_20260518.txt`

Result:

- Findings: 1,133
- Current checkpoint before these plans: `adc98f9f Update full repo advisor triage`

## Exclusions

- Deprecated Textual TUI findings are excluded from active refactor campaigns
  unless they are handled by deletion/deprecation cleanup.
- Cleanup-only readability/blank-line findings are batched separately and must
  not be confused with architecture blockers.
- Known-noise entries in `advisor_known_noise.md` remain excluded unless the
  underlying architecture changes.

## Campaign Queue

| Order | Status | Campaign | Plan File | Primary Gate |
| --- | --- | --- | --- | --- |
| 9 | Pending | Orchestrator stage split continuation | `orchestrator_stage_split_continuation_20260518.md` | focused orchestrator/debug tests + advisor on orchestrator |
| 10 | Pending | Runtime viewer and streaming protocol cleanup | `runtime_viewer_protocol_cleanup_20260518.md` | mocked Napari/Fiji imports + runtime viewer tests |
| 11 | Pending | Active PyQt residual decomposition | `active_pyqt_residual_decomposition_20260518.md` | Qt offscreen smoke + PyQt focused tests |
| 12 | Pending | Backend dimensional dispatch authority | `backend_dimensional_dispatch_authority_20260518.md` | focused backend tests + advisor on selected backend files |
| 13 | Pending | CellProfiler backend authority cleanup | `cellprofiler_backend_authority_cleanup_20260518.md` | CP compatibility/generated pipeline tests |
| 14 | Pending | Public API and export surface authority | `public_api_export_surface_authority_20260518.md` | import-surface tests + public API smoke |
| 15 | Pending | Active non-TUI cleanup batch | `active_non_tui_cleanup_batch_20260518.md` | targeted tests + full unit suite |

## Execution Rules

1. Start from the full-scan evidence in each plan.
2. Use file-specific advisor runs only as focused verification after selecting a
   campaign from the full scan.
3. Add characterization tests before changing risky runtime/compiler/GUI code.
4. Commit and push each completed campaign or coherent sub-campaign checkpoint.
5. Update this tracker with evidence, full unit results, and full advisor count.

## Full-Scan Evidence Summary

High-volume active, non-TUI areas from the full scan:

- CellProfiler backends: `thresholding.py`, `morphology.py`, `watershed.py`,
  `intensity_distribution.py`, `grid.py`, `zernike.py`, `illumination.py`,
  `colocalization.py`, `secondary.py`.
- Runtime viewers: `napari_stream_visualizer.py`, `napari_viewer_server.py`,
  `fiji_viewer_server.py`, `fiji_stream_visualizer.py`.
- Active PyQt: `image_browser.py`, `plate_view_widget.py`,
  `progress_tree_builder.py`, `dual_editor_window.py`,
  `step_parameter_editor.py`, `llm_pipeline_service.py`.
- Backend dimensional dispatch: `dxf_mask_pipeline.py`,
  `self_supervised_segmentation_3d.py`, `focus_torch.py`,
  `jax_nlm_processor.py`, `self_supervised_2d_deconvolution.py`,
  `self_supervised_3d_deconvolution.py`.
- Public/export surfaces and protocol probing: `__init__.py` modules,
  `unified_registry.py`, `func_registry.py`, `callable_contract.py`,
  `runtime_artifact_queries.py`.

