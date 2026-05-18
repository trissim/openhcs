# Full Repo Advisor Campaign Index - 2026-05-18

## Source Scan

The campaign set below is derived from the first successful full-package advisor
scan after advisor performance fixes:

```bash
python -m nominal_refactor_advisor openhcs
```

Scan timing:

- Parse: 1.478s
- Analysis: 59.554s
- Total: 61.032s

The scan emitted 1,140 findings. Many are cleanup-grade readability, blank-line,
string-dispatch, or helper-noise findings. This index filters for campaign-grade
architecture work where the finding points to a stable OpenHCS boundary.

## Campaign Files

1. `full_repo_orchestration_hubs_refactor_20260518.md`
   - Main target: `PipelineOrchestrator.execute_compiled_plate`
   - Secondary targets: large viewer/server event orchestration methods
   - Goal: split runtime orchestration into request, visualizer bootstrap,
     worker lifecycle, progress/event projection, result aggregation, and
     cancellation subsystems.

2. `runtime_artifact_query_axis_refactor_20260518.md`
   - Main target: `openhcs/core/runtime_artifact_queries.py`
   - Goal: replace mirrored slice/image-number query helpers with one typed row
     axis projection/query context.

3. `napari_streaming_handler_axis_refactor_20260518.md`
   - Main targets: `runtime/napari_stream_visualizer.py` and
     `runtime/napari_viewer_server.py`
   - Goal: collapse duplicated `StreamingDataType` handler tables into a shared
     typed handler registry/axis record.

4. `validation_registry_family_refactor_20260518.md`
   - Main target: `openhcs/validation/ast_validator.py`
   - Goal: move AST validators from loose subclass discovery/manual execution to
     a registered validator family with typed violation records and shared
     traversal behavior.

5. `preset_pipeline_spec_authority_refactor_20260518.md`
   - Main targets: `openhcs/processing/presets/pipelines/*`
   - Goal: replace copied pipeline Python files with authoritative specs plus
     variant overlays.

6. `backend_parameter_request_records_refactor_20260518.md`
   - Main targets: Ashlar CPU/GPU backends and repeated backend parameter
     families
   - Goal: introduce request/config records for large threaded parameter bundles.

7. `pyqt_gui_decomposition_refactor_20260518.md`
   - Main target: `PlateViewWidget.eventFilter`
   - Goal: split active PyQt event handling, selection state, rectangle
     rendering, and status presentation into coherent typed components.
   - Exclusion: deprecated Textual TUI findings are not refactor targets.

8. `full_repo_advisor_triage_policy_20260518.md`
   - Goal: define how to separate campaign-grade findings from accepted noise or
     cleanup-grade work during future full-repo scans.

## Recommended Execution Order

1. Runtime artifact query axis
2. Napari streaming handler axis
3. Validation registry family
4. Backend parameter request records
5. Preset pipeline spec authority
6. PyQt GUI decomposition
7. Orchestration hubs
8. Triage cleanup and known-noise ledger updates

This order starts with low-blast-radius typed abstractions and ends with the
highest-risk orchestrator and GUI decomposition work.

## Global Verification Gates

Run after each campaign:

```bash
.venv/bin/python -m pytest tests/unit -q
python -m nominal_refactor_advisor openhcs > /tmp/advisor_openhcs_after_campaign.txt
git diff --check
```

For PyQt GUI and runtime viewer campaigns, also run focused imports:

```bash
.venv/bin/python - <<'PY'
import openhcs.runtime.napari_stream_visualizer
import openhcs.runtime.napari_viewer_server
import openhcs.pyqt_gui.widgets.shared.plate_view_widget
PY
```

For CP/runtime-sensitive campaigns, rerun the CellProfiler generated pipeline
and compatibility tests:

```bash
.venv/bin/python -m pytest \
  tests/unit/test_cellprofiler_generated_pipeline_execution.py \
  tests/unit/test_runner_cellprofiler_compatibility.py \
  tests/unit/test_callable_contract.py -q
```
