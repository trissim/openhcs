# Remaining GUI And Runtime Authority Refactor - 2026-05-18

## Evidence

Refreshed active scan still reports production findings in:

- `openhcs/pyqt_gui/services/llm_pipeline_service.py`: 12 findings.
- `openhcs/pyqt_gui/dialogs/function_selector_dialog.py`: literal dispatch and
  selection/filter behavior findings.
- `openhcs/pyqt_gui/services/service_adapter.py`: reflective attribute access
  and async-operation authority findings.
- `openhcs/pyqt_gui/windows/synthetic_plate_generator_window.py`: cleanup/
  lifecycle probing.
- `openhcs/pyqt_gui/widgets/image_browser.py`: remaining broad facade findings.
- `openhcs/runtime/fiji_viewer_server.py`, `omero_instance_manager.py`, and
  `remote_orchestrator.py`: runtime dispatch/helper findings.

Previous PyQt checkpoints already split plate grid/selection, image-browser
catalog controls, progress tree building, and step-parameter file handling.

## Problem

The remaining active GUI/runtime debt is mostly service-boundary and command/
lifecycle authority, not simple widget extraction. Literal command dispatch,
reflective probing, broad facades, and helper reuse make GUI/runtime behavior
harder to test independently.

## Target Shape

- Command/filter/selection axes represented as typed action objects or closed
  dispatch families.
- Async service execution routed through nominal request/result records instead
  of reflective probes.
- Runtime viewer/server helpers grouped under explicit protocol/lifecycle
  authorities.
- Keep deprecated TUI out of scope; this plan covers active PyQt and runtime
  production code only.

## Phases

1. `function_selector_dialog.py`
   - Extract column-filter and tree-selection command dispatch into typed
     selection/filter action handlers.
   - Add focused tests if existing GUI tests do not cover behavior.
2. `service_adapter.py`
   - Replace reflective async-operation probing with a nominal executable
     operation/request protocol.
3. `llm_pipeline_service.py`
   - Split prompt/model/request/result concerns into service boundaries.
4. `image_browser.py`
   - Continue facade decomposition only where a real subsystem boundary remains,
     not just to appease broad-class noise.
5. Runtime files
   - `fiji_viewer_server.py`, `omero_instance_manager.py`,
     `remote_orchestrator.py`: extract command/platform dispatch authorities and
     lifecycle request records.

## Verification Gates

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/unit/pyqt_gui -q

.venv/bin/python - <<'PY'
import openhcs.runtime.fiji_viewer_server
import openhcs.runtime.omero_instance_manager
import openhcs.runtime.remote_orchestrator
PY

timeout 120 .venv/bin/python -m nominal_refactor_advisor \
  openhcs/pyqt_gui/dialogs/function_selector_dialog.py \
  openhcs/pyqt_gui/services/service_adapter.py \
  openhcs/pyqt_gui/services/llm_pipeline_service.py \
  openhcs/runtime/fiji_viewer_server.py \
  openhcs/runtime/omero_instance_manager.py \
  openhcs/runtime/remote_orchestrator.py
```

## Risks

- GUI behavior is easy to break without visual feedback. Add focused tests for
  extracted services and keep offscreen PyQt tests in every checkpoint.
- Do not refactor the deprecated Textual TUI as part of this campaign.
