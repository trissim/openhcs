# PyQt GUI Decomposition Refactor - 2026-05-18

## Advisor Evidence

Full-repo scan surfaced multiple UI decomposition findings. Textual TUI findings
are intentionally excluded because the TUI is deprecated; this plan covers the
active PyQt GUI path only.

- `pyqt_gui/widgets/shared/plate_view_widget.py`
  - `PlateViewWidget.eventFilter`
  - 167 lines, 20 branches, 68 calls, 28 callee families

## Current Problem

UI classes mix:

- event decoding;
- selection state transitions;
- rectangle geometry;
- visual updates;
- signal emission;
- service calls;
- navigation and rendering concerns.

This creates fragile integration points for source binding, debug workflows, and
pipeline editing.

## Target Shape

For `PlateViewWidget`, introduce:

- `PlateSelectionState`
- `PlateDragSelectionController`
- `PlateRectangleSelectionController`
- `PlateSelectionGeometry`
- `PlateSelectionPresenter`

The widget should remain a Qt widget, but event semantics should move to typed
controllers that can be unit tested without GUI event loops where possible.

## Phase 1: PlateView Characterization

Add focused tests for selection state if test harness exists. Otherwise, add
controller-level tests after extraction.

Current behavior to preserve:

- clicking active wells toggles selection;
- drag selection selects/deselects based on initial state;
- rectangle selection begins over active and empty buttons;
- status label updates after selection changes;
- signals emit on state changes.

## Phase 2: Extract Selection State

Create a state record/controller independent of Qt widgets:

```python
@dataclass(slots=True)
class PlateSelectionState:
    selected_wells: set[str]
    wells_with_images: set[str]
```

Methods:

- `toggle(well_id, selected)`
- `selection_mode_for(well_id)`
- `selected_count`
- `total_with_images`

## Phase 3: Extract Drag/Rectangle Controllers

Move eventFilter branch logic into controllers:

- mouse press starts drag/rectangle state;
- mouse move updates affected wells and rectangle;
- mouse release finalizes selection and emits changes.

The controllers should return instructions/events for the widget to apply,
rather than directly manipulating Qt widgets where possible.

## Phase 4: Presenter Boundary

Move status label and rectangle widget updates into presenter methods:

- `update_status(total_wells, selected_count)`
- `show_selection_rect(rect)`
- `hide_selection_rect()`

## Risks

- Qt event behavior is sensitive to object ownership and mouse grabs.
- Do not move Qt calls into pure state classes.
- GUI tests may require headless Qt configuration. Prefer controller tests for
  logic and smoke import tests for widgets.
- Deprecated Textual TUI findings should stay out of this campaign unless the
  TUI is revived or code is deleted.

## Verification Gates

```bash
.venv/bin/python -m pytest tests/unit -q
.venv/bin/python - <<'PY'
import openhcs.pyqt_gui.widgets.shared.plate_view_widget
PY
python -m nominal_refactor_advisor \
  openhcs/pyqt_gui/widgets/shared/plate_view_widget.py
```

## Completion Criteria

- `PlateViewWidget.eventFilter` is no longer a major orchestration hub.
- Selection semantics are testable outside a monolithic eventFilter.
- Deprecated TUI findings are ignored or handled only by deletion/deprecation
  cleanup, not refactoring investment.
