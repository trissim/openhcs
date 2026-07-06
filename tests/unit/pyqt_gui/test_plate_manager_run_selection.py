from types import SimpleNamespace

import pytest

from openhcs.core.orchestrator.orchestrator import OrchestratorState
from openhcs.pyqt_gui.widgets.shared.services.execution_state import (
    ManagerExecutionState,
)
from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget


class _ButtonRecorder:
    def __init__(self):
        self.enabled = None
        self.text_value = None

    def setEnabled(self, value):
        self.enabled = value

    def setText(self, value):
        self.text_value = value


class _SignalRecorder:
    def __init__(self):
        self.messages = []

    def emit(self, *args):
        self.messages.append(args)


class _BatchServiceRecorder:
    def __init__(self):
        self.items = None

    async def run_plates(self, items):
        self.items = list(items)


@pytest.mark.asyncio
async def test_run_plate_submits_all_valid_selected_plates_without_precompiled_gate(
    monkeypatch,
):
    from openhcs.config_framework.object_state import ObjectStateRegistry

    selected_items = [
        {"path": "/tmp/plate-a", "name": "plate-a"},
        {"path": "/tmp/plate-b", "name": "plate-b"},
    ]
    batch_service = _BatchServiceRecorder()

    widget = PlateManagerWidget.__new__(PlateManagerWidget)
    widget.plate_compiled_data = {"/tmp/plate-a": {"definition_pipeline": ["old"]}}
    widget._batch_workflow_service = batch_service
    widget.execution_error = _SignalRecorder()
    widget.get_selected_items = lambda: selected_items
    widget._get_current_pipeline_definition = lambda _plate_path: [object()]

    monkeypatch.setattr(
        ObjectStateRegistry,
        "get_object",
        lambda _scope: SimpleNamespace(state=OrchestratorState.READY),
    )

    await widget.action_run_plate()

    assert batch_service.items == selected_items
    assert widget.execution_error.messages == []


def test_run_button_enabled_for_valid_uncompiled_selection(monkeypatch):
    from openhcs.config_framework.object_state import ObjectStateRegistry

    selected_items = [
        {"path": "/tmp/plate-a", "name": "plate-a"},
        {"path": "/tmp/plate-b", "name": "plate-b"},
    ]
    buttons = {
        name: _ButtonRecorder()
        for name in [
            "del_plate",
            "edit_config",
            "init_plate",
            "compile_plate",
            "code_plate",
            "view_metadata",
            "run_plate",
        ]
    }

    widget = PlateManagerWidget.__new__(PlateManagerWidget)
    widget.buttons = buttons
    widget.execution_state = ManagerExecutionState.IDLE
    widget.plate_compiled_data = {}
    widget.get_selected_items = lambda: selected_items
    widget.is_any_plate_running = lambda: False
    widget._get_current_pipeline_definition = lambda _plate_path: [object()]

    monkeypatch.setattr(
        ObjectStateRegistry,
        "get_object",
        lambda _scope: SimpleNamespace(state=OrchestratorState.READY),
    )

    widget.update_button_states()

    assert buttons["run_plate"].enabled is True
    assert buttons["run_plate"].text_value == "Run"
