from __future__ import annotations

from dataclasses import dataclass

from openhcs.core.source_bindings import StepSourceBindingsConfig
from openhcs.pyqt_gui.windows.dual_editor_window import DualEditorWindow


@dataclass
class StepClone:
    func: object
    source_bindings: StepSourceBindingsConfig


class StepState:
    def __init__(self, parameters: dict):
        self.parameters = parameters


class StepEditor:
    def __init__(self, state: StepState):
        self.state = state


class ArtifactPreview:
    def __init__(self) -> None:
        self.calls = []

    def set_function_spec(self, func_spec, *, source_bindings):
        self.calls.append((func_spec, source_bindings))


def stale_func():
    return None


def restored_func():
    return None


def test_artifact_refresh_uses_restored_objectstate_function_spec() -> None:
    restored_bindings = StepSourceBindingsConfig()
    stale_bindings = StepSourceBindingsConfig()
    window = DualEditorWindow.__new__(DualEditorWindow)
    window.editing_step = StepClone(
        func=stale_func,
        source_bindings=stale_bindings,
    )
    window.step_editor = StepEditor(
        StepState(
            {
                "func": restored_func,
                "source_bindings": restored_bindings,
            }
        )
    )
    window.artifact_contract_preview = ArtifactPreview()

    window._refresh_artifact_contract_preview(window._current_function_spec())

    assert window.artifact_contract_preview.calls == [
        (restored_func, restored_bindings)
    ]
