from __future__ import annotations

from openhcs.pyqt_gui.services.main_window_workflows import (
    MainWindowTimeTravelWorkflow,
)


def test_time_travel_workflow_authorizes_before_registry_operation() -> None:
    operations: list[str] = []
    refreshes: list[bool] = []
    workflow = MainWindowTimeTravelWorkflow(
        refresh_time_travel_widget=lambda: refreshes.append(True),
        before_restore=lambda: (_ for _ in ()).throw(
            RuntimeError("mutation rejected")
        ),
    )

    result = workflow._run(lambda: operations.append("mutated"))

    assert not result
    assert operations == []
    assert refreshes == []
