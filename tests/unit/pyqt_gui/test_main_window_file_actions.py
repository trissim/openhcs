"""File-menu routing for whole-workspace orchestrator code documents."""

from __future__ import annotations

from types import SimpleNamespace

from openhcs.core.selection import SelectedAllSelectionMode
from openhcs.pyqt_gui.main import OpenHCSMainWindow


class _PlateManagerProbe:
    def __init__(self) -> None:
        self.selection_modes = []

    def action_code_plate(self, **kwargs) -> None:
        self.selection_modes.append(kwargs.get("selection_mode"))


def test_load_uses_normal_plate_manager_code_mode() -> None:
    plate_manager = _PlateManagerProbe()
    main_window = SimpleNamespace(plate_manager_widget=plate_manager)

    OpenHCSMainWindow.load_orchestrator_configuration(main_window)

    assert plate_manager.selection_modes == [None]


def test_save_exports_all_existing_orchestrators() -> None:
    plate_manager = _PlateManagerProbe()
    main_window = SimpleNamespace(plate_manager_widget=plate_manager)

    OpenHCSMainWindow.save_orchestrator_configuration(main_window)

    assert plate_manager.selection_modes == [SelectedAllSelectionMode.ALL]
