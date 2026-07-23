import pytest
import os
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QPushButton

# Skip in CPU-only mode
if os.getenv('OPENHCS_CPU_ONLY', 'false').lower() == 'true':
    pytest.skip('PyQt6 GUI tests skipped in CPU-only mode', allow_module_level=True)

from openhcs.pyqt_gui.config import PyQtGuiRuntimeContext, get_default_ui_config
from openhcs.pyqt_gui.main import OpenHCSMainWindow
from tests.pyqt_gui.integration.test_end_to_end_workflow_foundation import (
    WidgetFinder,
    TimingConfig,
    _wait_for_gui,
)

# Use environment-configurable timing
TIMING = TimingConfig.from_environment()

def test_my_workflow(qtbot):
    """
    Auto-generated test from GUI recording: my_workflow
    Recorded on: 2025-10-31 19:12:07
    Total events: 3
    """
    # Create main window
    main_window = OpenHCSMainWindow(
        runtime_context=PyQtGuiRuntimeContext(get_default_ui_config())
    )
    qtbot.addWidget(main_window)
    main_window.show()
    qtbot.wait(1500)

    # Replay recorded interactions
    qtbot.mouseClick(WidgetFinder.find_button_by_text(main_window, ["add"]), Qt.MouseButton.LeftButton)
    _wait_for_gui(TIMING.ACTION_DELAY)
    qtbot.mouseClick(WidgetFinder.find_button_by_text(main_window, ["init"]), Qt.MouseButton.LeftButton)
    _wait_for_gui(TIMING.ACTION_DELAY)
    qtbot.mouseClick(WidgetFinder.find_button_by_text(main_window, ["del"]), Qt.MouseButton.LeftButton)
    _wait_for_gui(TIMING.ACTION_DELAY)
