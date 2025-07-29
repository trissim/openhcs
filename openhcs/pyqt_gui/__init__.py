"""
OpenHCS PyQt6 GUI Implementation

Complete PyQt6 migration of the OpenHCS Textual TUI with full feature parity.
Provides native desktop integration while preserving all existing functionality.
"""

__version__ = "1.0.0"
__author__ = "OpenHCS Development Team"

from openhcs.pyqt_gui.main import OpenHCSMainWindow
from openhcs.pyqt_gui.app import OpenHCSPyQtApp

__all__ = [
    "OpenHCSMainWindow",
    "OpenHCSPyQtApp"
]
