"""OpenHCS PyQt desktop application."""

import logging
import sys
from typing import TYPE_CHECKING, Any

# CRITICAL: Check for SILENT mode BEFORE any OpenHCS imports
# This must be at MODULE LEVEL to run before main.py is imported
if "--log-level" in sys.argv:
    log_level_idx = sys.argv.index("--log-level")
    if log_level_idx + 1 < len(sys.argv) and sys.argv[log_level_idx + 1] == "SILENT":
        # Disable ALL logging before any imports
        logging.disable(logging.CRITICAL)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.CRITICAL + 1)

if TYPE_CHECKING:
    from openhcs.pyqt_gui.app import OpenHCSPyQtApp
    from openhcs.pyqt_gui.main import OpenHCSMainWindow

__all__ = ["OpenHCSMainWindow", "OpenHCSPyQtApp"]


def __getattr__(name: str) -> Any:
    """Resolve public GUI classes without importing the full UI at package load."""
    if name == "OpenHCSMainWindow":
        from openhcs.pyqt_gui.main import OpenHCSMainWindow

        globals()[name] = OpenHCSMainWindow
        return OpenHCSMainWindow
    if name == "OpenHCSPyQtApp":
        from openhcs.pyqt_gui.app import OpenHCSPyQtApp

        globals()[name] = OpenHCSPyQtApp
        return OpenHCSPyQtApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
