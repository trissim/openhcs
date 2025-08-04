"""
PyQt Log Detection Utils - Simple Re-exports

This module just re-exports core log utilities for PyQt GUI.
All logic is now in openhcs.core.log_utils.
"""

# Re-export everything from core utilities
from openhcs.core.log_utils import (
    LogFileInfo,
    discover_logs,
    classify_log_file,
    is_relevant_log_file,
    is_openhcs_log_file,
    infer_base_log_path,
    get_current_log_file_path
)
from pathlib import Path
from typing import Optional

# Simple compatibility alias
def get_current_tui_log_path() -> Path:
    """Get current log path as Path object."""
    log_path = get_current_log_file_path()
    return Path(log_path)

# Compatibility alias for old function name
def discover_all_logs(log_directory: Optional[Path] = None):
    """Discover all OpenHCS logs in a directory."""
    return discover_logs(log_directory=log_directory, include_main_log=False)
