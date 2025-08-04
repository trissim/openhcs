"""
Core Log Utilities for OpenHCS

Unified log discovery, classification, and monitoring utilities
shared between TUI and PyQt GUI implementations.
"""

import logging
import re
from pathlib import Path
from typing import Set, Dict, Optional, List
from dataclasses import dataclass

from openhcs.textual_tui.widgets.plate_manager import get_current_log_file_path as _get_current_log_file_path

logger = logging.getLogger(__name__)


@dataclass
class LogFileInfo:
    """Information about a discovered log file."""
    path: Path
    log_type: str  # "tui", "main", "worker", "unknown"
    worker_id: Optional[str] = None
    display_name: Optional[str] = None

    def __post_init__(self):
        """Generate display name if not provided."""
        if not self.display_name:
            if self.log_type == "tui":
                self.display_name = "Main Process"
            elif self.log_type == "main":
                self.display_name = "Main Subprocess"
            elif self.log_type == "worker" and self.worker_id:
                self.display_name = f"Worker {self.worker_id}"
            else:
                self.display_name = self.path.name


def get_current_log_file_path() -> str:
    """
    Get the current log file path from the logging system.
    
    This is a fail-loud wrapper around the core logging utility.
    
    Returns:
        str: Path to the current log file
        
    Raises:
        RuntimeError: If no log file found in logging configuration
    """
    log_path = _get_current_log_file_path()
    if log_path is None:
        raise RuntimeError("No file handler found in logging configuration")
    return log_path


def discover_logs(base_log_path: Optional[str] = None, include_main_log: bool = True,
                 log_directory: Optional[Path] = None) -> List[LogFileInfo]:
    """
    Discover OpenHCS log files and return as classified LogFileInfo objects.

    Args:
        base_log_path: Base path for specific subprocess logs (optional)
        include_main_log: Whether to include the current main process log
        log_directory: Directory to search (defaults to standard OpenHCS log directory)

    Returns:
        List of LogFileInfo objects for discovered log files
    """
    discovered_logs = []

    # Include current main process log if requested
    if include_main_log:
        try:
            main_log_path = get_current_log_file_path()
            main_log = Path(main_log_path)
            if main_log.exists():
                log_info = classify_log_file(main_log, base_log_path, include_main_log)
                discovered_logs.append(log_info)
        except Exception:
            pass  # Main log not available, continue

    # Discover subprocess logs if base_log_path is provided
    if base_log_path:
        base_path = Path(base_log_path)
        log_dir = base_path.parent
        if log_dir.exists():
            for log_file in log_dir.glob("*.log"):
                if is_relevant_log_file(log_file, base_log_path):
                    log_info = classify_log_file(log_file, base_log_path, include_main_log)
                    discovered_logs.append(log_info)

    # Discover all OpenHCS logs if no specific base_log_path
    elif log_directory or not base_log_path:
        if log_directory is None:
            log_directory = Path.home() / ".local" / "share" / "openhcs" / "logs"

        if log_directory.exists():
            for log_file in log_directory.glob("*.log"):
                if is_openhcs_log_file(log_file) and log_file not in [log.path for log in discovered_logs]:
                    # Infer base_log_path for proper classification
                    inferred_base = infer_base_log_path(log_file) if 'subprocess_' in log_file.name else None
                    log_info = classify_log_file(log_file, inferred_base, include_main_log)
                    discovered_logs.append(log_info)

    return discovered_logs


def classify_log_file(log_path: Path, base_log_path: Optional[str] = None, include_tui_log: bool = True) -> LogFileInfo:
    """
    Pure function: Classify a log file and extract metadata.

    Args:
        log_path: Path to log file
        base_log_path: Base path for subprocess log files
        include_tui_log: Whether to check for TUI log classification

    Returns:
        LogFileInfo with classification and metadata
    """
    file_name = log_path.name

    # Check if it's the current TUI log
    if include_tui_log:
        try:
            tui_log_path = get_current_log_file_path()
            if log_path == Path(tui_log_path):
                return LogFileInfo(log_path, "tui", display_name="Main Process")
        except RuntimeError:
            pass  # TUI log not found, continue with other classification

    # Check subprocess logs if base_log_path is provided
    if base_log_path:
        base_name = Path(base_log_path).name

        # Check if it's the main subprocess log: exact match
        if file_name == f"{base_name}.log":
            return LogFileInfo(log_path, "main", display_name="Main Subprocess")

        # Check if it's a worker log: {base_name}_worker_*.log
        if file_name.startswith(f"{base_name}_worker_") and file_name.endswith('.log'):
            # Extract worker ID (everything between _worker_ and .log)
            worker_part = file_name[len(f"{base_name}_worker_"):-4]  # Remove .log suffix
            worker_id = worker_part.split('_')[0]  # Take first part before any additional underscores
            return LogFileInfo(log_path, "worker", worker_id, display_name=f"Worker {worker_id}")

    # Unknown or malformed log file
    logger.debug(f"Unrecognized log file pattern: {file_name}")
    return LogFileInfo(log_path, "unknown")


def is_relevant_log_file(file_path: Path, base_log_path: Optional[str]) -> bool:
    """
    Check if file is a relevant log file for monitoring.

    Args:
        file_path: Path to file to check
        base_log_path: Base path for subprocess log files

    Returns:
        bool: True if file is relevant for monitoring
    """
    if not base_log_path:
        return False

    base_name = Path(base_log_path).name
    file_name = file_path.name

    # Check if it matches our patterns
    if file_name == f"{base_name}.log":
        return True

    if file_name.startswith(f"{base_name}_worker_") and file_name.endswith('.log'):
        return True

    return False


def is_openhcs_log_file(file_path: Path) -> bool:
    """
    Check if a file is an OpenHCS log file.

    Args:
        file_path: Path to file to check

    Returns:
        bool: True if file is an OpenHCS log file
    """
    if not file_path.name.endswith('.log'):
        return False

    file_name = file_path.name
    return (file_name.startswith('openhcs_') and
            ('unified_' in file_name or 'subprocess_' in file_name))


def infer_base_log_path(file_path: Path) -> str:
    """
    Infer the base log path from a subprocess log file name.

    Args:
        file_path: Path to subprocess log file

    Returns:
        str: Inferred base log path
    """
    file_name = file_path.name

    # Handle worker logs: remove _worker_* suffix
    if '_worker_' in file_name:
        base_name = file_name.split('_worker_')[0]
    else:
        # Handle main subprocess logs: remove .log extension
        base_name = file_path.stem

    return str(file_path.parent / base_name)






