"""
Log Detection Utilities for PyQt6 GUI

Extracted and adapted from ReactiveLogMonitor to provide standalone log detection
and classification functions for the PyQt6 log viewer. These functions operate
without instance dependencies and use explicit parameters.
"""

from typing import Optional, List, Tuple
from pathlib import Path
import logging
import re
from openhcs.textual_tui.widgets.reactive_log_monitor import LogFileInfo

logger = logging.getLogger(__name__)


def get_current_tui_log_path() -> Path:
    """
    Get current TUI log path by inspecting logging handlers.

    Extracted from ReactiveLogMonitor.get_current_tui_log_path() (lines 197-217).
    Inspects both root logger and openhcs logger to find FileHandler.

    Returns:
        Path: Path to the current TUI log file

    Raises:
        RuntimeError: If no FileHandler found in logging configuration
    """
    # Get the root logger and find the FileHandler (same as status bar)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return Path(handler.baseFilename)

    # Fallback: try to get from openhcs logger (same as status bar)
    openhcs_logger = logging.getLogger("openhcs")
    for handler in openhcs_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return Path(handler.baseFilename)

    raise RuntimeError("No file handler found in logging configuration")


def discover_logs(base_log_path: Optional[str], include_tui_log: bool = True) -> List[LogFileInfo]:
    """
    Discover existing log files.
    
    Extracted from ReactiveLogMonitor.discover_logs() (lines 219-256).
    Converted to standalone function with explicit parameters.
    
    Args:
        base_log_path: Base path for subprocess log files (file prefix, not directory)
        include_tui_log: Whether to include the current TUI process log
        
    Returns:
        List[LogFileInfo]: List of discovered log files with metadata
        
    Raises:
        RuntimeError: If TUI log file doesn't exist when include_tui_log=True
    """
    discovered_logs = []
    
    # Always include current TUI log if requested
    if include_tui_log:
        tui_log = get_current_tui_log_path()  # This will raise if not found
        # FAIL LOUD if log file doesn't exist
        if not tui_log.exists():
            raise RuntimeError(f"TUI log file does not exist: {tui_log}")
        tui_info = LogFileInfo(tui_log, "tui", display_name="TUI Process")
        discovered_logs.append(tui_info)
        logger.debug(f"Added TUI log to monitoring: {tui_log}")
    
    # Discover subprocess logs if base_log_path provided
    if base_log_path:
        base_path = Path(base_log_path)
        log_directory = base_path.parent
        
        if log_directory.exists():
            for log_file in log_directory.glob("*.log"):
                if log_file.exists() and is_relevant_log_file(log_file, base_log_path):
                    log_info = classify_log_file(log_file, base_log_path, include_tui_log)
                    discovered_logs.append(log_info)
                    logger.debug(f"Added subprocess log: {log_file}")
        else:
            logger.warning(f"Log directory does not exist: {log_directory}")
    
    return discovered_logs


def classify_log_file(log_path: Path, base_log_path: Optional[str], include_tui_log: bool = True) -> LogFileInfo:
    """
    Classify log file type and generate display name.
    
    Extracted from ReactiveLogMonitor.classify_log_file() (lines 258-297).
    Converted to standalone function with explicit parameters.
    
    Args:
        log_path: Path to log file to classify
        base_log_path: Base path for subprocess log files (file prefix)
        include_tui_log: Whether to check for TUI log classification
        
    Returns:
        LogFileInfo: Classified log file with metadata
    """
    file_name = log_path.name

    # Check if it's the current TUI log
    if include_tui_log:
        try:
            tui_log = get_current_tui_log_path()
            if log_path == tui_log:
                return LogFileInfo(log_path, "tui", display_name="TUI Process")
        except RuntimeError:
            pass  # TUI log not found, continue with other classification

    # Check subprocess logs if base_log_path is provided
    if base_log_path:
        base_name = Path(base_log_path).name

        # Check if it's the main subprocess log: exact match
        if file_name == f"{base_name}.log":
            return LogFileInfo(log_path, "main", display_name="Main Subprocess")

        # Check if it's a worker log: {base_name}_worker_{well_id}.log
        worker_pattern = rf"^{re.escape(base_name)}_worker_([A-Za-z0-9_]+)\.log$"
        match = re.match(worker_pattern, file_name)
        if match:
            well_id = match.group(1)
            # Validate well_id format (basic sanity check)
            if _is_valid_well_id(well_id):
                return LogFileInfo(log_path, "worker", well_id, display_name=f"Worker {well_id}")
            else:
                logger.warning(f"Invalid well ID format in log file: {file_name}")

    # Unknown or malformed log file
    logger.debug(f"Unrecognized log file pattern: {file_name}")
    return LogFileInfo(log_path, "unknown", display_name=log_path.name)


def is_relevant_log_file(file_path: Path, base_log_path: Optional[str]) -> bool:
    """
    Check if file is a relevant log file for monitoring.
    
    Extracted from ReactiveLogMonitor._is_relevant_log_file() (lines 299-330).
    Converted to standalone function with explicit parameters.
    
    Args:
        file_path: Path to file to check
        base_log_path: Base path for subprocess log files (file prefix)
        
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


def _is_valid_well_id(well_id: str) -> bool:
    """
    Validate well_id format (basic sanity check).
    
    Args:
        well_id: Well ID string to validate
        
    Returns:
        bool: True if well_id appears valid
    """
    # Basic validation: non-empty, reasonable length, alphanumeric + underscore
    return bool(well_id and len(well_id) <= 20 and re.match(r'^[A-Za-z0-9_]+$', well_id))
