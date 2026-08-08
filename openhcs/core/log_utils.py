"""
Core Log Utilities for OpenHCS

Unified log discovery, classification, and monitoring utilities
shared between TUI and PyQt GUI implementations.
"""

import logging
import time
from pathlib import Path
from typing import Optional, List

from pyqt_reactive.core.log_utils import LogFileInfo, LogType
from zmqruntime.execution.logs import ExecutionWorkerLogIdentity
from zmqruntime.messages import ProcessIdentity

logger = logging.getLogger(__name__)


def get_current_log_file_path() -> str:
    """Get the current log file path from the logging system."""
    try:
        # Get the root logger and find the FileHandler
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                return handler.baseFilename

        # Fallback: try to get from openhcs logger
        openhcs_logger = logging.getLogger("openhcs")
        for handler in openhcs_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                return handler.baseFilename

        # Last resort: create a default path
        log_dir = Path.home() / ".local" / "share" / "openhcs" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        return str(log_dir / f"openhcs_subprocess_{int(time.time())}.log")

    except Exception as e:
        logger.error(f"Failed to get current log file path: {e}")
        raise RuntimeError(f"Could not determine log file path: {e}")


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
    discovered_paths = set()

    # Include current main process log if requested
    if include_main_log:
        try:
            main_log_path = get_current_log_file_path()
            main_log = Path(main_log_path)
            if main_log.exists():
                log_info = classify_log_file(main_log, base_log_path, include_main_log)
                discovered_logs.append(log_info)
                discovered_paths.add(log_info.path)
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
                    if log_info.path not in discovered_paths:
                        discovered_logs.append(log_info)
                        discovered_paths.add(log_info.path)

    # Discover all OpenHCS logs if no specific base_log_path
    elif log_directory or not base_log_path:
        if log_directory is None:
            log_directory = Path.home() / ".local" / "share" / "openhcs" / "logs"

        if log_directory.exists():
            for log_file in log_directory.glob("*.log"):
                if is_openhcs_log_file(log_file) and log_file not in discovered_paths:
                    # Infer base_log_path for proper classification
                    inferred_base = infer_base_log_path(log_file) if 'subprocess_' in log_file.name else None
                    log_info = classify_log_file(log_file, inferred_base, include_main_log)
                    discovered_logs.append(log_info)
                    discovered_paths.add(log_info.path)

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
                return LogFileInfo(
                    log_path,
                    LogType.TUI,
                    process_identity=ProcessIdentity.current(),
                )
        except RuntimeError:
            pass  # TUI log not found, continue with other classification

    # Check for ZMQ server logs (openhcs_zmq_server_port_{port}_{timestamp}.log)
    if file_name.startswith('openhcs_zmq_server_port_'):
        # Extract port from filename
        parts = file_name.replace('openhcs_zmq_server_port_', '').replace('.log', '').split('_')
        port = parts[0] if parts else 'unknown'
        return LogFileInfo(
            log_path,
            LogType.ZMQ_SERVER,
            display_name=f"ZMQ Server (port {port})",
        )

    # Check for ZMQ worker logs
    worker_log = ExecutionWorkerLogIdentity.from_path(log_path)
    if worker_log is not None:
        worker_id = str(worker_log.worker_pid)
        return LogFileInfo(
            log_path,
            LogType.ZMQ_WORKER,
            worker_id,
            display_name=f"ZMQ Worker {worker_id}",
        )

    # Check for Napari viewer logs
    if file_name.startswith('napari_detached_port_'):
        port = file_name.replace('napari_detached_port_', '').replace('.log', '')
        return LogFileInfo(
            log_path,
            LogType.NAPARI,
            display_name=f"Napari Viewer (port {port})",
        )

    # Check subprocess logs if base_log_path is provided
    if base_log_path:
        base_name = Path(base_log_path).name

        # Check if it's the main subprocess log: exact match
        if file_name == f"{base_name}.log":
            return LogFileInfo(log_path, LogType.MAIN)

        # Check if it's a worker log: {base_name}_worker_*.log
        if file_name.startswith(f"{base_name}_worker_") and file_name.endswith('.log'):
            # Extract worker ID (everything between _worker_ and .log)
            worker_part = file_name[len(f"{base_name}_worker_"):-4]  # Remove .log suffix
            worker_id = worker_part.split('_')[0]  # Take first part before any additional underscores
            return LogFileInfo(log_path, LogType.WORKER, worker_id)

    # Unknown or malformed log file
    logger.debug(f"Unrecognized log file pattern: {file_name}")
    return LogFileInfo(log_path, LogType.UNKNOWN)


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

    # OpenHCS log patterns:
    # - openhcs_unified_*.log (main UI process)
    # - openhcs_subprocess_*.log (subprocess runner)
    # - openhcs_zmq_server_port_*.log (ZMQ execution server)
    # - pyqt_gui_subprocess_*.log (PyQt subprocess runner)
    # - zmq_worker_exec_*.log (ZMQ worker processes)
    # - napari_detached_port_*.log (Napari viewer)

    openhcs_patterns = [
        'openhcs_',
        'pyqt_gui_subprocess_',
        'zmq_worker_',
        'napari_detached_'
    ]

    return any(file_name.startswith(pattern) for pattern in openhcs_patterns)


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
