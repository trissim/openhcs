"""
Atomic file operations with locking for OpenHCS.

Provides utilities for atomic read-modify-write operations with file locking
to prevent concurrency issues in multiprocessing environments.
"""

import fcntl
import json
import logging
import os
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass(frozen=True)
class LockConfig:
    """Configuration constants for file locking operations."""
    DEFAULT_TIMEOUT: float = 30.0
    DEFAULT_POLL_INTERVAL: float = 0.1
    LOCK_SUFFIX: str = '.lock'
    TEMP_PREFIX: str = '.tmp'
    JSON_INDENT: int = 2


LOCK_CONFIG = LockConfig()


class FileLockError(Exception):
    """Raised when file locking operations fail."""
    pass


class FileLockTimeoutError(FileLockError):
    """Raised when file lock acquisition times out."""
    pass


@contextmanager
def file_lock(
    lock_path: Union[str, Path],
    timeout: float = LOCK_CONFIG.DEFAULT_TIMEOUT,
    poll_interval: float = LOCK_CONFIG.DEFAULT_POLL_INTERVAL
):
    """Context manager for exclusive file locking."""
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    lock_fd = None
    try:
        lock_fd = _acquire_lock_with_timeout(lock_path, timeout, poll_interval)
        yield
    except FileLockTimeoutError:
        raise
    except Exception as e:
        raise FileLockError(f"File lock operation failed for {lock_path}: {e}") from e
    finally:
        _cleanup_lock(lock_fd, lock_path)


def _acquire_lock_with_timeout(lock_path: Path, timeout: float, poll_interval: float) -> int:
    """Acquire file lock with timeout and return file descriptor."""
    deadline = time.time() + timeout

    while time.time() < deadline:
        if lock_fd := _try_acquire_lock(lock_path):
            return lock_fd
        time.sleep(poll_interval)

    raise FileLockTimeoutError(f"Failed to acquire lock {lock_path} within {timeout}s")


def _try_acquire_lock(lock_path: Path) -> Optional[int]:
    """Try to acquire lock once, return fd or None."""
    try:
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        logger.debug(f"Acquired file lock: {lock_path}")
        return lock_fd
    except (OSError, IOError):
        return None


def _cleanup_lock(lock_fd: Optional[int], lock_path: Path) -> None:
    """Clean up file lock resources."""
    if lock_fd is not None:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
            logger.debug(f"Released file lock: {lock_path}")
        except Exception as e:
            logger.warning(f"Error releasing lock {lock_path}: {e}")

    if lock_path.exists():
        try:
            lock_path.unlink()
        except Exception as e:
            logger.warning(f"Error removing lock file {lock_path}: {e}")


def atomic_write_json(
    file_path: Union[str, Path],
    data: Dict[str, Any],
    indent: int = LOCK_CONFIG.JSON_INDENT,
    ensure_directory: bool = True
) -> None:
    """Atomically write JSON data to file using temporary file + rename."""
    file_path = Path(file_path)

    if ensure_directory:
        file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        tmp_path = _write_to_temp_file(file_path, data, indent)
        os.rename(tmp_path, str(file_path))
        logger.debug(f"Atomically wrote JSON to {file_path}")
    except Exception as e:
        raise FileLockError(f"Atomic JSON write failed for {file_path}: {e}") from e


def _write_to_temp_file(file_path: Path, data: Dict[str, Any], indent: int) -> str:
    """Write data to temporary file and return path."""
    with tempfile.NamedTemporaryFile(
        mode='w',
        dir=file_path.parent,
        prefix=f"{LOCK_CONFIG.TEMP_PREFIX}{file_path.name}",
        suffix='.json',
        delete=False
    ) as tmp_file:
        json.dump(data, tmp_file, indent=indent)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
        return tmp_file.name


def atomic_update_json(
    file_path: Union[str, Path],
    update_func: Callable[[Optional[Dict[str, Any]]], Dict[str, Any]],
    lock_timeout: float = LOCK_CONFIG.DEFAULT_TIMEOUT,
    default_data: Optional[Dict[str, Any]] = None
) -> None:
    """Atomically update JSON file using read-modify-write with file locking."""
    file_path = Path(file_path)
    lock_path = file_path.with_suffix(f'{file_path.suffix}{LOCK_CONFIG.LOCK_SUFFIX}')

    with file_lock(lock_path, timeout=lock_timeout):
        current_data = _read_json_or_default(file_path, default_data)

        try:
            updated_data = update_func(current_data)
        except Exception as e:
            raise FileLockError(f"Update function failed for {file_path}: {e}") from e

        atomic_write_json(file_path, updated_data)
        logger.debug(f"Atomically updated JSON file: {file_path}")


def _read_json_or_default(file_path: Path, default_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Read JSON file or return default data if file doesn't exist or is invalid."""
    if not file_path.exists():
        return default_data

    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to read {file_path}, using default: {e}")
        return default_data
