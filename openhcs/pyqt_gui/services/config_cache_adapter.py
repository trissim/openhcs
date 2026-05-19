"""
PyQt-specific adapter for unified configuration cache.

Provides Qt-compatible interface for existing PyQt GUI code.
"""

from enum import Enum
from PyQt6.QtCore import QObject, pyqtSignal, QRunnable, QThreadPool
from openhcs.core.config_cache import (
    load_cached_global_config_sync,
    _sync_load_config,
    _sync_save_config
)
from openhcs.core.config import GlobalPipelineConfig
from pathlib import Path


class ConfigCacheOperation(Enum):
    """Closed cache operation axis for the Qt worker."""

    LOAD = "load"
    SAVE = "save"


class ConfigCacheWorker(QRunnable):
    """Qt worker for cache operations."""

    def __init__(
        self,
        operation: ConfigCacheOperation,
        cache_file: Path,
        config=None,
        callback=None,
    ):
        super().__init__()
        self.operation = operation
        self.cache_file = cache_file
        self.config = config
        self.callback = callback

    def run(self):
        result = CONFIG_CACHE_OPERATION_RUNNERS[self.operation](self)
        if self.callback:
            self.callback(result)


def _run_load_cache_operation(worker: ConfigCacheWorker):
    return _sync_load_config(worker.cache_file)


def _run_save_cache_operation(worker: ConfigCacheWorker):
    return _sync_save_config(worker.config, worker.cache_file)


CONFIG_CACHE_OPERATION_RUNNERS = {
    ConfigCacheOperation.LOAD: _run_load_cache_operation,
    ConfigCacheOperation.SAVE: _run_save_cache_operation,
}


class QtGlobalConfigCache(QObject):
    """Qt-specific wrapper for unified config cache."""

    config_loaded = pyqtSignal(object)  # GlobalPipelineConfig or None
    config_saved = pyqtSignal(bool)     # Success/failure

    def __init__(self, cache_file=None):
        super().__init__()
        if cache_file is None:
            from openhcs.core.xdg_paths import get_config_file_path
            cache_file = get_config_file_path("global_config.config")
        self.cache_file = cache_file
        self.thread_pool = QThreadPool()

    def load_cached_config_async(self):
        """Load cached config asynchronously with Qt threading."""
        worker = ConfigCacheWorker(
            ConfigCacheOperation.LOAD,
            self.cache_file,
            callback=self._on_load_finished,
        )
        self.thread_pool.start(worker)

    def save_config_to_cache_async(self, config: GlobalPipelineConfig):
        """Save config asynchronously with Qt threading."""
        worker = ConfigCacheWorker(
            ConfigCacheOperation.SAVE,
            self.cache_file,
            config,
            callback=self._on_save_finished,
        )
        self.thread_pool.start(worker)

    def _on_load_finished(self, result):
        self.config_loaded.emit(result)

    def _on_save_finished(self, result):
        self.config_saved.emit(result)


# Global instance for singleton pattern
_global_qt_config_cache = None


# Backward compatibility aliases
GlobalConfigCache = QtGlobalConfigCache


def get_global_config_cache() -> QtGlobalConfigCache:
    """Get the global config cache instance (PyQt compatibility)."""
    global _global_qt_config_cache
    if _global_qt_config_cache is None:
        _global_qt_config_cache = QtGlobalConfigCache()
    return _global_qt_config_cache


# Re-export sync function for startup
load_cached_global_config_sync = load_cached_global_config_sync
