"""
PyQt-specific adapter for unified configuration cache.

Provides Qt-compatible interface for existing PyQt GUI code.
"""

from PyQt6.QtCore import QObject, pyqtSignal, QRunnable, QThreadPool
from openhcs.core.config_cache import (
    load_cached_global_config_sync,
    get_global_config_cache as get_core_config_cache,
    QtExecutionStrategy,
    _sync_load_config,
    _sync_save_config
)
from openhcs.core.config import GlobalPipelineConfig
from pathlib import Path


class ConfigCacheWorker(QRunnable):
    """Qt worker for cache operations."""

    def __init__(self, operation: str, cache_file: Path, config=None, callback=None):
        super().__init__()
        self.operation = operation
        self.cache_file = cache_file
        self.config = config
        self.callback = callback

    def run(self):
        if self.operation == 'load':
            result = _sync_load_config(self.cache_file)
            if self.callback:
                self.callback(result)
        elif self.operation == 'save':
            result = _sync_save_config(self.config, self.cache_file)
            if self.callback:
                self.callback(result)


class QtGlobalConfigCache(QObject):
    """Qt-specific wrapper for unified config cache."""

    config_loaded = pyqtSignal(object)  # GlobalPipelineConfig or None
    config_saved = pyqtSignal(bool)     # Success/failure

    def __init__(self, cache_file=None):
        super().__init__()
        if cache_file is None:
            cache_file = Path.home() / ".openhcs" / "global_config.config"
        self.cache_file = cache_file
        self.thread_pool = QThreadPool()

    def load_cached_config_async(self):
        """Load cached config asynchronously with Qt threading."""
        worker = ConfigCacheWorker('load', self.cache_file, callback=self._on_load_finished)
        self.thread_pool.start(worker)

    def save_config_to_cache_async(self, config: GlobalPipelineConfig):
        """Save config asynchronously with Qt threading."""
        worker = ConfigCacheWorker('save', self.cache_file, config, callback=self._on_save_finished)
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
