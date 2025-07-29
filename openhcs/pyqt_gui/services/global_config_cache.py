"""
Global Configuration Cache Service for PyQt6 GUI - Persistent config storage.

Provides session-persistent global config caching using the same pickle-based
file format as the Textual TUI for consistency.
"""

import logging
import dill as pickle
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from PyQt6.QtCore import QObject, pyqtSignal, QThread, QRunnable, QThreadPool

from openhcs.core.config import GlobalPipelineConfig, get_default_global_config

logger = logging.getLogger(__name__)


class ConfigCacheWorker(QRunnable):
    """Worker for async config cache operations."""

    def __init__(self, operation, cache_file, config=None, callback=None):
        super().__init__()
        self.operation = operation  # 'load' or 'save'
        self.cache_file = cache_file
        self.config = config
        self.result = None
        self.error = None
        self.callback = callback
    
    def run(self):
        """Execute the cache operation."""
        try:
            if self.operation == 'load':
                self.result = self._sync_load_config()
            elif self.operation == 'save':
                self.result = self._sync_save_config()
        except Exception as e:
            self.error = e
        finally:
            if self.callback:
                self.callback(self)
    
    def _sync_load_config(self) -> Optional[GlobalPipelineConfig]:
        """Synchronous config loading."""
        if not self.cache_file.exists():
            return None
            
        try:
            with open(self.cache_file, 'rb') as f:
                config = pickle.load(f)
                
            # Validate it's the right type
            if not isinstance(config, GlobalPipelineConfig):
                logger.warning(f"Cached config is not GlobalPipelineConfig: {type(config)}")
                return None
                
            logger.info(f"Loaded cached global config from: {self.cache_file}")
            return config
            
        except pickle.PickleError as e:
            logger.warning(f"Failed to unpickle cached config: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to load cached config: {e}")
            return None
    
    def _sync_save_config(self) -> bool:
        """Synchronous config saving."""
        try:
            # Ensure cache directory exists
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save config using pickle
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.config, f)
                
            logger.info(f"Saved global config to cache: {self.cache_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save config to cache: {e}")
            return False


class GlobalConfigCache(QObject):
    """
    Persistent global configuration cache using pickle format (matches TUI).
    
    Uses Qt threading for non-blocking cache operations.
    """
    
    # Signals
    config_loaded = pyqtSignal(object)  # GlobalPipelineConfig or None
    config_saved = pyqtSignal(bool)     # Success/failure
    
    def __init__(self, cache_file: Optional[Path] = None):
        """
        Initialize global config cache.
        
        Args:
            cache_file: Optional custom cache file location
        """
        super().__init__()
        
        if cache_file is None:
            # Default cache location following TUI pattern
            cache_file = Path.home() / ".openhcs" / "global_config.config"
        
        self.cache_file = cache_file
        self.thread_pool = QThreadPool()
        logger.debug(f"GlobalConfigCache initialized with cache file: {self.cache_file}")
    
    def load_cached_config_async(self):
        """
        Load cached global config from disk asynchronously.

        Emits config_loaded signal when complete.
        """
        worker = ConfigCacheWorker('load', self.cache_file, callback=self._on_load_finished)
        self.thread_pool.start(worker)
    
    def save_config_to_cache_async(self, config: GlobalPipelineConfig):
        """
        Save global config to cache asynchronously.

        Args:
            config: GlobalPipelineConfig to cache

        Emits config_saved signal when complete.
        """
        worker = ConfigCacheWorker('save', self.cache_file, config, callback=self._on_save_finished)
        self.thread_pool.start(worker)
    
    def _on_load_finished(self, worker):
        """Handle load operation completion."""
        if worker.error:
            logger.error(f"Config load error: {worker.error}")
            self.config_loaded.emit(None)
        else:
            self.config_loaded.emit(worker.result)
    
    def _on_save_finished(self, worker):
        """Handle save operation completion."""
        if worker.error:
            logger.error(f"Config save error: {worker.error}")
            self.config_saved.emit(False)
        else:
            self.config_saved.emit(worker.result)


# Global instance for easy access (matches TUI pattern)
_global_config_cache = GlobalConfigCache()


def get_global_config_cache() -> GlobalConfigCache:
    """Get the global config cache instance."""
    return _global_config_cache


def load_cached_global_config_sync() -> GlobalPipelineConfig:
    """
    Load global config with cache fallback (synchronous version for startup).
    
    Tries to load from cache first, falls back to default config if cache
    is unavailable or invalid.
    
    Returns:
        GlobalPipelineConfig (cached or default)
    """
    try:
        cache_file = Path.home() / ".openhcs" / "global_config.config"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                config = pickle.load(f)
                
            if isinstance(config, GlobalPipelineConfig):
                logger.info("Using cached global configuration")
                return config
    except Exception as e:
        logger.warning(f"Failed to load cached config, using defaults: {e}")
    
    # Fallback to default config
    logger.info("Using default global configuration")
    return get_default_global_config()
