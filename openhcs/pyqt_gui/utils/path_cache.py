"""
Path caching system for PyQt6 file dialogs.

Mirrors the Textual TUI path caching system for consistent UX.
Persists last used paths across application runs.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class PathCacheKey(Enum):
    """Cache keys for different dialog contexts (mirrors Textual TUI)."""
    FILE_SELECTION = "file_selection"
    DIRECTORY_SELECTION = "directory_selection"
    PLATE_IMPORT = "plate_import"
    CONFIG_EXPORT = "config_export"
    GENERAL = "general"
    # Specific dialog types for better caching granularity
    FUNCTION_PATTERNS = "function_patterns"  # .func files
    PIPELINE_FILES = "pipeline_files"        # .pipeline files
    STEP_SETTINGS = "step_settings"          # .step files
    DEBUG_FILES = "debug_files"              # .pkl debug files


class PathCache:
    """Path caching with persistent storage (mirrors Textual TUI implementation)."""
    
    def __init__(self, cache_file: Optional[Path] = None):
        """Initialize path cache with optional custom cache file location."""
        if cache_file is None:
            # Default cache location in user's home directory
            cache_file = Path.home() / ".openhcs" / "path_cache.json"
        
        self.cache_file = cache_file
        self._cache: Dict[str, str] = {}
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load cache from disk with error handling."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    self._cache = json.load(f)
                logger.debug(f"Loaded path cache from {self.cache_file}")
            else:
                logger.debug(f"No existing cache file at {self.cache_file}")
        except Exception as e:
            logger.warning(f"Failed to load path cache: {e}")
            self._cache = {}
    
    def _save_cache(self) -> None:
        """Save cache to disk with error handling."""
        try:
            # Ensure cache directory exists
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.cache_file, 'w') as f:
                json.dump(self._cache, f, indent=2)
            logger.debug(f"Saved path cache to {self.cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save path cache: {e}")
    
    def get_cached_path(self, key: PathCacheKey) -> Optional[Path]:
        """Get cached path for given key."""
        cached_str = self._cache.get(key.value)
        if cached_str:
            cached_path = Path(cached_str)
            # Verify path still exists before returning
            if cached_path.exists():
                logger.debug(f"Using cached path for {key.value}: {cached_path}")
                return cached_path
            else:
                logger.debug(f"Cached path no longer exists: {cached_path}")
                # Remove invalid path from cache
                self._cache.pop(key.value, None)
                self._save_cache()
        
        return None
    
    def set_cached_path(self, key: PathCacheKey, path: Path) -> None:
        """Set cached path for given key."""
        if path and path.exists():
            self._cache[key.value] = str(path)
            self._save_cache()
            logger.debug(f"Cached path for {key.value}: {path}")
    
    def get_initial_path(self, key: PathCacheKey, fallback: Optional[Path] = None) -> Path:
        """Get initial path with intelligent fallback."""
        # Try cached path first
        cached = self.get_cached_path(key)
        if cached:
            return cached
        
        # Try fallback
        if fallback and fallback.exists():
            return fallback
        
        # Ultimate fallback to home directory
        return Path.home()


# Global cache instance
_global_cache: Optional[PathCache] = None


def get_path_cache() -> PathCache:
    """Get global path cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = PathCache()
    return _global_cache


def cache_dialog_path(key: PathCacheKey, path: Path) -> None:
    """Convenience function to cache a dialog path."""
    get_path_cache().set_cached_path(key, path)


def get_cached_dialog_path(key: PathCacheKey, fallback: Optional[Path] = None) -> Path:
    """Convenience function to get cached dialog path with fallback."""
    return get_path_cache().get_initial_path(key, fallback)
