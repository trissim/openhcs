"""
Unified Path Cache System

Provides shared path caching functionality for both TUI and PyQt GUI implementations.
Persists last used paths across application runs for improved user experience.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class PathCacheKey(Enum):
    """
    Enumeration of path cache keys for different UI contexts.
    
    Used to maintain separate cached paths for different file operations
    across both TUI and PyQt GUI implementations.
    """
    # Original keys from both implementations
    FILE_SELECTION = "file_selection"
    DIRECTORY_SELECTION = "directory_selection"
    PLATE_IMPORT = "plate_import"
    CONFIG_EXPORT = "config_export"
    GENERAL = "general"
    
    # Specific file type contexts
    FUNCTION_PATTERNS = "function_patterns"  # .func files
    PIPELINE_FILES = "pipeline_files"        # .pipeline files
    STEP_SETTINGS = "step_settings"          # .step files
    DEBUG_FILES = "debug_files"              # .pkl debug files
    
    # Additional contexts for future use
    PLATE_BROWSER = "plate_browser"
    FUNCTION_BROWSER = "function_browser"
    PIPELINE_BROWSER = "pipeline_browser"
    EXPORT_BROWSER = "export_browser"
    CONFIG_BROWSER = "config_browser"
    ANALYSIS_BROWSER = "analysis_browser"


class UnifiedPathCache:
    """
    Unified path cache for persisting directory paths across application sessions.
    
    Provides consistent caching behavior for both TUI browser widgets and
    PyQt GUI file dialogs.
    """
    
    def __init__(self, cache_file: Optional[Path] = None):
        """
        Initialize path cache.
        
        Args:
            cache_file: Optional custom cache file location
        """
        if cache_file is None:
            cache_file = Path.home() / ".openhcs" / "path_cache.json"
        
        self.cache_file = cache_file
        self._cache: Dict[str, str] = {}
        self._load_cache()
        logger.debug(f"UnifiedPathCache initialized with cache file: {self.cache_file}")
    
    def _load_cache(self) -> None:
        """Load cache from disk."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    self._cache = json.load(f)
                logger.debug(f"Loaded path cache with {len(self._cache)} entries")
            else:
                logger.debug("No existing path cache found, starting fresh")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load path cache: {e}")
            self._cache = {}
    
    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            # Ensure cache directory exists
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.cache_file, 'w') as f:
                json.dump(self._cache, f, indent=2)
            logger.debug(f"Saved path cache with {len(self._cache)} entries")
        except OSError as e:
            logger.warning(f"Failed to save path cache: {e}")
    
    def get_cached_path(self, key: PathCacheKey) -> Optional[Path]:
        """
        Get cached path for a specific key.
        
        Args:
            key: PathCacheKey identifying the context
            
        Returns:
            Cached Path if exists and valid, None otherwise
        """
        cached_str = self._cache.get(key.value)
        if cached_str:
            cached_path = Path(cached_str)
            if cached_path.exists():
                logger.debug(f"Retrieved cached path for {key.value}: {cached_path}")
                return cached_path
            else:
                # Remove invalid cached path
                logger.debug(f"Removing invalid cached path for {key.value}: {cached_path}")
                del self._cache[key.value]
                self._save_cache()
        
        return None
    
    def set_cached_path(self, key: PathCacheKey, path: Path) -> None:
        """
        Set cached path for a specific key.
        
        Args:
            key: PathCacheKey identifying the context
            path: Path to cache
        """
        if path and path.exists():
            self._cache[key.value] = str(path)
            self._save_cache()
            logger.debug(f"Cached path for {key.value}: {path}")
        else:
            logger.warning(f"Attempted to cache non-existent path for {key.value}: {path}")
    
    def get_initial_path(self, key: PathCacheKey, fallback: Optional[Path] = None) -> Path:
        """
        Get initial path with intelligent fallback hierarchy.
        
        Args:
            key: PathCacheKey identifying the context
            fallback: Optional fallback path if cached path unavailable
            
        Returns:
            Best available path (cached > fallback > home directory)
        """
        # Try cached path first
        cached = self.get_cached_path(key)
        if cached:
            return cached
        
        # Try fallback
        if fallback and fallback.exists():
            return fallback
        
        # Ultimate fallback to home directory
        return Path.home()
    
    def clear_cache(self) -> None:
        """Clear all cached paths."""
        self._cache.clear()
        self._save_cache()
        logger.info("Cleared all cached paths")
    
    def remove_cached_path(self, key: PathCacheKey) -> None:
        """
        Remove specific cached path.
        
        Args:
            key: PathCacheKey to remove
        """
        if key.value in self._cache:
            del self._cache[key.value]
            self._save_cache()
            logger.debug(f"Removed cached path for {key.value}")


# Global cache instance
_global_path_cache: Optional[UnifiedPathCache] = None


def get_path_cache() -> UnifiedPathCache:
    """Get global path cache instance."""
    global _global_path_cache
    if _global_path_cache is None:
        _global_path_cache = UnifiedPathCache()
    return _global_path_cache


def cache_path(key: PathCacheKey, path: Path) -> None:
    """
    Convenience function to cache a path.
    
    Args:
        key: PathCacheKey identifying the context
        path: Path to cache
    """
    get_path_cache().set_cached_path(key, path)


def get_cached_path(key: PathCacheKey) -> Optional[Path]:
    """
    Convenience function to get cached path.
    
    Args:
        key: PathCacheKey identifying the context
        
    Returns:
        Cached Path if exists and valid, None otherwise
    """
    return get_path_cache().get_cached_path(key)


def get_initial_path(key: PathCacheKey, fallback: Optional[Path] = None) -> Path:
    """
    Convenience function to get initial path with fallback.
    
    Args:
        key: PathCacheKey identifying the context
        fallback: Optional fallback path
        
    Returns:
        Best available path (cached > fallback > home directory)
    """
    return get_path_cache().get_initial_path(key, fallback)


# Backward compatibility aliases for existing code
def cache_browser_path(key: PathCacheKey, path: Path) -> None:
    """Backward compatibility alias for TUI code."""
    cache_path(key, path)


def cache_dialog_path(key: PathCacheKey, path: Path) -> None:
    """Backward compatibility alias for PyQt code."""
    cache_path(key, path)


def get_cached_browser_path(key: PathCacheKey, fallback: Optional[Path] = None) -> Path:
    """Backward compatibility alias for TUI code."""
    return get_initial_path(key, fallback)


def get_cached_dialog_path(key: PathCacheKey, fallback: Optional[Path] = None) -> Path:
    """Backward compatibility alias for PyQt code."""
    return get_initial_path(key, fallback)
