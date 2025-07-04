"""
Global Configuration Cache Service - Persistent config storage using pickle.

Provides session-persistent global config caching using the same pickle-based
file format as .func, .step, and other OpenHCS user data files.
"""

import asyncio
import dill as pickle
import logging
from pathlib import Path
from typing import Optional

from openhcs.core.config import GlobalPipelineConfig, get_default_global_config

logger = logging.getLogger(__name__)


class GlobalConfigCache:
    """
    Persistent global configuration cache using pickle format.
    
    Follows the same patterns as PatternFileService for consistency with
    existing OpenHCS file formats (.func, .step, etc.).
    """
    
    def __init__(self, cache_file: Optional[Path] = None):
        """
        Initialize global config cache.
        
        Args:
            cache_file: Optional custom cache file location
        """
        if cache_file is None:
            # Default cache location following path_cache pattern
            cache_file = Path.home() / ".openhcs" / "global_config.config"
        
        self.cache_file = cache_file
        logger.debug(f"GlobalConfigCache initialized with cache file: {self.cache_file}")
    
    async def load_cached_config(self) -> Optional[GlobalPipelineConfig]:
        """
        Load cached global config from disk.
        
        Returns:
            Cached config if available and valid, None otherwise
        """
        def _sync_load_config(path: Path) -> Optional[GlobalPipelineConfig]:
            """Synchronous config loading for executor."""
            if not path.exists() or not path.is_file():
                logger.debug(f"No cached config found at {path}")
                return None
            
            try:
                with open(path, "rb") as f:
                    config = pickle.load(f)
                
                # Validate that it's actually a GlobalPipelineConfig
                if not isinstance(config, GlobalPipelineConfig):
                    logger.warning(f"Invalid cached config type: {type(config)}")
                    return None
                
                logger.info(f"Loaded cached global config from {path}")
                return config
                
            except pickle.PickleError as e:
                logger.warning(f"Failed to unpickle cached config: {e}")
                return None
            except Exception as e:
                logger.warning(f"Failed to load cached config: {e}")
                return None
        
        # Use async executor to prevent blocking
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_load_config, self.cache_file)
    
    async def save_config_to_cache(self, config: GlobalPipelineConfig) -> bool:
        """
        Save global config to cache.
        
        Args:
            config: GlobalPipelineConfig to cache
            
        Returns:
            True if saved successfully, False otherwise
        """
        def _sync_save_config(config_data: GlobalPipelineConfig, path: Path) -> bool:
            """Synchronous config saving for executor."""
            try:
                # Ensure cache directory exists
                path.parent.mkdir(parents=True, exist_ok=True)
                
                # Validate config type
                if not isinstance(config_data, GlobalPipelineConfig):
                    logger.error(f"Invalid config type for caching: {type(config_data)}")
                    return False
                
                # Save using pickle (consistent with .func, .step files)
                with open(path, "wb") as f:
                    pickle.dump(config_data, f)
                
                logger.info(f"Saved global config to cache: {path}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to save config to cache: {e}")
                return False
        
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_save_config, config, self.cache_file)
    
    async def clear_cache(self) -> bool:
        """
        Clear cached config by removing the cache file.
        
        Returns:
            True if cleared successfully, False otherwise
        """
        def _sync_clear_cache(path: Path) -> bool:
            """Synchronous cache clearing for executor."""
            try:
                if path.exists():
                    path.unlink()
                    logger.info(f"Cleared config cache: {path}")
                return True
            except Exception as e:
                logger.error(f"Failed to clear config cache: {e}")
                return False
        
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_clear_cache, self.cache_file)
    
    def get_cache_info(self) -> dict:
        """
        Get information about the cache file.
        
        Returns:
            Dictionary with cache file information
        """
        info = {
            "cache_file": str(self.cache_file),
            "exists": self.cache_file.exists(),
            "size": None,
            "modified": None
        }
        
        if info["exists"]:
            try:
                stat = self.cache_file.stat()
                info["size"] = stat.st_size
                info["modified"] = stat.st_mtime
            except Exception as e:
                logger.debug(f"Failed to get cache file stats: {e}")
        
        return info


# Global instance for easy access
_global_config_cache = GlobalConfigCache()


async def load_cached_global_config() -> GlobalPipelineConfig:
    """
    Load global config with cache fallback.
    
    Tries to load from cache first, falls back to default config if cache
    is unavailable or invalid.
    
    Returns:
        GlobalPipelineConfig (cached or default)
    """
    try:
        cached_config = await _global_config_cache.load_cached_config()
        if cached_config is not None:
            logger.info("Using cached global configuration")
            return cached_config
    except Exception as e:
        logger.warning(f"Failed to load cached config, using defaults: {e}")
    
    # Fallback to default config
    logger.info("Using default global configuration")
    return get_default_global_config()


async def save_global_config_to_cache(config: GlobalPipelineConfig) -> bool:
    """
    Save global config to cache.
    
    Args:
        config: GlobalPipelineConfig to cache
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        return await _global_config_cache.save_config_to_cache(config)
    except Exception as e:
        logger.error(f"Failed to save global config to cache: {e}")
        return False


async def clear_global_config_cache() -> bool:
    """
    Clear the global config cache.
    
    Returns:
        True if cleared successfully, False otherwise
    """
    try:
        return await _global_config_cache.clear_cache()
    except Exception as e:
        logger.error(f"Failed to clear global config cache: {e}")
        return False


def get_global_config_cache_info() -> dict:
    """
    Get information about the global config cache.
    
    Returns:
        Dictionary with cache information
    """
    return _global_config_cache.get_cache_info()
