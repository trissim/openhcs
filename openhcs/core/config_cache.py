"""
Unified Global Configuration Cache System

Provides shared configuration caching logic with pluggable execution strategies
for different UI frameworks (async for TUI, Qt threading for PyQt).
"""

import logging
import dill as pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor

from openhcs.core.config import GlobalPipelineConfig, get_default_global_config

logger = logging.getLogger(__name__)


class CacheExecutionStrategy(ABC):
    """Abstract strategy for executing cache operations."""
    
    @abstractmethod
    async def execute_load(self, cache_file: Path) -> Optional[GlobalPipelineConfig]:
        """Execute cache load operation."""
        pass
    
    @abstractmethod
    async def execute_save(self, config: GlobalPipelineConfig, cache_file: Path) -> bool:
        """Execute cache save operation."""
        pass


class AsyncExecutionStrategy(CacheExecutionStrategy):
    """Async execution strategy for TUI."""
    
    async def execute_load(self, cache_file: Path) -> Optional[GlobalPipelineConfig]:
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_load_config, cache_file)
    
    async def execute_save(self, config: GlobalPipelineConfig, cache_file: Path) -> bool:
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_save_config, config, cache_file)


class QtExecutionStrategy(CacheExecutionStrategy):
    """Qt threading execution strategy for PyQt GUI."""
    
    def __init__(self, thread_pool=None):
        self.thread_pool = thread_pool or ThreadPoolExecutor(max_workers=2)
    
    async def execute_load(self, cache_file: Path) -> Optional[GlobalPipelineConfig]:
        # Convert to sync for Qt integration
        return _sync_load_config(cache_file)
    
    async def execute_save(self, config: GlobalPipelineConfig, cache_file: Path) -> bool:
        # Convert to sync for Qt integration  
        return _sync_save_config(config, cache_file)


def _sync_load_config(cache_file: Path) -> Optional[GlobalPipelineConfig]:
    """Synchronous config loading implementation."""
    try:
        if not cache_file.exists():
            return None
            
        with open(cache_file, 'rb') as f:
            config = pickle.load(f)
            
        if isinstance(config, GlobalPipelineConfig):
            logger.debug(f"Loaded cached config from: {cache_file}")
            return config
        else:
            logger.warning(f"Invalid config type in cache: {type(config)}")
            return None
            
    except pickle.PickleError as e:
        logger.warning(f"Failed to unpickle cached config: {e}")
        return None
    except Exception as e:
        logger.warning(f"Failed to load cached config: {e}")
        return None


def _sync_save_config(config: GlobalPipelineConfig, cache_file: Path) -> bool:
    """Synchronous config saving implementation."""
    try:
        # Ensure cache directory exists
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(cache_file, 'wb') as f:
            pickle.dump(config, f)
            
        logger.debug(f"Saved config to cache: {cache_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save config to cache: {e}")
        return False


class UnifiedGlobalConfigCache:
    """
    Unified global configuration cache with pluggable execution strategies.
    
    Supports both async (TUI) and Qt threading (PyQt) execution patterns
    while sharing the core caching logic.
    """
    
    def __init__(self, cache_file: Optional[Path] = None, strategy: Optional[CacheExecutionStrategy] = None):
        if cache_file is None:
            cache_file = Path.home() / ".openhcs" / "global_config.config"
        
        self.cache_file = cache_file
        self.strategy = strategy or AsyncExecutionStrategy()
        logger.debug(f"UnifiedGlobalConfigCache initialized with cache file: {self.cache_file}")
    
    async def load_cached_config(self) -> Optional[GlobalPipelineConfig]:
        """Load cached global config from disk."""
        return await self.strategy.execute_load(self.cache_file)
    
    async def save_config_to_cache(self, config: GlobalPipelineConfig) -> bool:
        """Save global config to cache."""
        return await self.strategy.execute_save(config, self.cache_file)
    
    async def clear_cache(self) -> bool:
        """Clear cached config by removing the cache file."""
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
                logger.info(f"Cleared config cache: {self.cache_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear config cache: {e}")
            return False


# Global instance for easy access
_global_config_cache: Optional[UnifiedGlobalConfigCache] = None


def get_global_config_cache(strategy: Optional[CacheExecutionStrategy] = None) -> UnifiedGlobalConfigCache:
    """Get global config cache instance with optional strategy."""
    global _global_config_cache
    if _global_config_cache is None or (strategy and _global_config_cache.strategy != strategy):
        _global_config_cache = UnifiedGlobalConfigCache(strategy=strategy)
    return _global_config_cache


async def load_cached_global_config(strategy: Optional[CacheExecutionStrategy] = None) -> GlobalPipelineConfig:
    """
    Load global config with cache fallback.
    
    Args:
        strategy: Optional execution strategy (defaults to async)
    
    Returns:
        GlobalPipelineConfig (cached or default)
    """
    try:
        cache = get_global_config_cache(strategy)
        cached_config = await cache.load_cached_config()
        if cached_config is not None:
            logger.info("Using cached global configuration")
            return cached_config
    except Exception as e:
        logger.warning(f"Failed to load cached config, using defaults: {e}")
    
    # Fallback to default config
    logger.info("Using default global configuration")
    return get_default_global_config()


def load_cached_global_config_sync() -> GlobalPipelineConfig:
    """
    Synchronous version for startup scenarios.
    
    Returns:
        GlobalPipelineConfig (cached or default)
    """
    try:
        cache_file = Path.home() / ".openhcs" / "global_config.config"
        cached_config = _sync_load_config(cache_file)
        if cached_config is not None:
            logger.info("Using cached global configuration")
            return cached_config
    except Exception as e:
        logger.warning(f"Failed to load cached config, using defaults: {e}")
    
    # Fallback to default config
    logger.info("Using default global configuration")
    return get_default_global_config()
