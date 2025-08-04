"""
TUI-specific adapter for unified configuration cache.

Provides async interface compatible with existing TUI code.
"""

from openhcs.core.config_cache import (
    load_cached_global_config, 
    get_global_config_cache,
    AsyncExecutionStrategy
)
from openhcs.core.config import GlobalPipelineConfig

# Re-export with TUI-specific strategy
async def load_cached_global_config_tui() -> GlobalPipelineConfig:
    """Load cached config with async strategy for TUI."""
    return await load_cached_global_config(AsyncExecutionStrategy())


def get_tui_config_cache():
    """Get config cache with async strategy for TUI."""
    return get_global_config_cache(AsyncExecutionStrategy())


# Additional TUI-specific functions for backward compatibility
async def save_global_config_to_cache(config: GlobalPipelineConfig) -> bool:
    """Save global config to cache (TUI compatibility)."""
    cache = get_tui_config_cache()
    return await cache.save_config_to_cache(config)


async def clear_global_config_cache() -> bool:
    """Clear the global config cache (TUI compatibility)."""
    cache = get_tui_config_cache()
    return await cache.clear_cache()


def get_global_config_cache_info() -> dict:
    """Get information about the global config cache (TUI compatibility)."""
    cache = get_tui_config_cache()
    info = {
        "cache_file": str(cache.cache_file),
        "exists": cache.cache_file.exists(),
        "size": None,
        "modified": None
    }
    
    if info["exists"]:
        try:
            stat = cache.cache_file.stat()
            info["size"] = stat.st_size
            info["modified"] = stat.st_mtime
        except Exception:
            pass
    
    return info
