"""
TUI-specific adapter for unified configuration cache.

Provides async interface compatible with existing TUI code.
"""

from openhcs.core.config_cache import load_cached_global_config, global_pipeline_config_cache
from openhcs.core.config import GlobalPipelineConfig

async def load_cached_global_config_tui() -> GlobalPipelineConfig:
    """Load cached config without blocking the Textual event loop."""
    return await load_cached_global_config()


async def save_global_config_to_cache(config: GlobalPipelineConfig) -> bool:
    """Save the global config without blocking the Textual event loop."""
    cache = global_pipeline_config_cache()
    return await cache.save(config)
