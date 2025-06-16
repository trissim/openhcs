#!/usr/bin/env python3
"""Test script for global config caching functionality."""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from openhcs.core.config import GlobalPipelineConfig, get_default_global_config
from openhcs.textual_tui.services.global_config_cache import (
    load_cached_global_config,
    save_global_config_to_cache,
    clear_global_config_cache,
    get_global_config_cache_info
)


async def test_config_cache():
    """Test the global config cache functionality."""
    print("🧪 Testing Global Config Cache")
    print("=" * 50)
    
    # Test 1: Check cache info
    print("\n1. Cache Info:")
    cache_info = get_global_config_cache_info()
    print(f"   Cache file: {cache_info['cache_file']}")
    print(f"   Exists: {cache_info['exists']}")
    if cache_info['exists']:
        print(f"   Size: {cache_info['size']} bytes")
    
    # Test 2: Clear cache to start fresh
    print("\n2. Clearing cache...")
    cleared = await clear_global_config_cache()
    print(f"   Cache cleared: {cleared}")
    
    # Test 3: Load config (should use defaults since cache is empty)
    print("\n3. Loading config (should use defaults)...")
    config1 = await load_cached_global_config()
    print(f"   Config loaded: {type(config1).__name__}")
    print(f"   num_workers: {config1.num_workers}")
    print(f"   microscope: {config1.microscope}")
    print(f"   vfs.default_intermediate_backend: {config1.vfs.default_intermediate_backend}")
    
    # Test 4: Modify config and save to cache
    print("\n4. Creating modified config and saving to cache...")
    from openhcs.core.config import VFSConfig, PathPlanningConfig
    from openhcs.constants import Microscope
    
    modified_config = GlobalPipelineConfig(
        num_workers=16,
        microscope=Microscope.ZEISS,
        vfs=VFSConfig(
            default_intermediate_backend="disk",
            default_materialization_backend="zarr",
            persistent_storage_root_path="/tmp/openhcs_test"
        ),
        path_planning=PathPlanningConfig(
            output_dir_suffix="_test_outputs",
            global_output_folder="/tmp/test_results"
        )
    )
    
    saved = await save_global_config_to_cache(modified_config)
    print(f"   Config saved to cache: {saved}")
    
    # Test 5: Load from cache (should get modified config)
    print("\n5. Loading config from cache...")
    config2 = await load_cached_global_config()
    print(f"   Config loaded: {type(config2).__name__}")
    print(f"   num_workers: {config2.num_workers}")
    print(f"   microscope: {config2.microscope}")
    print(f"   vfs.default_intermediate_backend: {config2.vfs.default_intermediate_backend}")
    print(f"   vfs.persistent_storage_root_path: {config2.vfs.persistent_storage_root_path}")
    print(f"   path_planning.output_dir_suffix: {config2.path_planning.output_dir_suffix}")
    
    # Test 6: Verify cache file exists
    print("\n6. Final cache info:")
    cache_info = get_global_config_cache_info()
    print(f"   Cache file: {cache_info['cache_file']}")
    print(f"   Exists: {cache_info['exists']}")
    if cache_info['exists']:
        print(f"   Size: {cache_info['size']} bytes")
    
    # Test 7: Verify configs are different
    print("\n7. Verification:")
    print(f"   Default config num_workers: {config1.num_workers}")
    print(f"   Cached config num_workers: {config2.num_workers}")
    print(f"   Configs are different: {config1 != config2}")
    
    print("\n✅ Config cache test completed!")


if __name__ == "__main__":
    asyncio.run(test_config_cache())
