"""
Unified caching utilities for external library function registries.

Provides common caching patterns extracted from scikit-image registry
for use by pyclesperanto, CuPy, and other external library registries.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Any, Callable

from openhcs.core.xdg_paths import get_cache_file_path

logger = logging.getLogger(__name__)


def get_library_cache_path(library_name: str) -> Path:
    """
    Get the cache file path for a specific library.

    Args:
        library_name: Name of the library (e.g., 'pyclesperanto', 'cupy')

    Returns:
        Path to the cache file
    """
    cache_filename = f"{library_name}_function_metadata.json"
    return get_cache_file_path(cache_filename)


def save_library_metadata(
    library_name: str,
    registry: Dict[str, Any],
    get_version_func: Callable[[], str],
    extract_cache_data_func: Callable[[Any], Dict[str, Any]]
) -> None:
    """
    Save library function metadata to cache.

    Args:
        library_name: Name of the library
        registry: Registry dictionary mapping function names to metadata objects
        get_version_func: Function that returns the library version string
        extract_cache_data_func: Function that extracts cacheable data from metadata object
    """
    cache_path = get_library_cache_path(library_name)

    # Get library version
    try:
        library_version = get_version_func()
    except Exception:
        library_version = "unknown"

    # Build cache data structure
    cache_data = {
        'cache_version': '1.0',
        'library_version': library_version,
        'timestamp': time.time(),
        'functions': {}
    }

    # Extract function metadata
    for full_name, func_meta in registry.items():
        try:
            cache_data['functions'][full_name] = extract_cache_data_func(func_meta)
        except Exception as e:
            logger.warning(f"Failed to extract cache data for {full_name}: {e}")

    # Save to disk
    try:
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
        logger.info(f"Saved {library_name} metadata cache: {len(cache_data['functions'])} functions")
    except Exception as e:
        logger.warning(f"Failed to save {library_name} metadata cache: {e}")


def load_library_metadata(
    library_name: str,
    get_version_func: Callable[[], str],
    max_age_days: int = 7
) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Load library function metadata from cache with validation.

    Args:
        library_name: Name of the library
        get_version_func: Function that returns the current library version
        max_age_days: Maximum age in days before cache is considered stale

    Returns:
        Dictionary of cached function metadata, or None if cache invalid
    """
    cache_path = get_library_cache_path(library_name)

    if not cache_path.exists():
        logger.debug(f"No {library_name} cache found at {cache_path}")
        return None

    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)

        # Handle old cache format (direct metadata dict)
        if 'functions' not in cache_data:
            logger.info(f"Found old {library_name} cache format - will rebuild")
            return None

        # Validate library version
        try:
            current_version = get_version_func()
        except Exception:
            current_version = "unknown"

        cached_version = cache_data.get('library_version', 'unknown')
        if cached_version != current_version:
            logger.info(f"{library_name} version changed ({cached_version} → {current_version}) - will rebuild cache")
            return None

        # Check cache age
        cache_timestamp = cache_data.get('timestamp', 0)
        cache_age_days = (time.time() - cache_timestamp) / (24 * 3600)
        if cache_age_days > max_age_days:
            logger.info(f"{library_name} cache is {cache_age_days:.1f} days old - will rebuild")
            return None

        functions = cache_data['functions']
        logger.info(f"Loaded valid {library_name} metadata cache: {len(functions)} functions")
        return functions

    except Exception as e:
        logger.warning(f"Failed to load {library_name} metadata cache: {e}")
        return None


def clear_library_cache(library_name: str) -> None:
    """
    Clear the library metadata cache to force rebuild on next startup.

    Args:
        library_name: Name of the library
    """
    cache_path = get_library_cache_path(library_name)
    try:
        if cache_path.exists():
            cache_path.unlink()
            logger.info(f"{library_name} metadata cache cleared")
        else:
            logger.info(f"No {library_name} metadata cache to clear")
    except Exception as e:
        logger.warning(f"Failed to clear {library_name} metadata cache: {e}")


def register_functions_from_cache(
    library_name: str,
    cached_metadata: Dict[str, Dict[str, Any]],
    get_function_func: Callable[[str, str], Any],
    register_function_func: Callable[[Any, str, str], None],
    memory_type: str
) -> tuple[int, int]:
    """
    Register library functions using cached metadata.

    Args:
        library_name: Name of the library
        cached_metadata: Dictionary of cached function metadata
        get_function_func: Function to get the actual function object (module_path, func_name) -> function
        register_function_func: Function to register the function (func, func_name, memory_type) -> None
        memory_type: Memory type for registration

    Returns:
        Tuple of (decorated_count, skipped_count)
    """
    logger.info(f"Registering {library_name} functions from metadata cache")

    decorated_count = 0
    skipped_count = 0

    for full_name, func_data in cached_metadata.items():
        try:
            func_name = func_data['name']
            module_path = func_data['module']
            contract = func_data['contract']

            # Skip functions with unknown or dimension-changing contracts
            if contract in ['unknown', 'dim_change']:
                skipped_count += 1
                continue

            # Get the actual function object
            original_func = get_function_func(module_path, func_name)
            if original_func is None:
                logger.warning(f"Could not find function {func_name} in {module_path}")
                skipped_count += 1
                continue

            # Register the function
            register_function_func(original_func, func_name, memory_type)
            decorated_count += 1

        except Exception as e:
            logger.error(f"Failed to register {full_name} from cache: {e}")
            skipped_count += 1

    logger.info(f"Registered {decorated_count} {library_name} functions from cache")
    logger.info(f"Skipped {skipped_count} functions (unknown/dim_change contracts or errors)")

    return decorated_count, skipped_count


def should_use_cache_for_library(library_name: str) -> bool:
    """
    Determine if cache should be used for a library based on environment.

    Args:
        library_name: Name of the library

    Returns:
        True if cache should be used, False if full discovery should run
    """
    import os

    # Always use cache in subprocess mode
    if os.environ.get('OPENHCS_SUBPROCESS_MODE'):
        logger.info(f"SUBPROCESS: Using cached metadata for {library_name} function registration")
        return True

    # Use cache for TUI speedup too
    logger.info(f"Checking for cached metadata to speed up {library_name} startup...")
    return True


def get_cache_status(library_name: str) -> Dict[str, Any]:
    """
    Get status information about a library's cache.

    Args:
        library_name: Name of the library

    Returns:
        Dictionary with cache status information
    """
    cache_path = get_library_cache_path(library_name)

    status = {
        'library': library_name,
        'cache_file': str(cache_path),
        'exists': cache_path.exists(),
        'size': None,
        'modified': None,
        'function_count': None,
        'library_version': None,
        'cache_age_days': None
    }

    if status['exists']:
        try:
            stat = cache_path.stat()
            status['size'] = stat.st_size
            status['modified'] = stat.st_mtime

            # Try to read cache data
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)

            if 'functions' in cache_data:
                status['function_count'] = len(cache_data['functions'])
                status['library_version'] = cache_data.get('library_version')

                cache_timestamp = cache_data.get('timestamp', 0)
                if cache_timestamp:
                    status['cache_age_days'] = (time.time() - cache_timestamp) / (24 * 3600)

        except Exception as e:
            logger.debug(f"Could not read cache status for {library_name}: {e}")

    return status


def run_cached_registration(library_name: str, register_from_cache_fn) -> bool:
    """
    Try to register functions for a library from cache based on environment heuristics.

    Returns True if registration was handled via cache (and caller should stop),
    otherwise False to indicate the caller should proceed with full discovery.
    """
    try:
        if should_use_cache_for_library(library_name):
            used = bool(register_from_cache_fn())
            return used
    except Exception as e:
        logger.warning(f"{library_name}: cache fast path failed with error; falling back to discovery: {e}")
    return False

