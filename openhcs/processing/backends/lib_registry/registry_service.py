"""
Registry Service - Clean function discovery and metadata access.

Provides unified access to all registry implementations with automatic discovery.
Follows OpenHCS generic solution principle - automatically adapts to new registries.
"""

import logging
import inspect
from collections.abc import Callable
from typing import Dict, Optional

from openhcs.constants import MemoryType
from openhcs.utils.environment import OpenHCSProcessEnvironment
from .unified_registry import LibraryRegistryBase, FunctionMetadata, LIBRARY_REGISTRIES

logger = logging.getLogger(__name__)


class RegistryService:
    """
    Clean service for registry discovery and function metadata access.
    
    Automatically discovers all registry implementations and provides
    unified access to their functions with caching.
    """
    
    _metadata_cache: Optional[Dict[str, FunctionMetadata]] = None
    
    @classmethod
    def get_all_functions_with_metadata(cls) -> Dict[str, FunctionMetadata]:
        """Get unified metadata for all functions from all registries."""
        if cls._metadata_cache is not None:
            logger.debug(f"🎯 REGISTRY SERVICE: Using cached metadata ({len(cls._metadata_cache)} functions)")
            return cls._metadata_cache

        logger.debug("🎯 REGISTRY SERVICE: Discovering functions from all registries...")
        all_functions = {}

        # Registries auto-discovered on first access to LIBRARY_REGISTRIES
        registry_classes = list(LIBRARY_REGISTRIES.values())
        logger.debug(f"🎯 REGISTRY SERVICE: Found {len(registry_classes)} registered library registries")

        # Load functions from each registry
        for registry_class in registry_classes:
            try:
                registry_instance = registry_class()

                if _cpu_only_mode_enabled() and not _registry_supports_cpu_only(
                    registry_instance
                ):
                    logger.info(
                        "CPU-only registry discovery skipping %s",
                        registry_instance.library_name,
                    )
                    continue

                # Skip if library not available
                if not registry_instance.is_library_available():
                    logger.warning(f"Library {registry_instance.library_name} not available, skipping")
                    continue

                # Get functions from this registry (with caching)
                logger.debug(f"🎯 REGISTRY SERVICE: Calling load_or_discover_functions for {registry_instance.library_name}")
                functions = registry_instance.load_or_discover_functions()
                logger.debug(f"🎯 REGISTRY SERVICE: Retrieved {len(functions)} {registry_instance.library_name} functions")

                # Use composite keys to prevent function name collisions between backends
                # Format: "backend:function_name" (e.g., "torch:stack_percentile_normalize")
                for metadata in functions.values():
                    all_functions[metadata.composite_key] = metadata

            except Exception as e:
                logger.warning(f"Failed to load registry {registry_class.__name__}: {e}")
                continue

        logger.info(f"Total functions discovered: {len(all_functions)}")
        cls._metadata_cache = all_functions
        return all_functions

    @classmethod
    def metadata_for_callable(
        cls,
        func: Callable,
    ) -> tuple[str, FunctionMetadata] | None:
        """Return the registry-owned callable projection for one declaration.

        Registry wrappers preserve their declaration through ``__wrapped__``.
        Comparing that nominal callable identity keeps transport code blind to
        library names, generated registry keys, and module naming conventions.
        """

        declared = inspect.unwrap(func)
        cached_metadata = cls._metadata_cache
        if cached_metadata is not None:
            for composite_key, metadata in cached_metadata.items():
                if inspect.unwrap(metadata.func) is declared:
                    return composite_key, metadata

        for composite_key, metadata in cls.get_all_functions_with_metadata().items():
            if inspect.unwrap(metadata.func) is declared:
                return composite_key, metadata
        return None

    @classmethod
    def registered_callable(cls, func: Callable) -> Callable:
        """Project a declaration onto its registered runtime owner when present."""

        declared = inspect.unwrap(func)
        cached_metadata = cls._metadata_cache
        if cached_metadata is not None:
            for metadata in cached_metadata.values():
                if inspect.unwrap(metadata.func) is declared:
                    return metadata.func

        # Local projection must not turn source parsing into catalog warmup.
        for registry_type in LibraryRegistryBase.loaded_registry_types():
            metadata = registry_type.metadata_for_declared_callable(func)
            if metadata is not None and inspect.unwrap(metadata.func) is declared:
                return metadata.func

        match = cls.metadata_for_callable(func)
        return func if match is None else match[1].func
    

    
    @classmethod
    def clear_metadata_cache(cls) -> None:
        """Clear cached metadata to force re-discovery."""
        cls._metadata_cache = None
        logger.info("Registry metadata cache cleared")


# Backward compatibility aliases
FunctionRegistryService = RegistryService
get_all_functions_with_metadata = RegistryService.get_all_functions_with_metadata
clear_metadata_cache = RegistryService.clear_metadata_cache


def _cpu_only_mode_enabled() -> bool:
    return OpenHCSProcessEnvironment.cpu_only_mode()


def _registry_supports_cpu_only(registry: LibraryRegistryBase) -> bool:
    memory_type = registry.get_memory_type()
    return memory_type is None or memory_type == MemoryType.NUMPY.value
