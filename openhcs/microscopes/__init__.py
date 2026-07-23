"""
Microscope-specific implementations for openhcs.

This package contains modules for different microscope types, each providing
concrete implementations of FilenameParser and MetadataHandler interfaces.

The package uses automatic discovery to find and register all handler implementations,
following OpenHCS generic solution principles. All handlers are automatically
discovered and registered via metaclass during discovery - no hardcoded imports needed.
"""

from importlib import import_module
from pkgutil import iter_modules


_DISCOVERY_EXCLUDED_MODULES = frozenset(
    {
        "handler_registry_service",
        "microscope_base",
        "microscope_interfaces",
        "tiff_metadata_mixin",
    }
)


def _load_microscope_modules() -> None:
    """Import microscope modules so nominal handler classes self-register."""
    for module_info in iter_modules(__path__):
        module_name = module_info.name
        if module_name.startswith("_") or module_name in _DISCOVERY_EXCLUDED_MODULES:
            continue
        import_module(f"{__name__}.{module_name}")


_load_microscope_modules()

# Import base components and factory function
from openhcs.microscopes.microscope_base import create_microscope_handler

# Import registry service for automatic discovery
from openhcs.microscopes.handler_registry_service import (
    get_all_handler_types,
    is_handler_available
)

__all__ = [
    # Factory function - primary public API
    'create_microscope_handler',
    # Registry service functions
    'get_all_handler_types',
    'is_handler_available',
]
