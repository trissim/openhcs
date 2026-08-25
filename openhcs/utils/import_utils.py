"""
Utility functions for handling optional imports in OpenHCS.

This module provides functions for importing optional dependencies
in a way that allows functions to be registered even if their
dependencies are not available at registration time.
"""

from typing import Any

from metaclass_registry import import_module_preserving_root_logging


class _ModulePlaceholder:
    """Falsy stand-in for annotations that reference an absent module."""

    def __init__(self, module_name: str) -> None:
        self._module_name = module_name

    def __bool__(self) -> bool:
        return False

    def __getattr__(self, name: str) -> "_ModulePlaceholder":
        return _ModulePlaceholder(f"{self._module_name}.{name}")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise ImportError(f"Module {self._module_name!r} is not available")

    def __repr__(self) -> str:
        return f"<ModulePlaceholder for {self._module_name!r}>"


def optional_import_or_none(module_name: str) -> Any | None:
    """
    Import a module if available, otherwise return None.

    Args:
        module_name: Name of the module to import (can be a dotted path)

    Returns:
        The imported module if available, None otherwise
    """
    try:
        return import_module_preserving_root_logging(module_name)
    except ImportError:
        return None


def optional_import_placeholder(module_name: str) -> Any:
    """Import an optional module or return a falsy annotation placeholder."""

    module = optional_import_or_none(module_name)
    return module if module is not None else _ModulePlaceholder(module_name)


def create_placeholder_class(
    name: str,
    base_class: type | None = None,
    required_library: str = "",
) -> type:
    """Return a supplied base class or an unavailable-library stand-in."""

    if base_class is not None:
        return base_class

    class Placeholder:
        _required_library_name = required_library or "An optional library"

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def __getattr__(self, item: str) -> Any:
            raise ImportError(
                f"{self._required_library_name} is required to use {item!r} on {name!r}"
            )

    Placeholder.__name__ = name
    Placeholder.__doc__ = (
        f"Placeholder for {name!r} when {required_library or 'its dependency'} "
        "is unavailable."
    )
    return Placeholder
