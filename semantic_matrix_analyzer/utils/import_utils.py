"""
Utility functions for handling optional imports in Semantic Matrix Analyzer.

This module provides functions for importing optional dependencies
in a way that allows components to be registered even if their
dependencies aren't available at registration time.
"""
import importlib
import logging
from typing import Optional, Any, Type, Dict, Set

logger = logging.getLogger(__name__)

# Track which optional libraries have been checked
_CHECKED_LIBRARIES: Set[str] = set()
_MISSING_LIBRARIES: Dict[str, str] = {}


def optional_import(module_name: str, verbose: bool = True) -> Optional[Any]:
    """
    Import a module if available, otherwise return None.

    Args:
        module_name: Name of the module to import (can be a dotted path)
        verbose: Whether to log a warning when the module is not found

    Returns:
        The imported module if available, None otherwise
    """
    # Check if we've already tried to import this module
    if module_name in _CHECKED_LIBRARIES:
        if module_name in _MISSING_LIBRARIES and verbose:
            logger.debug(f"Optional module '{module_name}' is not available: {_MISSING_LIBRARIES[module_name]}")
        return None if module_name in _MISSING_LIBRARIES else importlib.import_module(module_name)

    try:
        # Add to checked libraries
        _CHECKED_LIBRARIES.add(module_name)
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError, AttributeError) as e:
        # Record the error message
        _MISSING_LIBRARIES[module_name] = str(e)
        if verbose:
            logger.debug(f"Optional module '{module_name}' is not available: {e}")
        return None


def create_placeholder_class(name: str, base_class: Optional[Any] = None,
                            required_library: str = "") -> Type:
    """
    Create a placeholder class when a required library is not available.

    This function generates a placeholder class that can be used in place of a class
    that depends on an optional library. The placeholder class will raise an ImportError
    when any of its methods are called or attributes are accessed (excluding __init__, __name__, __doc__).

    Args:
        name: Name of the class to be created.
        base_class: Optional base class to inherit from if available
        required_library: Name of the required library (for error messages)

    Returns:
        Either the base_class itself if it's not None (meaning the library was available),
        or a newly created placeholder class that raises ImportError on use.

    Example:
        ```python
        # Import torch as an optional dependency
        torch = optional_import("torch")
        nn = optional_import("torch.nn") if torch else None

        # If nn and nn.Module are available, Module will be nn.Module
        # Otherwise, Module will be a placeholder.
        Module = create_placeholder_class(
            "Module",
            base_class=nn.Module if nn else None,
            required_library="PyTorch"
        )

        class MyModel(Module): # Inherits from nn.Module or Placeholder
            def __init__(self):
                super().__init__() # Works for both
                if torch is not None: # Check if real or placeholder
                    self.linear = nn.Linear(10,1) # Only if real
            
            def forward(self, x):
                # This would raise ImportError if Module is a placeholder and self.linear wasn't set
                # or if super().forward() was called on a placeholder.
                if torch is not None:
                    return self.linear(x)
                else:
                    # Placeholder specific behavior or raise error
                    raise ImportError(f"PyTorch is required to use MyModel.forward")
        ```
    """
    if base_class is not None:
        # The library and base class are available, return the actual base class.
        # The calling code will then define a class that inherits from this real base.
        return base_class
    else:
        # Create a placeholder class
        # This class will be used as a base for classes in other modules.
        # When methods of those derived classes are called (especially super().__init__ or super().method()),
        # or attributes are accessed, __getattr__ on this placeholder will be triggered if not found.
        class Placeholder:
            _required_library_name = required_library or "An optional library"

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                # The __init__ of a placeholder should generally do nothing or
                # just store args/kwargs if needed for some very specific placeholder logic.
                # It should NOT try to call super().__init__ if it's meant to be a root placeholder.
                pass

            def __getattr__(self, item: str) -> Any:
                # This will be called for any attribute not found on the instance.
                # This includes methods.
                raise ImportError(
                    f"{self._required_library_name} is required to use the attribute/method '{item}' "
                    f"of class '{name}' (which is a placeholder)."
                )

        Placeholder.__name__ = name
        Placeholder.__doc__ = (
            f"Placeholder for '{name}' when {required_library or 'its required library'} "
            "is not available. Accessing attributes or methods will raise an ImportError."
        )
        return Placeholder


def get_missing_libraries() -> Dict[str, str]:
    """
    Get a dictionary of libraries that were requested but not available.

    Returns:
        Dictionary mapping library names to error messages
    """
    return _MISSING_LIBRARIES.copy()


def clear_import_cache() -> None:
    """
    Clear the cache of checked libraries.
    
    This is mainly useful for testing.
    """
    _CHECKED_LIBRARIES.clear()
    _MISSING_LIBRARIES.clear()
