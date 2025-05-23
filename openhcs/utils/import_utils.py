"""
Utility functions for handling optional imports in OpenHCS.

This module provides functions for importing optional dependencies
in a way that allows functions to be registered even if their
dependencies aren't available at registration time.
"""
import importlib
from typing import Optional, Any, Type


def optional_import(module_name: str) -> Optional[Any]:
    """
    Import a module if available, otherwise return None.

    Args:
        module_name: Name of the module to import (can be a dotted path)

    Returns:
        The imported module if available, None otherwise
    """
    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError, AttributeError): # Added AttributeError for safety
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
        base_class: Optional base class to inherit from if the actual library is available.
                    If the library and thus the base_class are None, a placeholder is created.
        required_library: Name of the required library (for error messages).

    Returns:
        Either the base_class itself if it's not None (meaning the library was available),
        or a newly created placeholder class that raises ImportError on use.

    Example:
        ```python
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
                if isinstance(self, nn.Module): # Check if real or placeholder
                    self.linear = nn.Linear(10,1) # Only if real
            
            def forward(self, x):
                # This would raise ImportError if Module is a placeholder and self.linear wasn't set
                # or if super().forward() was called on a placeholder.
                if isinstance(self, nn.Module):
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
                # If this placeholder *itself* is meant to inherit from something, that's a different pattern.
                # For replacing e.g. nn.Module, this __init__ is fine.
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
