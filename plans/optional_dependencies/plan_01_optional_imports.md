# plan_01_optional_imports.md
## Component: Optional Dependency Handling

### Objective
Implement a simple approach for handling optional dependencies in OpenHCS that allows functions to be visible in the TUI even when their required libraries aren't installed, while maintaining the two-phase execution architecture (compilation and execution).

### Plan
1. Continue using `optional_import` for all library imports
2. Create an automated system for generating placeholder classes
3. Fix module-level code that tries to use optional libraries
4. Rely on the pipeline compiler to validate library availability during compilation

### Findings
The current implementation has several issues:

1. Some modules try to use optional libraries at module level (e.g., class definitions inheriting from `nn.Module`)
2. The `optional_import` function works for simple imports but needs improvement for handling complex import paths
3. The function registry imports all modules, which causes errors for modules with unavailable dependencies
4. We want functions to be visible in the TUI even when their required libraries aren't installed

The memory type decorators (`@torch_func`, `@cupy_func`, etc.) already mark which libraries are needed, and the pipeline compiler checks if the required libraries are available during compilation. We can leverage this existing architecture.

### Implementation Draft

#### 1. Fix the `optional_import` function
Update the `optional_import` function in `openhcs/core/utils.py` to use `importlib.import_module` for better handling of dotted import paths.

```python
def optional_import(module_name: str) -> Optional[Any]:
    """
    Import a module if available, otherwise return None.
    """
    try:
        import importlib
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError, AttributeError):
        return None
```

#### 2. Create an automated placeholder class system
Add a utility function to `openhcs/core/utils.py` that automatically generates placeholder classes for optional dependencies:

```python
def create_placeholder_class(name: str, base_class: Optional[Any] = None,
                            required_library: str = "") -> Type:
    """
    Create a placeholder class when a required library is not available.

    This function generates a placeholder class that can be used in place of a class
    that depends on an optional library. The placeholder class will raise an ImportError
    when any of its methods are called.

    Args:
        name: Name of the class
        base_class: Optional base class to inherit from if available
        required_library: Name of the required library

    Returns:
        Either a class that inherits from base_class if available,
        or a placeholder class that raises ImportError when methods are called

    Example:
        ```python
        # Import torch as an optional dependency
        torch = optional_import("torch")
        nn = optional_import("torch.nn") if torch is not None else None

        # Create the real or placeholder class
        MyNeuralNetwork = create_placeholder_class(
            "MyNeuralNetwork",
            base_class=nn.Module if nn is not None else None,
            required_library="PyTorch"
        )

        # Use the class as normal
        class MyModel(MyNeuralNetwork):
            def __init__(self):
                super().__init__()
                # Implementation
        ```
    """
    if base_class is not None:
        # Return a class that inherits from the base class
        return base_class
    else:
        # Create a placeholder class
        class PlaceholderClass:
            """Placeholder class when {required_library} is not available."""
            def __init__(self, *args, **kwargs):
                pass

            def __getattr__(self, name):
                raise ImportError(f"{required_library} is required for this class")

        PlaceholderClass.__name__ = name
        PlaceholderClass.__doc__ = f"Placeholder class when {required_library} is not available."
        return PlaceholderClass
```

#### 3. Fix module-level code
Identify and fix modules that try to use optional libraries at module level. Use the `create_placeholder_class` function to handle class definitions that depend on optional libraries.

Example for `n2v2_processor_torch.py`:
```python
# Import torch as an optional dependency
torch = optional_import("torch")
nn = optional_import("torch.nn") if torch is not None else None
F = optional_import("torch.nn.functional") if torch is not None else None

# Create placeholder classes for torch-dependent classes
Module = create_placeholder_class("Module",
                                base_class=nn.Module if nn is not None else None,
                                required_library="PyTorch")

# Define classes using the placeholder or real base class
class BlurPool3d(Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super().__init__()
        # Implementation
```

Modules to fix:
- `openhcs/processing/backends/enhance/n2v2_processor_torch.py`
- `openhcs/processing/backends/self_supervised_stitcher.py`
- Any other modules with similar issues

#### 4. Update import error handling in functions
Remove explicit checks for library availability in functions, as the compiler will ensure functions are only called when their dependencies are available.

Replace:
```python
if torch is None:
    raise ImportError("PyTorch is required for this function")
```

With:
```python
# The compiler will ensure this function is only called when PyTorch is available
# No need to check for PyTorch availability here
```

#### 5. Test the changes
Test the changes by:
1. Running the TUI with and without optional dependencies installed
2. Verifying that functions are visible in the TUI even when their dependencies aren't installed
3. Verifying that the pipeline compiler correctly validates library availability during compilation
4. Verifying that functions fail gracefully when called without their dependencies

This approach maintains the two-phase execution architecture while allowing functions to be visible in the TUI even when their dependencies aren't installed. The automated placeholder class system reduces code duplication and makes the pattern more consistent across the codebase.
