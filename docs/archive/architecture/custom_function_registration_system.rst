Custom Function Registration System
====================================

OpenHCS provides a dynamic custom function registration system that enables users to define custom processing functions via code editor and have them automatically integrated into the function registry with full type safety and validation.

**Why This Matters**: Scientific workflows often require specialized processing functions beyond standard libraries. OpenHCS enables users to create custom functions without modifying the codebase, with automatic memory type decoration, validation, persistence, and UI integration.

## Core Capabilities

The custom function registration system provides:

- **Code Editor Integration**: Create custom functions directly in the GUI via simple code editor
- **Automatic Registration**: Functions are automatically discovered and registered in the function registry
- **Multi-Backend Support**: Support for all memory types (numpy, cupy, torch, tensorflow, jax, pyclesperanto)
- **Persistent Storage**: Custom functions persist to disk and auto-load on startup
- **Type Safety**: 100% type-annotated with strict validation (no duck typing)
- **Security Validation**: Import validation prevents dangerous operations
- **UI Integration**: Automatic UI refresh via Qt signals when functions change

## Architecture Overview

The custom function system consists of five core modules with strict separation of concerns:

.. code-block:: text

   openhcs/processing/custom_functions/
   ├── manager.py           # CustomFunctionManager - lifecycle operations
   ├── validation.py        # Code validation with security checks
   ├── templates.py         # Memory type templates
   ├── signals.py          # Qt signals for UI updates
   └── __init__.py         # Public API exports

## Custom Function Manager

The ``CustomFunctionManager`` class manages the complete lifecycle of custom functions:

.. code-block:: python

   from openhcs.processing.custom_functions import CustomFunctionManager

   manager = CustomFunctionManager()

   # Register from code
   code = '''
   from openhcs.core.memory.decorators import numpy
   import numpy as np

   @numpy
   def my_function(image, threshold=0.5):
       """Custom thresholding function."""
       return np.where(image > threshold, image, 0)
   '''

   funcs = manager.register_from_code(code, persist=True)

   # Load all custom functions on startup
   count = manager.load_all_custom_functions()

   # List registered custom functions
   info_list = manager.list_custom_functions()

   # Delete custom function
   manager.delete_custom_function('my_function')

**Key Methods**:

- ``register_from_code(code, persist=True)``: Validate, execute, and register functions
- ``load_all_custom_functions()``: Auto-load from ~/.local/share/openhcs/custom_functions/
- ``delete_custom_function(func_name)``: Remove function and delete file
- ``list_custom_functions()``: Query registered custom functions

## Validation System

The validation system provides multi-stage validation with fail-loud error handling:

**Stage 1: Syntax Validation**

.. code-block:: python

   from openhcs.processing.custom_functions.validation import validate_syntax

   result = validate_syntax(code)
   if not result.is_valid:
       print(f"Syntax errors: {result.errors}")

Uses ``ast.parse()`` to validate Python syntax before execution.

**Stage 2: Import Validation**

.. code-block:: python

   from openhcs.processing.custom_functions.validation import validate_imports

   result = validate_imports(code)
   if not result.is_valid:
       print(f"Dangerous imports detected: {result.errors}")

Blocks dangerous imports (``os``, ``subprocess``, ``sys``, ``socket``, etc.) to prevent malicious code.

**Stage 3: Decorator Validation**

.. code-block:: python

   from openhcs.processing.custom_functions.validation import validate_decorator

   result = validate_decorator(code)
   if not result.is_valid:
       print(f"Missing decorators: {result.errors}")

Ensures at least one function has a memory type decorator (``@numpy``, ``@cupy``, etc.).

**Stage 4: Function Signature Validation**

.. code-block:: python

   from openhcs.processing.custom_functions.validation import validate_function

   result = validate_function(func)
   if not result.is_valid:
       print(f"Invalid signature: {result.errors}")

Validates that first parameter is named ``image`` and required memory type attributes exist.

**ValidationResult Dataclass**:

.. code-block:: python

   @dataclass(frozen=True)
   class ValidationResult:
       is_valid: bool
       errors: List[str]
       warnings: List[str]
       function_names: List[str]

**ValidationError Exception**:

.. code-block:: python

   class ValidationError(Exception):
       """Raised when custom function code is invalid."""

       def __init__(self, message: str, line_number: int = 0, code_snippet: str = ""):
           self.message = message
           self.line_number = line_number
           self.code_snippet = code_snippet

## Template System

The template system provides starter code for all memory types with proper imports, decorators, and documentation:

.. code-block:: python

   from openhcs.processing.custom_functions.templates import (
       get_default_template,
       get_template_for_memory_type,
       AVAILABLE_MEMORY_TYPES
   )

   # Get default numpy template
   template = get_default_template()

   # Get specific memory type template
   cupy_template = get_template_for_memory_type('cupy')
   torch_template = get_template_for_memory_type('torch')

**Available Memory Types**:

- ``numpy`` - CPU arrays with NumPy
- ``cupy`` - GPU arrays with CuPy (CUDA)
- ``torch`` - PyTorch tensors (CPU/GPU)
- ``tensorflow`` - TensorFlow tensors
- ``jax`` - JAX arrays with automatic differentiation
- ``pyclesperanto`` - GPU-accelerated OpenCL

**Example NumPy Template**:

.. code-block:: python

   from openhcs.core.memory.decorators import numpy
   import numpy as np

   @numpy
   def my_custom_function(image, scale: float = 1.0, offset: float = 0.0):
       """
       Custom image processing function using NumPy.

       Args:
           image: Input image as 3D numpy array (C, Y, X)
           scale: Scaling factor to multiply image values
           offset: Offset to add after scaling

       Returns:
           Processed image as 3D numpy array (C, Y, X)
       """
       # Your processing code here
       processed = image * scale + offset

       # Optional: return metadata alongside image
       # metadata = {"mean_intensity": float(np.mean(processed))}
       # return processed, metadata

       return processed

**Template Structure Requirements**:

- First parameter must be named ``image`` (3D array: C, Y, X)
- Must be decorated with memory type decorator
- Must include proper docstring with Args/Returns
- Should show example metadata return pattern

## Signal System

The signal system provides automatic UI refresh when custom functions change:

.. code-block:: python

   from openhcs.processing.custom_functions.signals import custom_function_signals

   # Connect to signal in UI components
   custom_function_signals.functions_changed.connect(self.refresh_function_list)

   # Signal emitted automatically after:
   # - register_from_code()
   # - load_all_custom_functions()
   # - delete_custom_function()

**CustomFunctionSignals Class**:

.. code-block:: python

   class CustomFunctionSignals(QObject):
       """Qt signals for custom function state changes."""

       # Emitted when custom functions are added, deleted, or reloaded
       functions_changed = pyqtSignal()

**Global Singleton Instance**:

.. code-block:: python

   # All components connect to this single instance
   custom_function_signals = CustomFunctionSignals()

## Storage and Persistence

Custom functions are stored in the XDG data directory:

**Storage Location**: ``~/.local/share/openhcs/custom_functions/``

**File Format**: One ``.py`` file per function, named ``{function_name}.py``

**Auto-Loading**: Functions are automatically loaded on OpenHCS startup via ``func_registry._auto_initialize_registry()``

**Example Storage Structure**:

.. code-block:: text

   ~/.local/share/openhcs/custom_functions/
   ├── my_threshold_function.py
   ├── custom_blur.py
   └── intensity_normalization.py

**File Contents**: Complete executable Python code with imports and decorators

.. code-block:: python

   # Example: my_threshold_function.py
   from openhcs.core.memory.decorators import numpy
   import numpy as np

   @numpy
   def my_threshold_function(image, threshold=0.5):
       """Custom adaptive thresholding."""
       return np.where(image > threshold, image, 0)

## Integration with Function Registry

Custom functions integrate seamlessly with the OpenHCS function registry:

**Registration Flow**:

1. User provides Python code via code editor
2. Code is validated (syntax, imports, decorators, signatures)
3. Code is executed in controlled namespace with memory decorators
4. Decorated functions are discovered via ``hasattr(obj, 'input_memory_type')``
5. Functions are validated for required attributes
6. Functions are registered via ``func_registry.register_function()``
7. Metadata caches are cleared (``RegistryService``, ``FunctionSelectorDialog``)
8. Qt signal is emitted for UI refresh

**Registry Integration**:

.. code-block:: python

   from openhcs.processing.func_registry import register_function

   # Custom functions registered as 'openhcs' backend
   register_function(func, backend='openhcs')

   # Appears in function selector with composite key: "openhcs:my_function"
   # Accessible alongside standard library functions

**Memory Type Attributes**:

Custom functions must have attributes set by memory type decorators:

.. code-block:: python

   @numpy
   def my_function(image):
       return image * 2

   # Decorator sets required attributes:
   # my_function.input_memory_type = 'numpy'
   # my_function.output_memory_type = 'numpy'
   # my_function.backend = 'numpy'

## OpenHCS Architecture Compliance

The custom function system follows all OpenHCS architectural principles:

**No Duck Typing**:

- Zero instances of ``getattr()`` with fallback defaults
- No ``hasattr()`` for guaranteed attributes (only for user code validation)
- Direct attribute access where contracts guarantee existence

.. code-block:: python

   # COMPLIANT: Direct access after validation
   memory_type = func.input_memory_type  # Validation confirmed it exists

   # NON-COMPLIANT: Would be defensive duck typing
   # memory_type = getattr(func, 'input_memory_type', 'numpy')  # ❌ Never do this

**100% Type Annotations**:

All public functions have complete type signatures:

.. code-block:: python

   def register_from_code(
       self,
       code: str,
       persist: bool = True
   ) -> List[Callable]:
       """Execute code and register all decorated functions found."""
       ...

**Fail-Loud Behavior**:

Validation errors raise exceptions immediately with clear messages:

.. code-block:: python

   # Syntax error
   raise ValidationError("Syntax error: invalid syntax on line 5")

   # Missing decorator
   raise ValidationError(
       "No valid functions found with memory type decorators. "
       "Functions must be decorated with one of: @numpy, @cupy, @torch, ..."
   )

   # Invalid signature
   raise ValidationError(
       "Function 'my_func' first parameter is 'img', but must be 'image'"
   )

**Frozen Dataclasses**:

Immutable data structures for validation results and metadata:

.. code-block:: python

   @dataclass(frozen=True)
   class ValidationResult:
       is_valid: bool
       errors: List[str]
       warnings: List[str]
       function_names: List[str]

   @dataclass(frozen=True)
   class CustomFunctionInfo:
       name: str
       file_path: Path
       memory_type: str
       doc: str

## Security Considerations

The custom function system implements basic security measures but does not provide full sandboxing:

**Import Validation**: Blocks dangerous modules (``os``, ``subprocess``, ``sys``, ``socket``, etc.)

**Execution Model**: Uses ``exec()`` with controlled namespace but with full Python builtins

**Threat Model**: Designed for trusted users, not untrusted code execution

**Security Limitations**:

- Custom functions execute with full Python privileges
- Import validation is not exhaustive
- No resource limits (CPU, memory, time)
- No process isolation or sandboxing

**Recommended Security Measures**:

- Only allow trusted users to create custom functions
- Review custom function code before deployment
- Use file system permissions to restrict write access to custom functions directory
- Consider running OpenHCS in containerized environment for isolation

## Performance Considerations

**Validation Overhead**: Minimal (< 10ms for typical functions) due to AST parsing only

**Registration Time**: Instant after validation (direct function registry modification)

**Startup Time**: Auto-loading adds ~1-5ms per custom function

**Memory Overhead**: Negligible (functions stored as Python objects)

**Execution Performance**: Identical to manually-written functions (no runtime overhead)

## Error Handling and Debugging

The system provides detailed error messages for common issues:

**Missing Decorator Error**:

.. code-block:: text

   ValidationError: No valid functions found with memory type decorators.
   Functions must be decorated with one of: @numpy, @cupy, @torch, ...

**Invalid Signature Error**:

.. code-block:: text

   ValidationError: Function 'process_image' first parameter is 'img', but must be 'image' (3D array: C, Y, X).

**Dangerous Import Error**:

.. code-block:: text

   ValidationError: Dangerous import detected: 'os'. Module 'os' is not allowed in custom functions.

**Execution Error**:

.. code-block:: text

   ValidationError: Code execution failed: NameError: name 'undefined_var' is not defined

## API Reference

**Manager API**:

.. code-block:: python

   class CustomFunctionManager:
       def __init__(self) -> None: ...

       def register_from_code(
           self,
           code: str,
           persist: bool = True
       ) -> List[Callable]: ...

       def load_all_custom_functions(self) -> int: ...

       def delete_custom_function(self, func_name: str) -> bool: ...

       def list_custom_functions(self) -> List[CustomFunctionInfo]: ...

**Validation API**:

.. code-block:: python

   def validate_code(code: str) -> ValidationResult: ...

   def validate_function(func: Callable) -> ValidationResult: ...

   def validate_syntax(code: str) -> ValidationResult: ...

   def validate_imports(code: str) -> ValidationResult: ...

   def validate_decorator(code: str) -> ValidationResult: ...

**Template API**:

.. code-block:: python

   def get_default_template() -> str: ...

   def get_template_for_memory_type(memory_type: str) -> str: ...

   AVAILABLE_MEMORY_TYPES: List[str] = [
       'numpy', 'cupy', 'torch', 'tensorflow', 'jax', 'pyclesperanto'
   ]

## Future Enhancements

Potential improvements for future versions:

**Enhanced Security**:

- Process isolation via multiprocessing
- Resource limits (CPU time, memory)
- Whitelist-based import system
- Code signing for trusted functions

**Advanced Features**:

- Function templates with parameter hints
- Custom function marketplace/sharing
- Version control integration
- Unit test generation
- Performance profiling integration

**UI Improvements**:

- Custom function management dialog
- Live preview of function output
- Parameter hints from docstrings
- Inline documentation viewer

## Related Documentation

**Core Systems**:

- :doc:`function_registry_system` - Main function registry architecture
- :doc:`memory_type_system` - Memory type decorators and conversion
- :doc:`function_pattern_system` - Function patterns and parameter handling

**User Guides**:

- :doc:`../user_guide/custom_functions` - Creating custom functions (user guide)
- :doc:`../user_guide/code_ui_editing` - Code/UI bidirectional editing

**Development**:

- :doc:`../development/respecting_codebase_architecture` - No duck typing principles
- :doc:`../development/refactoring_principles` - Type safety and fail-loud behavior
