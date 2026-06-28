==============================
Custom Function Management
==============================

*Module: openhcs.processing.custom_functions.manager*  
*Status: STABLE*

---

Overview
========

Custom functions are user-defined processing functions that integrate seamlessly with OpenHCS pipelines. The custom function management system provides atomic updates, module injection, and signal-based synchronization to keep UI and CLI in sync.

Quick Reference
===============

.. code-block:: python

    from openhcs.processing.custom_functions.manager import CustomFunctionManager
    
    # Create manager
    manager = CustomFunctionManager()
    
    # Add custom function
    function_code = '''
    from openhcs.core.memory import numpy
    import numpy as np
    from scipy.ndimage import gaussian_filter

    @numpy
    def my_custom_filter(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Apply custom Gaussian filter."""
        return gaussian_filter(image, sigma=sigma)
    '''
    
    manager.add_function('my_custom_filter', function_code)
    
    # Function is now available in pipelines
    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.processing.custom_functions import my_custom_filter

    step = FunctionStep(
        func=(my_custom_filter, {'sigma': 2.0}),
        name='my_custom_filter',
    )

End-to-End Flow
===============

The custom function lifecycle involves four stages:

1. **Code Editing**: User writes function code in GUI editor
2. **Validation**: Function signature and decorators validated
3. **Registration**: Function added to global registry
4. **Synchronization**: UI and CLI updated via signals

.. code-block:: python

    # Stage 1: User edits code in CustomFunctionManagerDialog
    dialog = CustomFunctionManagerDialog()
    dialog.show()
    
    # Stage 2: Validation on save
    # - Check for required decorators (@numpy, @cupy, etc.)
    # - Validate function signature (first arg must be image array)
    # - Verify return type annotation
    
    # Stage 3: Atomic registration
    manager.add_function(name, code)
    # - Execute code in isolated namespace
    # - Extract function object
    # - Inject module name
    # - Register in global function registry
    
    # Stage 4: Signal emission
    # - function_added signal emitted
    # - UI updates function list
    # - CLI sees new function immediately

Atomic Update Mechanism
========================

Custom function updates are atomic to prevent partial registration:

.. code-block:: python

    def add_function(self, name: str, code: str) -> None:
        """Atomically add or update custom function."""
        
        # Execute code in isolated namespace
        namespace = {}
        try:
            exec(code, namespace)
        except Exception as e:
            raise ValidationError(f"Code execution failed: {e}")
        
        # Extract function object
        if name not in namespace:
            raise ValidationError(f"Function '{name}' not found in code")
        
        func = namespace[name]
        
        # Validate function
        validation_result = validate_function(func)
        if not validation_result.is_valid:
            raise ValidationError(validation_result.errors)
        
        # Atomic update: Only register if all validation passes
        register_function(func, backend='openhcs')
        self._save_to_disk(name, code)
        
        # Emit signal for UI synchronization
        self.function_added.emit(name, func)

**Key insight**: If any step fails, the entire operation is rolled back. No partial registration.

Module Injection
================

Custom functions executed via ``exec()`` don't have ``__module__`` set properly. Module injection fixes this:

.. code-block:: python

    # Without module injection
    exec(code, namespace)
    func = namespace['my_function']
    print(func.__module__)  # '__main__' or None (WRONG!)
    
    # With module injection
    exec(code, namespace)
    func = namespace['my_function']
    
    # Inject proper module name
    if not hasattr(func, '__module__') or func.__module__ in (None, '__main__'):
        func.__module__ = 'openhcs.processing.custom_functions'
    
    print(func.__module__)  # 'openhcs.processing.custom_functions' (CORRECT!)

**Why this matters**:

- Function registry uses ``__module__`` for organization
- Code generation needs proper module paths
- Serialization requires valid module names

Signal-Based Synchronization
=============================

The manager emits signals to keep UI and CLI synchronized:

.. code-block:: python

    class CustomFunctionManager(QObject):
        """Manager with Qt signals for UI synchronization."""
        
        # Signals
        function_added = pyqtSignal(str, object)    # (name, function)
        function_removed = pyqtSignal(str)          # (name,)
        function_updated = pyqtSignal(str, object)  # (name, function)
        
        def add_function(self, name, code):
            # ... validation and registration ...
            
            # Emit signal
            self.function_added.emit(name, func)
        
        def remove_function(self, name):
            # ... unregistration ...
            
            # Emit signal
            self.function_removed.emit(name)

UI components connect to these signals:

.. code-block:: python

    class CustomFunctionManagerDialog(QDialog):
        def __init__(self):
            self.manager = CustomFunctionManager()
            
            # Connect signals
            self.manager.function_added.connect(self._on_function_added)
            self.manager.function_removed.connect(self._on_function_removed)
        
        def _on_function_added(self, name, func):
            """Update function list when function added."""
            self.function_list.addItem(name)
        
        def _on_function_removed(self, name):
            """Update function list when function removed."""
            items = self.function_list.findItems(name, Qt.MatchExactly)
            for item in items:
                self.function_list.takeItem(self.function_list.row(item))

**Key insight**: Signals decouple manager from UI, enabling multiple UI components to stay synchronized.

Function Validation
===================

Custom functions must meet specific requirements:

Required Decorators
-------------------

.. code-block:: python

    from openhcs.core.memory import numpy, cupy
    
    # Valid: Has memory type decorator
    @numpy
    def valid_function(image: np.ndarray) -> np.ndarray:
        return image
    
    # Invalid: Missing decorator
    def invalid_function(image: np.ndarray) -> np.ndarray:
        return image  # ValidationError!

**Validation check**:

.. code-block:: python

    if not hasattr(func, 'input_memory_type'):
        raise ValidationError("Function must have memory type decorator")

Signature Requirements
----------------------

.. code-block:: python

    # Valid: First parameter is image array
    def valid_function(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        return gaussian_filter(image, sigma=sigma)
    
    # Invalid: First parameter is not image
    def invalid_function(sigma: float, image: np.ndarray) -> np.ndarray:
        return gaussian_filter(image, sigma=sigma)  # ValidationError!

**Validation check**:

.. code-block:: python

    sig = inspect.signature(func)
    first_param = list(sig.parameters.values())[0]
    if first_param.annotation not in (np.ndarray, 'np.ndarray'):
        raise ValidationError("First parameter must be image array")

Return Type Annotation
----------------------

.. code-block:: python

    # Valid: Return type annotated
    def valid_function(image: np.ndarray) -> np.ndarray:
        return image
    
    # Invalid: Missing return type
    def invalid_function(image: np.ndarray):
        return image  # ValidationError!

**Validation check**:

.. code-block:: python

    if sig.return_annotation == inspect.Signature.empty:
        raise ValidationError("Function must have return type annotation")

Persistence and Loading
=======================

Custom functions are persisted to disk for reuse across sessions:

.. code-block:: python

    # Save to disk
    custom_functions_dir = Path.home() / '.openhcs' / 'custom_functions'
    function_file = custom_functions_dir / f'{name}.py'
    
    with open(function_file, 'w') as f:
        f.write(code)
    
    # Load on startup
    manager = CustomFunctionManager()
    manager.load_all_functions()
    
    # All custom functions available in pipelines

**Storage location**: ``~/.openhcs/custom_functions/``

Integration with Function Registry
===================================

Custom functions integrate with the global function registry:

.. code-block:: python

    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.processing.func_registry import register_function
    
    # Register custom function
    register_function(func, backend='openhcs')
    
    # Use in step
    step = FunctionStep(func=func, name=func.__name__)

**Backend organization**: Custom functions use ``backend='openhcs'`` to distinguish from library functions.

Common Patterns
===============

Adding Function from GUI
-------------------------

.. code-block:: python

    # User clicks "Add Function" button
    dialog = CustomFunctionManagerDialog()
    
    # User writes code in editor
    code = '''
    from openhcs.core.memory import numpy

    @numpy
    def my_filter(image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return image > threshold
    '''
    
    # User clicks "Save"
    dialog.manager.add_function('my_filter', code)
    
    # Function immediately available in pipeline editor

Updating Existing Function
---------------------------

.. code-block:: python

    # Load existing function
    existing_code = manager.get_function_code('my_filter')
    
    # User edits code
    updated_code = existing_code.replace('threshold: float = 0.5', 'threshold: float = 0.3')
    
    # Save update (atomic replacement)
    manager.add_function('my_filter', updated_code)
    
    # All pipelines using 'my_filter' now use updated version

Removing Function
-----------------

.. code-block:: python

    # Remove function
    manager.remove_function('my_filter')
    
    # Function no longer available in pipelines
    # Existing pipelines using 'my_filter' will fail validation

Implementation Notes
====================

**🔬 Source Code**: 

- Manager: ``openhcs/processing/custom_functions/manager.py`` (line 117)
- Registry integration: ``openhcs/processing/func_registry.py`` (line 33)
- GUI dialog: ``openhcs/pyqt_gui/dialogs/custom_function_manager_dialog.py`` (line 1)

**🏗️ Architecture**: 

- :doc:`../architecture/function-registry-system` - Function registry architecture
- :doc:`code_ui_editing` - Bidirectional code/UI editing

**📊 Performance**: 

- Function registration is fast (< 10ms)
- Disk persistence is asynchronous (doesn't block UI)
- Signal emission overhead is negligible

Key Design Decisions
====================

**Why atomic updates?**

Prevents partial registration that could leave system in inconsistent state. Either function is fully registered or not at all.

**Why inject module name?**

Functions executed via ``exec()`` don't have proper ``__module__``. Injection ensures proper organization and serialization.

**Why use signals instead of callbacks?**

Signals decouple manager from UI, enabling multiple components to stay synchronized without tight coupling.

Common Gotchas
==============

- **Don't forget memory type decorator**: Functions without ``@numpy`` or similar will fail validation
- **First parameter must be image**: Function signature must start with image array parameter
- **Return type annotation required**: Functions without return type annotation will fail validation
- **Function names must be unique**: Adding function with existing name replaces the old function
- **Removed functions break pipelines**: Pipelines using removed functions will fail validation

Debugging Custom Functions
===========================

Symptom: Function Not Appearing in Pipeline Editor
---------------------------------------------------

**Cause**: Validation failed during registration

**Diagnosis**:

.. code-block:: python

    # Check validation errors
    try:
        manager.add_function(name, code)
    except ValidationError as e:
        logger.error(f"Validation failed: {e}")

**Fix**: Ensure function meets all validation requirements

Symptom: Function Execution Fails
----------------------------------

**Cause**: Missing imports in function code

**Diagnosis**:

.. code-block:: python

    # Function code missing imports
    @numpy
    def my_filter(image: np.ndarray) -> np.ndarray:
        return gaussian_filter(image, sigma=1.0)  # NameError: gaussian_filter not defined

**Fix**: Add imports to function code:

.. code-block:: python

    from scipy.ndimage import gaussian_filter
    from openhcs.core.memory import numpy
    
    @numpy
    def my_filter(image: np.ndarray) -> np.ndarray:
        return gaussian_filter(image, sigma=1.0)
