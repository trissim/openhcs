===========================
Lazy Dataclass Utilities
===========================

*Module: openhcs.introspection.lazy_dataclass_utils*  
*Status: STABLE*

---

Overview
========

Code editors use ``exec()`` to create dataclass instances from user-edited code. Without constructor patching, lazy dataclasses resolve None values to concrete defaults during construction, making it impossible to distinguish between explicitly set values and inherited values.

Problem Context
===============

Lazy dataclasses have two-phase construction:

.. code-block:: python

    # Phase 1: Construction (should preserve None)
    config = LazyProcessingConfig(group_by=None)
    
    # Phase 2: Resolution (should resolve None to default)
    resolved = config.resolve()  # group_by → GroupBy.WELL (from global config)

Without patching, Phase 1 incorrectly resolves None to defaults:

.. code-block:: python

    # Without patching
    config = LazyProcessingConfig(group_by=None)
    print(config.group_by)  # GroupBy.WELL (WRONG! Should be None)
    
    # With patching
    config = LazyProcessingConfig(group_by=None)
    print(config.group_by)  # None (CORRECT!)

This breaks code editors that need to preserve None vs concrete distinction.

Solution: Constructor Patching
===============================

The ``patch_lazy_constructors`` context manager temporarily patches lazy dataclass constructors to preserve None values:

.. code-block:: python

    from openhcs.introspection.lazy_dataclass_utils import patch_lazy_constructors
    
    # Code editor execution
    with patch_lazy_constructors():
        # User code executed via exec()
        config = LazyProcessingConfig(
            group_by=None,           # Preserved as None
            variable_components=None # Preserved as None
        )
    
    # Outside context: normal lazy resolution
    config = LazyProcessingConfig(group_by=None)
    # group_by resolves to default

The patched constructor only sets fields explicitly provided in kwargs, leaving unprovided fields as None.

Discovery and Patching
======================

Automatic Type Discovery
------------------------

The system automatically discovers all lazy dataclass types without hardcoding:

.. code-block:: python

    from openhcs.introspection.lazy_dataclass_utils import discover_lazy_dataclass_types
    
    # Discover all lazy types from openhcs.core.config
    lazy_types = discover_lazy_dataclass_types()
    # Returns: [LazyProcessingConfig, LazyDtypeConfig, LazyNapariStreamingConfig, ...]

**Discovery logic**:

1. Inspect ``openhcs.core.config`` module
2. Find all classes with ``has_lazy_resolution()`` method
3. Return list of lazy dataclass types

This eliminates hardcoded type lists that become stale as new lazy types are added.

Constructor Patching Mechanism
------------------------------

.. code-block:: python

    @contextmanager
    def patch_lazy_constructors():
        """Patch lazy dataclass constructors to preserve None values."""
        
        # Discover all lazy types
        lazy_types = discover_lazy_dataclass_types()
        
        # Save original constructors
        original_inits = {}
        for lazy_type in lazy_types:
            original_inits[lazy_type] = lazy_type.__init__
        
        # Patch constructors
        for lazy_type in lazy_types:
            lazy_type.__init__ = _create_patched_init(lazy_type, original_inits[lazy_type])
        
        try:
            yield  # Code editor executes here
        finally:
            # Restore original constructors
            for lazy_type in lazy_types:
                lazy_type.__init__ = original_inits[lazy_type]

**Key insight**: Patching is temporary and scoped to code editor execution. Normal lazy resolution behavior is preserved outside the context.

Patched Constructor Behavior
-----------------------------

.. code-block:: python

    def _create_patched_init(lazy_type, original_init):
        """Create patched __init__ that preserves None values."""
        
        def patched_init(self, **kwargs):
            # Only set fields explicitly provided in kwargs
            for field_name, field_value in kwargs.items():
                object.__setattr__(self, field_name, field_value)
            
            # Leave unprovided fields as None (don't resolve to defaults)
        
        return patched_init

**Difference from original**:

- **Original**: Resolves None to defaults during construction
- **Patched**: Preserves None, only sets explicitly provided fields

Integration with Code Editors
==============================

Code Editor Form Updater (SIMPLIFIED 2024)
-------------------------------------------

The code editor form updater uses patched constructors to preserve None values. **Raw field values (via ``object.__getattribute__``) are the source of truth**:

.. code-block:: python

    from openhcs.introspection.lazy_dataclass_utils import patch_lazy_constructors

    class CodeEditorFormUpdater:
        def execute_code_and_update_form(self, code_text):
            """Execute user code and update form with results."""

            # Execute with patched constructors
            with patch_lazy_constructors():
                namespace = {}
                exec(code_text, namespace)

                # Extract dataclass instance
                config_instance = namespace.get('config')

            # Update ALL fields - form manager inspects raw values automatically
            self.form_manager.update_from_object(config_instance)

**Key Insight**: No need to track which fields were explicitly set or parse code with regex. The patched constructor already preserves the None vs concrete distinction in raw field values:

- **Raw None** (via ``object.__getattribute__``): Field inherits from parent config (show placeholder)
- **Raw concrete value**: Field explicitly set (show actual value)

This is the same pattern used in the uneval-based code serializer.

Shared Constructor Patching
----------------------------

All code editors share the same patching mechanism:

.. code-block:: python

    # Step editor
    with patch_lazy_constructors():
        exec(step_code, namespace)
    
    # Pipeline editor
    with patch_lazy_constructors():
        exec(pipeline_code, namespace)
    
    # Config window
    with patch_lazy_constructors():
        exec(config_code, namespace)

**Centralized location**: ``openhcs/introspection/lazy_dataclass_utils.py`` (line 1)

This eliminates duplicate patching logic across editors.

Common Patterns
===============

Code Editor Execution
---------------------

.. code-block:: python

    from openhcs.introspection.lazy_dataclass_utils import patch_lazy_constructors
    
    def execute_user_code(code_text):
        """Execute user code with lazy constructor patching."""
        
        with patch_lazy_constructors():
            namespace = {}
            exec(code_text, namespace)
            return namespace

Type Discovery for Validation
------------------------------

.. code-block:: python

    from openhcs.introspection.lazy_dataclass_utils import discover_lazy_dataclass_types
    
    def validate_config_type(config_instance):
        """Validate that config is a known lazy type."""
        
        lazy_types = discover_lazy_dataclass_types()
        if type(config_instance) not in lazy_types:
            raise TypeError(f"Unknown lazy type: {type(config_instance)}")

Form Manager Integration
------------------------

.. code-block:: python

    # Code editor creates instance with patched constructors
    with patch_lazy_constructors():
        config = LazyProcessingConfig(group_by=None)

    # Form manager inspects raw field values using object.__getattribute__
    # Raw None → shows placeholder, Raw concrete → shows actual value
    form_manager.update_from_object(config)
    # group_by field shows: "Pipeline default: GroupBy.WELL"

    # No need to track _explicitly_set_fields or parse code with regex
    # The raw field values ARE the source of truth

Implementation Notes
====================

**🔬 Source Code**: 

- Discovery: ``openhcs/introspection/lazy_dataclass_utils.py`` (line 17)
- Patching: ``openhcs/introspection/lazy_dataclass_utils.py`` (line 40)
- Code editor integration: ``openhcs/ui/shared/code_editor_form_updater.py`` (line 189)

**🏗️ Architecture**: 

- :doc:`../architecture/configuration-management-system` - Lazy dataclass system
- :doc:`../user_guide/code_ui_editing` - Bidirectional code/UI editing

**📊 Performance**: 

- Discovery is cached (runs once per code editor session)
- Patching overhead is negligible (simple function replacement)
- Context manager ensures cleanup even on exceptions

Key Design Decisions
====================

**Why use context manager instead of permanent patching?**

Lazy dataclasses need normal resolution behavior outside code editors. Permanent patching would break lazy resolution everywhere.

**Why discover types instead of hardcoding?**

New lazy types are added frequently. Discovery eliminates maintenance burden and prevents stale type lists.

**Why patch __init__ instead of using factory functions?**

Code editors use ``exec()`` which calls constructors directly. Factory functions would require changing user code patterns.

Common Gotchas
==============

- **Don't use patched constructors outside code editors**: Normal code should use standard lazy resolution
- **Discovery only finds types in openhcs.core.config**: Lazy types in other modules won't be discovered
- **Patching is not thread-safe**: Don't use in multi-threaded code editor contexts
- **Context manager must complete**: Exceptions during exec() will restore original constructors (cleanup guaranteed)

Debugging Patching Issues
==========================

Symptom: None Values Resolving to Defaults
-------------------------------------------

**Cause**: Code executing outside patched context

**Diagnosis**:

.. code-block:: python

    # Check if patching is active
    from openhcs.introspection.lazy_dataclass_utils import discover_lazy_dataclass_types
    
    lazy_types = discover_lazy_dataclass_types()
    for lazy_type in lazy_types:
        logger.debug(f"{lazy_type.__name__}.__init__: {lazy_type.__init__}")
        # Should show patched_init during code execution

**Fix**: Ensure code executes within ``with patch_lazy_constructors():`` block

Symptom: Type Not Discovered
-----------------------------

**Cause**: Lazy type not in ``openhcs.core.config`` module

**Diagnosis**:

.. code-block:: python

    lazy_types = discover_lazy_dataclass_types()
    logger.debug(f"Discovered types: {[t.__name__ for t in lazy_types]}")

**Fix**: Move lazy type to ``openhcs.core.config`` or extend discovery to other modules

Advanced Usage
==============

Custom Discovery Scope
----------------------

.. code-block:: python

    def discover_lazy_types_in_module(module):
        """Discover lazy types in custom module."""
        from openhcs.config_framework.placeholder import LazyDefaultPlaceholderService
        
        lazy_types = []
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and LazyDefaultPlaceholderService.has_lazy_resolution(obj):
                lazy_types.append(obj)
        
        return lazy_types

Selective Patching
------------------

.. code-block:: python

    @contextmanager
    def patch_specific_types(types_to_patch):
        """Patch only specific lazy types."""
        
        original_inits = {t: t.__init__ for t in types_to_patch}
        
        for lazy_type in types_to_patch:
            lazy_type.__init__ = _create_patched_init(lazy_type, original_inits[lazy_type])
        
        try:
            yield
        finally:
            for lazy_type in types_to_patch:
                lazy_type.__init__ = original_inits[lazy_type]
