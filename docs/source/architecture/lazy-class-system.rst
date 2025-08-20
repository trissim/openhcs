Lazy Class System
=================

Dynamic dataclass generation with lazy field resolution and thread-local context integration.

Overview
--------

The lazy class system solves a fundamental problem in configuration management: how do you create objects that can get their values from different sources depending on context, while still preserving user edits?

**The Challenge:** OpenHCS has complex configuration hierarchies. A pipeline step might get its configuration from global defaults, pipeline-specific overrides, or user edits. Traditional dataclasses are static - once created, their values don't change. But we needed objects that could dynamically resolve values based on the current context.

**The Solution:** Lazy dataclasses are generated at runtime with custom resolution logic. They look like normal dataclasses to consuming code, but behind the scenes they can resolve field values from thread-local contexts, preserve user edits, and maintain proper hierarchy relationships.

**Real-World Impact:** This system eliminated the "lost edits" bug where user changes would revert to defaults, and made it possible to have context-aware placeholder text throughout the UI.

LazyDataclassFactory
--------------------

The LazyDataclassFactory is the heart of the system. It takes a regular dataclass and generates a new class that has the same interface but with lazy resolution behavior.

**Why a Factory?** We can't modify existing dataclasses (they might be from external libraries), and we need different resolution strategies for different use cases. The factory pattern lets us generate customized lazy classes on demand.

Core Factory Methods
~~~~~~~~~~~~~~~~~~~~

The factory provides two main creation methods, each optimized for different use cases:

.. code-block:: python

    # Thread-local lazy dataclass
    PipelineConfig = LazyDataclassFactory.make_lazy_thread_local(
        base_class=GlobalPipelineConfig,
        global_config_type=GlobalPipelineConfig,
        field_path=None,  # Root instance
        use_recursive_resolution=True
    )
    
    # Nested field lazy dataclass  
    LazyStepConfig = LazyDataclassFactory.make_lazy_thread_local(
        base_class=StepMaterializationConfig,
        global_config_type=GlobalPipelineConfig,
        field_path="materialization_defaults"
    )

Dynamic Class Generation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Creates dataclass with introspected fields
    lazy_class = make_dataclass(
        lazy_class_name,
        LazyDataclassFactory._introspect_dataclass_fields(
            base_class, debug_template, global_config_type, parent_field_path
        ),
        frozen=True
    )
    
    # Bind resolution methods declaratively
    LazyDataclassFactory._bind_methods_to_class(lazy_class, base_class, resolution_config)

Resolution Mechanisms
---------------------

Field Value Resolution
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def _resolve_field_value(self, field_name: str) -> Any:
        """Resolve field value from hierarchy: user → thread-local → defaults."""
        
        # 1. Check for user-set value (highest priority)
        if hasattr(self, f'_{field_name}_user_set'):
            return getattr(self, f'_{field_name}')
        
        # 2. Resolve from thread-local context
        global_config = get_current_global_config(self._global_config_type)
        if global_config and self._field_path:
            nested_value = self._navigate_field_path(global_config, self._field_path)
            if nested_value and hasattr(nested_value, field_name):
                return getattr(nested_value, field_name)
        
        # 3. Fall back to static defaults
        return self._get_static_default(field_name)

Recursive Resolution
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Automatic nested lazy dataclass creation
    if self._use_recursive_resolution and dataclasses.is_dataclass(field_type):
        nested_field_path = f"{self._field_path}.{field_name}" if self._field_path else field_name
        
        nested_lazy_class = LazyDataclassFactory.make_lazy_thread_local(
            base_class=field_type,
            global_config_type=self._global_config_type,
            field_path=nested_field_path
        )
        return nested_lazy_class()

Structure Preservation
----------------------

User Value Tracking
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def _preserve_lazy_structure_if_needed(self, field_name: str, value: Any) -> Any:
        """Preserve user-set values vs None values in lazy structures."""
        
        # Mark as user-set to prevent lazy resolution override
        object.__setattr__(self, f'_{field_name}_user_set', True)
        object.__setattr__(self, f'_{field_name}', value)
        
        # Handle nested dataclass preservation
        if dataclasses.is_dataclass(value):
            return self._rebuild_nested_dataclass_instance(value, field_name)
        
        return value

Safe Instance Handling
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def _get_raw_field_value(self, field_name: str) -> Any:
        """Get raw field value without triggering lazy resolution."""
        # Prevents infinite recursion during introspection
        return object.__getattribute__(self, f'_{field_name}')

Lifecycle Management
--------------------

Instantiation Pattern
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Lazy dataclass instantiation
    config = PipelineConfig()  # Creates instance with lazy resolution
    
    # Field access triggers resolution
    value = config.some_field  # Resolves from thread-local → defaults
    
    # User assignment preserves values
    config = config.replace(some_field=user_value)  # Marks as user-set

Thread-Local Integration
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Set thread-local context
    set_current_global_config(GlobalPipelineConfig, global_config)
    
    # All lazy instances in this thread resolve from global_config
    config1 = PipelineConfig()
    config2 = LazyStepConfig()
    
    # Both resolve from the same thread-local context

Benefits
--------

- **Lazy Resolution**: Values computed only when accessed
- **Context Awareness**: Automatic thread-local context integration
- **Structure Preservation**: User edits preserved across operations
- **Type Safety**: Generated classes maintain original type contracts
- **Recursive Support**: Automatic nested lazy dataclass creation
