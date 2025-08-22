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

Advanced Inheritance Patterns
-----------------------------

The UI refactor introduced sophisticated inheritance mechanisms that enable complex configuration scenarios while maintaining simplicity for consuming code.

Multi-Level Resolution Chains
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenHCS implements a hierarchical resolution system where configuration values flow through multiple levels:

**Resolution Hierarchy:**

1. **Step Level**: Individual step configuration (highest priority for user edits)
2. **Orchestrator Level**: Pipeline-specific configuration
3. **Global Level**: Application-wide defaults (lowest priority)

.. code-block:: python

    # Example: output_dir_suffix resolution chain
    # 1. Step level (None) → 2. Pipeline level ("_custom") → 3. Global level ("_openhcs")

    step_config = LazyStepMaterializationConfig()  # All None values
    step_config.output_dir_suffix  # Resolves to "_custom" from pipeline level

**Real-World Resolution Example:**

.. code-block:: python

    # Global configuration (application defaults)
    global_config = GlobalPipelineConfig(
        path_planning=PathPlanningConfig(output_dir_suffix="_openhcs"),
        materialization_defaults=StepMaterializationConfig(output_dir_suffix="_openhcs")
    )

    # Pipeline configuration (user overrides)
    pipeline_config = PipelineConfig(
        path_planning=LazyPathPlanningConfig(output_dir_suffix="_pipeline_custom"),
        materialization_defaults=LazyStepMaterializationConfig()  # None values
    )

    # Step configuration (inherits from pipeline)
    step_config = LazyStepMaterializationConfig()

    # Resolution chain:
    # step_config.output_dir_suffix (None)
    # → pipeline.materialization_defaults.output_dir_suffix (None)
    # → pipeline.path_planning.output_dir_suffix ("_pipeline_custom") ✅

Sibling Inheritance Mechanisms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One of the most sophisticated features is **sibling inheritance** - where fields can inherit from related configurations at the same hierarchy level.

**Sibling Inheritance Pattern:**

.. code-block:: python

    # StepMaterializationConfig inherits shared fields from PathPlanningConfig
    # when those fields are None in the materialization config

    class StepMaterializationConfig:
        output_dir_suffix: Optional[str] = None  # Can inherit from PathPlanningConfig
        sub_dir: Optional[str] = None            # Own field, no inheritance

    class PathPlanningConfig:
        output_dir_suffix: Optional[str] = "_openhcs"  # Shared field

**How Sibling Inheritance Works:**

1. **Field Classification**: Fields are classified as "inherited" (shared with siblings) or "own" (unique to this config)
2. **Hierarchy Building**: Resolution paths include both direct paths and sibling paths
3. **Context-Aware Resolution**: Uses current context (pipeline config) and global context separately

.. code-block:: python

    # Hierarchy paths for StepMaterializationConfig.output_dir_suffix:
    hierarchy_paths = [
        ('current', 'materialization_defaults'),  # Direct path
        ('current', 'path_planning'),             # Sibling inheritance ✅
        ('global', 'materialization_defaults'),   # Global direct
        ('global', 'path_planning')               # Global sibling
    ]

**Sibling Inheritance Example:**

.. code-block:: python

    # User sets path_planning.output_dir_suffix = "_custom"
    # materialization_defaults.output_dir_suffix = None (inherits from sibling)

    pipeline_config = PipelineConfig(
        path_planning=LazyPathPlanningConfig(output_dir_suffix="_custom"),
        materialization_defaults=LazyStepMaterializationConfig()  # None values
    )

    # Sibling inheritance in action:
    value = pipeline_config.materialization_defaults.output_dir_suffix
    # Result: "_custom" (inherited from sibling path_planning)

Context-Aware Resolution Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The system uses **context providers** to enable sophisticated resolution scenarios where the resolution context can be different from the global thread-local context.

**Context Provider Pattern:**

.. code-block:: python

    def create_context_aware_lazy_class(base_class, parent_instance):
        """Create lazy class that resolves from specific parent instance."""

        def context_provider():
            return parent_instance  # Use specific instance, not global context

        return LazyDataclassFactory.make_lazy_with_field_level_auto_hierarchy(
            base_class=base_class,
            global_config_type=GlobalPipelineConfig,
            field_path="materialization_defaults",
            context_provider=context_provider  # Custom context
        )

**Context Propagation in Nested Resolution:**

.. code-block:: python

    # Parent instance provides context for nested lazy classes
    def nested_context_provider():
        if parent_instance_provider:
            parent_instance = parent_instance_provider()
            if parent_instance:
                return parent_instance  # Use parent's context

        # Fall back to global config
        return get_current_global_config(global_config_type)

This enables scenarios where nested configurations resolve from their immediate parent rather than the global thread-local context, crucial for step editor functionality.

Preservation of User Edits
---------------------------

One of the most critical aspects of the lazy class system is preserving user edits while maintaining lazy resolution capabilities. This was a major source of bugs before the UI refactor.

Structure Preservation System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The system uses a three-method preservation approach to handle the complex interaction between user edits and lazy resolution:

.. code-block:: python

    def _preserve_lazy_structure_if_needed(self, field_name: str, value: Any) -> Any:
        """Core preservation logic - maintains user intent vs lazy resolution."""

        # Mark as user-set to prevent lazy resolution override
        object.__setattr__(self, f'_{field_name}_user_set', True)
        object.__setattr__(self, f'_{field_name}', value)

        # Handle nested dataclass preservation
        if dataclasses.is_dataclass(value):
            return self._rebuild_nested_dataclass_instance(value, field_name)

        return value

    def _convert_to_lazy_dataclass(self, value: Any, field_type: Type) -> Any:
        """Safe conversion to lazy dataclass when needed."""
        if LazyDefaultPlaceholderService.has_lazy_resolution(field_type):
            # Already a lazy dataclass - preserve as-is
            return value
        else:
            # Convert to lazy version for proper inheritance
            return self._create_lazy_version(value, field_type)

    def _rebuild_nested_dataclass_instance(self, nested_values: Dict[str, Any],
                                         nested_type: Type, param_name: str) -> Any:
        """Recursive reconstruction of nested dataclass instances."""
        nested_type_is_lazy = LazyDefaultPlaceholderService.has_lazy_resolution(nested_type)

        if nested_type_is_lazy:
            # Lazy dataclass: preserve None values for lazy resolution
            # This maintains "lazy mixed" pattern - some concrete, some None
            return nested_type(**nested_values)
        else:
            # Non-lazy dataclass: filter out None values
            filtered_values = {k: v for k, v in nested_values.items() if v is not None}
            return nested_type(**filtered_values) if filtered_values else nested_type()

Mixed State Management
~~~~~~~~~~~~~~~~~~~~~~

A key innovation is **mixed state management** - the ability for a single dataclass instance to have some fields with concrete user values and other fields with None values that resolve lazily.

.. code-block:: python

    # Example: Mixed state in StepMaterializationConfig
    step_config = LazyStepMaterializationConfig(
        output_dir_suffix="_user_custom",  # Concrete user value
        sub_dir=None,                      # Lazy resolution from hierarchy
        force_disk_output=True             # Concrete user value
    )

    # Field access behavior:
    step_config.output_dir_suffix  # Returns "_user_custom" (user-set)
    step_config.sub_dir           # Resolves from pipeline → global hierarchy
    step_config.force_disk_output # Returns True (user-set)

**Why Mixed State Matters:**

1. **User Intent Preservation**: User edits are never lost, even when other fields change
2. **Selective Inheritance**: Users can override specific fields while inheriting others
3. **Context Sensitivity**: Same instance behaves differently based on resolution context

Recursive Reconstruction
~~~~~~~~~~~~~~~~~~~~~~~~

When nested dataclasses are modified, the system recursively rebuilds the entire structure while preserving user edits at every level:

.. code-block:: python

    def rebuild_lazy_config_with_new_global_reference(
        current_config: Any,
        new_global_config: Any,
        global_config_type: Type
    ) -> Any:
        """Rebuild entire config hierarchy with new global reference."""

        current_field_values = {}

        for field_obj in fields(type(current_config)):
            raw_value = _get_raw_field_value(current_config, field_obj.name)

            if raw_value is not None and hasattr(raw_value, '__dataclass_fields__'):
                # Nested dataclass - recursively rebuild
                rebuilt_nested_value = rebuild_lazy_config_with_new_global_reference(
                    raw_value, new_global_config, global_config_type
                )
                current_field_values[field_obj.name] = rebuilt_nested_value
            else:
                # Regular field - preserve as-is
                current_field_values[field_obj.name] = raw_value

        return type(current_config)(**current_field_values)

This ensures that when global configuration changes, all existing lazy instances are updated while preserving their user-set values.

Real-World Inheritance Examples
-------------------------------

These examples demonstrate the complex inheritance scenarios that the lazy class system handles in practice.

Example 1: Step Editor Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Scenario**: User editing a step's materialization configuration in the step editor.

.. code-block:: python

    # Global configuration (application defaults)
    global_config = GlobalPipelineConfig(
        path_planning=PathPlanningConfig(output_dir_suffix="_openhcs"),
        materialization_defaults=StepMaterializationConfig(
            output_dir_suffix="_openhcs",
            sub_dir="processed",
            force_disk_output=False
        )
    )

    # Pipeline configuration (user customizations)
    pipeline_config = PipelineConfig(
        path_planning=LazyPathPlanningConfig(output_dir_suffix="_pipeline_custom"),
        materialization_defaults=LazyStepMaterializationConfig()  # All None - inherits
    )

    # Step configuration (step-specific overrides)
    step_config = LazyStepMaterializationConfig(
        sub_dir="_step_specific"  # User override for this step only
    )

    # Resolution results:
    step_config.output_dir_suffix  # "_pipeline_custom" (from pipeline path_planning)
    step_config.sub_dir           # "_step_specific" (user override)
    step_config.force_disk_output # False (from global defaults)

**Resolution Chain Analysis:**

1. ``output_dir_suffix``: None (step) → None (pipeline materialization) → "_pipeline_custom" (pipeline path_planning) ✅
2. ``sub_dir``: "_step_specific" (step user override) ✅
3. ``force_disk_output``: None (step) → None (pipeline materialization) → False (global materialization) ✅

Example 2: Complex Sibling Inheritance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Scenario**: Multiple configuration types sharing fields with different inheritance patterns.

.. code-block:: python

    # User sets path planning configuration
    pipeline_config = PipelineConfig(
        path_planning=LazyPathPlanningConfig(
            output_dir_suffix="_user_custom",
            input_dir_prefix="raw_",
            temp_dir_suffix="_temp"
        ),
        materialization_defaults=LazyStepMaterializationConfig(),  # Inherits from path_planning
        vfs=LazyVFSConfig()  # Also inherits shared fields
    )

    # Sibling inheritance results:
    # StepMaterializationConfig inherits output_dir_suffix from PathPlanningConfig
    pipeline_config.materialization_defaults.output_dir_suffix  # "_user_custom"

    # VFSConfig inherits different fields from PathPlanningConfig
    pipeline_config.vfs.temp_dir_suffix  # "_temp"

    # Non-shared fields resolve independently
    pipeline_config.materialization_defaults.sub_dir  # None → resolves from global
    pipeline_config.vfs.backend_type  # None → resolves from global

Example 3: Context-Aware Step Editor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Scenario**: Step editor showing placeholders that reflect the actual inheritance chain.

.. code-block:: python

    # Set up pipeline context
    set_current_global_config(GlobalPipelineConfig, global_config)

    # Create step editor with context-aware lazy config
    def create_step_editor_config(pipeline_config):
        """Create step config that resolves from pipeline context."""

        def context_provider():
            return pipeline_config  # Use pipeline as resolution context

        return LazyDataclassFactory.make_lazy_with_field_level_auto_hierarchy(
            base_class=StepMaterializationConfig,
            global_config_type=GlobalPipelineConfig,
            field_path="materialization_defaults",
            context_provider=context_provider
        )

    # Step editor configuration
    StepEditorConfig = create_step_editor_config(pipeline_config)
    step_editor_config = StepEditorConfig()

    # UI placeholder text generation:
    # "Pipeline default: _user_custom" (shows actual pipeline value)
    placeholder_text = LazyDefaultPlaceholderService.get_lazy_resolved_placeholder(
        StepEditorConfig, "output_dir_suffix", placeholder_prefix="Pipeline default"
    )

Example 4: Mixed State Preservation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Scenario**: User makes partial edits to a configuration, system preserves user intent.

.. code-block:: python

    # Initial state: all lazy resolution
    config = LazyStepMaterializationConfig()

    # User edits one field
    config = config.replace(output_dir_suffix="_user_override")

    # System state after edit:
    # - output_dir_suffix: "_user_override" (concrete user value)
    # - sub_dir: None (still lazy, resolves from hierarchy)
    # - force_disk_output: None (still lazy, resolves from hierarchy)

    # Global config changes
    new_global = GlobalPipelineConfig(
        materialization_defaults=StepMaterializationConfig(
            output_dir_suffix="_new_global",
            sub_dir="updated",
            force_disk_output=True
        )
    )

    # After global config update:
    config.output_dir_suffix  # "_user_override" (preserved user edit)
    config.sub_dir           # "updated" (new global value)
    config.force_disk_output # True (new global value)

Example 5: Compiler Context Resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Scenario**: Pipeline compilation with different resolution context than UI.

.. code-block:: python

    # UI context: Uses pipeline config with None values (enables sibling inheritance)
    ui_context = pipeline_config  # Has None values in materialization_defaults

    # Compiler context: Uses resolved effective config
    effective_config = pipeline_config.to_base_config()  # All values resolved

    # Different resolution results:

    # UI resolution (correct):
    with LazyConfigContext(ui_context):
        step_config = LazyStepMaterializationConfig()
        ui_value = step_config.output_dir_suffix  # "_pipeline_custom" (from path_planning)

    # Compiler resolution (was broken, now fixed):
    with LazyConfigContext(ui_context):  # Use unresolved context, not effective
        step_config = LazyStepMaterializationConfig()
        compiler_value = step_config.output_dir_suffix  # "_pipeline_custom" (same as UI)

This example shows how the context-aware resolution system ensures consistency between UI and compilation phases.

Benefits
--------

- **Lazy Resolution**: Values computed only when accessed
- **Context Awareness**: Automatic thread-local context integration
- **Structure Preservation**: User edits preserved across operations
- **Type Safety**: Generated classes maintain original type contracts
- **Recursive Support**: Automatic nested lazy dataclass creation
- **Multi-Level Hierarchy**: Step → Pipeline → Global resolution chains
- **Sibling Inheritance**: Cross-configuration field inheritance
- **Context Propagation**: Flexible resolution context management

See Also
--------

- :doc:`configuration-resolution` - Thread-local context management and resolution patterns
- :doc:`step-editor-generalization` - How step editors use lazy dataclass patterns
- :doc:`field-path-detection` - Automatic field path discovery for lazy config generation
- :doc:`service-layer-architecture` - Framework-agnostic business logic that works with lazy configs
- :doc:`../development/ui-patterns` - UI patterns that leverage lazy dataclass systems
