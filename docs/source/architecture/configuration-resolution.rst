Configuration Resolution System
===============================

Thread-local context system and hierarchical configuration resolution introduced in the UI refactor.

Overview
--------

Thread-local configuration context system for hierarchical value resolution. Enables lazy dataclasses to resolve values from different sources based on execution context without explicit parameter passing.

Thread-Local Context System
----------------------------

Generic thread-local storage for configuration contexts. Different operations require different resolution behavior - global editing shows defaults, pipeline editing shows resolved values.

Generic Storage
~~~~~~~~~~~~~~~

Works with any configuration type via type-keyed storage.

.. code-block:: python

    # Generic thread-local storage
    _global_config_contexts: Dict[Type, threading.local] = {}
    
    def set_current_global_config(config_type: Type, config_instance: Any) -> None:
        """Set current global config for any dataclass type."""
        if config_type not in _global_config_contexts:
            _global_config_contexts[config_type] = threading.local()
        _global_config_contexts[config_type].value = config_instance
    
    def get_current_global_config(config_type: Type) -> Optional[Any]:
        """Get current global config for any dataclass type."""
        context = _global_config_contexts.get(config_type)
        return getattr(context, 'value', None) if context else None

Usage Pattern
~~~~~~~~~~~~~

.. code-block:: python

    # Set thread-local context
    set_current_global_config(GlobalPipelineConfig, global_config)
    
    # Lazy dataclasses automatically resolve from thread-local context
    pipeline_config = PipelineConfig()  # Resolves from thread-local
    value = pipeline_config.some_field  # Thread-local resolution

Configuration Resolution Hierarchy
-----------------------------------

Resolution Authority
~~~~~~~~~~~~~~~~~~~~

Configuration resolution follows a clear hierarchy:

1. **User-set values** - Explicit user input (highest priority)
2. **Thread-local context** - Current global config instance  
3. **Static defaults** - Dataclass field defaults (lowest priority)

Context-Driven Behavior
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Global config editing: Use actual default values
    is_global_config_editing = not LazyDefaultPlaceholderService.has_lazy_resolution(dataclass_type)
    
    # Lazy config editing: Use None to show placeholder text
    if is_global_config_editing:
        return actual_default_value
    else:
        return None  # Shows "Pipeline default: value" placeholder

Field Path Resolution
---------------------

Explicit Path Navigation
~~~~~~~~~~~~~~~~~~~~~~~~

Eliminates algorithmic field name conversion bugs:

.. code-block:: python

    # Dot-separated paths for nested field access
    LazyStepMaterializationConfig = make_lazy_thread_local(
        StepMaterializationConfig,
        GlobalPipelineConfig,
        field_path="materialization_defaults"  # Explicit path
    )

Automatic Field Path Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Replaces hardcoded field paths
    field_path = FieldPathDetector.find_field_path_for_type(
        parent_type=GlobalPipelineConfig,
        child_type=StepMaterializationConfig
    )
    # Returns: "materialization_defaults"

UI Context Scoping Patterns
---------------------------

Three UI context types with different resolution behavior.

Context Types
~~~~~~~~~~~~~

**1. Global Config Editing**

Shows actual default values.

.. code-block:: python

    # Global config editor setup
    set_current_global_config(GlobalPipelineConfig, global_defaults)

    # Shows actual default values, no lazy resolution

**2. Pipeline Config Editing**

Shows orchestrator values with global fallback.

.. code-block:: python

    # Pipeline editor setup
    orchestrator_config = orchestrator.get_effective_config()
    set_current_global_config(GlobalPipelineConfig, orchestrator_config)

    # Lazy resolution from orchestrator config, None values use global defaults

**3. Step Config Editing (Scoped)**

Temporary context for step editing with orchestrator→global fallback.

.. code-block:: python

    @contextlib.contextmanager
    def _scoped_orchestrator_context(self):
        """Temporarily set orchestrator context for step editing."""
        original_config = get_current_global_config(GlobalPipelineConfig)
        orchestrator = self._get_current_orchestrator()
        if orchestrator:
            effective_config = orchestrator.get_effective_config()
            set_current_global_config(GlobalPipelineConfig, effective_config)

        try:
            yield  # Step editor operates in this context
        finally:
            set_current_global_config(GlobalPipelineConfig, original_config)

Context Manager Pattern
~~~~~~~~~~~~~~~~~~~~~~~

Ensures proper context isolation and cleanup.

.. code-block:: python

    # Step editor usage
    with self._scoped_orchestrator_context():
        step_form = ParameterFormManager(...)
        # LazyStepMaterializationConfig resolves from orchestrator context

    # Context automatically restored

Context managers provide isolation, cleanup, and safe nesting.

Resolution Hierarchy
~~~~~~~~~~~~~~~~~~~~

Step editor resolution chain:

.. code-block:: python

    # Step's LazyStepMaterializationConfig resolution chain:

    # 1. Check step's materialization_config (user-set values)
    step_value = step.materialization_config.backend
    if step_value is not None:
        return step_value

    # 2. Resolve from orchestrator's materialization_defaults (thread-local)
    orchestrator_config = get_current_global_config(GlobalPipelineConfig)
    orchestrator_value = orchestrator_config.materialization_defaults.backend
    if orchestrator_value is not None:
        return orchestrator_value

    # 3. Fall back to global defaults
    global_default = GlobalPipelineConfig().materialization_defaults.backend
    return global_default

Example: Global=DISK, Orchestrator=None, Step=None → Result: DISK

Thread-Local Requirement
~~~~~~~~~~~~~~~~~~~~~~~~~

Thread-local storage eliminates parameter threading. Without it, every function needs explicit context parameters:

.. code-block:: python

    # Without thread-local (parameter threading nightmare):
    def create_step_form(step, orchestrator_config, global_config, editing_mode):
        return ParameterFormManager(
            step.parameters, orchestrator_config, global_config, editing_mode
        )

    def create_parameter_widget(param, orchestrator_config, global_config, editing_mode):
        placeholder = get_placeholder(param, orchestrator_config, global_config, editing_mode)
        return widget

    def get_placeholder(param, orchestrator_config, global_config, editing_mode):
        # Every function needs all context parameters
        pass

With thread-local, context is ambient:

.. code-block:: python

    def create_step_form(step):
        return ParameterFormManager(step.parameters)  # Context is ambient

    def get_placeholder(param):
        config = get_current_global_config(GlobalPipelineConfig)
        return resolve_from_context(param, config)

Lazy dataclasses require thread-local for resolution:

.. code-block:: python

    class LazyStepMaterializationConfig:
        def _resolve_field_value(self, field_name: str) -> Any:
            # This is why thread-local is required:
            # Lazy resolution happens without explicit parameters
            global_config = get_current_global_config(GlobalPipelineConfig)
            if global_config and self._field_path:
                nested_value = self._navigate_field_path(global_config, self._field_path)
                if nested_value and hasattr(nested_value, field_name):
                    return getattr(nested_value, field_name)
            return self._get_static_default(field_name)

Benefits
--------

- **Thread Safety**: Each thread has isolated configuration context
- **Explicit Resolution**: Clear hierarchy eliminates ambiguous behavior
- **Type Safety**: Generic storage maintains type information
- **Fail-Loud**: Configuration errors surface immediately
- **Context Awareness**: Behavior adapts to editing context
- **Clean API**: No parameter threading through function calls
- **UI Operation Scoping**: Enables sophisticated context switching for different editing modes
- **Lazy Loading Foundation**: Provides the ambient context necessary for clean lazy resolution
- **Context Isolation**: Different operations can have different configuration contexts
- **Testability**: Easy to set up test contexts without complex mocking
