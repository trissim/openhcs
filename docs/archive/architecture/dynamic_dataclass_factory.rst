Dynamic Dataclass Factory System
================================

**Runtime dataclass generation with contextvars-based lazy resolution.**

*Status: STABLE*
*Module: openhcs.config_framework.lazy_factory*

The Problem: Fixed Dataclass Behavior
--------------------------------------

Traditional dataclasses have fixed behavior at definition time: fields always return stored values. But lazy configuration requires runtime behavior customization based on context. For example, a step configuration field might need to return the global default when not explicitly set, but return the explicit value when the user has configured it. Without dynamic behavior, you need separate dataclass types for each context level, leading to code duplication and maintenance overhead.

The Solution: Runtime Dataclass Generation with Context-Aware Resolution
--------------------------------------------------------------------------

The dynamic factory system generates dataclasses with custom resolution methods that use Python's contextvars to look up values from the current configuration context. This enables the same dataclass type to behave differently depending on which configuration context is active, eliminating the need for separate types.

Overview
--------
Traditional dataclasses have fixed behavior at definition time, but lazy configuration requires runtime behavior customization based on context. The dynamic factory system generates dataclasses with custom resolution methods that use Python's contextvars to look up values from the current configuration context.

:py:meth:`~openhcs.config_framework.lazy_factory.LazyDataclassFactory.make_lazy_simple` creates a lazy dataclass from a regular dataclass. When you access a field on a lazy dataclass instance, instead of returning a stored value, it triggers resolution logic that looks up the value from the current context using :py:func:`~openhcs.config_framework.context_manager.config_context`. This enables the same dataclass interface with different resolution behavior for different contexts - step editors resolve against pipeline config, pipeline configs resolve against global config, and global configs use static defaults.

LazyDataclassFactory Architecture
---------------------------------
The factory uses a simplified creation pattern focused on contextvars-based resolution.

Core Factory Method
~~~~~~~~~~~~~~~~~~
:py:meth:`~openhcs.config_framework.lazy_factory.LazyDataclassFactory.make_lazy_simple` is the primary public API. It takes a regular dataclass and generates a new lazy class with the same fields and interface. The generated class uses :py:func:`~dataclasses.make_dataclass` to create a new class that inherits from the base dataclass, then attaches custom methods that implement lazy resolution behavior using contextvars.

Method Binding System
~~~~~~~~~~~~~~~~~~~~
:py:class:`~openhcs.config_framework.lazy_factory.LazyMethodBindings` acts like a method factory. It creates the actual functions that get attached to generated classes:

- ``__getattribute__()`` - Intercepts field access and triggers resolution
- ``_resolve_field_value()`` - Looks up field values from current context
- ``to_base_config()`` - Converts lazy instances back to concrete values

These methods are created as closures that capture the resolution logic, then attached to the class using :py:func:`setattr`.

Contextvars-Based Resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The factory uses Python's :py:mod:`contextvars` module for context management. When a field is accessed on a lazy dataclass, the ``__getattribute__()`` method calls :py:func:`~openhcs.config_framework.context_manager.current_temp_global.get()` to retrieve the current merged configuration context. The :py:func:`~openhcs.config_framework.dual_axis_resolver.resolve_field_inheritance` function then searches this context for the field value using a two-axis resolution strategy.

Recursive Lazy Dataclass Creation
---------------------------------
The factory automatically creates lazy versions of nested dataclasses.

:py:meth:`~openhcs.config_framework.lazy_factory.LazyDataclassFactory._introspect_dataclass_fields` examines each field of the base dataclass. When it finds a field whose type is itself a dataclass, it recursively calls :py:meth:`~openhcs.config_framework.lazy_factory.LazyDataclassFactory.make_lazy_simple` to create a lazy version of that nested type. This creates a tree of lazy dataclasses where each level can have its own resolution behavior while maintaining the original nested structure.

For example, if ``GlobalPipelineConfig`` has a field ``well_filter_config: WellFilterConfig``, the factory automatically creates a lazy version of ``WellFilterConfig`` and uses that as the field type in the lazy ``GlobalPipelineConfig``. When you access ``lazy_global_config.well_filter_config``, you get a lazy instance that resolves its fields from the current context.

Type Registry Integration
------------------------
Generated classes are automatically registered for type mapping.

:py:func:`~openhcs.config_framework.lazy_factory.register_lazy_type_mapping` maintains a bidirectional mapping between lazy classes and their base classes. This allows the system to recognize that ``LazyPipelineConfig`` instances should be treated as ``PipelineConfig`` for type checking purposes, and enables conversion functions to automatically find the right base type when serializing lazy configs back to concrete values.

The registry is populated automatically when :py:meth:`~openhcs.config_framework.lazy_factory.LazyDataclassFactory.make_lazy_simple` creates a new lazy class. You can retrieve the base type using :py:func:`~openhcs.config_framework.lazy_factory.get_base_type_for_lazy`.

Dataclass Reconstruction (Serialization)
--------------------------------------

OpenHCS uses the ObjectState lazy framework for config/lazy dataclass resolution and serialization.
When resolved lazy dataclasses are converted back into concrete dataclasses (e.g. for pickling),
ObjectState reconstructs objects from a resolved ``{field_name: value}`` mapping.

If a dataclass uses a custom constructor (common with ``@dataclass(init=False)``), reconstruction
via ``Type(**fields)`` may fail. In that case, the dataclass should provide a rebuild hook:

.. code-block:: python

    @classmethod
    def __objectstate_rebuild__(cls, **fields):
        return cls(*fields["outputs"], primary=fields.get("primary", 0))

This keeps custom constructors clean while still supporting lazy resolution and serialization.

Dual-Axis Resolution Strategy
-----------------------------
The factory uses a two-axis resolution strategy to find field values.

X-Axis: Context Hierarchy
~~~~~~~~~~~~~~~~~~~~~~~~~
The X-axis searches up the configuration hierarchy. When resolving a field, the system first checks the current context (from contextvars), then checks parent contexts if available. For example, when editing a step config, it first checks the step config's values, then the pipeline config's values, then the global config's values.

Y-Axis: Sibling Inheritance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The Y-axis searches across related configuration types at the same level. The :py:func:`~openhcs.config_framework.dual_axis_resolver.resolve_field_inheritance` function uses the MRO (Method Resolution Order) to find concrete values in related config types. This enables fields to inherit from sibling configs when the current config doesn't have a concrete value.

The resolution strategy is implemented in :py:mod:`openhcs.config_framework.dual_axis_resolver` as a pure function that takes the object, field name, and available configs, then returns the resolved value.

Context Management with Contextvars
-----------------------------------
The factory integrates with Python's contextvars system for context scoping.

Context Scoping
~~~~~~~~~~~~~~~
The :py:func:`~openhcs.config_framework.context_manager.config_context` context manager creates a new scope where a specific configuration is merged into the current context. When you enter a ``config_context(pipeline_config)`` block, the pipeline config's fields are merged into the current global config, and this merged config becomes the active context for all lazy dataclass resolutions within that block.

Config Merging
~~~~~~~~~~~~~~
The :py:func:`~openhcs.config_framework.context_manager.merge_configs` function recursively merges nested dataclass fields. When merging, None values are treated as "don't override" by default, allowing inheritance to work correctly. This enables step configs to override only specific fields while inheriting others from the pipeline config.

Usage Pattern
~~~~~~~~~~~~~
The typical usage pattern is:

.. code-block:: python

    from openhcs.config_framework.context_manager import config_context

    # Create lazy versions of configs
    lazy_global = LazyDataclassFactory.make_lazy_simple(GlobalPipelineConfig)
    lazy_pipeline = LazyDataclassFactory.make_lazy_simple(PipelineConfig)

    # Use config_context to set the active context
    with config_context(pipeline_config):
        # Within this block, lazy_pipeline fields resolve from pipeline_config
        # and inherit from global_config for missing values
        value = lazy_pipeline.some_field  # Resolves from context

See Also
--------
- :doc:`configuration_framework` - Configuration framework overview
- :doc:`concurrency_model` - Contextvars and thread-local context system
- :doc:`code_ui_interconversion` - How lazy configs are used in UI code generation

**Implementation References:**

- ``openhcs/config_framework/lazy_factory.py`` - LazyDataclassFactory and LazyMethodBindings
- ``openhcs/config_framework/dual_axis_resolver.py`` - Dual-axis resolution strategy
- ``openhcs/config_framework/context_manager.py`` - Contextvars-based context management
