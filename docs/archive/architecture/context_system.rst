Context Management System
=========================

Configuration resolution requires tracking which configs are active at any point during execution. OpenHCS uses Python's ``contextvars`` for clean context stacking without frame introspection.

.. code-block:: python

   from openhcs.config_framework import config_context

   with config_context(global_config):
       with config_context(pipeline_config):
           # Both configs available for resolution
           lazy_instance.field_name  # Resolves through both contexts

The ``config_context()`` manager extracts dataclass fields and merges them into the context stack, enabling lazy resolution without explicit parameter passing.

Context Stacking
----------------

Contexts stack via ``contextvars.ContextVar``:

.. code-block:: python

   # openhcs/config_framework/context_manager.py
   _config_context_base: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
       'config_context_base',
       default=None
   )

   # Track types pushed via config_context() for hierarchy queries
   context_type_stack: ContextVar[Tuple[Type, ...]] = ContextVar(
       'context_type_stack',
       default=()
   )

   @contextmanager
   def config_context(obj):
       """Stack a configuration context."""
       # Extract all dataclass fields from obj
       new_configs = extract_all_configs(obj)

       # Get current context
       current = _config_context_base.get()

       # Merge with current context
       merged = merge_configs(current, new_configs) if current else new_configs

       # Track type in stack for hierarchy queries
       current_types = context_type_stack.get()
       new_types = current_types + (type(obj),) if obj is not None else current_types

       # Set new context
       token = _config_context_base.set(merged)
       type_token = context_type_stack.set(new_types)
       try:
           yield
       finally:
           _config_context_base.reset(token)
           context_type_stack.reset(type_token)

Each ``with config_context()`` block adds configs to the stack. On exit, the context is automatically restored.

Context Type Stack
------------------

The ``context_type_stack`` ContextVar tracks which types have been pushed via ``config_context()``. This enables generic hierarchy queries without hardcoding type names:

.. code-block:: python

   from openhcs.config_framework import get_context_type_stack

   with config_context(global_config):
       with config_context(pipeline_config):
           with config_context(step):
               # Query the active type stack
               stack = get_context_type_stack()
               # Returns: (GlobalPipelineConfig, PipelineConfig, StepType)

Hierarchy Registry
------------------

For cross-window updates (when parent config windows are open but not actively resolving),
the system maintains a persistent hierarchy registry:

.. code-block:: python

   # Registry maps child_type → parent_type
   _known_hierarchy: dict = {}

   def register_hierarchy_relationship(parent_type, child_type):
       """Register that parent_type is the parent of child_type in the config hierarchy."""
       parent_base = _normalize_type(parent_type)
       child_base = _normalize_type(child_type)
       if parent_base != child_base and not _is_global_type(parent_base):
           _known_hierarchy[child_base] = parent_base

   def unregister_hierarchy_relationship(child_type):
       """Remove hierarchy entry when form closes."""
       child_base = _normalize_type(child_type)
       _known_hierarchy.pop(child_base, None)

Form managers register their hierarchy when they open:

.. code-block:: python

   # Step editor opens with context_obj=pipeline_config
   register_hierarchy_relationship(type(context_obj), type(object_instance))
   # Registers: PipelineConfig → Step

This allows ``get_types_before_in_stack()`` to return correct ancestors even when
the parent window isn't actively inside a ``config_context()`` call.

Hierarchy Query Functions
-------------------------

The config framework provides generic functions to query the hierarchy:

.. code-block:: python

   from openhcs.config_framework import (
       get_types_before_in_stack,
       is_ancestor_in_context,
       is_same_type_in_context,
       get_ancestors_from_hierarchy
   )

   # Get all types that come before Step in the hierarchy
   ancestors = get_types_before_in_stack(Step)
   # Returns: [PipelineConfig] (uses active stack or registry fallback)

   # Check if PipelineConfig is an ancestor of Step
   is_ancestor = is_ancestor_in_context(PipelineConfig, Step)
   # Returns: True

   # Check if two types are equivalent (handles lazy wrappers)
   is_same = is_same_type_in_context(LazyPipelineConfig, PipelineConfig)
   # Returns: True

These functions first check the active ``context_type_stack``, then fall back to
the ``_known_hierarchy`` registry for cross-window scenarios.

Context Extraction
-----------------

The system extracts all dataclass fields from the provided object:

.. code-block:: python

   def extract_all_configs(context_obj) -> Dict[str, Any]:
       """Extract all dataclass configs from an object."""
       configs = {}

       # Extract dataclass fields
       if is_dataclass(context_obj):
           for field in fields(context_obj):
               value = getattr(context_obj, field.name)
               if is_dataclass(value):
                   # Store by type name
                   configs[type(value).__name__] = value

       return configs

This flattens nested configs into a single dictionary keyed by type name.

Global Config Context
--------------------

Global config uses thread-local storage for stability:

.. code-block:: python

   # Thread-local storage for global config
   _global_config_storage: Dict[Type, threading.local] = {}

   def ensure_global_config_context(config_type: Type, config_instance: Any):
       """Establish global config as base context."""
       # Store in thread-local
       set_current_global_config(config_type, config_instance)

       # Also inject into contextvars base
       with config_context(config_instance):
           # Global context now available

This hybrid approach uses thread-local for the global base and contextvars for dynamic stacking.

Resolution Integration
---------------------

The dual-axis resolver receives the merged context:

.. code-block:: python

   def resolve_field_inheritance(obj, field_name, available_configs):
       """Resolve field through dual-axis algorithm.

       available_configs: Merged context from config_context() stack
       """
       # Walk MRO
       for mro_class in type(obj).__mro__:
           # Check if this MRO class has a config instance in context
           for config_name, config_instance in available_configs.items():
               if type(config_instance) == mro_class:
                   value = object.__getattribute__(config_instance, field_name)
                   if value is not None:
                       return value
       return None

The ``available_configs`` dict contains all configs from the context stack, flattened and ready for MRO traversal.

Usage Pattern
------------

From ``tests/integration/test_main.py``:

.. code-block:: python

   # Establish global context
   global_config = GlobalPipelineConfig(num_workers=4)
   ensure_global_config_context(GlobalPipelineConfig, global_config)

   # Create pipeline config
   pipeline_config = PipelineConfig(
       path_planning_config=LazyPathPlanningConfig(output_dir_suffix="_custom")
   )

   # Stack contexts
   with config_context(pipeline_config):
       # Both global and pipeline configs available
       # Lazy fields resolve through merged context
       orchestrator = Orchestrator(pipeline_config)

The orchestrator and all lazy configs inside it can resolve fields through both ``global_config`` and ``pipeline_config`` contexts.

Context Cleanup
--------------

Contextvars automatically handle cleanup:

.. code-block:: python

   with config_context(pipeline_config):
       # Context active
       pass
   # Context automatically restored to previous state

No manual cleanup needed - Python's context manager protocol handles it.

Framework-Agnostic Context Stack Building
-----------------------------------------

For UI placeholder resolution, the ``build_context_stack()`` function provides a
framework-agnostic way to build complete context stacks:

.. code-block:: python

   from openhcs.config_framework import build_context_stack

   # Build context stack for placeholder resolution
   stack = build_context_stack(
       context_obj=pipeline_config,           # Parent context
       overlay=manager.parameters,            # Current form values
       dataclass_type=manager.dataclass_type, # Type being edited
       live_context=live_context_dict,        # Live values from other forms
       root_form_values=root_values,          # Root form's values (for sibling inheritance)
       root_form_type=root_type,              # Root form's dataclass type
   )

   with stack:
       # Context layers are active
       placeholder = resolve_placeholder(field_name)

The stack builds layers in order:

1. **Global context layer** - Thread-local global config or live editor values
2. **Intermediate layers** - Ancestors from ``get_types_before_in_stack()``
3. **Parent context** - The ``context_obj`` parameter
4. **Root form layer** - For sibling inheritance (see below)
5. **Overlay** - Current form values

Sibling Inheritance via Root Form
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When nested configs need to inherit from siblings (e.g., ``well_filter_config``
inheriting from ``step_well_filter_config``), the root form's values enable this:

.. code-block:: python

   # Root form (Step) contains both sibling configs
   root_values = {
       'step_well_filter_config': LazyStepWellFilterConfig(well_filter=123),
       'well_filter_config': LazyWellFilterConfig(well_filter=None),
       ...
   }

   # When resolving well_filter_config.well_filter:
   # 1. stack includes root_values
   # 2. LazyWellFilterConfig.well_filter resolution walks MRO
   # 3. Finds StepWellFilterConfig (superclass) in context
   # 4. Uses step_well_filter_config.well_filter = 123

For non-dataclass roots (e.g., ``FunctionStep``), the function wraps values in
``SimpleNamespace`` to maintain a unified code path:

.. code-block:: python

   if root_form_type and is_dataclass(root_form_type):
       root_instance = root_form_type(**root_form_values)
   else:
       # Non-dataclass root - wrap in SimpleNamespace
       from types import SimpleNamespace
       root_instance = SimpleNamespace(**root_form_values)

   stack.enter_context(config_context(root_instance))

This enables sibling inheritance for any root type, including function step parameters.

See :doc:`field_change_dispatcher` for how the UI uses this for live placeholder updates.
