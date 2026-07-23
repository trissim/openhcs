Configuration Framework
=======================

Traditional configuration systems lose user edits when switching contexts because they can't distinguish between "use parent default" and "I specifically want this value". OpenHCS solves this through lazy dataclass resolution with dual-axis inheritance.

.. code-block:: python

   @auto_create_decorator
   @dataclass(frozen=True)
   class GlobalPipelineConfig:
       num_workers: int = 1

   # Decorator automatically creates PipelineConfig with None defaults:
   # class PipelineConfig:
   #     num_workers: int | None = None  # Inherits from GlobalPipelineConfig

The ``@auto_create_decorator`` generates lazy dataclasses where ``None`` means "inherit from parent context" and explicit values mean "use this specific value". This preserves user intent across context switches.

Dual-Axis Resolution
--------------------

Resolution combines context flattening (X-axis) with MRO traversal (Y-axis):

.. code-block:: python

   # X-axis: Context hierarchy flattened into available_configs dict
   with config_context(global_config):           # GlobalPipelineConfig
       with config_context(pipeline_config):     # PipelineConfig
           with config_context(step_config):     # StepMaterializationConfig
               # All three merged into available_configs dict

   # Y-axis: MRO determines priority
   # StepMaterializationConfig.__mro__ = [StepMaterializationConfig, StepWellFilterConfig,
   #                                       PathPlanningConfig, WellFilterConfig, ...]
   # Walk MRO, check available_configs for each type, return first concrete value

This enables sophisticated inheritance patterns without hardcoded priority functions - Python's MRO IS the priority.

Configuration Hierarchy
-----------------------

OpenHCS has two configuration levels with automatic lazy generation:

.. code-block:: python

   # Level 1: GlobalPipelineConfig (application-wide defaults)
   @auto_create_decorator
   @dataclass(frozen=True)
   class GlobalPipelineConfig:
       num_workers: int = 1

   # Level 2: PipelineConfig (auto-generated with None defaults)
   # Automatically created by decorator

The decorator system eliminates boilerplate - you only define ``GlobalPipelineConfig`` with concrete defaults, and ``PipelineConfig`` is generated automatically with ``None`` defaults for inheritance.

Nested Configuration Pattern
----------------------------

Nested configurations use the ``@global_pipeline_config`` decorator to inject fields into both ``GlobalPipelineConfig`` and ``PipelineConfig``:

.. code-block:: python

   @global_pipeline_config
   @dataclass(frozen=True)
   class PathPlanningConfig(WellFilterConfig):
       output_dir_suffix: str = ""
       sub_dir: str = ""

Nested configs inherit through both their own MRO and the parent config hierarchy.

Sibling Inheritance via MRO
---------------------------

Multiple inheritance enables sibling field inheritance:

.. code-block:: python

   @global_pipeline_config
   @dataclass(frozen=True)
   class WellFilterConfig:
       well_filter: Optional[Union[List[str], str, int]] = None

   @global_pipeline_config
   @dataclass(frozen=True)
   class PathPlanningConfig(WellFilterConfig):
       output_dir_suffix: str = ""
       sub_dir: str = ""

   @global_pipeline_config
   @dataclass(frozen=True)
   class StepWellFilterConfig(WellFilterConfig):
       well_filter: Optional[Union[List[str], str, int]] = 1  # Override default

   @global_pipeline_config
   @dataclass(frozen=True)
   class StepMaterializationConfig(StepWellFilterConfig, PathPlanningConfig):
       backend: MaterializationBackend = MaterializationBackend.AUTO
       # Inherits well_filter from StepWellFilterConfig (comes first in MRO)
       # Inherits output_dir_suffix, sub_dir from PathPlanningConfig

The MRO for ``StepMaterializationConfig`` is:

.. code-block:: python

   StepMaterializationConfig.__mro__ = (
       StepMaterializationConfig,
       StepWellFilterConfig,
       PathPlanningConfig,
       WellFilterConfig,
       object
   )

When resolving ``well_filter`` field with all values set to ``None``:

1. Check ``StepMaterializationConfig`` - no override (inherits)
2. Check ``StepWellFilterConfig`` - has ``well_filter = 1`` → **use this**
3. Never reaches ``PathPlanningConfig`` or ``WellFilterConfig``

This is pure Python MRO - no custom priority logic needed.

Real-World Usage
---------------

From ``tests/integration/test_main.py``:

.. code-block:: python

   # Create global config with concrete values
   global_config = GlobalPipelineConfig(
       num_workers=4,
       path_planning_config=PathPlanningConfig(
           sub_dir="processed",
           output_dir_suffix="_output"
       )
   )

   # Establish global context
   ensure_global_config_context(GlobalPipelineConfig, global_config)

   # Create pipeline config with lazy configs for inheritance
   pipeline_config = PipelineConfig(
       path_planning_config=LazyPathPlanningConfig(
           output_dir_suffix="_custom"  # Override global
           # sub_dir=None (implicit) - inherits "processed" from global
       )
   )

The lazy configs resolve through dual-axis algorithm: check ``PipelineConfig`` context first, then ``GlobalPipelineConfig`` context, walking MRO at each level.

Framework Modules
----------------

The framework is extracted to ``openhcs.config_framework`` for reuse:

**lazy_factory.py**
  Generates lazy dataclasses with ``__getattribute__`` interception

**dual_axis_resolver.py**
  Pure MRO-based resolution - no priority functions

**context_manager.py**
  Contextvars-based context stacking via ``config_context()``

**placeholder.py**
  UI placeholder generation showing inherited values

**global_config.py**
  Thread-local storage for global configuration

**config.py**
  Framework initialization - ``set_base_config_type(GlobalPipelineConfig)``

Backward compatibility shims at old paths (``openhcs.core.lazy_config``, etc.) re-export from framework.

Thread-Local Context Synchronization
------------------------------------

The configuration framework maintains thread-local context for GlobalPipelineConfig to support lazy placeholder resolution across the application.

**Context Lifecycle**

1. **Initialization**: Context set during GUI startup
2. **Live Updates**: Context synchronized with form edits in real-time
3. **Restoration**: Original context restored on cancel
4. **Cleanup**: Context cleared on window close

**GUI Integration**

The configuration window implements special handling for GlobalPipelineConfig to ensure thread-local context stays synchronized with form edits. Each field change triggers immediate context update, ensuring placeholder values in other windows update immediately.

When users cancel config editing, the original context is restored to prevent context pollution from experimental changes.

**Implementation Details**

See :doc:`code_ui_interconversion` for detailed implementation of context synchronization during code editing.

Token-Based Caching Infrastructure
-----------------------------------

The configuration framework includes reusable caching abstractions that eliminate redundant context resolution operations, particularly in GUI scenarios where the same values are resolved repeatedly.

**Core Abstractions**

**TokenCache<T>**
  Generic multi-key cache with automatic invalidation when a token changes. Useful when caching multiple related values that should all invalidate together.

  .. code-block:: python

     from openhcs.config_framework import TokenCache, CacheKey

     # Create cache with token provider
     cache = TokenCache(lambda: global_token_counter)

     # Get or compute value
     value = cache.get_or_compute(
         key=CacheKey.from_args('scope', 'param_name'),
         compute_fn=lambda: expensive_computation()
     )

     # Cache automatically invalidates when token changes

**SingleValueTokenCache<T>**
  Simplified variant for caching a single value. More efficient when you only need one cached value per token.

  .. code-block:: python

     from openhcs.config_framework import SingleValueTokenCache

     cache = SingleValueTokenCache(lambda: global_token_counter)
     value = cache.get_or_compute(lambda: expensive_computation())

**LiveContextResolver**
  Pure service for resolving config attributes through context hierarchies with caching. Completely UI-agnostic - works with any dataclasses and context stacks.

  .. code-block:: python

     from openhcs.config_framework import LiveContextResolver

     resolver = LiveContextResolver()

     # Resolve attribute through context stack
     resolved_value = resolver.resolve_config_attr(
         config_obj=step_config,
         attr_name='enabled',
         context_stack=[global_config, pipeline_config, step],
         live_context={PipelineConfig: {'num_workers': 4}},
         cache_token=current_token
     )

  **Critical None Value Semantics**: The resolver passes ``None`` values through during live context merge. When a field is reset to ``None`` in a form, the ``None`` value overrides the saved concrete value via ``dataclasses.replace()``. This triggers MRO resolution which walks up the context hierarchy to find the inherited value from parent context (e.g., GlobalPipelineConfig).

**Architecture Principles**

1. **Token-based invalidation**: O(1) cache invalidation across all caches by incrementing a single counter
2. **Separation of concerns**: Caching logic is independent of domain concepts (steps, pipelines, etc.)
3. **Caller responsibility**: UI layer provides live context and token; service layer handles resolution
4. **Incremental updates**: Only changed values trigger token increment, enabling fine-grained cache control

**Performance Impact**

Token-based caching eliminates redundant operations:

- **Context stack building**: Avoided when token unchanged (was happening 20+ times per UI refresh)
- **Placeholder text resolution**: Cached expensive string formatting operations
- **Entire refresh operations**: Skipped when inputs haven't changed

Measured improvements: 60ms → 1ms for pipeline editor preview updates.

**When to Use**

Use ``TokenCache`` when:

- Multiple related values need synchronized invalidation
- Cache keys have multiple components (scope, parameter name, etc.)
- You need explicit cache key management

Use ``SingleValueTokenCache`` when:

- Only one value needs caching per token
- Simpler API is preferred
- Cache key is implicit (always the same computation)

Use ``LiveContextResolver`` when:

- Resolving config attributes through context hierarchies
- Need to merge live values from multiple sources
- Want to cache resolution results automatically

Cross-Window Update System
---------------------------

The configuration framework includes a real-time cross-window update system that propagates changes between open configuration windows using type-based inheritance filtering and targeted field refresh.

**Performance Characteristics**

- **Type-based filtering**: Only refresh configs that inherit from the changed type via MRO
- **Targeted field refresh**: Only refresh the specific field that changed, not all fields
- **Widget signature checks**: Skip UI updates when placeholder text hasn't changed
- **Update latency**: <10ms per change (down from ~200ms before optimization)

**Architecture**

The system uses class-level signals (``context_value_changed``, ``context_refreshed``) to propagate changes between windows. When a field changes:

1. Emitting window sends field path (e.g., ``"PipelineConfig.well_filter_config.well_filter"``)
2. Receiving windows check if they're affected using MRO inheritance checks
3. Affected windows extract relevant field name from path for their level
4. Only fields inheriting from the changed field's type are refreshed
5. Widget signature checks prevent redundant UI updates

**Reset Propagation**

When fields are reset to ``None``, the system tracks them in a ``reset_fields`` set and includes them in live context even though their value is ``None``. The ``LiveContextResolver`` filters out ``None`` values during merge to preserve MRO inheritance semantics.

For detailed information about cross-window update optimization, see :doc:`cross_window_update_optimization`.
