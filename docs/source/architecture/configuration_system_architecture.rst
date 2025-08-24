Configuration System Architecture
=================================

Overview
--------

OpenHCS implements a hierarchical configuration system that flows from global application settings down to individual step execution. The system combines immutable dataclasses, thread-local context management, dynamic lazy resolution, and automatic field path detection to provide a unified configuration architecture.

**The Problem This Solves**: Traditional configuration systems in scientific software suffer from the "lost edits" problem - when users modify settings in a UI, their changes get overwritten by defaults or disappear when switching contexts. This happens because most systems use static configuration objects that can't adapt their behavior based on the current editing context.

**Key Feature**: The configuration system addresses the "lost edits" problem through lazy dataclasses that can resolve values from different sources based on context while preserving user modifications. When a user is editing global settings, the system shows actual default values. When editing pipeline-specific settings, it shows inherited values from the global config. When editing step-specific settings, it shows values inherited from both pipeline and global levels.

**Architectural Approach**: The system treats configuration as a compilation target. Configuration objects are immutable once created, but the system can generate different "views" of the same underlying data depending on the editing context. This provides consistency while allowing flexibility for different workflow requirements.

Core Philosophy
---------------

The configuration system is built on four fundamental principles that address common challenges in scientific computing workflows:

1. **Hierarchical Flow**: Configuration flows from Global → Pipeline → Step levels

   *Why This Matters*: Scientific experiments often involve running the same analysis pipeline with slight variations. Researchers need to set global defaults (like number of GPU workers), override them at the pipeline level (for specific experiments), and further customize them at the step level (for particular processing operations). The hierarchical flow ensures that changes propagate logically while allowing fine-grained control.

2. **Immutable Dataclasses**: All configuration objects are frozen dataclasses ensuring consistency

   *Why This Matters*: In scientific computing, reproducibility is critical. Once a configuration is created, it should never change unexpectedly. Immutable dataclasses prevent accidental modifications that could lead to irreproducible results. If you need different settings, you create a new configuration object rather than modifying an existing one.

3. **Lazy Resolution**: Values are resolved dynamically based on execution context

   *Why This Matters*: The same configuration field might need different values depending on whether you're editing global settings, pipeline settings, or step settings. Lazy resolution allows the system to show the most appropriate value for the current context without requiring complex parameter passing or global state management.

4. **Zero Hardcoding**: Field relationships are discovered automatically through type introspection

   *Why This Matters*: As OpenHCS evolves, new configuration types are added and existing ones are modified. Hardcoded field mappings would require manual updates every time the configuration structure changes. Automatic discovery through type introspection means the system adapts to configuration changes without requiring code updates.

Configuration Hierarchy
------------------------

The configuration hierarchy addresses a common problem in scientific software: how to provide sensible defaults while allowing users to override them at multiple levels of granularity. OpenHCS addresses this through a three-tier resolution system that reflects typical experimental parameter organization.

Resolution Authority
~~~~~~~~~~~~~~~~~~~~

Configuration resolution follows a clear hierarchy that reflects real-world usage patterns:

1. **User-set values** - Explicit user input (highest priority)

   *When This Applies*: When a user explicitly sets a value in a form or configuration file, that value takes absolute precedence. This ensures that deliberate user choices are never overridden by the system.

2. **Thread-local context** - Current global config instance

   *When This Applies*: When a user hasn't explicitly set a value, the system looks to the current context. If you're editing a pipeline, it inherits from the global pipeline config. If you're editing a step, it inherits from the current pipeline config. This provides intelligent defaults based on the current working context.

3. **Static defaults** - Dataclass field defaults (lowest priority)

   *When This Applies*: When no user value exists and no context is available, the system falls back to the hardcoded defaults defined in the dataclass. These represent the "factory settings" that work for most common use cases.

Hierarchical Structure
~~~~~~~~~~~~~~~~~~~~~~

::

   GlobalPipelineConfig (Root)
   ├── VFSConfig (Virtual File System settings)
   ├── PathPlanningConfig (Directory naming conventions)
   ├── ZarrConfig (Zarr storage backend settings)
   ├── num_workers (Parallel execution settings)
   └── microscope (Default microscope type)

Configuration Dataclasses
~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenHCS configuration is built using immutable dataclasses that compose together to form the complete configuration hierarchy. Each dataclass represents a specific domain of configuration (VFS settings, path planning, etc.) and can be used independently or as part of larger configuration objects.

**Dataclass Composition Pattern**: Configuration objects are composed by declaring one dataclass as a field of another. This creates clear ownership relationships and enables automatic field path detection.

**Inheritance vs Composition**: OpenHCS uses composition (dataclass contains other dataclasses) rather than inheritance (dataclass extends another dataclass). This provides more flexibility and clearer relationships between configuration domains.

.. code:: python

   @dataclass(frozen=True)
   class VFSConfig:
       """Configuration for Virtual File System operations."""
       intermediate_backend: Backend = Backend.MEMORY
       """Backend for intermediate step results (not explicitly materialized)."""

       materialization_backend: MaterializationBackend = MaterializationBackend.DISK
       """Backend for explicitly materialized outputs (final results, user saves)."""

       persistent_storage_root_path: Optional[str] = None
       """Root path for persistent storage backends."""

   @dataclass(frozen=True)
   class PathPlanningConfig:
       """Configuration for pipeline path planning."""
       output_dir_suffix: str = "_outputs"
       """Default suffix for general step output directories."""

       positions_dir_suffix: str = "_positions"
       """Suffix for position generation result directories."""

       stitched_dir_suffix: str = "_stitched"
       """Suffix for stitched image result directories."""

   @dataclass(frozen=True)
   class StepMaterializationConfig(PathPlanningConfig):
       """Configuration for step-level materialization behavior.

       Inherits from PathPlanningConfig to ensure all required path planning fields
       are available for the lazy loading system.
       """
       save_intermediate: bool = False
       """Whether to save intermediate processing results."""

       well_filter_mode: WellFilterMode = WellFilterMode.INCLUDE
       """Well filtering mode for step materialization."""

       # Override PathPlanningConfig defaults to prevent collisions
       output_dir_suffix: str = ""  # Uses same output plate path as main pipeline
       sub_dir: str = "checkpoints"  # vs global "images"

   @dataclass(frozen=True)
   class GlobalPipelineConfig:
       """Root configuration object demonstrating composition architecture."""
       # Core execution settings
       num_workers: int = field(default_factory=lambda: os.cpu_count() or 1)
       use_threading: bool = True
       microscope: Microscope = Microscope.AUTO

       # Composed configuration domains
       path_planning: PathPlanningConfig = field(default_factory=PathPlanningConfig)
       vfs: VFSConfig = field(default_factory=VFSConfig)
       zarr: ZarrConfig = field(default_factory=ZarrConfig)
       analysis_consolidation: AnalysisConsolidationConfig = field(default_factory=AnalysisConsolidationConfig)
       plate_metadata: PlateMetadataConfig = field(default_factory=PlateMetadataConfig)
       function_registry: FunctionRegistryConfig = field(default_factory=FunctionRegistryConfig)
       materialization_defaults: StepMaterializationConfig = field(default_factory=StepMaterializationConfig)
       tui: Optional[TUIConfig] = field(default_factory=lambda: TUIConfig())

**Architectural Composition Pattern**: The GlobalPipelineConfig demonstrates the composition-over-inheritance principle by aggregating specialized configuration objects rather than inheriting from them. Each domain (storage, UI, analysis) has its own configuration object with clear boundaries.

**Additional Configuration Domains**: The GlobalPipelineConfig includes several specialized configuration objects for different system domains:

- **ZarrConfig**: Compression, chunking, and OME-ZARR metadata settings
- **AnalysisConsolidationConfig**: Automatic analysis results consolidation
- **PlateMetadataConfig**: Plate metadata handling and formats
- **FunctionRegistryConfig**: Function registry behavior and caching
- **TUIConfig**: Text User Interface preferences

**Configuration Enums**: The system uses enums for type-safe configuration choices:

- **MaterializationBackend**: ZARR vs DISK storage selection
- **ZarrCompressor**: Compression algorithms (ZSTD, LZ4, GZIP)
- **ZarrChunkStrategy**: Array chunking strategies
- **WellFilterMode**: Well filtering for selective materialization

For complete configuration details, field descriptions, and usage examples, see :doc:`../api/config`.

**Key Design Principles**:

- **Frozen Dataclasses**: All configuration objects are immutable once created, ensuring consistency
- **Composition Over Inheritance**: Complex configurations are built by composing simpler dataclasses
- **Default Factories**: Each field uses a factory to create independent instances
- **Clear Ownership**: Each configuration domain has a single responsible dataclass
- **Inheritance Where Appropriate**: StepMaterializationConfig inherits from PathPlanningConfig for field reuse

Thread-Local Context System
----------------------------

The thread-local context system addresses a common problem in multi-user scientific computing environments: how to maintain separate configuration contexts for different operations without complex parameter passing or global state management.

**The Challenge**: In scientific computing environments, multiple users might be running different experiments simultaneously, or the same user might be working on multiple projects. Each context needs its own configuration state, but passing configuration objects through every function call creates complex APIs and tight coupling.

**The Solution**: Thread-local storage provides isolated configuration contexts that are automatically available to any code running in that thread, without requiring explicit parameter passing. Each thread maintains its own configuration state, ensuring that different operations don't interfere with each other.

Generic Storage
~~~~~~~~~~~~~~~

The system uses type-keyed thread-local storage for configuration contexts, which provides both type safety and flexibility. This approach allows any dataclass type to be stored and retrieved from thread-local context without requiring hardcoded type mappings.

**Type-Keyed Storage**: Each dataclass type (like `GlobalPipelineConfig`) serves as its own key in the thread-local storage. This ensures type safety while allowing the system to work with any configuration dataclass.

**Thread Isolation**: Each thread maintains its own independent configuration context, preventing interference between concurrent operations.

.. code:: python

   # Actual OpenHCS thread-local storage implementation
   _global_config_contexts: Dict[Type, threading.local] = {}
   _lazy_type_registry: Dict[Type, Type] = {}

   def set_current_global_config(config_type: Type, config_instance: Any) -> None:
       """Set current global config for any dataclass type."""
       if config_type not in _global_config_contexts:
           _global_config_contexts[config_type] = threading.local()
       _global_config_contexts[config_type].value = config_instance

   def get_current_global_config(config_type: Type) -> Optional[Any]:
       """Get current global config for any dataclass type."""
       context = _global_config_contexts.get(config_type)
       return getattr(context, 'value', None) if context else None

   def register_lazy_type_mapping(lazy_type: Type, base_type: Type) -> None:
       """Register mapping between lazy and base types for automatic resolution."""
       _lazy_type_registry[lazy_type] = base_type

   def get_base_type_for_lazy(lazy_type: Type) -> Optional[Type]:
       """Get the base type for a lazy type."""
       return _lazy_type_registry.get(lazy_type)

**Configuration Cache System**: OpenHCS includes a sophisticated caching system for configuration persistence:

.. code:: python

   class UnifiedGlobalConfigCache:
       """Unified global configuration cache with pluggable execution strategies."""

       def __init__(self, cache_file: Optional[Path] = None, strategy: Optional[CacheExecutionStrategy] = None):
           # Supports both async (TUI) and Qt threading (PyQt) execution patterns
           self.strategy = strategy or AsyncExecutionStrategy()

       async def load_cached_config(self) -> Optional[GlobalPipelineConfig]:
           """Load cached global config from disk."""
           return await self.strategy.execute_load(self.cache_file)

       async def save_config_to_cache(self, config: GlobalPipelineConfig) -> bool:
           """Save global config to cache."""
           return await self.strategy.execute_save(config, self.cache_file)

**Usage Pattern**: The system supports both synchronous and asynchronous configuration operations, with automatic caching and cross-framework compatibility.

Context-Driven Behavior
~~~~~~~~~~~~~~~~~~~~~~~

The thread-local context system enables different behavior based on the current editing context. The same configuration field can show different values depending on whether you're editing global settings, pipeline settings, or step settings.

**Context Detection**: The system determines the current editing context by checking whether lazy resolution is active for a given dataclass type. This allows the UI to adapt its behavior automatically.

**Placeholder Logic**: When editing a lazy configuration, the system shows `None` values as placeholder text that indicates what the inherited value would be. When editing a global configuration, it shows the actual default values.

.. code:: python

   # Global config editing: Use actual default values
   is_global_config_editing = not LazyDefaultPlaceholderService.has_lazy_resolution(dataclass_type)

   # Lazy config editing: Use None to show placeholder text
   if is_global_config_editing:
       return actual_default_value
   else:
       return None  # Shows "Pipeline default: value" placeholder

**Real-World Example**: When editing a step's materialization config, a `None` value for `output_dir_suffix` would show placeholder text like "Pipeline default: _outputs" indicating that this step would inherit the pipeline's default suffix.

UI Context Scoping
~~~~~~~~~~~~~~~~~~~

The system supports three distinct UI editing contexts, each with different resolution behavior that matches the user's mental model of configuration hierarchy.

**1. Global Config Editing**
When editing global settings, users expect to see the actual default values that will be used throughout the system.

.. code:: python

   # Global config editor setup
   set_current_global_config(GlobalPipelineConfig, global_defaults)
   # Shows actual default values, no lazy resolution

**2. Pipeline Config Editing**
When editing pipeline-specific settings, users expect to see values inherited from global config for fields they haven't customized.

.. code:: python

   # Pipeline editor setup
   orchestrator_config = orchestrator.get_effective_config()
   set_current_global_config(GlobalPipelineConfig, orchestrator_config)
   # Lazy resolution from orchestrator config, None values use global defaults

**3. Step Config Editing**
When editing individual step settings, users expect to see values inherited from both pipeline and global levels.

.. code:: python

   # Step editor setup
   step_config = step.get_effective_config()
   set_current_global_config(GlobalPipelineConfig, step_config)
   # Multi-level lazy resolution: step → pipeline → global

**Context Switching**: The system automatically switches between these contexts as users navigate between different configuration editors, ensuring that the displayed values always match the user's current editing scope.

Lazy Dataclass Factory
-----------------------

The LazyDataclassFactory is a key component of the configuration system's flexibility. It addresses a tension in configuration management: how to create objects that behave like normal dataclasses (with predictable APIs and type safety) but can dynamically change their behavior based on context.

**The Problem**: Traditional dataclasses are static - once created, their field values are fixed. In scientific computing, you often want the same configuration object to show different values depending on the editing context. For example, when editing global settings, you want to see the actual default values. When editing pipeline settings, you want to see values inherited from the global config.

**The Solution**: The LazyDataclassFactory generates runtime dataclasses with custom resolution logic. They appear as normal dataclasses to consuming code (same API, same type hints, same behavior) but can resolve field values from thread-local contexts while preserving user edits. This provides predictable APIs for developers and context-aware behavior for users.

Auto-Hierarchy Constructor
~~~~~~~~~~~~~~~~~~~~~~~~~~

The LazyDataclassFactory uses an auto-hierarchy approach for creating lazy dataclasses with sophisticated field-level inheritance that can resolve values from multiple sources in the configuration hierarchy. It automatically discovers field paths and enables sibling inheritance patterns.

.. code:: python

   # Auto-hierarchy lazy config with field-level inheritance
   LazyStepMaterializationConfig = LazyDataclassFactory.make_lazy_with_field_level_auto_hierarchy(
       base_class=StepMaterializationConfig,
       global_config_type=GlobalPipelineConfig,
       field_path="materialization_defaults",
       lazy_class_name="LazyStepMaterializationConfig",
       context_provider=None  # Optional context provider
   )

   # Root-level config (field_path=None for top-level configs)
   PipelineConfig = LazyDataclassFactory.make_lazy_with_field_level_auto_hierarchy(
       base_class=GlobalPipelineConfig,
       global_config_type=GlobalPipelineConfig,
       field_path=None,  # Root instance
       lazy_class_name="PipelineConfig"
   )

**Key Features**:

- **Automatic Field Path Discovery**: Uses type introspection to find configuration relationships
- **Sibling Inheritance**: Supports inheritance between related configuration objects
- **Context-Aware Resolution**: Resolves values from thread-local storage with hierarchy fallbacks
- **Universal Algorithm**: Works for any dataclass structure without manual configuration

**Additional Factory Methods**: The LazyDataclassFactory provides several convenience methods for common use cases:

.. code:: python

   # High-level configuration editing
   def create_config_for_editing(config_type: Type, current_config: Any) -> Any:
       """Create a lazy configuration instance for UI editing."""
       set_current_global_config(GlobalPipelineConfig, current_config)
       return LazyDataclassFactory.make_lazy_with_field_level_auto_hierarchy(
           base_class=config_type,
           global_config_type=GlobalPipelineConfig,
           field_path=FieldPathDetector.find_field_path_for_type(GlobalPipelineConfig, config_type),
           lazy_class_name=f"Lazy{config_type.__name__}"
       )()

   # Context management
   def ensure_global_config_context(config_type: Type) -> Any:
       """Ensure global config context exists for the given type."""
       existing = get_current_global_config(config_type)
       if existing is None:
           default_config = get_default_global_config()
           set_current_global_config(config_type, default_config)
           return default_config
       return existing

   # Unified lazy class creation
   def _create_unified_lazy_class(
       base_class: Type,
       global_config_type: Type,
       field_name: str,
       lazy_class_name: str,
       parent_field_path: Optional[str],
       parent_instance_provider: Optional[Callable]
   ) -> Type:
       """Create lazy version with automatic inheritance detection and context propagation."""
       # Automatically handles nested lazy dataclass creation with proper context inheritance

Dynamic Class Generation
~~~~~~~~~~~~~~~~~~~~~~~~

The LazyDataclassFactory generates new dataclass types at runtime that have the same fields as the base class but with custom resolution behavior. This process involves introspecting the base class structure and creating a new class with modified field access methods.

**Field Introspection**: The system examines the base dataclass to understand its field structure, types, and default values. This information is used to create an equivalent lazy dataclass.

**Method Binding**: After creating the new dataclass structure, the factory binds custom resolution methods that implement the lazy behavior while maintaining the same API as the original dataclass.

.. code:: python

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

**Runtime Generation**: This approach allows the system to create lazy versions of any dataclass without requiring manual implementation for each configuration type.

Field Value Resolution
~~~~~~~~~~~~~~~~~~~~~~

The core of the lazy dataclass system is the custom field resolution logic that implements the hierarchical fallback behavior. When a field is accessed, the system follows a specific resolution chain to determine the appropriate value.

**Resolution Chain**: The system tries multiple sources in order of priority: user-set values (highest priority), thread-local context values, and static defaults (lowest priority).

**Transparent Operation**: This resolution happens transparently when fields are accessed, so consuming code doesn't need to know whether it's working with a lazy or regular dataclass.

.. code:: python

   def __getattribute__(self, name):
       """Lazy field resolution with fallback chain."""
       if name.startswith('_') or name in ['__class__', '__dict__']:
           return object.__getattribute__(self, name)

       # Try user-set value first
       user_value = object.__getattribute__(self, f'_{name}')
       if user_value is not None:
           return user_value

       # Fall back to thread-local resolution
       context_instance = self._instance_provider()
       if context_instance and hasattr(context_instance, name):
           return getattr(context_instance, name)

       # Final fallback to static defaults
       return self._get_static_default(name)

**Field Access Pattern**: Each field is stored with a private name (prefixed with `_`) to hold user-set values, while the public field name triggers the resolution logic.

Field Path Detection System
---------------------------

The field path detection system addresses a maintenance challenge common in complex configuration systems: hardcoded field mappings that break when the configuration structure changes.

**The Problem**: Traditional approaches require manual mapping of relationships between configuration types. When you add a new configuration field or change the structure of existing ones, you have to update hardcoded mappings throughout the codebase. This creates maintenance overhead and introduces bugs when mappings get out of sync with the actual configuration structure.

**The Solution**: Automatic field path detection uses Python's type introspection capabilities to discover relationships between configuration types at runtime. Instead of hardcoding that "StepMaterializationConfig lives at the 'materialization_defaults' field of GlobalPipelineConfig," the system discovers this relationship by examining the type annotations of GlobalPipelineConfig's fields.

**Why This Works**: Python's type system provides all the information needed to navigate configuration hierarchies. By treating type annotations as the authoritative source of truth, the system can automatically adapt to configuration structure changes without requiring code updates.

Core Algorithm
~~~~~~~~~~~~~~

The field path detection algorithm uses a two-phase approach: first checking for direct field matches, then recursively searching nested dataclass structures. This ensures that the system can find configuration types regardless of how deeply they're nested in the hierarchy.

**Direct Matching**: The algorithm first examines all fields of the parent dataclass to see if any field's type directly matches the target type. This handles simple composition cases.

**Recursive Search**: If no direct match is found, the algorithm recursively searches within any nested dataclass fields, building dot-notation paths as it traverses the structure.

.. code:: python

   @staticmethod
   def find_field_path_for_type(parent_type: Type, child_type: Type) -> Optional[str]:
       """Find field path for child_type within parent_type structure."""

       # Direct field matching
       for field in dataclasses.fields(parent_type):
           field_type = FieldPathDetector._unwrap_optional_type(field.type)
           if field_type == child_type:
               return field.name

       # Nested field matching (recursive search)
       for field in dataclasses.fields(parent_type):
           field_type = FieldPathDetector._unwrap_optional_type(field.type)
           if dataclasses.is_dataclass(field_type):
               nested_path = FieldPathDetector.find_field_path_for_type(field_type, child_type)
               if nested_path:
                   return f"{field.name}.{nested_path}"

       return None

   @staticmethod
   def find_all_field_paths_for_type(parent_type: Type, target_type: Type) -> List[str]:
       """Find ALL field paths that contain the target type in the parent config structure.

       This enables automatic hierarchy discovery for lazy resolution by recursively
       searching through nested dataclass structures to find all instances of a target type.
       """
       paths = []

       def _recursive_search(current_type: Type, current_path: str = ""):
           if not dataclasses.is_dataclass(current_type):
               return

           for field in dataclasses.fields(current_type):
               field_type = FieldPathDetector._unwrap_optional_type(field.type)
               field_path = f"{current_path}.{field.name}" if current_path else field.name

               # Direct type match
               if field_type == target_type:
                   paths.append(field_path)
               # Recursive search in nested dataclasses
               elif dataclasses.is_dataclass(field_type):
                   _recursive_search(field_type, field_path)

       _recursive_search(parent_type)
       return paths

**Advanced Features**: The field path detection system supports comprehensive hierarchy discovery, enabling automatic detection of all instances of a configuration type within the global configuration structure.

Type Matching Logic
~~~~~~~~~~~~~~~~~~~

The type matching logic handles the complexities of Python's type system, including Optional types and Union types that are commonly used in configuration dataclasses.

**Direct Type Matching**: The simplest case where the field type exactly matches the target type.

**Optional Type Handling**: Many configuration fields are declared as `Optional[SomeConfig]` to allow None values. The system recognizes that `Optional[T]` is equivalent to `Union[T, None]` and can match the underlying type `T`.

**Union Type Support**: The system can handle more complex Union types by checking if the target type is one of the union members.

.. code:: python

   @staticmethod
   def _types_match(field_type: Type, target_type: Type) -> bool:
       """Check if field_type matches target_type, handling Optional and Union."""

       # Direct match
       if field_type == target_type:
           return True

       # Handle Optional[Type] (Union[Type, None])
       if hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
           union_args = field_type.__args__
           return target_type in union_args

       return False

**Real-World Usage**: This allows the system to correctly identify that a field declared as `Optional[VFSConfig]` should match when searching for `VFSConfig`.

Automatic Discovery Usage
~~~~~~~~~~~~~~~~~~~~~~~~~

The automatic discovery system replaces hardcoded field mappings with dynamic type-based discovery. This means that when new configuration types are added or existing ones are moved, the system automatically adapts without requiring code changes.

**Dynamic Path Resolution**: Instead of maintaining a manual mapping of configuration types to field paths, the system discovers these relationships by examining the dataclass structure at runtime.

**Maintenance-Free Operation**: When developers add new configuration types or restructure existing ones, the field path detection automatically finds the new relationships.

.. code:: python

   # Automatic discovery replaces hardcoded mappings
   field_path = FieldPathDetector.find_field_path_for_type(
       GlobalPipelineConfig, StepMaterializationConfig
   )
   # Returns: "materialization_defaults"

   field_path = FieldPathDetector.find_field_path_for_type(
       GlobalPipelineConfig, VFSConfig
   )
   # Returns: "vfs"

**Zero Configuration**: Developers simply declare their dataclass structure using standard Python type annotations, and the system automatically understands the relationships.

Advanced Resolution Patterns
----------------------------

The advanced resolution patterns in OpenHCS address complex configuration scenarios that arise in scientific computing environments. These patterns go beyond simple inheritance to provide behavior that matches how researchers work with experimental setups.

Multi-Level Resolution Chains
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenHCS implements hierarchical resolution where configuration values flow through multiple levels, reflecting the natural hierarchy of scientific experiments:

**The Scientific Context**: In a typical high-content screening experiment, you might have global settings for your entire lab (like default GPU allocation), pipeline-specific settings for a particular type of analysis (like neurite tracing parameters), and step-specific settings for individual processing operations (like specific filter parameters for noisy images).

**The Technical Challenge**: How do you provide a configuration system that allows overrides at each level while maintaining clear precedence rules and avoiding configuration conflicts?

**Resolution Hierarchy:**

The multi-level resolution system reflects the natural hierarchy of scientific experiments, where settings can be customized at different levels of granularity.

1. **Step Level**: Individual step configuration (highest priority for user edits)
2. **Orchestrator Level**: Pipeline-specific configuration
3. **Global Level**: Application-wide defaults (lowest priority)

**Resolution Path Example**: When a step needs to determine its output directory suffix, the system follows a specific resolution chain that checks multiple sources in order of priority.

.. code:: python

   # Example resolution chain for StepMaterializationConfig.output_dir_suffix:
   resolution_paths = [
       ('step', 'materialization_config'),        # Step-specific override
       ('pipeline', 'materialization_defaults'),  # Pipeline-level default
       ('pipeline', 'path_planning'),             # Pipeline-level sibling inheritance
       ('global', 'materialization_defaults'),    # Global default
       ('global', 'path_planning')                # Global sibling inheritance
   ]

**Breadth-First Traversal**: The system uses breadth-first traversal, checking all sources at the current level before moving to the next level. This ensures that step-level settings always take precedence over pipeline-level settings, which take precedence over global settings.

Sibling Inheritance
~~~~~~~~~~~~~~~~~~~

Sibling inheritance addresses a usability issue: related configuration objects often share common fields, and users expect sensible defaults that reflect these relationships.

**The Problem**: Consider materialization configuration (which controls how processing results are saved) and path planning configuration (which controls directory naming). Both have an "output_dir_suffix" field. Users expect that if they set a global directory suffix for path planning, it should also apply to materialization unless they explicitly override it. But these are separate configuration objects with no direct inheritance relationship.

**The Solution**: Sibling inheritance allows fields to inherit from related configurations when not explicitly set. The system recognizes that `StepMaterializationConfig` and `PathPlanningConfig` are "siblings" within `GlobalPipelineConfig` and allows cross-inheritance between them.

**Dataclass Declaration Enables Inheritance**: Sibling inheritance works because both configuration types are declared as fields within the same parent dataclass. The system can discover these relationships through type introspection of the parent's field declarations.

**Inheritance Through Composition**: This pattern works at any level where two dataclasses are composed together within a parent dataclass. If a step constructor had two optional dataclass parameters, they could inherit from each other using the same mechanism.

.. code:: python

   # Example: Pipeline-level sibling inheritance
   # Both materialization_defaults and path_planning are fields in GlobalPipelineConfig

   @dataclass(frozen=True)
   class GlobalPipelineConfig:
       path_planning: PathPlanningConfig = field(default_factory=PathPlanningConfig)
       materialization_defaults: StepMaterializationConfig = field(default_factory=StepMaterializationConfig)
       # ... other fields

   # Resolution hierarchy for materialization_defaults.output_dir_suffix:
   hierarchy_paths = [
       ('current', 'materialization_defaults'),  # Direct path
       ('current', 'path_planning'),             # Sibling inheritance ✅
       ('global', 'materialization_defaults'),   # Global direct
       ('global', 'path_planning')               # Global sibling
   ]

**Step-Level Example**: If a step constructor had both `materialization_config` and `path_config` parameters, they would also be siblings and could inherit from each other at the step level.

Benefits and Design Principles
------------------------------

**System Benefits:**

- **Thread Safety**: Each thread has isolated configuration context
- **Explicit Resolution**: Clear hierarchy eliminates ambiguous behavior
- **Type Safety**: Generic storage maintains type information
- **Fail-Loud**: Configuration errors surface immediately
- **Context Awareness**: Behavior adapts to editing context
- **Clean API**: No parameter threading through function calls
- **Zero Hardcoding**: Eliminates all hardcoded field path mappings
- **Automatic Discovery**: Finds relationships through type introspection
- **Maintenance-Free**: New config types work automatically without code changes

**Design Principles:**

- **Single Source of Truth**: Type annotations are authoritative
- **Lazy Resolution**: Values computed only when needed
- **Immutable Structures**: Configuration objects cannot be modified after creation
- **Context-Driven**: Behavior adapts based on usage context
- **Hierarchical Inheritance**: Clear precedence rules for value resolution

Integration with UI Systems
---------------------------

The configuration system's integration with UI frameworks demonstrates how architectural design enables clean separation of concerns while providing functionality across different interfaces.

**The Challenge**: OpenHCS supports both PyQt6 (for desktop users) and Textual (for terminal/SSH users). Each framework has different patterns for form creation, validation, and data binding. Without careful design, this could lead to duplicating configuration logic between frameworks or creating tight coupling between business logic and UI code.

**The Solution**: The configuration system integrates with both PyQt6 and Textual UI frameworks through the service layer architecture. The configuration logic is framework-agnostic, while framework-specific adapters handle the UI details.

**Framework Integration Pattern**: The configuration system provides a standard pattern for creating configuration editors that work across different UI frameworks.

**Automatic Field Path Discovery**: The integration uses the field path detection system to automatically determine where a configuration type lives within the global configuration hierarchy.

**Context Setup**: The pattern establishes the appropriate thread-local context and creates a lazy dataclass that will resolve values according to the current editing context.

.. code:: python

   # Framework-agnostic configuration editing
   def create_config_editor(dataclass_type: Type, current_config: Any):
       """Create configuration editor for any dataclass type."""

       # Set up thread-local context
       set_current_global_config(GlobalPipelineConfig, current_config)

       # Create lazy dataclass for editing
       lazy_config = LazyDataclassFactory.make_lazy_with_field_level_auto_hierarchy(
           base_class=dataclass_type,
           global_config_type=GlobalPipelineConfig,
           field_path=FieldPathDetector.find_field_path_for_type(
               GlobalPipelineConfig, dataclass_type
           ),
           lazy_class_name=f"Lazy{dataclass_type.__name__}"
       )

       return lazy_config()

**Universal Pattern**: This same pattern works for any configuration dataclass, whether it's a simple VFS config or a complex nested step configuration.

See Also
--------

- :doc:`../api/config` - Complete configuration reference with field descriptions and examples
- :doc:`service-layer-architecture` - Framework-agnostic service patterns for configuration
- :doc:`step-editor-generalization` - Step editors that use configuration system patterns
- :doc:`pipeline_compilation_system` - How configuration flows through compilation phases
- :doc:`../development/ui-patterns` - UI patterns that leverage configuration architecture

Archived Documentation
-----------------------

The following documents were consolidated into this unified architecture document:

- ``configuration_management_system.rst`` - Hierarchical configuration flow and dataclass structures
- ``configuration-resolution.rst`` - Thread-local context and resolution hierarchy
- ``lazy-class-system.rst`` - Dynamic dataclass generation with lazy resolution
- ``field-path-detection.rst`` - Automatic field path discovery for configs

These archived documents are available in ``docs/source/architecture/archive/`` for reference.
