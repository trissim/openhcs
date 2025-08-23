Field Path Detection System
============================

Automatic field path detection using type introspection to eliminate hardcoded field mappings and provide a single source of truth for type discovery across the UI system.

Overview
--------

The field path detection system solves a fundamental problem in configuration management: how do you automatically discover relationships between dataclass types without hardcoding field names?

OpenHCS has complex configuration hierarchies where different dataclass types need to reference each other. Traditional approaches require manual mapping of type relationships, creating maintenance overhead and breaking when configuration structures change.

A type introspection system automatically discovers field paths by analyzing dataclass annotations. The system can find direct relationships, nested paths, and inheritance relationships without any hardcoded mappings.

This eliminated hardcoded field path mappings, enabled automatic lazy config generation, and provided foundation for UI patterns.

Automatic Field Path Determination
-----------------------------------

The system uses type introspection to automatically discover field relationships.

Basic Field Path Discovery
~~~~~~~~~~~~~~~~~~~~~~~~~~

The core functionality finds field paths by analyzing type annotations:

.. code-block:: python

    class FieldPathDetector:
        @staticmethod
        def find_field_path_for_type(parent_type: Type, child_type: Type) -> Optional[str]:
            """Find field path by inspecting parent type annotations."""
            
            if not dataclasses.is_dataclass(parent_type):
                return None
            
            # Analyze all fields in parent type
            for field in dataclasses.fields(parent_type):
                field_type = FieldPathDetector._unwrap_optional_type(field.type)
                
                # Direct type match
                if field_type == child_type:
                    return field.name
            
            return None

**Real-World Usage:**

.. code-block:: python

    # Automatic discovery replaces hardcoded mappings
    field_path = FieldPathDetector.find_field_path_for_type(
        GlobalPipelineConfig, StepMaterializationConfig
    )
    # Returns: "materialization_defaults"
    
    field_path = FieldPathDetector.find_field_path_for_type(
        GlobalPipelineConfig, VFSConfig
    )
    # Returns: "vfs"

Optional Type Unwrapping
~~~~~~~~~~~~~~~~~~~~~~~~

The system handles complex type annotations including Optional types:

.. code-block:: python

    @staticmethod
    def _unwrap_optional_type(field_type: Type) -> Type:
        """Convert Optional[T] -> T, Union[T, None] -> T, etc."""
        
        # Handle Optional types (Union[Type, None])
        if get_origin(field_type) is Union:
            args = get_args(field_type)
            if len(args) == 2 and type(None) in args:
                # Return the non-None type
                return args[0] if args[1] is type(None) else args[1]
        
        return field_type

**Type Annotation Examples:**

.. code-block:: python

    # All these annotations are handled automatically:
    
    @dataclass
    class PipelineConfig:
        vfs: VFSConfig                                    # Direct type
        materialization_defaults: StepMaterializationConfig  # Direct type
        optional_config: Optional[CustomConfig] = None   # Optional type
        union_config: Union[ConfigA, None] = None        # Union type

Comprehensive Path Discovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For complex scenarios, the system can find all instances of a type:

.. code-block:: python

    @staticmethod
    def find_all_field_paths_for_type(parent_type: Type, target_type: Type) -> list[str]:
        """Find ALL field paths that contain the target type."""
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

**Nested Path Discovery:**

.. code-block:: python

    # Find all instances of StepMaterializationConfig
    paths = FieldPathDetector.find_all_field_paths_for_type(
        GlobalPipelineConfig, StepMaterializationConfig
    )
    # Returns: ['materialization_defaults', 'nested.step_config', ...]

Dataclass Field Analysis
-------------------------

The system provides sophisticated analysis of dataclass structures and relationships.

Inheritance Relationship Discovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The system can automatically discover inheritance relationships:

.. code-block:: python

    @staticmethod
    def find_inheritance_relationships(target_type: Type) -> list[Type]:
        """Find all parent dataclasses that target_type inherits from."""
        inheritance_chain = []
        
        for base in target_type.__bases__:
            if base != object and dataclasses.is_dataclass(base):
                inheritance_chain.append(base)
                # Recursively find parent relationships
                inheritance_chain.extend(
                    FieldPathDetector.find_inheritance_relationships(base)
                )
        
        return inheritance_chain

**Inheritance Analysis Example:**

.. code-block:: python

    # Discover inheritance relationships automatically
    parents = FieldPathDetector.find_inheritance_relationships(StepMaterializationConfig)
    # Returns: [PathPlanningConfig] (if StepMaterializationConfig inherits from PathPlanningConfig)

Type Introspection Algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The system uses robust type introspection that handles edge cases:

.. code-block:: python

    # Handles complex type scenarios:
    
    # Generic types
    field_type: List[CustomConfig]
    field_type: Dict[str, ConfigType]
    
    # Nested Optional types
    field_type: Optional[List[Optional[ConfigType]]]
    
    # Forward references
    field_type: 'ForwardReferencedConfig'
    
    # Type variables
    field_type: TypeVar('T', bound=BaseConfig)

**Robust Error Handling:**

.. code-block:: python

    try:
        if not dataclasses.is_dataclass(parent_type):
            return None
        
        # Type introspection logic
        for field in dataclasses.fields(parent_type):
            # Safe type analysis
            field_type = FieldPathDetector._unwrap_optional_type(field.type)
            
    except Exception:
        # Fail gracefully for any type introspection issues
        return None

Integration with Lazy Dataclass Factory
----------------------------------------

The field path detection system integrates seamlessly with automatic lazy config generation.

Automatic Lazy Config Creation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Field paths discovered by the detection system are used to create lazy configs automatically:

.. code-block:: python

    # Automatic lazy config generation using detected field paths
    for field in dataclasses.fields(GlobalPipelineConfig):
        if dataclasses.is_dataclass(field.type):
            # Use field path detection to find the path
            field_path = FieldPathDetector.find_field_path_for_type(
                GlobalPipelineConfig, field.type
            )
            
            if field_path:
                # Create lazy config using detected path
                lazy_name = f"Lazy{field.type.__name__}"
                lazy_config = LazyDataclassFactory.make_lazy_with_field_level_auto_hierarchy(
                    base_class=field.type,
                    global_config_type=GlobalPipelineConfig,
                    field_path=field_path,  # Automatically detected
                    lazy_class_name=lazy_name
                )
                globals()[lazy_name] = lazy_config

Sibling Inheritance Path Discovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The system discovers sibling inheritance paths for complex resolution scenarios:

.. code-block:: python

    # Discover sibling inheritance paths automatically
    parent_types = FieldPathDetector.find_inheritance_relationships(base_class)
    sibling_paths = []
    
    for parent_type in parent_types:
        # Find all paths where parent types appear
        sibling_paths.extend(
            FieldPathDetector.find_all_field_paths_for_type(global_config_type, parent_type)
        )
    
    # Use sibling paths for inheritance hierarchy
    hierarchy_paths = [
        ('current', field_path),      # Direct path
        *[('current', path) for path in sibling_paths],  # Sibling paths
        ('global', field_path),       # Global direct
        *[('global', path) for path in sibling_paths]    # Global sibling
    ]

Context-Aware Path Resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Field paths are used in context-aware resolution scenarios:

.. code-block:: python

    def resolve_field_through_hierarchy(self, field_name, hierarchy_paths):
        """Resolve field using automatically detected paths."""
        
        for context_type, path in hierarchy_paths:
            if context_type == 'current':
                config = self.context_provider()
            else:
                config = get_current_global_config(self.global_config_type)
            
            # Navigate to instance using detected path
            instance = FieldPathNavigator.navigate_to_instance(config, path)
            if instance:
                value = getattr(instance, field_name, None)
                if value is not None:
                    return value
        
        return None

Elimination of Hardcoded Mappings
----------------------------------

The field path detection system completely eliminates the need for manual field mappings.

Before: Manual Mapping Approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Traditional approaches required extensive hardcoded mappings:

.. code-block:: python

    # Manual mapping - brittle and maintenance-heavy
    FIELD_PATH_MAPPINGS = {
        'StepMaterializationConfig': 'materialization_defaults',
        'VFSConfig': 'vfs',
        'PathPlanningConfig': 'path_planning',
        'ZarrConfig': 'zarr',
        # ... manual mapping for every type
    }

    def get_field_path(config_type):
        """Manual lookup - breaks when config structure changes."""
        return FIELD_PATH_MAPPINGS.get(config_type.__name__)

**Problems with Manual Mapping:**

1. **Maintenance Overhead**: Every new config type requires manual mapping
2. **Brittle**: Breaks when field names change in dataclass definitions
3. **Error-Prone**: Typos in mappings cause runtime failures
4. **Duplication**: Same mappings scattered across multiple files
5. **No Validation**: No way to verify mappings are correct

After: Automatic Discovery Approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The new system discovers relationships automatically:

.. code-block:: python

    # Automatic discovery - robust and maintenance-free
    def get_field_path(parent_type, child_type):
        """Automatic discovery - adapts to config structure changes."""
        return FieldPathDetector.find_field_path_for_type(parent_type, child_type)

    # Usage examples:
    path = get_field_path(GlobalPipelineConfig, StepMaterializationConfig)
    # Returns: "materialization_defaults" (discovered automatically)

    path = get_field_path(PipelineConfig, VFSConfig)
    # Returns: "vfs" (discovered automatically)

**Benefits of Automatic Discovery:**

1. **Zero Maintenance**: New config types work automatically
2. **Self-Updating**: Adapts when field names change
3. **Type-Safe**: Uses actual type annotations as source of truth
4. **Validated**: Fails fast if relationships don't exist
5. **Centralized**: Single discovery algorithm across entire system

Type Matching Algorithms
-------------------------

The system uses sophisticated algorithms for type discovery and matching.

Direct Type Matching
~~~~~~~~~~~~~~~~~~~~~

The primary algorithm matches types directly:

.. code-block:: python

    def _direct_type_match(field_type: Type, target_type: Type) -> bool:
        """Direct type equality check."""

        # Handle Optional types first
        unwrapped_type = FieldPathDetector._unwrap_optional_type(field_type)

        # Direct equality check
        if unwrapped_type == target_type:
            return True

        # String comparison for type identity issues
        if str(unwrapped_type) == str(target_type):
            return True

        return False

String-Based Type Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For complex type scenarios, string comparison provides robustness:

.. code-block:: python

    # Handle type identity issues with string comparison
    for field in dataclasses.fields(PipelineConfig):
        if str(field.type) == str(target_type):
            return field.name

**Why String Comparison:**

.. code-block:: python

    # Type identity can be tricky with dynamic imports
    from module_a import ConfigType as ConfigA
    from module_b import ConfigType as ConfigB

    # These might be the same type but fail identity check
    ConfigA == ConfigB  # Might be False
    str(ConfigA) == str(ConfigB)  # More reliable

Frame Inspection Support
~~~~~~~~~~~~~~~~~~~~~~~~

The system supports frame inspection for dynamic type discovery:

.. code-block:: python

    def discover_types_in_context():
        """Discover types available in current execution context."""
        import inspect

        # Get current frame
        frame = inspect.currentframe()
        try:
            # Analyze local and global variables for dataclass types
            local_vars = frame.f_locals
            global_vars = frame.f_globals

            discovered_types = []
            for var_dict in [local_vars, global_vars]:
                for name, value in var_dict.items():
                    if inspect.isclass(value) and dataclasses.is_dataclass(value):
                        discovered_types.append((name, value))

            return discovered_types
        finally:
            del frame

Recursive Type Analysis
~~~~~~~~~~~~~~~~~~~~~~~

For nested structures, the system uses recursive analysis:

.. code-block:: python

    def _analyze_nested_structure(dataclass_type: Type, max_depth: int = 5) -> Dict[str, Any]:
        """Recursively analyze nested dataclass structure."""

        if max_depth <= 0:
            return {}

        structure = {}

        for field in dataclasses.fields(dataclass_type):
            field_type = FieldPathDetector._unwrap_optional_type(field.type)

            if dataclasses.is_dataclass(field_type):
                # Recursive analysis of nested dataclass
                structure[field.name] = {
                    'type': field_type,
                    'nested': _analyze_nested_structure(field_type, max_depth - 1)
                }
            else:
                structure[field.name] = {
                    'type': field_type,
                    'nested': None
                }

        return structure

Real-World Usage Examples
-------------------------

These examples show how the field path detection system works in practice.

Example 1: Step Editor Parameter Mapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Step editor automatically maps parameters to pipeline fields
    def _find_pipeline_field_by_type(self, target_type):
        """Generic type-based discovery - no hardcoding."""

        # Use field path detection instead of manual mapping
        pipeline_field_name = FieldPathDetector.find_field_path_for_type(
            PipelineConfig, target_type
        )

        return pipeline_field_name

    # Usage in step editor:
    if self._is_optional_lazy_dataclass_in_pipeline(param_type, param_name):
        # Automatically find corresponding pipeline field
        pipeline_field = self._find_pipeline_field_by_type(inner_type)
        if pipeline_field:
            # Create step-level config with automatic inheritance
            step_config = self._create_step_level_config(param_name, param_type)

Example 2: Automatic Lazy Config Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Automatic generation of all lazy configs for GlobalPipelineConfig
    _step_lazy_configs = {}

    for field in dataclasses.fields(GlobalPipelineConfig):
        if dataclasses.is_dataclass(field.type):
            # Use field path detection to find the path
            field_path = FieldPathDetector.find_field_path_for_type(
                GlobalPipelineConfig, field.type
            )

            if field_path:
                lazy_name = f"Lazy{field.type.__name__}"
                lazy_config = LazyDataclassFactory.make_lazy_with_field_level_auto_hierarchy(
                    base_class=field.type,
                    global_config_type=GlobalPipelineConfig,
                    field_path=field_path,  # Automatically detected
                    lazy_class_name=lazy_name
                )
                _step_lazy_configs[lazy_name] = lazy_config
                globals()[lazy_name] = lazy_config

Example 3: Sibling Inheritance Discovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Discover sibling inheritance paths for StepMaterializationConfig
    parent_types = FieldPathDetector.find_inheritance_relationships(StepMaterializationConfig)
    # Returns: [PathPlanningConfig]

    sibling_paths = []
    for parent_type in parent_types:
        paths = FieldPathDetector.find_all_field_paths_for_type(
            GlobalPipelineConfig, parent_type
        )
        sibling_paths.extend(paths)
    # sibling_paths: ['path_planning']

    # Use for inheritance hierarchy
    hierarchy_paths = [
        ('current', 'materialization_defaults'),  # Direct path
        ('current', 'path_planning'),             # Sibling inheritance
        ('global', 'materialization_defaults'),   # Global direct
        ('global', 'path_planning')               # Global sibling
    ]

Benefits
--------

- **Zero Hardcoding**: Eliminates all hardcoded field path mappings
- **Automatic Discovery**: Finds relationships through type introspection
- **Robust Type Handling**: Handles Optional, Union, and complex type annotations
- **Inheritance Support**: Discovers inheritance relationships automatically
- **Nested Path Support**: Finds paths through nested dataclass structures
- **Integration Ready**: Seamlessly integrates with lazy config generation
- **Fail-Safe**: Graceful error handling for type introspection edge cases
- **Single Source of Truth**: Centralized type discovery logic across UI system
- **Maintenance-Free**: New config types work automatically without code changes
- **Type-Safe Discovery**: Uses actual type annotations as authoritative source
- **Performance Optimized**: Efficient algorithms for type matching and discovery

See Also
--------

- :doc:`step-editor-generalization` - Step editors that use field path detection for zero-hardcoding
- :doc:`lazy-class-system` - Lazy dataclass factory that uses field path detection
- :doc:`configuration-resolution` - Context resolution that works with detected field paths
- :doc:`service-layer-architecture` - Service layer patterns that leverage type discovery
- :doc:`../development/ui-patterns` - UI patterns that benefit from automatic field discovery
