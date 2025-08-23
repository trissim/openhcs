Configuration Management System
===============================

Overview
--------

OpenHCS implements a hierarchical configuration system that flows from
global application settings down to individual step execution. The
system uses immutable dataclasses to ensure configuration consistency
and provides structured access patterns throughout the pipeline
lifecycle.

**Note**: All configuration examples reflect the actual OpenHCS
implementation and are verified against the current codebase.

Configuration Architecture
--------------------------

Configuration Hierarchy
~~~~~~~~~~~~~~~~~~~~~~~

::

   GlobalPipelineConfig (Root)
   ├── VFSConfig (Virtual File System settings)
   ├── PathPlanningConfig (Directory naming conventions)
   ├── ZarrConfig (Zarr storage backend settings)
   ├── num_workers (Parallel execution settings)
   └── microscope (Default microscope type)

Configuration Dataclasses
~~~~~~~~~~~~~~~~~~~~~~~~~

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
   class GlobalPipelineConfig:
       """Root configuration object for OpenHCS pipeline session."""
       num_workers: int = field(default_factory=lambda: os.cpu_count() or 1)
       """Number of worker processes/threads for parallelizable tasks."""
       
       path_planning: PathPlanningConfig = field(default_factory=PathPlanningConfig)
       """Configuration for path planning (directory suffixes)."""
       
       vfs: VFSConfig = field(default_factory=VFSConfig)
       """Configuration for Virtual File System behavior."""

       microscope: Microscope = Microscope.AUTO
       """Default microscope type for auto-detection."""

Configuration Flow
------------------

1. Application Startup
~~~~~~~~~~~~~~~~~~~~~~

Configuration is initialized at application startup:

.. code:: python

   # Default configuration
   def get_default_global_config() -> GlobalPipelineConfig:
       """Provides default configuration instance."""
       logger.info("Initializing with default GlobalPipelineConfig.")
       return GlobalPipelineConfig(
           path_planning=PathPlanningConfig(),
           vfs=VFSConfig()
           # num_workers uses field default_factory
       )

   # Application initialization
   global_config = get_default_global_config()
   # OR
   global_config = GlobalPipelineConfig(
       num_workers=8,
       vfs=VFSConfig(default_intermediate_backend="memory"),
       path_planning=PathPlanningConfig(output_dir_suffix="_processed")
   )

2. Orchestrator Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The orchestrator receives and stores the global configuration:

.. code:: python

   class PipelineOrchestrator:
       """Pipeline orchestrator with configuration management."""
       
       def __init__(self, plate_path, workspace_path=None, *, global_config=None):
           # Configuration initialization
           if global_config is None:
               self.global_config = get_default_global_config()
               logger.info("Using default global configuration.")
           else:
               self.global_config = global_config
           
           # Store configuration for context creation
           self.plate_path = plate_path
           self.workspace_path = workspace_path
       
       def create_context(self, well_id: str) -> ProcessingContext:
           """Create context with global configuration."""
           context = ProcessingContext(
               global_config=self.global_config,  # Pass config to context
               well_id=well_id,
               filemanager=self.filemanager
           )
           # Add orchestrator-specific attributes
           context.orchestrator = self
           context.input_dir = self.input_dir
           return context

3. Context Configuration Access
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ProcessingContext provides structured access to configuration:

.. code:: python

   class ProcessingContext:
       """Processing context with configuration access."""
       
       def __init__(self, global_config: GlobalPipelineConfig, **kwargs):
           self.global_config = global_config  # Store global config
           self.step_plans = {}
           self.well_id = kwargs.get('well_id')
           # ... other initialization
       
       def get_vfs_config(self) -> VFSConfig:
           """Get VFS configuration."""
           if not hasattr(self, 'global_config') or self.global_config is None:
               raise RuntimeError("GlobalPipelineConfig not set on ProcessingContext.")
           return self.global_config.vfs
       
       def get_path_planning_config(self) -> PathPlanningConfig:
           """Get path planning configuration."""
           if not hasattr(self, 'global_config') or self.global_config is None:
               raise RuntimeError("GlobalPipelineConfig not set on ProcessingContext.")
           return self.global_config.path_planning
       
       def get_num_workers(self) -> int:
           """Get number of workers configuration."""
           return self.global_config.num_workers

Configuration Usage in Compilation
----------------------------------

Path Planning Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PathPlanner uses configuration for directory naming:

.. code:: python

   class PipelinePathPlanner:
       """Path planner with configuration-driven behavior."""
       
       @staticmethod
       def prepare_pipeline_paths(context: ProcessingContext, pipeline_definition):
           """Prepare paths using configuration."""
           
           # Access configuration through context
           path_config = context.get_path_planning_config()
           
           # Use configuration for directory naming
           for i, step in enumerate(pipeline_definition):
               step_id = step.uid
               
               # Generate output directory with configured suffix
               if i == 0:  # First step
                   base_output_dir = Path(context.input_dir).parent
                   step_output_dir = base_output_dir / f"{context.well_id}{path_config.output_dir_suffix}"
               else:
                   # Subsequent steps use previous step's output
                   prev_step_id = pipeline_definition[i-1].uid
                   prev_output = context.step_plans[prev_step_id]["output_dir"]
                   step_output_dir = Path(prev_output).parent / f"{context.well_id}_step{i:02d}{path_config.output_dir_suffix}"
               
               # Store in step plan
               context.step_plans[step_id] = {
                   "output_dir": str(step_output_dir),
                   # ... other path information
               }

VFS Configuration
~~~~~~~~~~~~~~~~~

The MaterializationFlagPlanner uses VFS configuration for backend
selection:

.. code:: python

   class MaterializationFlagPlanner:
       """Materialization planner with VFS configuration."""
       
       @staticmethod
       def plan_materialization_flags(context: ProcessingContext, pipeline_definition):
           """Plan materialization using VFS configuration."""
           
           # Access VFS configuration
           vfs_config = context.get_vfs_config()
           
           for step in pipeline_definition:
               step_plan = context.step_plans[step.uid]
               
               # Determine read backend
               # Use configured intermediate backend for all steps except first
               step_plan["read_backend"] = vfs_config.default_intermediate_backend

               # Determine write backend
               if step_plan.get("force_disk_output", False):
                   # Use configured materialization backend
                   step_plan["write_backend"] = vfs_config.default_materialization_backend
               else:
                   # Use configured intermediate backend
                   step_plan["write_backend"] = vfs_config.default_intermediate_backend
               
               logger.debug(
                   f"Step {step.name}: read_backend={step_plan['read_backend']}, "
                   f"write_backend={step_plan['write_backend']}"
               )

Parallel Execution Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The orchestrator uses num_workers for parallel execution:

.. code:: python

   def execute_compiled_plate(self, pipeline_definition, compiled_contexts, 
                             max_workers=None, visualizer=None):
       """Execute with configuration-driven parallelism."""
       
       # Use configured number of workers if not overridden
       actual_max_workers = max_workers if max_workers is not None else self.global_config.num_workers
       
       if actual_max_workers <= 0:
           actual_max_workers = 1
       
       logger.info(f"Starting execution with max_workers={actual_max_workers}")
       
       if actual_max_workers > 1 and len(compiled_contexts) > 1:
           # Parallel execution using configured worker count
           with concurrent.futures.ThreadPoolExecutor(max_workers=actual_max_workers) as executor:
               # ... parallel execution logic
       else:
           # Sequential execution
           # ... sequential execution logic

Live Configuration Updates
--------------------------

Configuration Replacement
~~~~~~~~~~~~~~~~~~~~~~~~~

The orchestrator supports live configuration updates:

.. code:: python

   async def apply_new_global_config(self, new_config: GlobalPipelineConfig):
       """Apply new configuration to running orchestrator."""
       
       if not isinstance(new_config, GlobalPipelineConfig):
           logger.error(f"Invalid config type {type(new_config)}")
           return
       
       logger.info(
           f"Applying new config. Old num_workers: {self.global_config.num_workers}, "
           f"New num_workers: {new_config.num_workers}"
       )
       
       # Replace configuration
       self.global_config = new_config
       
       # Note: Components that cache config values would need explicit updates
       # Currently, most components access config through context on each use
       
       logger.info("New GlobalPipelineConfig applied.")

Configuration Propagation Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The system uses **pull-based configuration access** rather than
push-based updates:

.. code:: python

   # ✅ Pull-based (current approach)
   def some_planner_method(context):
       vfs_config = context.get_vfs_config()  # Always gets current config
       backend = vfs_config.default_intermediate_backend
       # ... use backend

   # ❌ Push-based (not used)
   class SomePlanner:
       def __init__(self, vfs_config):
           self.cached_backend = vfs_config.default_intermediate_backend  # Stale if config changes

**Benefits of Pull-Based Approach**: - No need to update cached values
when configuration changes - Always uses current configuration - Simpler
to implement and maintain - Natural immutability enforcement

Configuration Validation
------------------------

Compile-Time Validation
~~~~~~~~~~~~~~~~~~~~~~~

Configuration is validated during compilation:

.. code:: python

   def validate_configuration(global_config: GlobalPipelineConfig):
       """Validate configuration consistency."""
       
       # Validate VFS configuration
       vfs_config = global_config.vfs
       
       # Check backend compatibility
       if vfs_config.default_intermediate_backend not in ["memory", "disk", "zarr"]:
           raise ValueError(f"Invalid intermediate backend: {vfs_config.default_intermediate_backend}")
       
       if vfs_config.default_materialization_backend not in ["disk", "zarr"]:
           raise ValueError(f"Invalid materialization backend: {vfs_config.default_materialization_backend}")
       
       # Validate path planning configuration
       path_config = global_config.path_planning
       
       # Check suffix validity
       for suffix_name, suffix_value in [
           ("output_dir_suffix", path_config.output_dir_suffix),
           ("positions_dir_suffix", path_config.positions_dir_suffix),
           ("stitched_dir_suffix", path_config.stitched_dir_suffix)
       ]:
           if not suffix_value or not isinstance(suffix_value, str):
               raise ValueError(f"Invalid {suffix_name}: {suffix_value}")
       
       # Validate worker count
       if global_config.num_workers <= 0:
           raise ValueError(f"Invalid num_workers: {global_config.num_workers}")

Runtime Configuration Checks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   def check_configuration_compatibility(context: ProcessingContext):
       """Check configuration compatibility during execution."""
       
       vfs_config = context.get_vfs_config()
       
       # Check if configured backends are available
       available_backends = context.filemanager.get_available_backends()
       
       if vfs_config.default_intermediate_backend not in available_backends:
           raise RuntimeError(
               f"Configured intermediate backend '{vfs_config.default_intermediate_backend}' "
               f"not available. Available: {available_backends}"
           )
       
       if vfs_config.default_materialization_backend not in available_backends:
           raise RuntimeError(
               f"Configured materialization backend '{vfs_config.default_materialization_backend}' "
               f"not available. Available: {available_backends}"
           )

Configuration Override Mechanisms
---------------------------------

Step-Level Overrides
~~~~~~~~~~~~~~~~~~~~

Individual steps can override global configuration:

.. code:: python

   class FunctionStep(AbstractStep):
       """Function step with configuration override capability."""
       
       def __init__(self, *, force_disk_output=False, **kwargs):
           super().__init__(**kwargs)
           self.force_disk_output = force_disk_output  # Override global VFS config
       
       def process(self, context):
           """Process with potential configuration overrides."""
           step_plan = context.get_step_plan(self.step_id)
           
           # Step-level override takes precedence
           if self.force_disk_output:
               # Override global VFS configuration
               step_plan["write_backend"] = "disk"
               step_plan["force_disk_output"] = True
           
           # ... continue processing

Compilation-Time Overrides
~~~~~~~~~~~~~~~~~~~~~~~~~~

The compiler can apply global overrides:

.. code:: python

   def apply_global_visualizer_override_for_context(context, global_enable_visualizer):
       """Apply global override to all steps."""
       
       if global_enable_visualizer:
           for step_id, plan in context.step_plans.items():
               plan["visualize"] = True  # Override step-level visualization settings
               logger.info(f"Global override: Step '{plan.get('step_name', step_id)}' marked for visualization")

Current Implementation Status
-----------------------------

Implemented Features
~~~~~~~~~~~~~~~~~~~~

1. **Hierarchical Configuration**: Fully implemented with lazy resolution chains
2. **Configuration Inheritance**: StepMaterializationConfig inherits from PathPlanningConfig
3. **Plugin Configuration**: Metaprogramming-based plugin system already exists
4. **Thread-Local Configuration**: Context-aware configuration resolution
5. **Lazy Configuration**: Automatic resolution through inheritance chains
6. **Configuration Validation**: Dataclass-based validation with type hints

The configuration system is feature-complete for current OpenHCS requirements. All configuration is handled programmatically through dataclass instantiation with comprehensive lazy resolution and inheritance support.
