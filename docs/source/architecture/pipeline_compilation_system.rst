Pipeline Compilation System Architecture
========================================

Overview
--------

OpenHCS implements a declarative, compile-time pipeline system that
treats configuration as a first-class compilation target. This
architecture separates pipeline definition from execution, enabling
compile-time validation, resource optimization, and reproducible
execution.

**Note**: Function patterns shown use real OpenHCS API with function
objects and parameter tuples, matching TUI-generated script patterns.

Core Philosophy
---------------

The system is designed around three fundamental principles:

1. **Declaration Phase**: Functions declare their contracts via
   decorators
2. **Compilation Phase**: Multi-pass compiler builds execution plans
3. **Execution Phase**: Stateless execution against immutable contexts

This approach is analogous to a programming language compiler, but for
data processing pipelines.

Multi-Pass Compiler Architecture
--------------------------------

The pipeline compiler operates in five sequential phases, each building
upon the previous:

Phase 1: Step Plan Initialization (``initialize_step_plans_for_context``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Establishes the data flow topology and initializes step plans

-  **Input**: Step definitions + ProcessingContext (well_id, input_dir)
-  **Output**: Initialized ``step_plans`` with input/output directories and
   special I/O paths
-  **Responsibilities**:

   -  Creates basic step plan structure for each step
   -  Calls ``PipelinePathPlanner.prepare_pipeline_paths()`` for path resolution
   -  Determines input/output directories for each step
   -  Creates VFS paths for special I/O (cross-step communication)
   -  Links special outputs from one step to special inputs of another
   -  Handles chain breaker logic and input source detection

**Key Error**:
``"Context step_plans must be initialized before path planning"`` -
Indicates this phase failed to properly initialize the step_plans
structure

Phase 2: ZARR Store Declaration (``declare_zarr_stores_for_context``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Declares ZARR stores for steps that will use ZARR backend

-  **Input**: ``step_plans`` from Phase 1 with path information
-  **Output**: ZARR store declarations in step plans (``zarr_config`` entries)
-  **Responsibilities**:

   -  Identifies steps that will use ZARR materialization backend
   -  Declares ZARR stores with appropriate configuration metadata
   -  Sets up well coordination information for multi-well ZARR stores
   -  Provides configuration for runtime store creation

**Key Insight**: This phase doesn't create ZARR stores - it declares
which steps will need them and provides the metadata for runtime store
creation. The actual ZARR stores are created during execution when the
first well writes data.

Phase 3: Materialization Planning (``plan_materialization_flags_for_context``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Decides where data lives (VFS backend strategy)

-  **Input**: ``step_plans`` from Phase 2 with ZARR store declarations
-  **Output**: Backend selection (disk vs memory) for each step
-  **Strategy**:

   -  First step: Always reads from disk (input images)
   -  Last step: Always writes to disk (final outputs)
   -  Middle steps: Can use memory backend for speed
   -  FunctionSteps: Can use intermediate backends
   -  Non-FunctionSteps: Must use persistent backends

Phase 4: Memory Contract Validation (``validate_memory_contracts_for_context``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Ensures memory type compatibility across the pipeline AND
stores function patterns

-  **Input**: ``step_plans`` + function decorators + modified function
   patterns from path planner
-  **Output**: Memory type validation, function pattern storage, and
   injection into step_plans
-  **Key Responsibility**: **Stores the function pattern (potentially
   modified by path planner) in ``step_plans['func']``**
-  **Implementation**: Uses ``FuncStepContractValidator`` internally
-  **Validation**:

   -  All functions must have explicit memory type declarations
   -  Functions in the same step must have consistent memory types
   -  Memory types must be valid (numpy, cupy, torch, tensorflow, jax)

-  **Storage**: Returns
   ``{'input_memory_type': ..., 'output_memory_type': ..., 'func': func_pattern}``

Phase 5: GPU Resource Assignment (``assign_gpu_resources_for_context``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Resource allocation for GPU-accelerated steps

-  **Input**: ``step_plans`` with memory types
-  **Output**: GPU device assignments
-  **Implementation**: Uses ``GPUMemoryTypeValidator`` internally
-  **Logic**:

   -  Identifies steps requiring GPU memory types
   -  Assigns available GPU devices
   -  Validates GPU resource availability

Function Pattern System
-----------------------

The Sacred Four Patterns
~~~~~~~~~~~~~~~~~~~~~~~~

OpenHCS supports four fundamental function execution patterns that
provide unified handling of different processing strategies:

1. Single Function Pattern
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   from openhcs.processing.backends.processors.cupy_processor import tophat
   FunctionStep(func=(tophat, {'selem_radius': 50}))

-  **Use Case**: Apply function with parameters to all data
-  **Execution**: ``tophat(image_stack, selem_radius=50)`` for each
   pattern group

2. Sequential Function Chain
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   from openhcs.processing.backends.processors.cupy_processor import stack_percentile_normalize, tophat
   FunctionStep(func=[
       (stack_percentile_normalize, {'low_percentile': 1.0}),
       (tophat, {'selem_radius': 50})
   ])

-  **Use Case**: Apply multiple functions in sequence
-  **Execution**:
   ``tophat(stack_percentile_normalize(image_stack, low_percentile=1.0), selem_radius=50)``
   for each pattern group

3. Component-Specific Functions (Dict Pattern)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   from openhcs.processing.backends.analysis.cell_counting_cpu import count_cells_single_channel
   from openhcs.processing.backends.analysis.skan_axon_analysis import skan_axon_skeletonize_and_analyze
   FunctionStep(func={
       '1': (count_cells_single_channel, {'min_sigma': 1.0}),
       '2': skan_axon_skeletonize_and_analyze  # No parameters needed
   })

-  **Use Case**: Different processing per component (channel, site,
   etc.)
-  **Execution**: ``count_cells_single_channel`` for channel 1 data,
   ``skan_axon_skeletonize_and_analyze`` for channel 2 data

4. Bare Function Pattern (Simple Cases)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   from openhcs.processing.backends.assemblers.assemble_stack_cupy import assemble_stack_cupy
   FunctionStep(func=assemble_stack_cupy)

-  **Use Case**: Function with no parameters
-  **Execution**: ``assemble_stack_cupy(image_stack)`` for each pattern
   group

Pattern Resolution Flow
~~~~~~~~~~~~~~~~~~~~~~~

1. **Pattern Detection**: ``microscope_handler.auto_detect_patterns()``
   finds image files matching well/component criteria
2. **Pattern Grouping**: ``prepare_patterns_and_functions()`` groups
   patterns by component and resolves func patterns
3. **Execution**: For each pattern group: load images → stack → process
   → unstack → save

Decorator System
----------------

Memory Type Decorators
~~~~~~~~~~~~~~~~~~~~~~

Functions declare their memory interface using decorators:

.. code:: python

   @torch(input_type="torch", output_type="torch")
   def my_function(image_stack):
       return processed_stack

   @numpy  # Shorthand for numpy input/output
   def another_function(data):
       return result

**Supported Memory Types**: ``numpy``, ``cupy``, ``torch``,
``tensorflow``, ``jax``, ``pyclesperanto``

**Benefits**: - No runtime overhead - pure metadata - Enables
compile-time memory type checking - Supports automatic memory type
conversion planning

Special I/O Decorators
~~~~~~~~~~~~~~~~~~~~~~

Functions declare cross-step dependencies:

.. code:: python

   @special_outputs("positions", "metadata")
   def generate_positions(image_stack):
       return processed_stack, positions, metadata

   @special_inputs("positions")
   def stitch_images(image_stack, positions):
       return stitched_stack

**Compiler Behavior**: - Automatically links outputs to inputs - Creates
VFS paths for intermediate data - Validates dependency chains at compile
time

Chain Breaker Decorator
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   @chain_breaker
   def independent_function(image_stack):
       return result

Forces the next step to read from the pipeline’s original input
directory rather than the previous step’s output.

Virtual File System (VFS)
-------------------------

Abstraction Layer
~~~~~~~~~~~~~~~~~

The VFS provides a unified interface for all storage operations:

.. code:: python

   # Same API regardless of backend
   filemanager.save(data, "path/to/data", "memory")
   filemanager.save(data, "path/to/data", "disk")
   data = filemanager.load("path/to/data", "memory")

Backend Types
~~~~~~~~~~~~~

-  **Memory Backend**: Fast intermediate data (numpy arrays, tensors)
-  **Disk Backend**: Persistent data (images, final outputs)
-  **Zarr Backend**: Chunked array storage (future)

Location Transparency
~~~~~~~~~~~~~~~~~~~~~

Data can be moved between backends without changing application code.
The materialization planner decides optimal storage locations based on:
- Step position in pipeline - Step type (FunctionStep vs others) -
Resource constraints - Performance requirements

ProcessingContext Lifecycle
---------------------------

1. Creation
~~~~~~~~~~~

.. code:: python

   context = ProcessingContext(
       global_config=config,
       well_id="A01",
       filemanager=filemanager
   )

2. Population (Compilation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Phase 1: Path planning
   PipelinePathPlanner.prepare_pipeline_paths(context, steps)

   # Phase 2: Materialization planning  
   MaterializationFlagPlanner.prepare_pipeline_flags(context, steps)

   # Phase 3: Memory contract validation + function pattern storage
   memory_types = FuncStepContractValidator.validate_pipeline(steps, context)
   # memory_types includes: input_memory_type, output_memory_type, AND func
   # Inject memory types AND function patterns into context.step_plans
   for step_id, types_and_func in memory_types.items():
       context.step_plans[step_id].update(types_and_func)  # Includes 'func' key!

   # Phase 4: GPU resource assignment
   GPUMemoryTypeValidator.validate_step_plans(context.step_plans)

3. Freezing
~~~~~~~~~~~

.. code:: python

   context.freeze()  # Makes context immutable

4. Execution
~~~~~~~~~~~~

.. code:: python

   for step in steps:
       step.process(context)  # Read-only access to frozen context

Step Plans Structure
--------------------

Each step gets a comprehensive execution plan:

.. code:: python

   context.step_plans[step_id] = {
       # Basic metadata
       "step_name": "Z-Stack Flattening",
       "step_type": "FunctionStep",
       "well_id": "A01",

       # I/O configuration
       "input_dir": "/path/to/input",
       "output_dir": "/path/to/output",
       "read_backend": "disk",
       "write_backend": "memory",

       # Memory configuration
       "input_memory_type": "numpy",
       "output_memory_type": "torch",
       "gpu_id": 0,

       # Function pattern (CRITICAL: stored by FuncStepContractValidator)
       "func": function_pattern,  # The actual function pattern (potentially modified by path planner)

       # Special I/O
       "special_inputs": {
           "positions": {"path": "/vfs/positions.pkl", "backend": "memory"}
       },
       "special_outputs": {
           "metadata": {"path": "/vfs/metadata.pkl", "backend": "memory"}
       },

       # Flags
       "force_disk_output": False,
       "visualize": False
   }

Execution Model
---------------

Stateless Steps
~~~~~~~~~~~~~~~

After compilation, step objects become pure templates: - All
configuration lives in ``context.step_plans[step_id]`` - Same step
definition reused across wells with different configs - Functional
programming approach to pipeline execution

VFS-Based Data Flow
~~~~~~~~~~~~~~~~~~~

-  No direct data passing between steps
-  All data flows through VFS paths specified in step_plans
-  Location transparency: data can be in memory or on disk
-  Automatic serialization/deserialization based on backend

Benefits of This Architecture
-----------------------------

1. **Compile-Time Safety**: Catch errors before expensive execution
2. **Resource Optimization**: Global view enables smart resource
   allocation
3. **Reproducibility**: Immutable contexts ensure consistent results
4. **Scalability**: Stateless execution enables easy parallelization
5. **Debuggability**: Can inspect and modify plans before execution
6. **Flexibility**: VFS abstraction allows different storage strategies
7. **Performance**: Memory-aware planning optimizes data movement

Error Handling
--------------

The system is designed to **fail fast** during compilation rather than
during execution:

-  Missing memory type declarations → Compilation error
-  Incompatible memory types → Compilation error
-  Missing special input dependencies → Compilation error
-  Invalid step plan structure → Compilation error

This approach prevents expensive pipeline failures after processing has
begun.

See Also
--------

**Core Integration**:

- :doc:`function_pattern_system` - Function patterns compiled by this system
- :doc:`memory_type_system` - Memory type validation and conversion
- :doc:`special_io_system` - Cross-step communication compilation

**Practical Usage**:

- :doc:`../guides/pipeline_compilation_workflow` - Complete compilation workflow guide
- :doc:`../api/orchestrator` - PipelineOrchestrator compilation integration
- :doc:`../concepts/pipeline` - Basic pipeline concepts

**Advanced Topics**:

- :doc:`compilation_system_detailed` - Deep dive into 5-phase compilation
- :doc:`pipeline_debugging_guide` - Debugging compilation issues
- :doc:`gpu_resource_management` - GPU resource assignment details
