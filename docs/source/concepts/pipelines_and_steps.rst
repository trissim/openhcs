Pipelines and Steps
===================

OpenHCS organizes image processing through two fundamental concepts: pipelines and steps. Understanding these building blocks is essential for creating effective analysis workflows.

What is a Pipeline?
------------------

A pipeline in OpenHCS is simply a Python list of processing steps that execute in sequence. Each step processes data and passes results to the next step, creating an automated workflow that runs across all your experimental data.

.. code-block:: python

   from openhcs.core.steps.function_step import FunctionStep
   from openhcs.core.pipeline import Pipeline

   # A pipeline is a list of steps
   my_pipeline = Pipeline([
       FunctionStep(func=normalize_images, name="normalize"),
       FunctionStep(func=segment_cells, name="segment"), 
       FunctionStep(func=measure_features, name="measure")
   ])

**Key Characteristics**:

- **Sequential execution**: Steps run in the order you define them
- **Automatic data flow**: Output from one step becomes input to the next
- **Parallel processing**: Each step processes multiple wells/sites simultaneously
- **Declarative**: You describe what to do, not how to do it

What is a Step?
--------------

A step is a ``FunctionStep`` - the basic processing unit in OpenHCS. Each step contains one or more functions to execute and defines how data should be organized for processing.

.. code-block:: python

   from openhcs.constants.constants import VariableComponents

   # A step that normalizes images, processing each site separately
   normalize_step = FunctionStep(
       func=(stack_percentile_normalize, {
           'low_percentile': 1.0,
           'high_percentile': 99.0
       }),
       name="normalize",
       variable_components=[VariableComponents.SITE]
   )

**Step Components**:

- **func**: The function(s) to execute (can be single function, list, or dictionary)
- **name**: Human-readable identifier for the step
- **variable_components**: Defines how data is grouped for processing
- **Additional parameters**: Passed directly to the function(s)

Why These Abstractions vs Regular Python Scripts?
------------------------------------------------

OpenHCS pipelines provide significant advantages over traditional custom scripts:

Automatic Parallelization
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Traditional approach - manual loop
   for well in wells:
       for site in sites:
           result = process_image(load_image(well, site))
           save_result(result, well, site)

   # OpenHCS approach - automatic parallelization
   step = FunctionStep(
       func=(process_image, {}),
       variable_components=[VariableComponents.SITE]
   )
   # Automatically processes all wells and sites in parallel

**Benefits**: OpenHCS automatically discovers all wells and sites in your data and processes them in parallel across multiple CPU cores, without requiring manual coordination.

Memory Management
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Traditional approach - manual memory management
   try:
       gpu_data = cupy.asarray(cpu_data)
       result = gpu_function(gpu_data)
       cpu_result = cupy.asnumpy(result)
   except cupy.cuda.memory.OutOfMemoryError:
       # Manual fallback to CPU
       cpu_result = cpu_function(cpu_data)

   # OpenHCS approach - automatic memory management
   step = FunctionStep(func=(gpu_function, {}))  # Automatic GPU memory management

**Benefits**: OpenHCS automatically handles GPU memory allocation, transfers between CPU and GPU, and fallback to CPU when GPU memory is exhausted.

Format Abstraction
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Traditional approach - format-specific code
   if microscope_type == "ImageXpress":
       files = glob.glob(f"{plate_dir}/TimePoint_1/**/A01_s1_w*.tif")
   elif microscope_type == "OperaPhenix":
       files = glob.glob(f"{plate_dir}/Images/r01c01f*-ch*.tiff")
   # ... different logic for each format

   # OpenHCS approach - format-agnostic
   step = FunctionStep(func=(process_image, {}))  # Works with any format

**Benefits**: OpenHCS automatically detects microscope formats and handles file discovery, so your analysis code works with data from any supported microscope.

Reproducibility
~~~~~~~~~~~~~~

.. code-block:: python

   # Traditional approach - hardcoded parameters
   def analyze_cells():
       threshold = 0.5  # Hardcoded
       min_size = 100   # Hardcoded
       # ... analysis logic

   # OpenHCS approach - declarative configuration
   step = FunctionStep(
       func=(analyze_cells, {
           'threshold': 0.5,
           'min_size': 100
       }),
       name="cell_analysis"
   )

**Benefits**: All parameters are explicitly declared and can be easily modified, shared, or systematically varied without changing code.

Scalability
~~~~~~~~~~

.. code-block:: python

   # Traditional approach - limited by single machine
   for image_file in image_files:
       result = process_large_image(image_file)  # May exceed memory

   # OpenHCS approach - scalable processing
   step = FunctionStep(
       func=(process_large_image, {}),
       variable_components=[VariableComponents.SITE]
   )
   # Automatically chunks data and manages memory usage

**Benefits**: OpenHCS handles large datasets that exceed available memory by processing data in chunks and using efficient storage backends.

Pipeline Execution Model
------------------------

When you run a pipeline, OpenHCS follows a systematic execution model:

1. **Discovery**: Automatically finds all wells, sites, channels in your data
2. **Compilation**: Converts your pipeline definition into an optimized execution plan
3. **Parallel Execution**: Processes multiple wells simultaneously across CPU cores
4. **Memory Management**: Automatically handles data loading, GPU transfers, and storage
5. **Result Organization**: Saves outputs in organized directory structure

.. code-block:: python

   from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator

   # Define pipeline
   pipeline = Pipeline([
       FunctionStep(func=preprocess, name="preprocess"),
       FunctionStep(func=analyze, name="analyze"),
       FunctionStep(func=assemble, name="assemble")
   ])

   # Execute across entire dataset
   orchestrator = PipelineOrchestrator(
       plate_path="/path/to/microscope/data",
       global_config=config
   )
   orchestrator.run_pipeline(pipeline)

This execution model handles all the complexity of parallel processing, memory management, and file organization, allowing you to focus on defining the analysis logic rather than managing infrastructure.

Pipeline Orchestrator Execution Details
---------------------------------------

The PipelineOrchestrator is the central execution engine that manages the entire pipeline workflow through a sophisticated compilation and execution process.

Compilation Process
~~~~~~~~~~~~~~~~~~

The orchestrator transforms your pipeline definition into an optimized execution plan through a 5-phase compilation process:

1. **Step Plan Initialization**: Creates a basic plan for each step, resolving input/output paths within the VFS
2. **ZARR Store Declaration**: If Zarr is the materialization backend, declares necessary Zarr stores
3. **Materialization Planning**: Determines which steps require output written to persistent storage
4. **Memory Validation**: Checks memory requirements against available system resources
5. **GPU Assignment**: Assigns specific GPU devices to processing tasks for balanced utilization

.. code-block:: python

   # The orchestrator handles compilation automatically
   orchestrator = PipelineOrchestrator(plate_path, global_config=config)

   # Three-phase execution workflow
   orchestrator.initialize()                                    # Environment setup
   compiled_contexts = orchestrator.compile_pipelines(pipeline) # 5-phase compilation
   results = orchestrator.execute_compiled_plate(              # Parallel execution
       pipeline_definition=pipeline,
       compiled_contexts=compiled_contexts,
       max_workers=config.num_workers
   )

Parallel Execution Model
~~~~~~~~~~~~~~~~~~~~~~~

The orchestrator executes pipelines with sophisticated resource management:

- **Multi-well parallelization**: Processes multiple wells simultaneously across worker processes
- **GPU resource management**: Automatically assigns and balances GPU devices
- **Memory optimization**: Manages memory usage across parallel workers
- **Error handling**: Provides detailed error reporting and recovery mechanisms

The pipeline approach scales from simple single-step processing to complex multi-stage analysis workflows, providing a consistent framework that grows with your analysis needs.
