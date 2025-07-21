Pipeline Compilation Workflow
=============================

OpenHCS uses a sophisticated 5-phase compilation system that transforms pipeline definitions into optimized execution plans. This guide explains how the compilation process works and how it integrates with the overall system.

Compilation Overview
--------------------

The pipeline compilation follows a multi-pass compiler architecture similar to programming language compilers:

1. **Declaration Phase**: Functions declare their contracts via decorators
2. **Compilation Phase**: 5-phase compiler builds execution plans  
3. **Execution Phase**: Stateless execution against immutable contexts

This approach enables powerful optimizations, validation, and resource management before any processing begins.

The 5 Compilation Phases
------------------------

The ``PipelineCompiler`` executes five sequential phases for each well:

.. code-block:: python

    # Actual compilation sequence from orchestrator
    for well_id in wells_to_process:
        context = self.create_context(well_id)

        # 5-Phase Compilation
        PipelineCompiler.initialize_step_plans_for_context(
            context, pipeline_definition, metadata_writer=is_responsible, plate_path=self.plate_path
        )
        PipelineCompiler.declare_zarr_stores_for_context(
            context, pipeline_definition, self
        )
        PipelineCompiler.plan_materialization_flags_for_context(
            context, pipeline_definition, self
        )
        PipelineCompiler.validate_memory_contracts_for_context(
            context, pipeline_definition, self
        )
        PipelineCompiler.assign_gpu_resources_for_context(context)

        context.freeze()
        compiled_contexts[well_id] = context

Phase 1: Step Plan Initialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Purpose**: Establishes the data flow topology and I/O paths

**Implementation**: ``PipelineCompiler.initialize_step_plans_for_context()``

**Key Operations**:
- Creates step plans for each pipeline step
- Calls ``PipelinePathPlanner.prepare_pipeline_paths()`` for path resolution
- Handles special I/O path linking between steps
- Sets up chainbreaker status for steps that break the pipeline flow

.. code-block:: python

    # Path planning example
    steps = [
        FunctionStep(func=normalize_images, name="normalize"),
        FunctionStep(func=segment_cells, name="segment"),
        FunctionStep(func=count_cells, name="count")
    ]

    # Phase 1 creates step plans:
    # step_plans = {
    #     "normalize_id": {
    #         "input_dir": "/data/plate/well_A01",
    #         "output_dir": "/memory/normalize_output",
    #         "special_inputs": {},
    #         "special_outputs": {}
    #     },
    #     "segment_id": {
    #         "input_dir": "/memory/normalize_output", 
    #         "output_dir": "/memory/segment_output",
    #         ...
    #     }
    # }

Phase 2: ZARR Store Declaration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Purpose**: Configures ZARR storage for large datasets

**Implementation**: ``PipelineCompiler.declare_zarr_stores_for_context()``

**Key Operations**:
- Identifies steps that require ZARR storage (large outputs, final results)
- Configures ZARR store parameters (compression, chunking)
- Sets up shared ZARR stores across wells for efficiency

.. code-block:: python

    # ZARR configuration for large datasets
    if will_use_zarr:
        step_plan["zarr_config"] = {
            "store_name": "images.zarr",
            "all_wells": all_wells,
            "needs_initialization": True,
            "compressor": "lz4",
            "chunk_strategy": "adaptive"
        }

Phase 3: Materialization Flag Planning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Purpose**: Determines read/write backends for each step

**Implementation**: ``PipelineCompiler.plan_materialization_flags_for_context()``

**Key Operations**:
- Sets read backends (disk for first step, memory for intermediate steps)
- Sets write backends (memory for intermediate, materialization backend for final)
- Handles backend compatibility with microscope formats

.. code-block:: python

    # Backend selection logic
    for i, step in enumerate(pipeline_definition):
        if i == 0:  # First step
            step_plan["read_backend"] = "disk"  # Read from microscope files
        else:
            step_plan["read_backend"] = "memory"  # Read from previous step
        
        if i == len(pipeline_definition) - 1:  # Last step
            step_plan["write_backend"] = "zarr"  # Final output
        else:
            step_plan["write_backend"] = "memory"  # Intermediate output

Phase 4: Memory Contract Validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Purpose**: Validates memory types and stores function patterns

**Implementation**: ``PipelineCompiler.validate_memory_contracts_for_context()``

**Key Operations**:
- Validates memory type compatibility between steps
- Stores resolved function patterns in step plans
- Injects memory type conversion information

.. code-block:: python

    # Memory contract validation
    memory_types = FuncStepContractValidator.validate_pipeline(steps, context)
    
    # Injects into step plans:
    # step_plan.update({
    #     "input_memory_type": "numpy",
    #     "output_memory_type": "torch", 
    #     "func": resolved_function_pattern
    # })

Phase 5: GPU Resource Assignment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Purpose**: Assigns GPU resources and validates GPU memory requirements

**Implementation**: ``PipelineCompiler.assign_gpu_resources_for_context()``

**Key Operations**:
- Assigns GPU devices to GPU-enabled steps
- Validates GPU memory requirements
- Sets up GPU resource scheduling

.. code-block:: python

    # GPU resource assignment
    for step_id, step_plan in context.step_plans.items():
        if step_plan["output_memory_type"] in VALID_GPU_MEMORY_TYPES:
            step_plan["gpu_id"] = assigned_gpu_id
            step_plan["gpu_memory_required"] = estimated_memory

Context Freezing and Immutability
----------------------------------

After compilation, contexts are frozen to ensure immutability during execution:

.. code-block:: python

    # After all compilation phases
    context.freeze()
    
    # Context becomes immutable - no further modifications allowed
    # Execution phase operates on frozen, validated contexts

Function Pattern Resolution
---------------------------

The compilation system resolves complex function patterns:

**Single Function Pattern**:

.. code-block:: python

    FunctionStep(func=normalize_images)
    # Resolves to: normalize_images(image_stack)

**Function with Parameters**:

.. code-block:: python

    FunctionStep(func=(normalize_images, {'percentile': 99.0}))
    # Resolves to: normalize_images(image_stack, percentile=99.0)

**Function Chain Pattern**:

.. code-block:: python

    FunctionStep(func=[
        (normalize_images, {'percentile': 99.0}),
        (apply_filter, {'sigma': 2.0})
    ])
    # Resolves to: apply_filter(normalize_images(image_stack, percentile=99.0), sigma=2.0)

**Dict Pattern (Channel-specific)**:

.. code-block:: python

    FunctionStep(func={
        '1': count_nuclei,
        '2': trace_neurites
    }, variable_components=[VariableComponents.CHANNEL])
    # Resolves to different functions per channel

Special I/O Integration
-----------------------

The compilation system handles cross-step communication:

**Special Outputs Declaration**:

.. code-block:: python

    @special_outputs("positions", "metadata")
    def generate_positions(image_stack):
        return processed_stack, positions, metadata

**Special Inputs Consumption**:

.. code-block:: python

    @special_inputs("positions")
    def assemble_images(image_tiles, positions):
        return assembled_image

**Compilation Integration**:

.. code-block:: python

    # Compilation links special I/O paths
    step_plans["generate_positions"]["special_outputs"] = {
        "positions": "/memory/positions_data",
        "metadata": "/memory/metadata_data"
    }
    
    step_plans["assemble_images"]["special_inputs"] = {
        "positions": "/memory/positions_data"
    }

Execution Phase Integration
---------------------------

Compiled contexts enable stateless execution:

.. code-block:: python

    # Execution retrieves everything from frozen context
    def process(self, context: ProcessingContext):
        step_id = get_step_id(self)
        step_plan = context.step_plans[step_id]
        
        # All execution parameters from compilation
        func_from_plan = step_plan['func']
        input_memory_type = step_plan['input_memory_type']
        output_memory_type = step_plan['output_memory_type']
        gpu_id = step_plan.get('gpu_id')
        
        # Execute with compiled parameters
        result = func_from_plan(input_data)

Performance Benefits
--------------------

The compilation system provides significant performance benefits:

**Validation at Compile Time**: Catch errors before processing begins
**Resource Optimization**: Optimal GPU and memory allocation
**Path Optimization**: Efficient I/O path planning
**Memory Type Planning**: Minimize conversions between memory types
**Parallel Compilation**: Compile multiple wells simultaneously

Error Handling
--------------

Compilation failures are caught early with detailed error messages:

.. code-block:: python

    try:
        compiled_contexts = orchestrator.compile_pipelines(pipeline_definition)
    except ValidationError as e:
        print(f"Pipeline validation failed: {e}")
        # Fix pipeline before execution
    except MemoryTypeError as e:
        print(f"Memory type incompatibility: {e}")
        # Adjust memory types or add conversions

Best Practices
--------------

**Pipeline Design**:
- Group functions by memory type to minimize conversions
- Use special I/O for cross-step communication
- Design for parallel execution across wells

**Function Development**:
- Always use memory type decorators
- Declare special inputs/outputs explicitly
- Follow 3D array output conventions

**Resource Management**:
- Configure appropriate ZARR settings for large datasets
- Set memory limits to trigger automatic ZARR usage
- Use GPU memory types for compute-intensive operations

See Also
--------

- :doc:`../architecture/compilation_system_detailed` - Detailed compilation architecture
- :doc:`../architecture/pipeline_compilation_system` - Compilation system overview
- :doc:`memory_type_integration` - Memory type system integration
- :doc:`../api/orchestrator` - PipelineOrchestrator API documentation
