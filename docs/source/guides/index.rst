Integration Guides
==================

These guides explain how OpenHCS systems work together to provide seamless bioimage analysis workflows. Each guide focuses on the integration between major system components.

System Integration Guides
--------------------------

.. toctree::
   :maxdepth: 2

   memory_type_integration
   pipeline_compilation_workflow

Memory Type Integration
^^^^^^^^^^^^^^^^^^^^^^^

Learn how OpenHCS automatically converts between NumPy, CuPy, PyTorch, JAX, TensorFlow, and pyclesperanto arrays. Understand GPU device management, zero-copy conversions, and memory type decorators.

:doc:`memory_type_integration`

Pipeline Compilation Workflow  
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Understand the 5-phase compilation system that transforms pipeline definitions into optimized execution plans. Learn about path planning, materialization, memory contract validation, and GPU resource assignment.

:doc:`pipeline_compilation_workflow`

Quick Reference
---------------

**Memory Type Decorators**:

.. code-block:: python

    from openhcs.core.memory.decorators import numpy, cupy, torch, jax, pyclesperanto

    @numpy
    def cpu_function(image): pass

    @torch(oom_recovery=True)  
    def gpu_function(image): pass

**Pipeline Compilation**:

.. code-block:: python

    # Automatic compilation in orchestrator
    orchestrator = PipelineOrchestrator(
        plate_paths=plate_paths,
        steps=steps,
        global_config=global_config
    )
    
    # 5-phase compilation happens automatically
    orchestrator.run()

**Function Patterns**:

.. code-block:: python

    # Single function
    FunctionStep(func=my_function)
    
    # Function with parameters
    FunctionStep(func=(my_function, {'param': value}))
    
    # Function chain
    FunctionStep(func=[
        (func1, {'param1': value1}),
        (func2, {'param2': value2})
    ])
    
    # Dict pattern (channel-specific)
    FunctionStep(func={
        '1': nuclei_function,
        '2': neurite_function
    }, variable_components=[VariableComponents.CHANNEL])

Integration Patterns
--------------------

**Memory Type + Function Registry**:
Functions are automatically discovered and registered with their memory type contracts, enabling automatic conversion planning during compilation.

**Pipeline Compilation + GPU Management**:
The compilation system assigns GPU resources and validates memory requirements before execution begins.

**Special I/O + VFS System**:
Cross-step communication uses the VFS system for efficient data transfer between pipeline steps.

**Configuration + All Systems**:
The configuration system provides unified settings that affect memory management, compilation, and execution across all components.

See Also
--------

- :doc:`../architecture/index` - Detailed architecture documentation
- :doc:`../api/index` - API reference documentation  
- :doc:`../user_guide/index` - User guides and tutorials
