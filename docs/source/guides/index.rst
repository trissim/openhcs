Integration Guides
==================

These guides explain how OpenHCS systems work together to provide seamless bioimage analysis workflows. Each guide focuses on the integration between major system components.

Complete Examples
-----------------

.. toctree::
   :maxdepth: 2

   complete_examples
   example_corpus_map

**Working Examples**: Comprehensive scripts demonstrating major OpenHCS functionality and practical usage patterns.

:doc:`complete_examples`
:doc:`example_corpus_map`

System Integration Guides
--------------------------

.. toctree::
   :maxdepth: 2

   memory_type_integration
   pipeline_compilation_workflow
   omero_integration
   viewer_management
   fiji_viewer_management
   testing_guide

Memory Type Integration
^^^^^^^^^^^^^^^^^^^^^^^

Learn how OpenHCS automatically converts between NumPy, CuPy, PyTorch, JAX, TensorFlow, and pyclesperanto arrays. Understand GPU device management, zero-copy conversions, and memory type decorators.

:doc:`memory_type_integration`

Pipeline Compilation Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Understand the 5-phase compilation system that transforms pipeline definitions into optimized execution plans. Learn about path planning, materialization, memory contract validation, and GPU resource assignment.

:doc:`pipeline_compilation_workflow`

OMERO Integration
^^^^^^^^^^^^^^^^^

Complete server-side execution support for OpenHCS on OMERO servers with zero data transfer overhead. Learn about virtual backends, multiprocessing-safe connection management, and automatic instance management.

:doc:`omero_integration`

Viewer Management
^^^^^^^^^^^^^^^^^

Learn how to manage Napari and Fiji viewer processes across OpenHCS components. Understand viewer reuse, parallel startup, persistent viewers, and automatic reconnection for real-time visualization workflows.

:doc:`viewer_management`

Fiji Viewer Management
^^^^^^^^^^^^^^^^^^^^^^

Learn how to manage Fiji/ImageJ viewer processes for OpenHCS visualization. Understand PyImageJ integration, automatic hyperstack building, persistent viewers, and ZMQ-based streaming for leveraging the ImageJ ecosystem.

:doc:`fiji_viewer_management`

Testing Guide
^^^^^^^^^^^^^

Comprehensive guide for running OpenHCS integration tests with different configurations. Learn how to test with Napari/Fiji visualizers, OMERO backend, and various execution modes using VSCode test discovery or command-line.

:doc:`testing_guide`

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
